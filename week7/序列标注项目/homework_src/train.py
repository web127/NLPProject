"""
BERT NER 训练脚本 - 作业版

使用 peoples_daily 数据集进行训练
支持 BertNER（线性头）和 BertCRFNER（CRF头）

使用方式：
  python homework_src/train.py              # 训练 BERT+Linear
  python homework_src/train.py --use_crf    # 训练 BERT+CRF
"""

import os

# 兼容部分 macOS 环境下 OpenMP 重复加载报错。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

# 添加项目根目录到路径，保证直接运行脚本时也能正确导入本项目模块。
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import json
import time
import argparse

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizerFast as BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from seqeval.metrics import f1_score as seqeval_f1

from homework_src.dataset import build_label_schema, build_dataloaders
from homework_src.model import build_model


ROOT = Path(__file__).parent.parent
# 预训练 BERT 路径。
BERT_PATH = ROOT / "pretrain_models" / "bert-base-chinese"
# 数据集路径。
DATA_DIR = ROOT / "data" / "peoples_daily"
# 模型 checkpoint 输出目录。
CKPT_DIR = ROOT / "homework_outputs" / "checkpoints"
# 训练日志输出目录。
LOG_DIR = ROOT / "homework_outputs" / "logs"


def evaluate_epoch(
        model: nn.Module,
        loader,
        id2label: dict,
        device: torch.device,
        use_crf: bool,
        grad_accum: int,
) -> tuple[float, float]:
    """在 loader 上评估，返回 (avg_loss, entity_f1)。"""
    # 切换到评估模式，关闭 dropout 等训练时行为。
    model.eval()
    # 累积 loss 以便最后计算平均值。
    total_loss = 0.0
    all_preds: list[list[str]] = []
    all_golds: list[list[str]] = []

    # 评估阶段不需要梯度。
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            # CRF 与线性头的前向和解码逻辑略有不同。
            if use_crf:
                emissions, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids, labels)
            else:
                logits, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = logits.argmax(dim=-1).tolist()

            total_loss += loss.item() / grad_accum

            labels_np = labels.cpu().tolist()
            # attention mask 也转为列表；当前函数中未直接使用，但保留便于调试扩展。
            mask_np = attention_mask.cpu().tolist()
            for i in range(len(input_ids)):
                gold_seq = []
                pred_seq = []
                token_labels = labels_np[i]
                pred_ids = pred_ids_list[i]

                # CRF decode 返回的是压缩后的有效标签序列，需要单独对齐。
                if use_crf:
                    pred_ptr = 0
                    for j, gold_id in enumerate(token_labels):
                        if gold_id == -100:
                            continue
                        gold_seq.append(id2label[gold_id])
                        pred_id = pred_ids[pred_ptr] if pred_ptr < len(pred_ids) else 0
                        pred_seq.append(id2label.get(pred_id, "O"))
                        pred_ptr += 1
                else:
                    # Linear 模式：直接索引
                    for j, gold_id in enumerate(token_labels):
                        if gold_id == -100:
                            continue
                        gold_seq.append(id2label[gold_id])
                        pred_seq.append(id2label.get(pred_ids[j], "O"))

                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    avg_loss = total_loss / len(loader)
    entity_f1 = seqeval_f1(all_golds, all_preds)
    return avg_loss, entity_f1


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    grad_accum: int,
) -> float:
    # 切换到训练模式。
    model.train()
    # 累积整个 epoch 的原始 loss。
    total_loss = 0.0
    # 训练开始前清空优化器中的梯度。
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        _, loss = model(input_ids, attention_mask, token_type_ids, labels)

        # 梯度累积时，将 loss 按累积步数缩放后再反传。
        (loss / grad_accum).backward()
        total_loss += loss.item()

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # 处理最后不足 grad_accum 的批次
    remainder = len(loader) % grad_accum
    if remainder != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(loader)
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    # 标签体系
    # 从数据目录读取标签定义，并构建映射关系。
    labels, label2id, id2label = build_label_schema(DATA_DIR)
    num_labels = len(labels)
    print(f"BIO 标签数：{num_labels}（{labels}）")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(str(args.bert_path))

    # DataLoader
    # 整理各数据划分允许使用的样本数上限。
    max_samples = {
        "train": args.train_samples,
        "val": args.val_samples,
        "test": args.test_samples,
    }
    train_loader, val_loader, _ = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=args.max_length,
        data_dir=DATA_DIR,
        max_samples=max_samples,
    )

    # 模型
    # 根据命令行选项构建线性头或 CRF 模型。
    model = build_model(
        use_crf=args.use_crf,
        bert_path=str(args.bert_path),
        num_labels=num_labels,
        dropout=args.dropout,
        id2label=id2label,
    ).to(device)

    # 分层学习率：移除无用dropout参数
    # 收集 BERT 主干参数。
    bert_params = list(model.bert.parameters())
    # 收集分类头参数。
    head_params = list(model.classifier.parameters())
    # 若使用 CRF，再把 CRF 层参数也加入头部参数组。
    if args.use_crf:
        head_params += list(model.crf.parameters())
    optimizer = AdamW(
        [
            {"params": bert_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * args.head_lr_mult},
        ],
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"训练步数：{total_steps}，预热步数：{warmup_steps}")

    run_tag = "crf" if args.use_crf else "linear"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"best_{run_tag}.pt"
    log_path = LOG_DIR / f"train_{run_tag}.json"

    best_f1 = -1.0
    log_records = []

    print(f"\n开始训练（{'BERT+CRF' if args.use_crf else 'BERT+Linear'}）...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, args.epochs, args.grad_accum
        )
        # 补齐grad_accum参数
        # 在验证集上评估模型性能。
        val_loss, val_f1 = evaluate_epoch(model, val_loader, id2label, device, args.use_crf, args.grad_accum)
        elapsed = time.time() - t0

        # 打印本轮训练与验证指标。
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_entity_f1={val_f1:.4f} | "
            f"time={elapsed:.0f}s"
        )

        # 把本轮关键指标写入日志列表。
        log_records.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6) if isinstance(train_loss, float) else 0.0,
            "val_loss": round(val_loss, 6) if isinstance(val_loss, float) else 0.0,
            "val_entity_f1": round(val_f1, 6) if isinstance(val_f1, float) else 0.0,
            "elapsed_s": round(elapsed, 1),
        })

        if val_f1 > best_f1 :
            best_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "use_crf": args.use_crf,
                    "state_dict": model.state_dict(),
                    "val_entity_f1": val_f1,
                    "label2id": label2id,
                    "id2label": id2label,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  ★ 新最优 F1={val_f1:.4f}，已保存 → {ckpt_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成！最优 val_entity_f1={best_f1:.4f}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  训练日志: {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="训练 BERT NER 模型 - 作业版")
    parser.add_argument("--use_crf", action="store_true", help="使用 CRF 层（否则使用线性头）")
    parser.add_argument("--bert_path", type=Path, default=BERT_PATH)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5, help="BERT 层学习率")
    parser.add_argument("--head_lr_mult", type=float, default=5.0, help="分类头学习率倍数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train_samples", type=int, default=100, help="训练集样本数")
    parser.add_argument("--val_samples", type=int, default=50, help="验证集样本数")
    parser.add_argument("--test_samples", type=int, default=1000, help="测试集样本数")
    return parser.parse_args()


if __name__ == "__main__":
    main()
