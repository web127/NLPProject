"""
在测试集上评估 BERT NER 模型 - 作业版

评估 BertNER 和 BertCRFNER，对比非法序列统计

使用方式：
  python homework_src/evaluate.py              # 评估 BERT+Linear
  python homework_src/evaluate.py --use_crf    # 评估 BERT+CRF
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from transformers import BertTokenizer
from seqeval.metrics import (
    f1_score, precision_score, recall_score,
    classification_report as seqeval_report,
)

from homework_src.dataset import build_label_schema, build_dataloaders
from homework_src.model import build_model

ROOT = Path(__file__).parent.parent
BERT_PATH = ROOT / "pretrain_models" / "bert-base-chinese"
DATA_DIR = ROOT / "data" / "peoples_daily"
CKPT_DIR = ROOT / "homework_outputs" / "checkpoints"
LOG_DIR = ROOT / "homework_outputs" / "logs"


def count_illegal_sequences(pred_seqs: list[list[str]]) -> dict:
    """统计非法 BIO 序列数量。

    非法类型：
      - illegal_start：序列以 I-X 开头
      - illegal_transition：B-X 或 I-X 后面跟 I-Y（X≠Y）
    """
    stats = {"illegal_start": 0, "illegal_transition": 0, "total_seqs": len(pred_seqs)}
    for seq in pred_seqs:
        if not seq:
            continue
        # 检查开头
        if seq[0].startswith("I-"):
            stats["illegal_start"] += 1

        # 检查转移
        for i in range(1, len(seq)):
            prev, curr = seq[i-1], seq[i]
            if curr.startswith("I-"):
                curr_type = curr[2:]
                if prev == "O":
                    stats["illegal_transition"] += 1
                elif prev.startswith("B-") or prev.startswith("I-"):
                    prev_type = prev[2:]
                    if prev_type != curr_type:
                        stats["illegal_transition"] += 1

    total_illegal = stats["illegal_start"] + stats["illegal_transition"]
    stats["total_illegal"] = total_illegal
    return stats


def run_inference(
    model,
    loader,
    id2label: dict,
    device: torch.device,
    use_crf: bool,
) -> tuple[list[list[str]], list[list[str]]]:
    """在 loader 上推理，返回 (all_preds, all_golds)，每个元素为字符串标签列表。"""
    model.eval()
    all_preds = []
    all_golds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            if use_crf:
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
            else:
                logits, _ = model(input_ids, attention_mask, token_type_ids)
                pred_ids_list = logits.argmax(dim=-1).tolist()

            labels_list = labels.cpu().tolist()

            for i in range(len(input_ids)):
                gold_seq = []
                pred_seq = []
                token_labels = labels_list[i]

                for j, gold_id in enumerate(token_labels):
                    if gold_id == -100:
                        continue
                    gold_seq.append(id2label[gold_id])
                    if use_crf:
                        pred_seq.append(id2label.get(pred_ids_list[i][j] if j < len(pred_ids_list[i]) else 0, "O"))
                    else:
                        pred_seq.append(id2label.get(pred_ids_list[i][j], "O"))

                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    return all_preds, all_golds


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_tag = "crf" if args.use_crf else "linear"
    ckpt_path = CKPT_DIR / f"best_{run_tag}.pt"

    if not ckpt_path.exists():
        print(f"找不到 checkpoint：{ckpt_path}")
        print(f"请先运行：python homework_src/train.py {'--use_crf' if args.use_crf else ''}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    labels, label2id, id2label = build_label_schema(DATA_DIR)

    model = build_model(
        use_crf=args.use_crf,
        bert_path=str(args.bert_path),
        num_labels=len(labels),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"加载 checkpoint（epoch={ckpt['epoch']}，val_f1={ckpt['val_entity_f1']:.4f}）")

    tokenizer = BertTokenizer.from_pretrained(str(args.bert_path))
    max_samples = {
        "train": args.train_samples,
        "val": args.val_samples,
        "test": args.test_samples,
    }
    _, val_loader, test_loader = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=ckpt["args"].get("max_length", 128),
        data_dir=DATA_DIR,
        max_samples=max_samples,
    )
    loader = val_loader if args.split == "validation" else test_loader
    split_name = args.split

    print(f"\n正在在 [{split_name}] 集上推理...")
    all_preds, all_golds = run_inference(model, loader, id2label, device, args.use_crf)

    # seqeval entity-level 指标
    p = precision_score(all_golds, all_preds) if all_golds else 0.0
    r = recall_score(all_golds, all_preds) if all_golds else 0.0
    f1 = f1_score(all_golds, all_preds) if all_golds else 0.0

    print("\n" + "=" * 70)
    print(f"模型：{'BERT+CRF' if args.use_crf else 'BERT+Linear'}  |  评估集：{split_name}")
    print("=" * 70)
    print(f"Entity-level Precision: {p:.4f}")
    print(f"Entity-level Recall:    {r:.4f}")
    print(f"Entity-level F1:        {f1:.4f}")

    print("\n【逐类型 F1】")
    if all_golds:
        print(seqeval_report(all_golds, all_preds, digits=4))
    else:
        print("(无数据)")

    # 非法序列统计（核心教学对比点）
    illegal_stats = count_illegal_sequences(all_preds)
    print("【非法 BIO 序列统计】")
    print(f"  总序列数：{illegal_stats['total_seqs']}")
    print(f"  非法开头（I-X 开头）：{illegal_stats['illegal_start']} 条")
    print(f"  非法转移（B-X/I-X → I-Y, X≠Y）：{illegal_stats['illegal_transition']} 条")
    print(f"  合计非法序列：{illegal_stats['total_illegal']} 条")
    pct = illegal_stats["total_illegal"] / max(illegal_stats['total_seqs'], 1) * 100
    if args.use_crf:
        if illegal_stats["total_illegal"] == 0:
            print("  → CRF Viterbi 解码：非法序列 0 条 ✓（转移矩阵已充分学习约束）")
        else:
            print(f"  → CRF 非法序列 {illegal_stats['total_illegal']} 条（{pct:.1f}%）")
            print(f"  → 提示：训练 epoch 不足时转移矩阵尚未收敛；充分训练后可降至 0")
    else:
        print(f"  → 线性头约 {pct:.1f}% 的序列含非法转移，充分训练的 CRF 可完全消除")

    # 保存评估结果 JSON
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "model": "BERT+CRF" if args.use_crf else "BERT+Linear",
        "split": split_name,
        "precision": round(p, 6) if isinstance(p, float) else 0.0,
        "recall": round(r, 6) if isinstance(r, float) else 0.0,
        "f1": round(f1, 6) if isinstance(f1, float) else 0.0,
        "illegal_stats": illegal_stats,
    }
    out_path = LOG_DIR / f"eval_{run_tag}_{split_name}.json"
    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存 → {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="评估 BERT NER 模型 - 作业版")
    parser.add_argument("--use_crf", action="store_true")
    parser.add_argument("--bert_path", type=Path, default=BERT_PATH)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    # 数据集大小参数
    parser.add_argument("--train_samples", type=int, default=2000, help="训练集样本数")
    parser.add_argument("--val_samples", type=int, default=500, help="验证集样本数")
    parser.add_argument("--test_samples", type=int, default=500, help="测试集样本数")
    return parser.parse_args()


if __name__ == "__main__":
    main()
