"""
在测试集上评估 BERT NER 模型 - 作业版

评估 BertNER 和 BertCRFNER，对比非法序列统计

使用方式：
  python homework_src/evaluate.py   --show_illegal_details           # 评估 BERT+Linear
  python homework_src/evaluate.py --use_crf  --show_illegal_details  # 评估 BERT+CRF
"""

import os

# 兼容部分环境中的 OpenMP 重复加载问题。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

# 添加项目根目录到模块搜索路径，确保脚本独立运行时也能导入本地模块。
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import json
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast as BertTokenizer
from seqeval.metrics import (
    f1_score, precision_score, recall_score,
    classification_report as seqeval_report,
)

from homework_src.dataset import PeoplesDailyDataset, load_records
from homework_src.model import build_model

BERT_PATH = ROOT / "pretrain_models" / "bert-base-chinese"
DATA_DIR = ROOT / "data" / "peoples_daily"
CKPT_DIR = ROOT / "homework_outputs" / "checkpoints"
LOG_DIR = ROOT / "homework_outputs" / "logs"
# 评估时用于过滤超长样本的阈值。
EVAL_FILTER_MAX_LENGTH = 128


def count_illegal_sequences(
    pred_seqs: list[list[str]],
    gold_seqs: list[list[str]] | None = None,
    raw_records: list[dict] | None = None,
    include_details: bool = False,
) -> dict:
    """统计非法 BIO 序列数量，并返回非法序列明细。

    非法类型：
      - illegal_start：序列以 I-X 开头
      - illegal_transition：B-X 或 I-X 后面跟 I-Y（X≠Y）
    """
    # 初始化非法序列统计字典。
    stats = {"illegal_start": 0, "illegal_transition": 0, "total_seqs": len(pred_seqs)}
    # 若需要输出明细，则把每条非法样本的信息放到这里。
    illegal_cases = []
    illegal_sequence_count = 0

    for seq_idx, seq in enumerate(pred_seqs):
        if not seq:
            continue

        violations = []
        gold_seq = gold_seqs[seq_idx] if gold_seqs is not None else None

        # 检查开头
        # BIO 序列若以 I- 开头，视为非法起始。
        if seq[0].startswith("I-"):
            stats["illegal_start"] += 1
            violations.append({
                "type": "illegal_start",
                "position": 0,
                "prev": None,
                "curr": seq[0],
            })

        # 检查转移
        # 从第二个标签开始检查相邻转移是否合法。
        for i in range(1, len(seq)):
            prev, curr = seq[i - 1], seq[i]
            if curr.startswith("I-"):
                curr_type = curr[2:]
                if prev == "O":
                    stats["illegal_transition"] += 1
                    violations.append({
                        "type": "illegal_transition",
                        "position": i,
                        "prev": prev,
                        "curr": curr,
                    })
                elif prev.startswith("B-") or prev.startswith("I-"):
                    prev_type = prev[2:]
                    if prev_type != curr_type:
                        stats["illegal_transition"] += 1
                        violations.append({
                            "type": "illegal_transition",
                            "position": i,
                            "prev": prev,
                            "curr": curr,
                        })

        if violations:
            illegal_sequence_count += 1

            if include_details:
                raw_tokens = raw_records[seq_idx]["tokens"] if raw_records is not None else None
                label_errors = []
                if raw_tokens is not None and gold_seq is not None:
                    for pos, (token, gold_label, pred_label) in enumerate(zip(raw_tokens, gold_seq, seq)):
                        if gold_label != pred_label:
                            label_errors.append({
                                "position": pos,
                                "token": token,
                                "gold": gold_label,
                                "pred": pred_label,
                            })

                illegal_cases.append({
                    "sequence_index": seq_idx,
                    "violations": violations,
                    "pred_seq": seq,
                    "gold_seq": gold_seq,
                    "raw_text": "".join(raw_tokens) if raw_tokens is not None else None,
                    "label_errors": label_errors,
                })

    total_illegal = stats["illegal_start"] + stats["illegal_transition"]
    stats["total_illegal"] = total_illegal
    stats["illegal_sequence_count"] = illegal_sequence_count
    if include_details:
        stats["illegal_cases"] = illegal_cases
    return stats


def print_illegal_sequences(illegal_stats: dict):
    """打印非法 BIO 序列明细。"""
    illegal_cases = illegal_stats.get("illegal_cases", [])
    print(f"【非法 BIO 序列明细】共 {len(illegal_cases)} 条")

    if not illegal_cases:
        print("  无非法序列")
        return

    for case in illegal_cases:
        seq_idx = case["sequence_index"]
        violation_desc = []
        for violation in case["violations"]:
            if violation["type"] == "illegal_start":
                violation_desc.append(
                    f"位置{violation['position']}：以 {violation['curr']} 开头"
                )
            else:
                violation_desc.append(
                    f"位置{violation['position']}：{violation['prev']} -> {violation['curr']}"
                )

        print(f"  序列 {seq_idx}:")
        print(f"    违规原因: {'; '.join(violation_desc)}")
        if case.get("raw_text") is not None:
            print(f"    Text: {case['raw_text']}")
        label_errors = case.get("label_errors", [])
        if label_errors:
            print("    预测错误标签:")
            for error in label_errors:
                print(
                    f"      位置{error['position']}: token={error['token']} | "
                    f"gold={error['gold']} | pred={error['pred']}"
                )
        print(f"    Pred: {case['pred_seq']}")
        if case["gold_seq"] is not None:
            print(f"    Gold: {case['gold_seq']}")


def run_inference(
    model,
    loader,
    id2label: dict,
    device: torch.device,
    use_crf: bool,
) -> tuple[list[list[str]], list[list[str]]]:
    """在 loader 上推理，返回 (all_preds, all_golds)，每个元素为字符串标签列表。"""
    # 切换模型到评估模式。
    model.eval()
    all_preds = []
    all_golds = []

    # 评估时不跟踪梯度。
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            if use_crf:
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids, labels)
            else:
                logits, _ = model(input_ids, attention_mask, token_type_ids)
                pred_ids_list = logits.argmax(dim=-1).tolist()

            # 将标签移回 CPU 便于后续逐样本处理。
            labels_list = labels.cpu().tolist()

            for i in range(len(input_ids)):
                gold_seq = []
                pred_seq = []
                token_labels = labels_list[i]

                # CRF 返回的是压缩后的有效标签位置序列，需要额外对齐。
                if use_crf:
                    pred_ids = pred_ids_list[i]
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
                    # 线性头的预测结果与原 token 位置一一对应。
                    for j, gold_id in enumerate(token_labels):
                        if gold_id == -100:
                            continue
                        gold_seq.append(id2label[gold_id])
                        pred_seq.append(id2label.get(pred_ids_list[i][j], "O"))

                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    return all_preds, all_golds


def build_eval_loader(
    split: str,
    tokenizer: BertTokenizer,
    label2id: dict,
    batch_size: int,
    max_length: int,
    max_samples: int | None,
) -> tuple[DataLoader, int, int, list[dict]]:
    """构建评估 DataLoader，并过滤原始 tokens 长度超过 128 的样本。"""
    records = load_records(split, DATA_DIR, max_samples)
    total_before_filter = len(records)
    # 过滤掉长度超过评估上限的样本，避免与训练设定不一致。
    filtered_records = [row for row in records if len(row["tokens"]) <= EVAL_FILTER_MAX_LENGTH]
    filtered_out = total_before_filter - len(filtered_records)

    dataset = PeoplesDailyDataset(filtered_records, tokenizer, label2id, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, total_before_filter, filtered_out, filtered_records


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

    # ✅ 使用checkpoint里保存的标签映射！
    # 读取标签到 id 的映射。
    label2id = ckpt["label2id"]
    # 读取 id 到标签的映射。
    id2label = ckpt["id2label"]
    # 计算标签总数。
    num_labels = len(label2id)
    print(f"加载 checkpoint（epoch={ckpt['epoch']}，val_f1={ckpt['val_entity_f1']:.4f}）")
    print(f"标签数: {num_labels}")
    print(f"标签: {[id2label[i] for i in range(num_labels)]}")

    model = build_model(
        use_crf=args.use_crf,
        bert_path=str(args.bert_path),
        num_labels=num_labels,
        id2label=id2label,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])

    tokenizer = BertTokenizer.from_pretrained(str(args.bert_path))
    model_max_length = ckpt["args"].get("max_length", 128)
    split_max_samples = args.val_samples if args.split == "validation" else args.test_samples
    loader, total_before_filter, filtered_out, filtered_records = build_eval_loader(
        split=args.split,
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=model_max_length,
        max_samples=split_max_samples,
    )
    split_name = args.split

    print(f"\n正在在 [{split_name}] 集上推理...")
    print(
        f"评估前样本数：{total_before_filter}，过滤掉长度 > {EVAL_FILTER_MAX_LENGTH} 的样本：{filtered_out}，"
        f"实际参与评估：{len(loader.dataset)}"
    )
    all_preds, all_golds = run_inference(model, loader, id2label, device, args.use_crf)

    # 调试输出
    # 打印前几个样本的预测结果，便于快速人工检查。
    print(f"\n样本预测:")
    for i in range(min(3, len(all_golds))):
        print(f"Sample {i}:")
        print(f"  Gold: {all_golds[i]}")
        print(f"  Pred: {all_preds[i]}")

    p = precision_score(all_golds, all_preds) if all_golds else 0.0
    r = recall_score(all_golds, all_preds) if all_golds else 0.0
    f1 = f1_score(all_golds, all_preds) if all_golds else 0.0

    print("\n" + "=" * 70)
    print(f"模型：{'BERT + CRF' if args.use_crf else 'BERT + Linear'}  |  评估集：{split_name}")
    print("=" * 70)
    print(f"Entity-level Precision: {p:.4f}")
    print(f"Entity-level Recall:    {r:.4f}")
    print(f"Entity-level F1:        {f1:.4f}")

    print("\n【逐类型 F1】")
    if all_golds:
        print(seqeval_report(all_golds, all_preds, digits=4))
    else:
        print("(无数据)")

    # 非法序列统计
    # 统计预测序列中的 BIO 非法模式。
    illegal_stats = count_illegal_sequences(
        all_preds,
        all_golds,
        filtered_records,
        include_details=args.show_illegal_details,
    )
    print("【非法 BIO 序列统计】")
    print(f"  总序列数：{illegal_stats['total_seqs']}")
    print(f"  非法开头 (I-X开头)：{illegal_stats['illegal_start']} 条")
    print(f"  非法转移 (B-X/I-X → I-Y, X≠Y)：{illegal_stats['illegal_transition']} 条")
    print(f"  合计非法序列：{illegal_stats['total_illegal']} 条")
    pct = illegal_stats['total_illegal'] / max(illegal_stats['total_seqs'], 1) * 100
    if args.use_crf:
        if illegal_stats['total_illegal'] == 0:
            print("  → CRF Viterbi解码：非法序列 0 条 ✓（转移矩阵已充分学习约束）")
        else:
            print(f"  → CRF非法序列 {illegal_stats['total_illegal']} 条（{pct:.1f}%）")
            print(f"  → 提示：训练 epoch不足时转移矩阵尚未收敛；充分训练后可降到0")
    else:
        print(f"  → 线性头约 {pct:.1f}% 的序列含非法转移，充分训练的CRF可完全消除")

    if args.show_illegal_details:
        print_illegal_sequences(illegal_stats)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "model": "BERT + CRF" if args.use_crf else "BERT + Linear",
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--train_samples", type=int, default=100, help="训练集样本数")
    parser.add_argument("--val_samples", type=int, default=500, help="验证集样本数")
    parser.add_argument("--test_samples", type=int, default=500, help="测试集样本数")
    parser.add_argument(
        "--show_illegal_details",
        action="store_true",
        help="打印非法序列的 text、错误标签位置以及 Pred/Gold 对比",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
