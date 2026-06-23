
"""调试 CRF 预测对齐问题"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import torch
from transformers import BertTokenizerFast as BertTokenizer

from homework_src.dataset import build_label_schema, build_dataloaders
from homework_src.model import build_model

BERT_PATH = ROOT / "pretrain_models" / "bert-base-chinese"
DATA_DIR = ROOT / "data" / "peoples_daily"


def debug_alignment():
    """调试对比两种对齐方式"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    # 标签体系
    labels, label2id, id2label = build_label_schema(DATA_DIR)
    num_labels = len(labels)
    print(f"BIO 标签数：{num_labels}（{labels}）")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))

    # 只加载少量样本
    max_samples = {"train": 10, "val": 5, "test": 5}
    train_loader, val_loader, _ = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=2,
        max_length=128,
        data_dir=DATA_DIR,
        max_samples=max_samples,
    )

    # 创建一个小模型用于测试
    model = build_model(
        use_crf=True,
        bert_path=str(BERT_PATH),
        num_labels=num_labels,
        dropout=0.1,
    ).to(device)
    model.eval()

    print("\n" + "="*80)
    print("测试 CRF decode 输出和对齐逻辑")
    print("="*80)

    # 取一个 batch
    batch = next(iter(val_loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    labels = batch["labels"].to(device)

    with torch.no_grad():
        emissions, loss = model(input_ids, attention_mask, token_type_ids, labels)
        pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)

    print(f"\n【decode 返回的 pred_ids_list】")
    print(f"  类型: {type(pred_ids_list)}")
    print(f"  长度: {len(pred_ids_list)} (batch size)")
    for i, pred_seq in enumerate(pred_ids_list):
        print(f"  序列 {i}: 长度={len(pred_seq)}, 内容={pred_seq}")

    print(f"\n【attention_mask 形状】: {attention_mask.shape}")
    print(f"【labels 形状】: {labels.shape}")

    # 对比两种对齐方式
    print("\n" + "="*80)
    print("对比两种对齐方式")
    print("="*80)

    labels_np = labels.cpu().tolist()
    mask_np = attention_mask.cpu().tolist()

    for i in range(len(input_ids)):
        print(f"\n--- 样本 {i} ---")
        token_labels = labels_np[i]
        valid_mask = mask_np[i]
        pred_ids = pred_ids_list[i]

        # 方式1: train.py 中的方式（有问题）
        print("\n方式1: train.py 的对齐逻辑（pred_idx 递增）")
        gold_seq1 = []
        pred_seq1 = []
        pred_idx = 0
        for j, gold_id in enumerate(token_labels):
            if valid_mask[j] == 0:
                continue
            if gold_id == -100:
                continue
            gold_seq1.append(id2label[gold_id])
            if pred_idx < len(pred_ids):
                pred_seq1.append(id2label[pred_ids[pred_idx]])
            else:
                pred_seq1.append("O")
            pred_idx += 1
        print(f"  Gold: {gold_seq1}")
        print(f"  Pred: {pred_seq1}")
        print(f"  对齐长度: gold={len(gold_seq1)}, pred={len(pred_seq1)}")

        # 方式2: evaluate.py 中的方式（正确）
        print("\n方式2: evaluate.py 的对齐逻辑（mask位置映射）")
        # 先收集所有 mask=True 的位置索引
        mask_true_indices = [j for j, mask_val in enumerate(valid_mask) if mask_val]
        print(f"  mask_true_indices (attention_mask=1的位置): {mask_true_indices}")
        # 建立映射：原位置 j -> 预测的 id
        pos_to_pred_id = {}
        for mask_idx, orig_j in enumerate(mask_true_indices):
            if mask_idx < len(pred_ids):
                pos_to_pred_id[orig_j] = pred_ids[mask_idx]
        print(f"  pos_to_pred_id: {pos_to_pred_id}")

        gold_seq2 = []
        pred_seq2 = []
        for j, gold_id in enumerate(token_labels):
            if gold_id == -100:
                continue
            gold_seq2.append(id2label[gold_id])
            pred_id = pos_to_pred_id.get(j, 0)
            pred_seq2.append(id2label.get(pred_id, "O"))
        print(f"  Gold: {gold_seq2}")
        print(f"  Pred: {pred_seq2}")
        print(f"  对齐长度: gold={len(gold_seq2)}, pred={len(pred_seq2)}")

        # 验证 gold_seq1 和 gold_seq2 相同
        assert gold_seq1 == gold_seq2, "Gold 序列应该一致"

        # 统计有效标签位置
        print(f"\n  有效标签位置 (gold_id != -100):")
        valid_label_positions = [j for j, gold_id in enumerate(token_labels) if gold_id != -100]
        print(f"    位置索引: {valid_label_positions}")
        print(f"    这些位置在 pos_to_pred_id 中吗?")
        for pos in valid_label_positions:
            in_map = pos in pos_to_pred_id
            print(f"      位置 {pos}: {'✓' if in_map else '✗'}")


if __name__ == "__main__":
    debug_alignment()
