
"""测试修复 - 直接使用 evaluate.py 的推理逻辑"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import torch
from transformers import BertTokenizerFast as BertTokenizer
from seqeval.metrics import f1_score as seqeval_f1

from homework_src.dataset import build_label_schema, build_dataloaders
from homework_src.model import build_model

BERT_PATH = ROOT / "pretrain_models" / "bert-base-chinese"
DATA_DIR = ROOT / "data" / "peoples_daily"


def test_evaluate_logic():
    """测试用 evaluate.py 的逻辑进行评估"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    # 标签体系
    labels, label2id, id2label = build_label_schema(DATA_DIR)
    num_labels = len(labels)
    print(f"BIO 标签数：{num_labels}（{labels}）")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))

    # 只加载少量样本
    max_samples = {"train": 200, "val": 50, "test": 50}
    train_loader, val_loader, _ = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=16,
        max_length=128,
        data_dir=DATA_DIR,
        max_samples=max_samples,
    )

    # 创建模型
    model = build_model(
        use_crf=True,
        bert_path=str(BERT_PATH),
        num_labels=num_labels,
        dropout=0.1,
    ).to(device)

    # 尝试用 evaluate.py 的方式进行推理
    print("\n" + "="*80)
    print("用 evaluate.py 的逻辑进行推理")
    print("="*80)

    model.eval()
    all_preds = []
    all_golds = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)

            labels_list = labels.cpu().tolist()
            attention_list = attention_mask.cpu().tolist()

            for i in range(len(input_ids)):
                gold_seq = []
                pred_seq = []
                token_labels = labels_list[i]

                # 这是 evaluate.py 的对齐逻辑
                mask_true_indices = [j for j, mask_val in enumerate(attention_list[i]) if mask_val]
                pos_to_pred_id = {}
                for mask_idx, orig_j in enumerate(mask_true_indices):
                    if mask_idx < len(pred_ids_list[i]):
                        pos_to_pred_id[orig_j] = pred_ids_list[i][mask_idx]

                for j, gold_id in enumerate(token_labels):
                    if gold_id == -100:
                        continue
                    gold_seq.append(id2label[gold_id])
                    pred_id = pos_to_pred_id.get(j, 0)
                    pred_seq.append(id2label.get(pred_id, "O"))

                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    # 统计一下数据
    print(f"\n总序列数: {len(all_golds)}")
    print(f"前3个序列:")
    for i in range(3):
        print(f"\n序列 {i}:")
        print(f"  Gold: {all_golds[i]}")
        print(f"  Pred: {all_preds[i]}")

    # 统计有实体的样本
    has_entity_count = 0
    for gold in all_golds:
        for tag in gold:
            if tag != 'O':
                has_entity_count += 1
                break
    print(f"\n有实体的样本数: {has_entity_count}/{len(all_golds)}")

    # 计算 F1
    f1 = seqeval_f1(all_golds, all_preds)
    print(f"\nseqeval F1: {f1:.4f}")

    return f1


if __name__ == "__main__":
    test_evaluate_logic()
