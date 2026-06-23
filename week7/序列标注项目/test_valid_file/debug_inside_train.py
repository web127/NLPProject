
"""直接调试 evaluate_epoch 的对齐"""

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


def debug_alignment_detailed():
    """详细调试对齐"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    # 标签体系
    labels, label2id, id2label = build_label_schema(DATA_DIR)
    num_labels = len(labels)
    print(f"BIO 标签数：{num_labels}（{labels}）")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))

    # 加载少量数据
    max_samples = {"train": 10, "val": 10, "test": 10}
    train_loader, val_loader, _ = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=1,
        max_length=128,
        data_dir=DATA_DIR,
        max_samples=max_samples,
    )

    # 加载之前训练好的 checkpoint（从 outputs/logs/train_crf.json 看之前是能用的）
    # 先确认数据集，然后训练一个简单的
    model = build_model(
        use_crf=True,
        bert_path=str(BERT_PATH),
        num_labels=num_labels,
        dropout=0.1,
    ).to(device)

    # 直接测试对齐逻辑
    print("\n" + "="*80)
    print("测试对齐逻辑")
    print("="*80)

    model.eval()
    batch = next(iter(val_loader))

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    labels = batch["labels"].to(device)

    print(f"input_ids shape: {input_ids.shape}")
    print(f"labels shape: {labels.shape}")

    with torch.no_grad():
        emissions, _ = model(input_ids, attention_mask, token_type_ids)
        pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)

    print(f"\npred_ids_list 类型: {type(pred_ids_list)}")
    print(f"pred_ids_list 长度: {len(pred_ids_list)}")
    print(f"pred_ids_list[0] 内容: {pred_ids_list[0]}")
    print(f"pred_ids_list[0] 长度: {len(pred_ids_list[0])}")

    # 对比 attention_mask=1 的位置数量
    mask_list = attention_mask.cpu().tolist()[0]
    mask_true_count = sum(1 for m in mask_list if m == 1)
    print(f"\nmask_true_count (attention_mask=1 的数量): {mask_true_count}")

    # 对比 decode 输出的长度
    print(f"decode 输出长度: {len(pred_ids_list[0])}")
    assert len(pred_ids_list[0]) == mask_true_count, "decode 输出长度应该等于 mask=1 的数量"

    # 获取有效标签数量
    label_list = labels.cpu().tolist()[0]
    valid_label_count = sum(1 for l in label_list if l != -100)
    print(f"valid_label_count (label != -100 的数量): {valid_label_count}")

    # 详细看对齐
    print(f"\n" + "="*80)
    print(f"详细对齐分析")
    print(f"="*80)

    # 1. 收集 mask=True 的位置
    mask_true_indices = [j for j, mask_val in enumerate(mask_list) if mask_val == 1]
    print(f"\nmask_true_indices:")
    print(f"  数量: {len(mask_true_indices)}")
    print(f"  内容: {mask_true_indices}")

    # 2. 建立位置 -> 预测id 的映射
    pos_to_pred_id = {}
    for mask_idx, orig_j in enumerate(mask_true_indices):
        if mask_idx < len(pred_ids_list[0]):
            pos_to_pred_id[orig_j] = pred_ids_list[0][mask_idx]
    print(f"\npos_to_pred_id:")
    print(f"  数量: {len(pos_to_pred_id)}")

    # 3. 对齐
    gold_seq = []
    pred_seq = []

    print(f"\n对齐过程:")
    print(f"  {'j':<4} {'gold_id':<8} {'in_map':<6} {'pred_id':<8}")

    for j, gold_id in enumerate(label_list):
        if gold_id == -100:
            continue
        in_map = j in pos_to_pred_id
        pred_id = pos_to_pred_id.get(j, -1)
        print(f"  {j:<4} {gold_id:<8} {str(in_map):<6} {pred_id:<8}")
        gold_seq.append(id2label[gold_id])
        pred_seq.append(id2label.get(pred_id, "O"))

    print(f"\n最终:")
    print(f"  Gold: {gold_seq}")
    print(f"  Pred: {pred_seq}")

    # 试试另一种对齐方式 - 先收集 label!=-100 的位置，然后预测也只取对应位置
    print(f"\n" + "="*80)
    print("另一种对齐思路")
    print("="*80)

    valid_label_positions = [j for j, l in enumerate(label_list) if l != -100]
    print(f"valid_label_positions: {valid_label_positions}")

    # 创建一个列表：只有在 valid_label_positions 中的位置才取预测
    # 首先，mask_true_indices 是所有被 mask 住的位置（包括 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>, [SEP] 等）
    # pred_ids_list 是按 mask_true_indices 的顺序预测的

    # 现在要找到：valid_label_positions 中的每一个位置，在 mask_true_indices 中的索引
    # 因为 pred_ids_list 正是按 mask_true_indices 的顺序输出的！

    # 建立从位置 j 到其在 mask_true_indices 中的索引
    j_to_mask_idx = {j:i for i,j in enumerate(mask_true_indices)}

    print(f"\n对齐:")
    gold_seq2 = []
    pred_seq2 = []

    for j in valid_label_positions:
        gold_id = label_list[j]
        mask_idx = j_to_mask_idx.get(j)
        if mask_idx is not None and mask_idx < len(pred_ids_list[0]):
            pred_id = pred_ids_list[0][mask_idx]
        else:
            pred_id = 0
        gold_seq2.append(id2label[gold_id])
        pred_seq2.append(id2label.get(pred_id, "O"))
        print(f"  j={j}: gold={id2label[gold_id]}, pred={id2label.get(pred_id, 'O')}")

    print(f"\nGold: {gold_seq2}")
    print(f"Pred: {pred_seq2}")

    # 这两个序列应该一致！
    print(f"\n两个序列相等: {gold_seq == gold_seq2}")
    print(f"两个预测序列相等: {pred_seq == pred_seq2}")


if __name__ == "__main__":
    debug_alignment_detailed()
