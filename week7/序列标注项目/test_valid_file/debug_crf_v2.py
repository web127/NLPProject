
"""调试 CRF 预测对齐问题 - 更详细的版本"""

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
    """调试对比两种对齐方式，使用已训练的checkpoint"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    # 标签体系
    labels, label2id, id2label = build_label_schema(DATA_DIR)
    num_labels = len(labels)
    print(f"BIO 标签数：{num_labels}（{labels}）")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))

    # 只加载少量样本
    max_samples = {"train": 10, "val": 10, "test": 5}
    train_loader, val_loader, _ = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=1,
        max_length=128,
        data_dir=DATA_DIR,
        max_samples=max_samples,
    )

    # 尝试加载已训练的 checkpoint
    ckpt_path = ROOT / "homework_outputs" / "checkpoints" / "best_crf.pt"
    if ckpt_path.exists():
        print(f"\n加载 checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(
            use_crf=True,
            bert_path=str(BERT_PATH),
            num_labels=num_labels,
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        print(f"\n未找到 checkpoint，使用随机初始化的模型")
        model = build_model(
            use_crf=True,
            bert_path=str(BERT_PATH),
            num_labels=num_labels,
            dropout=0.1,
        ).to(device)

    model.eval()

    # 查看 dataset 如何处理样本
    print("\n" + "="*80)
    print("查看 dataset 中的样本")
    print("="*80)
    from homework_src.dataset import PeoplesDailyDataset, load_records
    val_records = load_records("validation", DATA_DIR, max_samples=5)
    val_dataset = PeoplesDailyDataset(val_records, tokenizer, label2id, max_length=128)

    for sample_idx in range(3):
        print(f"\n\n{'='*80}")
        print(f"样本 {sample_idx}")
        print('='*80)

        print(f"\n原始数据:")
        print(f"  tokens: {''.join(val_records[sample_idx]['tokens'])}")
        print(f"  ner_tags: {val_records[sample_idx]['ner_tags']}")

        # 获取处理后的样本
        item = val_dataset[sample_idx]
        input_ids = item["input_ids"]
        labels = item["labels"]
        attention_mask = item["attention_mask"]

        # 重新编码以获取 word_ids
        encoding = tokenizer(
            val_records[sample_idx]['tokens'],
            is_split_into_words=True,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0)

        print(f"\n处理后的标签 (labels):")
        print(f"  形状: {labels.shape}")
        print(f"  前30个值: {labels[:30].tolist()}")

        print(f"\nword_ids:")
        print(f"  前30个值: {word_ids[:30]}")

        print(f"\n对齐详细信息:")
        print(f"  {'索引':<6} {'token':<15} {'word_id':<8} {'label_id':<10} {'label_str':<15}")
        print("  " + "-"*60)
        valid_positions = []
        for idx, (t_id, wid, l_id, mask) in enumerate(zip(input_ids, word_ids, labels, attention_mask)):
            if mask == 0:
                continue
            token = tokenizer.convert_ids_to_tokens([t_id])[0]
            label_str = id2label[l_id] if l_id != -100 else "-100"
            print(f"  {idx:<6} {token:<15} {str(wid):<8} {l_id:<10} {label_str:<15}")
            if l_id != -100 and token not in ["<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>", "[SEP]"]:
                valid_positions.append(idx)

        print(f"\n有效标签位置 (label != -100): {valid_positions}")
        print(f"有效标签数量: {len(valid_positions)}")
        print(f"对应原始标签数量: {len(val_records[sample_idx]['ner_tags'])}")

    # 测试推理和对齐
    print("\n" + "="*80)
    print("测试推理和对齐逻辑")
    print("="*80)

    batch = next(iter(val_loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    labels = batch["labels"].to(device)

    with torch.no_grad():
        emissions, loss = model(input_ids, attention_mask, token_type_ids, labels)
        pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)

    print(f"\ndecode 返回:")
    print(f"  长度: {len(pred_ids_list)}")
    print(f"  第一个序列长度: {len(pred_ids_list[0])}")
    print(f"  第一个序列内容: {pred_ids_list[0]}")

    # 对齐逻辑测试
    i = 0
    token_labels = labels[i].cpu().tolist()
    mask_list = attention_mask[i].cpu().tolist()
    pred_ids = pred_ids_list[i]

    print(f"\n样本 {i} 对齐测试:")
    print(f"  token_labels 长度: {len(token_labels)}")
    print(f"  mask_list 长度: {len(mask_list)}")
    print(f"  pred_ids 长度: {len(pred_ids)}")

    # 先收集所有 mask=True 的位置索引
    mask_true_indices = [j for j, mask_val in enumerate(mask_list) if mask_val == 1]
    print(f"\nmask_true_indices (attention_mask=1 的位置):")
    print(f"  长度: {len(mask_true_indices)}")
    print(f"  内容: {mask_true_indices}")

    # 建立映射：原位置 j -> 预测的 id
    pos_to_pred_id = {}
    for mask_idx, orig_j in enumerate(mask_true_indices):
        if mask_idx < len(pred_ids):
            pos_to_pred_id[orig_j] = pred_ids[mask_idx]
    print(f"\npos_to_pred_id (位置 -> 预测id):")
    print(f"  长度: {len(pos_to_pred_id)}")
    print(f"  内容: {pos_to_pred_id}")

    # 对齐标签
    gold_seq = []
    pred_seq = []
    print(f"\n对齐过程:")
    print(f"  {'j':<4} {'gold_id':<8} {'gold_label':<12} {'in_map':<6} {'pred_id':<8} {'pred_label':<12}")
    print("  " + "-"*60)
    for j, gold_id in enumerate(token_labels):
        if gold_id == -100:
            continue
        gold_label = id2label[gold_id]
        in_map = j in pos_to_pred_id
        if in_map:
            pred_id = pos_to_pred_id[j]
            pred_label = id2label.get(pred_id, "O")
        else:
            pred_id = -1
            pred_label = "N/A"
        print(f"  {j:<4} {gold_id:<8} {gold_label:<12} {str(in_map):<6} {pred_id:<8} {pred_label:<12}")
        gold_seq.append(gold_label)
        pred_seq.append(pred_label if in_map else "O")

    print(f"\n最终序列:")
    print(f"  Gold: {gold_seq}")
    print(f"  Pred: {pred_seq}")

    # 用 seqeval 计算 f1
    from seqeval.metrics import f1_score
    print(f"\nseqeval F1: {f1_score([gold_seq], [pred_seq])}")


if __name__ == "__main__":
    debug_alignment()
