#!/usr/bin/env python3
"""调试 CRF 解码和标签对齐的问题 """

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
CKPT_DIR = ROOT / "homework_outputs" / "checkpoints"


def debug_crf_decode():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 checkpoint
    ckpt_path = CKPT_DIR / "best_crf.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    print("Checkpoint 信息:")
    print(f"  epoch: {ckpt['epoch']}")
    print(f"  val_f1: {ckpt['val_entity_f1']:.4f}")
    print(f"  label2id: {ckpt['label2id']}")
    print(f"  id2label: {ckpt['id2label']}")

    # 构建模型
    label2id = ckpt["label2id"]
    id2label = ckpt["id2label"]
    num_labels = len(label2id)

    model = build_model(
        use_crf=True,
        bert_path=str(BERT_PATH),
        num_labels=num_labels,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 加载数据
    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))
    _, _, test_loader = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=1,
        max_samples={"train": 10, "val": 5, "test": 2},
        data_dir=DATA_DIR,
    )

    print("\n" + "="*80)
    print("检查 CRF 解码输出和标签对齐问题")
    print("="*80)

    for batch_idx, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        print(f"\n--- 样本 {batch_idx}:")

        # 查看 attention mask
        print(f"  attention_mask: {attention_mask[0].tolist()}")
        print(f"  有效 token 数: {attention_mask[0].sum().item()}")

        # 查看 labels
        print(f"  labels (原始): {labels[0].tolist()}")

        # 解码预测
        with torch.no_grad():
            emissions, _ = model(input_ids, attention_mask, token_type_ids, labels)
            pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)

        print(f"  emissions shape: {emissions.shape}")
        print(f"  pred_ids_list (decode输出): {pred_ids_list}")
        print(f"  pred_ids_list 长度: {len(pred_ids_list[0])}")

        # 检查对齐
        print("\n  对齐过程:")
        labels_list = labels.cpu().tolist()
        token_labels = labels_list[0]
        gold_seq = []
        pred_seq = []
        for j, gold_id in enumerate(token_labels):
            if gold_id == -100:
                print(f"    j={j}: gold=-100 (跳过)")
                continue
            gold_seq.append(id2label[gold_id])
            if j < len(pred_ids_list[0]):
                pred_label = id2label.get(pred_ids_list[0][j], "O")
                pred_seq.append(pred_label)
                print(f"    j={j}: gold={id2label[gold_id]}, pred={pred_label} (使用 pred_ids_list[0][{j}]={pred_ids_list[0][j]})")
            else:
                pred_seq.append("O")
                print(f"    j={j}: gold={id2label[gold_id]}, pred=O (超出 pred_ids_list 长度)")

        print(f"\n  Gold: {gold_seq}")
        print(f"  Pred: {pred_seq}")

        if batch_idx >= 1:
            break


def debug_mask_vs_crf_output():
    """检查 attention_mask 和 CRF decode 输出长度的关系"""
    print("\n" + "="*80)
    print("检查 attention_mask 和 CRF decode 输出长度的关系")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 checkpoint
    ckpt_path = CKPT_DIR / "best_crf.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 构建模型
    label2id = ckpt["label2id"]
    id2label = ckpt["id2label"]
    num_labels = len(label2id)

    model = build_model(
        use_crf=True,
        bert_path=str(BERT_PATH),
        num_labels=num_labels,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 加载数据
    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))
    _, _, test_loader = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=2,
        max_samples={"train": 10, "val": 5, "test": 2},
        data_dir=DATA_DIR,
    )

    batch = next(iter(test_loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)

    with torch.no_grad():
        emissions = model._get_emissions(input_ids, attention_mask, token_type_ids)
        mask = attention_mask.bool()
        pred_ids_list = model.crf.decode(emissions, mask=mask)

    print(f"\nemissions shape: {emissions.shape}")
    print(f"mask shape: {mask.shape}")

    for i in range(len(input_ids)):
        print(f"\n样本 {i}:")
        print(f"  mask: {mask[i].tolist()}")
        print(f"  mask.sum(): {mask[i].sum().item()}")
        print(f"  decode 输出长度: {len(pred_ids_list[i])}")
        print(f"  decode 输出: {pred_ids_list[i]}")


if __name__ == "__main__":
    debug_crf_decode()
    debug_mask_vs_crf_output()
