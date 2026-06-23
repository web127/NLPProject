
"""测试 Linear 模型，对比两种对齐方式"""

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


def test_linear_evaluate():
    """测试 Linear 模型的评估"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    # 标签体系
    labels, label2id, id2label = build_label_schema(DATA_DIR)
    num_labels = len(labels)
    print(f"BIO 标签数：{num_labels}（{labels}）")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))

    # 只加载少量样本
    max_samples = {"train": 100, "val": 50, "test": 50}
    train_loader, val_loader, _ = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=16,
        max_length=128,
        data_dir=DATA_DIR,
        max_samples=max_samples,
    )

    # 创建 Linear 模型
    model = build_model(
        use_crf=False,
        bert_path=str(BERT_PATH),
        num_labels=num_labels,
        dropout=0.1,
    ).to(device)

    # 简单训练 1 个 epoch 看看能否学到东西
    print("\n" + "="*80)
    print("训练 Linear 模型 1 个 epoch")
    print("="*80)

    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    model.train()
    for epoch in range(1):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits, loss = model(input_ids, attention_mask, token_type_ids, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}, loss: {total_loss / len(train_loader):.4f}")

    # 评估 - Linear 模式
    print("\n" + "="*80)
    print("评估 Linear 模型")
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

            logits, _ = model(input_ids, attention_mask, token_type_ids)
            pred_ids_list = logits.argmax(dim=-1).tolist()

            labels_list = labels.cpu().tolist()
            attention_list = attention_mask.cpu().tolist()

            for i in range(len(input_ids)):
                gold_seq = []
                pred_seq = []
                token_labels = labels_list[i]
                pred_ids = pred_ids_list[i]

                # Linear 模式的对齐
                for j, gold_id in enumerate(token_labels):
                    if gold_id == -100:
                        continue
                    gold_seq.append(id2label[gold_id])
                    pred_seq.append(id2label.get(pred_ids[j], "O"))

                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    f1 = seqeval_f1(all_golds, all_preds)
    print(f"Linear F1: {f1:.4f}")

    # 打印一些样本
    print(f"\n前3个预测样本:")
    for i in range(3):
        print(f"\n样本 {i}:")
        print(f"  Gold: {all_golds[i]}")
        print(f"  Pred: {all_preds[i]}")


if __name__ == "__main__":
    test_linear_evaluate()
