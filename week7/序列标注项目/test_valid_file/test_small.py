"""
测试数据加载是否正常
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import torch
from transformers import BertTokenizer
from src.dataset import build_label_schema, build_dataloaders, CluenerDataset

print("测试 tokenizer...")
tokenizer = BertTokenizer.from_pretrained(str(ROOT / "pretrain_models/bert-base-chinese"))
print("Tokenizer OK")

print("\n测试 cluener 数据加载...")
labels, label2id, id2label = build_label_schema()
print(f"Labels: {labels}")

# 小样本测试
import json
with open(ROOT / "data/cluener/train.json", "r", encoding="utf-8") as f:
    records = json.load(f)[:2]

ds = CluenerDataset(records, tokenizer, label2id, max_length=32)
print(f"Dataset size: {len(ds)}")

sample = ds[0]
print(f"Sample keys: {list(sample.keys())}")
print(f"input_ids shape: {sample['input_ids'].shape}")
print(f"attention_mask shape: {sample['attention_mask'].shape}")
print(f"labels shape: {sample['labels'].shape}")
print(f"labels: {sample['labels']}")

# 测试 word_ids
chars = list(records[0]["text"])
encoding = tokenizer(chars, is_split_into_words=True, max_length=32, truncation=True, padding="max_length", return_tensors="pt")
word_ids = encoding.word_ids(batch_index=0)
print(f"\nword_ids: {word_ids}")

print("\n所有测试通过！")
