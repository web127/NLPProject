"""
NER 数据集类：支持 cluener 和 peoples_daily 两种数据集格式

教学重点：
  1. cluener2020 的 span 格式转为 BIO 格式
     - span: {"name": {"叶老桂": [[9, 11]]}}
     - BIO:  ['O','O',...,'B-name','I-name','I-name',...]
  2. peoples_daily 已有 tokens + ner_tags 的 BIO 格式
  3. BERT 子词对齐（word_ids 策略）
     - 中文字符通常一字一token，但 [UNK] 和特殊字符可能例外
     - 非首子词标记为 -100，在 loss 计算中被忽略
  3. DataLoader 工厂函数统一封装

使用方式：
  from dataset import build_label_schema, build_dataloaders
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "cluener"

# cluener 的实体类型
CLUENER_ENTITY_TYPES = [
    "address", "book", "company", "game",
    "government", "movie", "name", "organization",
    "position", "scene",
]

# peoples_daily 的实体类型
PEOPLES_DAILY_ENTITY_TYPES = ["PER", "ORG", "LOC"]


def detect_dataset_format(data_dir: Path) -> str:
    """检测数据集格式：通过检查 label_names.json 或数据文件内容"""
    label_names_path = data_dir / "label_names.json"
    if label_names_path.exists():
        with open(label_names_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        # peoples_daily 有 PER, ORG, LOC
        if any("PER" in lbl for lbl in labels) or any("LOC" in lbl for lbl in labels):
            return "peoples_daily"

    # 检查训练文件格式
    train_path = data_dir / "train.json"
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            first = json.load(f)
            if isinstance(first, list) and len(first) > 0:
                if "tokens" in first[0] and "ner_tags" in first[0]:
                    return "peoples_daily"

    return "cluener"


def build_label_schema(data_dir: Optional[Path] = None) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """构建 BIO 标签体系，返回 (labels, label2id, id2label)。"""
    if data_dir is not None:
        label_names_path = data_dir / "label_names.json"
        if label_names_path.exists():
            with open(label_names_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
            label2id = {lbl: i for i, lbl in enumerate(labels)}
            id2label = {i: lbl for lbl, i in label2id.items()}
            return labels, label2id, id2label

    # 默认使用 cluener 标签
    labels = ["O"]
    for etype in CLUENER_ENTITY_TYPES:
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")

    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label


def span_to_bio(text: str, label_dict: dict, label2id: dict) -> list[int]:
    """将 cluener2020 的 span 格式标注转为逐字符 BIO 标签 id 列表。"""
    n = len(text)
    bio = ["O"] * n

    if not label_dict:
        return [label2id[t] for t in bio]

    for etype, spans in label_dict.items():
        b_tag = f"B-{etype}"
        i_tag = f"I-{etype}"
        for surface, positions in spans.items():
            for start, end in positions:
                if start >= n or end >= n:
                    continue
                bio[start] = b_tag
                for idx in range(start + 1, end + 1):
                    bio[idx] = i_tag

    return [label2id.get(t, 0) for t in bio]


class CluenerDataset(Dataset):
    """cluener2020 的 PyTorch Dataset。"""

    def __init__(
        self,
        records: list,
        tokenizer: BertTokenizer,
        label2id: dict,
        max_length: int = 128,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        text: str = row["text"]
        label_dict: dict = row.get("label") or {}

        # 1. span → 字符级 BIO id 列表
        char_labels = span_to_bio(text, label_dict, self.label2id)

        # 2. 将文本拆为字符列表，传入 tokenizer
        chars = list(text)
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # 3. 子词对齐
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                if wid < len(char_labels):
                    aligned_labels.append(char_labels[wid])
                else:
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                aligned_labels.append(-100)

        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": labels_tensor,
        }


class PeoplesDailyDataset(Dataset):
    """peoples_daily 的 PyTorch Dataset。

    数据格式：[{"tokens": [...], "ner_tags": [...]}]
    ner_tags 已经是 BIO 格式的字符串标签
    """

    def __init__(
        self,
        records: list,
        tokenizer: BertTokenizer,
        label2id: dict,
        max_length: int = 128,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        tokens: list = row["tokens"]
        ner_tags: list = row["ner_tags"]

        # 将标签字符串转为 id
        char_labels = [self.label2id.get(tag, 0) for tag in ner_tags]

        # tokens 已经是分词后的列表，直接传入 tokenizer
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # 子词对齐
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                if wid < len(char_labels):
                    aligned_labels.append(char_labels[wid])
                else:
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                aligned_labels.append(-100)

        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": labels_tensor,
        }


def load_records(split: str, data_dir: Optional[Path] = None, max_samples: int = 100) -> list:
    d = data_dir or DATA_DIR
    with open(d / f"{split}.json", "r", encoding="utf-8") as f:
        records = json.load(f)
    # 只取部分数据，加快训练速度
    if len(records) > max_samples:
        print(f"  加载 {split} 集：{len(records)} 条 -> 取前 {max_samples} 条")
        return records[:max_samples]
    print(f"  加载 {split} 集：{len(records)} 条")
    return records


def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 128,
    data_dir: Optional[Path] = None,
    max_samples: Optional[dict] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练/验证/测试 DataLoader，返回 (train_loader, val_loader, test_loader)。"""
    data_dir = data_dir or DATA_DIR
    dataset_format = detect_dataset_format(data_dir)

    # 解析 max_samples
    train_max = max_samples.get("train", 100) if max_samples else 100
    val_max = max_samples.get("val", 100) if max_samples else 100
    test_max = max_samples.get("test", 100) if max_samples else 100

    train_records = load_records("train", data_dir, max_samples=train_max)
    val_records = load_records("validation", data_dir, max_samples=val_max)
    test_records = load_records("test", data_dir, max_samples=test_max)

    if dataset_format == "peoples_daily":
        DatasetClass = PeoplesDailyDataset
        print(f"数据集格式：peoples_daily (PER, ORG, LOC)")
    else:
        DatasetClass = CluenerDataset
        print(f"数据集格式：cluener")

    train_ds = DatasetClass(train_records, tokenizer, label2id, max_length)
    val_ds = DatasetClass(val_records, tokenizer, label2id, max_length)
    test_ds = DatasetClass(test_records, tokenizer, label2id, max_length)

    print(f"数据集规模：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
