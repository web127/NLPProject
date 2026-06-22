"""
NER 数据集类：处理 peoples_daily 数据集（已有 BIO 格式）+ BERT 子词对齐

教学重点：
  1. peoples_daily 数据集格式：tokens + ner_tags（已 BIO）
  2. BERT 子词对齐（word_ids 策略）
     - 中文字符通常一字一token，但 [UNK] 和特殊字符可能例外
     - 非首子词标记为 -100，在 loss 计算中被忽略
  3. DataLoader 工厂函数统一封装，支持 max_samples 参数

使用方式：
  from homework_src.dataset import build_label_schema, build_dataloaders
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast as BertTokenizer


def build_label_schema(data_dir: Path) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """从 label_names.json 构建标签体系，返回 (labels, label2id, id2label)。"""
    with open(data_dir / "label_names.json", "r", encoding="utf-8") as f:
        labels = json.load(f)

    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label


class PeoplesDailyDataset(Dataset):
    """peoples_daily 数据集的 PyTorch Dataset。

    数据格式：
      {
        "tokens": ["海", "钓", ...],
        "ner_tags": ["O", "O", ...]
      }

    处理流程：
      tokens + ner_tags → 字符级 BIO ids
           → BertTokenizer (is_split_into_words=True)
           → 用 word_ids() 对齐子词标签（非首子词设为 -100）
           → 返回 input_ids / attention_mask / token_type_ids / labels
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
        tokens: list[str] = row["tokens"]
        ner_tags: list[str] = row["ner_tags"]

        # 1. 将 BIO 标签转为 id 列表
        char_labels = [self.label2id.get(tag, 0) for tag in ner_tags]

        # 2. 传入 tokenizer（已经是字符列表）
        #    is_split_into_words=True：把 word_ids() 与字符索引对齐
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # 3. 子词对齐：取每个 token 对应的字符索引
        #    - word_ids() 返回 [None, 0, 0, 1, 2, 2, ..., None]
        #      None 对应 [CLS]/[SEP]/[PAD]
        #    - 一个中文字符通常只有 1 个子词，但 ##xx 子词是非首子词
        #    - 非首子词、特殊token 标记为 -100，cross_entropy 的 ignore_index
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                # 首次出现这个字符索引：使用 BIO 标签
                if wid < len(char_labels):
                    aligned_labels.append(char_labels[wid])
                else:
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                # 同一字符的后续子词（中文通常不会出现，但保留正确处理）
                aligned_labels.append(-100)

        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": labels_tensor,
        }


def load_records(split: str, data_dir: Path, max_samples: Optional[int] = None) -> list:
    """加载指定 split 的数据，可选限制样本数。"""
    with open(data_dir / f"{split}.json", "r", encoding="utf-8") as f:
        records = json.load(f)

    if max_samples is not None and max_samples > 0:
        records = records[:max_samples]

    return records


def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 128,
    data_dir: Optional[Path] = None,
    max_samples: Optional[dict[str, int]] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练/验证/测试 DataLoader，返回 (train_loader, val_loader, test_loader)。

    Args:
        max_samples: 可选，例如 {"train": 2000, "val": 500, "test": 500}
    """
    if max_samples is None:
        max_samples = {}

    train_records = load_records("train", data_dir, max_samples.get("train"))
    val_records = load_records("validation", data_dir, max_samples.get("val"))
    test_records = load_records("test", data_dir, max_samples.get("test"))

    train_ds = PeoplesDailyDataset(train_records, tokenizer, label2id, max_length)
    val_ds = PeoplesDailyDataset(val_records, tokenizer, label2id, max_length)
    test_ds = PeoplesDailyDataset(test_records, tokenizer, label2id, max_length)

    print(f"数据集规模：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def debug_dataset_sample(dataset: PeoplesDailyDataset, id2label: dict, idx: int = 0):
    """调试：可视化查看一条数据的处理过程。"""
    row = dataset.records[idx]
    tokens = row["tokens"]
    ner_tags = row["ner_tags"]

    print("=" * 80)
    print(f"【样本 {idx}】原始数据")
    print("=" * 80)
    print(f"文本 (tokens): {''.join(tokens)}")
    print(f"原始标签: {ner_tags}")
    print()

    # 获取处理后的样本
    item = dataset[idx]
    input_ids = item["input_ids"]
    labels = item["labels"]

    # 重新编码以获取 word_ids
    tokenizer = dataset.tokenizer
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        max_length=dataset.max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    word_ids = encoding.word_ids(batch_index=0)

    print("=" * 80)
    print(f"【样本 {idx}】处理后数据（子词对齐）")
    print("=" * 80)
    print(f"{'索引':<6} {'Token':<15} {'WordID':<8} {'LabelID':<10} {'LabelStr':<15}")
    print("-" * 80)

    for i, (token, wid, label_id) in enumerate(zip(decoded_tokens, word_ids, labels.tolist())):
        if token == "[PAD]":
            break  # 后面的 padding 不显示
        label_str = id2label[label_id] if label_id != -100 else "-100 (skip)"
        wid_str = str(wid) if wid is not None else "None"
        print(f"{i:<6} {token:<15} {wid_str:<8} {label_id:<10} {label_str:<15}")

    print()
    print("=" * 80)
    print(f"【样本 {idx}】有效标签（排除 -100）")
    print("=" * 80)

    valid_tokens = []
    valid_labels = []
    for token, wid, label_id in zip(decoded_tokens, word_ids, labels.tolist()):
        if label_id != -100 and token not in ["[CLS]", "[SEP]", "[PAD]"]:
            valid_tokens.append(token)
            valid_labels.append(id2label[label_id])

    print(f"有效 tokens: {' '.join(valid_tokens)}")
    print(f"有效 labels: {' '.join(valid_labels)}")
    print()

    # 对比原始和处理后
    print("=" * 80)
    print(f"【样本 {idx}】原始 vs 处理后对比")
    print("=" * 80)
    print(f"{'原始字符':<10} {'原始标签':<15} | {'处理后Token':<15} {'处理后标签':<15}")
    print("-" * 80)

    # 建立 word_id -> (token, label) 的映射
    token_label_map = {}
    for token, wid, label_id in zip(decoded_tokens, word_ids, labels.tolist()):
        if wid is not None and label_id != -100:
            token_label_map[wid] = (token, id2label[label_id])

    for i, (char, orig_tag) in enumerate(zip(tokens, ner_tags)):
        if i in token_label_map:
            token, label = token_label_map[i]
            print(f"{char:<10} {orig_tag:<15} | {token:<15} {label:<15}")
        else:
            print(f"{char:<10} {orig_tag:<15} | {'(truncated)':<15} {'(truncated)':<15}")
    print()


if __name__ == "__main__":
    # 调试代码：运行此文件可查看数据处理过程
    from pathlib import Path

    ROOT = Path(__file__).parent.parent
    DATA_DIR = ROOT / "data" / "peoples_daily"
    BERT_PATH = ROOT / "pretrain_models" / "bert-base-chinese"

    print("正在加载标签和 tokenizer...")
    labels, label2id, id2label = build_label_schema(DATA_DIR)
    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))

    print("正在加载数据...")
    records = load_records("train", DATA_DIR, max_samples=10)
    dataset = PeoplesDailyDataset(records, tokenizer, label2id, max_length=128)

    # 调试查看前 2 条样本
    for i in range(min(5, len(dataset))):
        debug_dataset_sample(dataset, id2label, idx=i)
