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
from seqeval.metrics.sequence_labeling import get_entities

def build_label_schema(data_dir: Path) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """从 label_names.json 构建标签体系，返回 (labels, label2id, id2label)。"""
    # 打开标签定义文件。
    with open(data_dir / "label_names.json", "r", encoding="utf-8") as f:
        # 读取标签列表，例如 ['O', 'B-PER', 'I-PER', ...]。
        labels = json.load(f)

    # 构建标签字符串到标签 id 的映射。
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    # 构建标签 id 到标签字符串的逆映射。
    id2label = {i: lbl for lbl, i in label2id.items()}
    # 返回标签列表及双向映射。
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
        # 保存原始记录列表。
        self.records = records
        # 保存 tokenizer 以便 __getitem__ 中调用。
        self.tokenizer = tokenizer
        # 保存标签到 id 的映射。
        self.label2id = label2id
        # 保存最大截断长度。
        self.max_length = max_length

    def __len__(self):
        # 返回数据集样本数。
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        # 取出第 idx 条原始样本。
        row = self.records[idx]
        # 读取字符级 token 列表。
        tokens: list[str] = row["tokens"]
        # 读取对应的 BIO 标签列表。
        ner_tags: list[str] = row["ner_tags"]

        # 1. 将 BIO 标签转为 id 列表
        # 若遇到未知标签，则默认映射到 0（通常对应 O）。
        char_labels = [self.label2id.get(tag, 0) for tag in ner_tags]

        # 2. 传入 tokenizer（已经是字符列表）
        #    is_split_into_words=True：把 word_ids() 与字符索引对齐
        # 对字符列表做 BERT 编码，并保持字符级对齐信息。
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
        # 获取每个子词对应的原始字符索引。
        word_ids = encoding.word_ids(batch_index=0)
        # 保存对齐后的标签 id 列表。
        aligned_labels = []
        # 记录上一个子词对应的字符索引，用于区分首子词和非首子词。
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                # 首次出现这个字符索引：使用 BIO 标签
                # 正常情况下 wid 一定落在 char_labels 范围内，这里保留防御式判断。
                if wid < len(char_labels):
                    aligned_labels.append(char_labels[wid])
                else:
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                # 同一字符的后续子词（中文通常不会出现，但保留正确处理）
                aligned_labels.append(-100)

        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        # encoding = {
        #     "input_ids": [101, 2769, 4263, 0, 0, 0],  # [CLS] + 真实token + [PAD]
        #     "attention_mask": [1, 1, 1, 0, 0, 0],  # 1=有效, 0=padding
        #     "token_type_ids": [0, 0, 0, 0, 0, 0],  # 句子类型
        # }
        return {
            # squeeze(0) 去掉 tokenizer 产生的 batch 维。
            "input_ids": encoding["input_ids"].squeeze(0),
            # 返回 attention mask。
            "attention_mask": encoding["attention_mask"].squeeze(0),
            # 返回 token_type_ids。
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            # 返回与子词对齐后的标签张量。
            "labels": labels_tensor,
        }


def load_records(split: str, data_dir: Path, max_samples: Optional[int] = None) -> list:
    """加载指定 split 的数据，可选限制样本数。"""
    # 打开指定划分的 JSON 文件。
    with open(data_dir / f"{split}.json", "r", encoding="utf-8") as f:
        # 读取原始记录列表。
        records = json.load(f)

    if max_samples is not None and max_samples > 0:
        records = records[:max_samples]

    return records
def get_max_sequence_length(self) -> int:
    """获取数据集中最长序列的长度（原始 tokens 数量）。"""
    # 初始化最大长度为 0。
    max_len = 0
    for record in self.records:
        # 统计当前样本的 token 数。
        seq_len = len(record["tokens"])
        # 若当前样本更长，则更新最大长度。
        if seq_len > max_len:
            max_len = seq_len
    # 返回最大序列长度。
    return max_len

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
    # 若未传入 max_samples，则使用空字典表示不限制样本数。
    if max_samples is None:
        max_samples = {}

    train_records = load_records("train", data_dir, max_samples.get("train"))
    val_records = load_records("validation", data_dir, max_samples.get("val"))
    test_records = load_records("test", data_dir, max_samples.get("test"))

    # 用训练集记录构建 Dataset。
    train_ds = PeoplesDailyDataset(train_records, tokenizer, label2id, max_length)
    # 用验证集记录构建 Dataset。
    val_ds = PeoplesDailyDataset(val_records, tokenizer, label2id, max_length)
    # 用测试集记录构建 Dataset。
    test_ds = PeoplesDailyDataset(test_records, tokenizer, label2id, max_length)

    # 打印各数据集规模。
    print(f"数据集规模：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}")

    # 训练集 DataLoader 开启 shuffle 打乱顺序。
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    # 验证集 DataLoader 不打乱顺序。
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    # 测试集 DataLoader 不打乱顺序。
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def debug_dataset_sample(dataset: PeoplesDailyDataset, id2label: dict, idx: int = 0):
    """调试：可视化查看一条数据的处理过程。"""
    # 取出原始样本。
    row = dataset.records[idx]
    # 原始字符序列。
    tokens = row["tokens"]
    # 原始 BIO 标签序列。
    ner_tags = row["ner_tags"]

    print(f"id2label: {id2label}")
    print("=" * 80)
    print(f"【样本 {idx}】原始数据")
    print("=" * 80)
    print(f"文本 (tokens): {''.join(tokens)}")
    print(f"原始标签: {ner_tags}")
    print()

    # 获取处理后的样本
    # 调用数据集的 __getitem__，获取模型实际看到的样本格式。
    item = dataset[idx]
    input_ids = item["input_ids"]
    labels = item["labels"]

    # 重新编码以获取 word_ids
    # 直接复用数据集内部 tokenizer。
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
    # 用于把原始字符位置映射到处理后的 token 与标签。
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
    # 添加：使用 seqeval 识别并打印实体
    print("=" * 80)
    print(f"【样本 {idx}】识别到的实体（seqeval）")
    print("=" * 80)



    # 使用 seqeval 提取实体
    entities = get_entities(valid_labels)

    if entities:
        for ent in entities:
            ent_type, start_idx, end_idx = ent
            ent_tokens = valid_tokens[start_idx:end_idx + 1]
            print(f"实体类型: {ent_type}, 位置: [{start_idx}-{end_idx}], 文本: {' '.join(ent_tokens)}")
    else:
        print("未识别到实体")
    print()


if __name__ == "__main__":
    from pathlib import Path

    ROOT = Path(__file__).parent.parent
    DATA_DIR = ROOT / "data" / "peoples_daily"
    BERT_PATH = ROOT / "pretrain_models" / "bert-base-chinese"

    print("正在加载标签和 tokenizer...")
    labels, label2id, id2label = build_label_schema(DATA_DIR)
    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))
    # 加载各数据集并显示大小
    # 逐个划分统计样本数。
    for split in ["train", "validation", "test"]:
        records = load_records(split, DATA_DIR)
        print(f"  {split} 数据集大小: {len(records)} 条")

    print("正在加载数据...")
    records = load_records("train", DATA_DIR, max_samples=10)
    dataset = PeoplesDailyDataset(records, tokenizer, label2id, max_length=128)

    # 调试查看前 2 条样本
    # 这里实际只查看 1 条样本，因为 range(min(1, len(dataset))) 上限为 1。
    for i in range(min(1, len(dataset))):
        debug_dataset_sample(dataset, id2label, idx=i)
