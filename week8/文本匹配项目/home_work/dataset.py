"""
文本匹配数据集类

这个文件可以理解为“把原始 JSONL 文本数据，整理成 PyTorch 能直接喂给模型的张量”。
如果把训练脚本看成“做菜”，那这里就是“洗菜、切菜、装盘”的步骤。

教学重点：
  1. PairDataset — 句对数据集，用于 CosineEmbeddingLoss 训练和评估
  2. TripletDataset — 三元组数据集，用于 TripletLoss 训练
     离线构建方式：从正样本对出发，为每个 anchor 查找负样本
  3. CrossEncoderDataset — 交互型数据集，句对拼接为单序列送入 BERT

一个 JSONL 样本长这样：
  {"sentence1": "花呗可以提现吗", "sentence2": "花呗能提现吗", "label": 1}

初学者可以重点关注两件事：
  - Dataset 决定“单条样本长什么样”
  - DataLoader 决定“一个 batch 怎么打包”

使用方式：
  from dataset import PairDataset, TripletDataset, CrossEncoderDataset, build_pair_loaders

  # 例 1：构建句对数据集
  ds = PairDataset("../data/afqmc/train.jsonl", tokenizer, max_length=64)
  sample = ds[0]
  # sample.keys() -> input_ids_a / attention_mask_a / token_type_ids_a / ... / label

  # 例 2：构建 DataLoader
  train_loader, val_loader, test_loader = build_pair_loaders("../data/afqmc", tokenizer)
  first_batch = next(iter(train_loader))
  # first_batch["input_ids_a"].shape 约为 [batch_size, max_length]

依赖：
  pip install torch transformers
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

random.seed(42)


# ── 工具函数 ──────────────────────────────────────────────────────────────

def load_jsonl(path):
    """读取 JSONL 文件并返回 list[dict]。

    JSONL = 每一行都是一个独立 JSON 对象，适合大规模文本数据集。

    例如文件前两行可能是：
      {"sentence1": "A", "sentence2": "B", "label": 1}
      {"sentence1": "C", "sentence2": "D", "label": 0}
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode_single(tokenizer, text, max_length):
    """把单个句子编码成 BERT 需要的输入张量。

    例子：
      text = "花呗能提现吗"
      返回后通常包含 3 个长度均为 max_length 的张量：
        input_ids      -> token 编号
        attention_mask -> 哪些位置是真实 token，哪些是 padding
        token_type_ids -> 句子段编号（单句时通常全 0）

    为什么 return_tensors="pt"？
      因为后面要直接送入 PyTorch 模型，所以这里直接转成 torch.Tensor。
    """
    # 对单句进行编码。tokenizer 会自动做：分词、转 id、截断、补齐。
    enc = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    # tokenizer 输出默认带一个 batch 维度，例如 [1, 64]。
    # 但 Dataset 的 __getitem__ 只返回“单条样本”，所以这里把最前面的 1 去掉，变成 [64]。
    return {
        "input_ids":      enc["input_ids"].squeeze(0),
        "attention_mask": enc["attention_mask"].squeeze(0),
        "token_type_ids": enc["token_type_ids"].squeeze(0),
    }


# ── PairDataset ───────────────────────────────────────────────────────────

class PairDataset(Dataset):
    """
    句对数据集：每个样本是 (sentence1, sentence2, label)

    用途：
      - CosineEmbeddingLoss 训练（label 0/1 在训练脚本中转为 -1/+1）
      - 评估时计算余弦相似度分布

    参数：
      data_path  : JSONL 文件路径，字段 sentence1 / sentence2 / label
      tokenizer  : HuggingFace tokenizer
      max_length : 单句最大 token 数
    """

    def __init__(self, data_path, tokenizer, max_length=64):
        self.rows      = load_jsonl(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        # 先拿到原始文本，再分别编码 sentence1 和 sentence2。
        r = self.rows[idx]
        enc_a = encode_single(self.tokenizer, r["sentence1"], self.max_length)
        enc_b = encode_single(self.tokenizer, r["sentence2"], self.max_length)
        # 返回结果中的 a/b 表示句对的两侧。
        # 训练时模型会分别编码它们，然后再比较两向量的相似度。
        return {
            "input_ids_a":      enc_a["input_ids"],
            "attention_mask_a": enc_a["attention_mask"],
            "token_type_ids_a": enc_a["token_type_ids"],
            "input_ids_b":      enc_b["input_ids"],
            "attention_mask_b": enc_b["attention_mask"],
            "token_type_ids_b": enc_b["token_type_ids"],
            "label": torch.tensor(r["label"], dtype=torch.long),
        }


# ── TripletDataset ────────────────────────────────────────────────────────

class TripletDataset(Dataset):
    """
    三元组数据集：每个样本是 (anchor, positive, negative)

    构建逻辑（离线静态构建）：
      1. 扫描所有正样本对 label=1，得到 (anchor=s1, positive=s2)
      2. 为每个 anchor 收集其配对的负样本（同一 s1 出现在 label=0 对中的 s2）
      3. 若 anchor 无自身负样本，从全局负样本池随机选取
      → 最终 triplet 数量 ≈ 正样本对数量（AFQMC 约 10K 条）

    教学说明：
      这是"离线随机负采样"策略，简单但效果基础。
      进阶做法是"在线难负样本挖掘"（Online Hard Negative Mining）：
      同一 batch 内，选余弦相似度最高的非正例对作为负样本。

    参数：
      data_path  : JSONL 文件路径
      tokenizer  : HuggingFace tokenizer
      max_length : 单句最大 token 数
    """

    def __init__(self, data_path, tokenizer, max_length=64):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.triplets   = self._build_triplets(load_jsonl(data_path))

    def _build_triplets(self, rows):
        # 第 1 步：为每个句子建立“它见过哪些负样本”的索引表。
        # 例如：
        #   "花呗能提现吗" -> ["借呗怎么开通", "余额宝收益怎么算"]
        # 这样后面遇到一个正样本 anchor 时，就能更快找到 negative。
        neg_by_sent = defaultdict(list)
        all_sents   = set()
        for r in rows:
            all_sents.add(r["sentence1"])
            all_sents.add(r["sentence2"])
            if r["label"] == 0:
                neg_by_sent[r["sentence1"]].append(r["sentence2"])
                neg_by_sent[r["sentence2"]].append(r["sentence1"])

        # 第 2 步：准备一个全局句子池，给“找不到局部负样本”的 anchor 兜底。
        global_pool = list(all_sents)

        triplets = []
        for r in rows:
            if r["label"] != 1:
                continue
            # 只从正样本对出发构造 triplet。
            # 逻辑是：
            #   anchor   = sentence1
            #   positive = 与它语义相似的 sentence2
            #   negative = 与 anchor 不相似的另一个句子
            anchor   = r["sentence1"]
            positive = r["sentence2"]

            negs = neg_by_sent.get(anchor, [])
            if negs:
                # 优先选“和这个 anchor 真实出现过的不相似句子”，更合理。
                negative = random.choice(negs)
            else:
                # 如果这个 anchor 在原数据里没有负样本，就退化为全局随机采样。
                # 这里要避免采到自己本身，或者采到 positive。
                negative = anchor
                while negative in (anchor, positive):
                    negative = random.choice(global_pool)

            triplets.append((anchor, positive, negative))

        print(f"  TripletDataset: 构建 {len(triplets):,} 个三元组")
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]
        # 三元组训练时，模型要同时看 3 句话：
        #   anchor：基准句
        #   positive：应该靠近 anchor
        #   negative：应该远离 anchor
        enc_a = encode_single(self.tokenizer, anchor,   self.max_length)
        enc_p = encode_single(self.tokenizer, positive, self.max_length)
        enc_n = encode_single(self.tokenizer, negative, self.max_length)
        return {
            "input_ids_a":      enc_a["input_ids"],
            "attention_mask_a": enc_a["attention_mask"],
            "token_type_ids_a": enc_a["token_type_ids"],
            "input_ids_p":      enc_p["input_ids"],
            "attention_mask_p": enc_p["attention_mask"],
            "token_type_ids_p": enc_p["token_type_ids"],
            "input_ids_n":      enc_n["input_ids"],
            "attention_mask_n": enc_n["attention_mask"],
            "token_type_ids_n": enc_n["token_type_ids"],
        }


# ── CrossEncoderDataset ───────────────────────────────────────────────────

class CrossEncoderDataset(Dataset):
    """
    交互型数据集：sentence1 与 sentence2 拼接为单序列

    BERT tokenizer 自动生成：
      [CLS] sentence1 [SEP] sentence2 [SEP]
      token_type_ids: 0000...0  1111...1

    教学对比：
      相比 BiEncoder（两路独立编码），CrossEncoder 让两句在每一层都交互，
      表达能力更强，但无法预计算句向量，推理时每对都要过一次 BERT。

    参数：
      data_path  : JSONL 文件路径
      tokenizer  : HuggingFace tokenizer
      max_length : 句对总最大 token 数（两句拼接后）
    """

    def __init__(self, data_path, tokenizer, max_length=128):
        self.rows       = load_jsonl(data_path)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        # CrossEncoder 的关键区别：不是分别编码两句话，
        # 而是把 sentence1 和 sentence2 一起交给 tokenizer。
        # BERT 最终看到的形式大致是：
        #   [CLS] sentence1 [SEP] sentence2 [SEP]
        enc = self.tokenizer(
            r["sentence1"],
            r["sentence2"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc["token_type_ids"].squeeze(0),
            "label": torch.tensor(r["label"], dtype=torch.long),
        }


# ── DataLoader 工厂函数 ───────────────────────────────────────────────────

def build_pair_loaders(data_dir, tokenizer, max_length=64, batch_size=32):
    """
    为 BiEncoder（CosineEmbeddingLoss / 评估）构建 train/val/test DataLoader。
    注意：AFQMC test 集无正样本标签，实际评估用 val。

    初学者可以记住：
      PairDataset 决定“单条样本是什么样子”；
      DataLoader 决定“多少条样本组成一个 batch”。
    """
    data_dir = Path(data_dir)
    train_ds = PairDataset(data_dir / "train.jsonl",      tokenizer, max_length)
    val_ds   = PairDataset(data_dir / "validation.jsonl", tokenizer, max_length)
    test_ds  = PairDataset(data_dir / "test.jsonl",       tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  train : {len(train_ds):>7,} 条, {len(train_loader):>5} batch")
    print(f"  val   : {len(val_ds):>7,} 条, {len(val_loader):>5} batch")
    print(f"  test  : {len(test_ds):>7,} 条, {len(test_loader):>5} batch  (AFQMC test 无正样本，仅供参考)")
    return train_loader, val_loader, test_loader


def build_triplet_loader(data_dir, tokenizer, max_length=64, batch_size=32):
    """为 TripletLoss 训练构建 DataLoader，val/test 仍用 PairDataset。

    这样设计的原因是：
      - 训练 TripletLoss 需要 (anchor, positive, negative)
      - 但验证时我们依然关心“句对分类效果”，所以验证集继续用 PairDataset
    """
    data_dir = Path(data_dir)
    train_ds = TripletDataset(data_dir / "train.jsonl", tokenizer, max_length)
    val_ds   = PairDataset(data_dir / "validation.jsonl", tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  triplet train : {len(train_ds):>7,} 三元组, {len(train_loader):>5} batch")
    print(f"  val (pair)    : {len(val_ds):>7,} 对,     {len(val_loader):>5} batch")
    return train_loader, val_loader


def build_crossencoder_loaders(data_dir, tokenizer, max_length=128, batch_size=32):
    """为 CrossEncoder 构建 train/val/test DataLoader。

    例子：
      train_loader, val_loader, test_loader = build_crossencoder_loaders(
          "../data/afqmc", tokenizer, max_length=128, batch_size=16
      )
    """
    data_dir = Path(data_dir)
    train_ds = CrossEncoderDataset(data_dir / "train.jsonl",      tokenizer, max_length)
    val_ds   = CrossEncoderDataset(data_dir / "validation.jsonl", tokenizer, max_length)
    test_ds  = CrossEncoderDataset(data_dir / "test.jsonl",       tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  train : {len(train_ds):>7,} 条, {len(train_loader):>5} batch")
    print(f"  val   : {len(val_ds):>7,} 条, {len(val_loader):>5} batch")
    print(f"  test  : {len(test_ds):>7,} 条, {len(test_loader):>5} batch")
    return train_loader, val_loader, test_loader
