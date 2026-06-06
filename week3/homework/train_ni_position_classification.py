"""
train_ni_position_classification.py
"你"字位置分类 - RNN 和 LSTM 版本

任务：给定 5 字包含"你"的文本，分类"你"在第几位（0-4）
模型：Embedding → RNN/LSTM → MaxPool → BN → Dropout → Linear(5)
优化：Adam (lr=1e-3)  损失：CrossEntropyLoss
评估：准确率 + 精确率 + 召回率 + F1-score

依赖：torch >= 2.0, scikit-learn
"""

import random
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 4000  # 每类 800 样本
MAXLEN      = 5
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8
DROPOUT     = 0.3

random.seed(SEED)
torch.manual_seed(SEED)

# 常用中文字符集
COMMON_CHARS = '的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感见明问力理尔点文几定本公特做外孩相西果走将月十实向声车全信重机工物气每并别真打太新比才便夫再书部水像眼少家经'

# ─── 1. 数据生成 ────────────────────────────────────────────
def generate_sample(fixed_position=None):
    """生成单个训练样本：5字文本，"你"在指定位置或随机位置"""
    if fixed_position is not None:
        ni_position = fixed_position
    else:
        ni_position = random.randint(0, 4)
    chars = []
    for i in range(5):
        if i == ni_position:
            chars.append('你')
        else:
            chars.append(random.choice(COMMON_CHARS))
    return ''.join(chars), ni_position


def build_dataset(num_samples=N_SAMPLES):
    """构建数据集，保证每个类别样本数均衡"""
    samples_per_class = num_samples // 5
    data = []
    for pos in range(5):
        for _ in range(samples_per_class):
            sent, label = generate_sample(fixed_position=pos)
            data.append((sent, label))
    random.shuffle(data)
    return data


# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    """构建字符级词表"""
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab, maxlen=MAXLEN):
    """将文本编码为 id 序列"""
    ids = [vocab.get(ch, 1) for ch in sent]
    ids = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids
