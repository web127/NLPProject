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
