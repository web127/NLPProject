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
