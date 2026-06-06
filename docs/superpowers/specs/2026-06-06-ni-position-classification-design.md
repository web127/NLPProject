# "你"字位置分类任务设计文档

**日期**: 2026-06-06
**项目**: NLPProject week3 作业

## 概述

完成一个中文文本分类任务：给定一个包含"你"字的 5 字文本，"你"在第几位（从 0 开始），就属于第几类。

## 任务规格

- **输入**: 5 个中文字符组成的文本，必须包含"你"字
- **输出**: 5 分类，类别 0-4
- **标签规则**: y = "你"字在文本中的位置索引（从 0 开始）

## 实现方案

本项目将提供两个版本的实现，通过参数选择使用 RNN 或 LSTM：

- **RNN 版本**: 简单 RNN 架构，与参考代码风格一致
- **LSTM 版本**: LSTM 架构，更好地捕捉序列信息

两个版本共享数据生成、词表构建、评估逻辑等公共代码。

---

## 详细设计

### 1. 数据生成模块

```python
# 常用中文字符集（用于生成随机文本）
COMMON_CHARS = '的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感见明问力理尔点文几定本公特做外孩相西果走将月十实向声车全信重机工物气每并别真打太新比才便夫再书部水像眼少家经'

def generate_sample():
    """生成单个训练样本：5字文本，"你"在随机位置"""
    ni_position = random.randint(0, 4)
    chars = []
    for i in range(5):
        if i == ni_position:
            chars.append('你')
        else:
            chars.append(random.choice(COMMON_CHARS))
    return ''.join(chars), ni_position

def build_dataset(num_samples):
    """构建数据集，保证每个类别样本数均衡"""
    samples_per_class = num_samples // 5
    data = []
    for pos in range(5):
        for _ in range(samples_per_class):
            # 生成"你"在指定位置的样本
            chars = []
            for i in range(5):
                if i == pos:
                    chars.append('你')
                else:
                    chars.append(random.choice(COMMON_CHARS))
            data.append((''.join(chars), pos))
    random.shuffle(data)
    return data
```

### 2. 词表构建与编码

```python
def build_vocab(data):
    """构建字符级词表"""
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

def encode(sent, vocab, maxlen=5):
    """将文本编码为 id 序列"""
    ids = [vocab.get(ch, 1) for ch in sent]
    ids = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids
```

### 3. 模型架构

**基础配置**:
- 输入: 5 字符文本
- Embedding 维度: 64
- 隐藏层维度: 64
- Dropout: 0.3
- 输出: 5 类 logits

#### RNN 版本

```python
class NiPositionRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        # x: (batch, seq_len)
        e, _ = self.rnn(self.embedding(x))  # (B, L, hidden_dim)
        pooled = e.max(dim=1)[0]  # (B, hidden_dim)
        pooled = self.dropout(self.bn(pooled))
        out = self.fc(pooled)  # (B, 5)
        return out
```

#### LSTM 版本

```python
class NiPositionLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        # x: (batch, seq_len)
        e, _ = self.lstm(self.embedding(x))  # (B, L, hidden_dim)
        pooled = e.max(dim=1)[0]  # (B, hidden_dim)
        pooled = self.dropout(self.bn(pooled))
        out = self.fc(pooled)  # (B, 5)
        return out
```

### 4. 评估指标

计算每类的精确率、召回率、F1-score，以及宏平均：

```python
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def evaluate(model, loader):
    """评估模型，返回准确率和各类指标"""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1, 2, 3, 4], average=None
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'support': support
    }
```

### 5. 训练配置

| 参数 | 值 |
|------|-----|
| 总样本数 | 4000 |
| 训练集比例 | 0.8 |
| Batch size | 64 |
| Epochs | 20 |
| 学习率 | 1e-3 |
| 最大序列长度 | 5 |
| Embedding 维度 | 64 |
| 隐藏层维度 | 64 |
| Dropout | 0.3 |

### 6. 训练流程

1. 生成数据集（每类 800 样本）
2. 构建词表
3. 划分训练集/验证集
4. 根据 `model_type` 参数选择初始化 RNN 或 LSTM 模型
5. 使用 CrossEntropyLoss 训练
6. 每个 epoch 评估并打印详细指标
7. 展示推理示例

### 7. 文件结构

```
week3/
├── homework/
│   ├── job.desc
│   ├── chinese_classification.py
│   └── train_ni_position_classification.py  # 新建
└── train_chinese_cls_rnn.py (参考)
```

## 验收标准

- 两个模型（RNN 和 LSTM）在验证集上的准确率应接近 100%
- 每类的精确率、召回率、F1-score 都应接近 1.0
- 训练曲线显示 loss 稳定下降
- 能够通过命令行参数选择使用 RNN 或 LSTM
- 代码风格与参考文件一致
