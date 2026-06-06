"""
train_ni_position_rnn.py
"你"字位置分类 - RNN 版本

任务：给定 5 字包含"你"的文本，分类"你"在第几位（0-4）
模型：Embedding → RNN → MaxPool → BN → Dropout → Linear(5)
优化：Adam (lr=1e-3)  损失：CrossEntropyLoss
评估：准确率 + 精确率 + 召回率 + F1-score

依赖：torch >= 2.0, scikit-learn
"""

import random
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


# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [label for _, label in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.long)
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
class NiPositionRNN(nn.Module):
    """
    "你"字位置分类器 - RNN 版本
    架构：Embedding → RNN → MaxPool → BN → Dropout → Linear(5)
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        # x: (batch, seq_len)
        #e = self.embedding(x)  # 先 embedding
        #out_rnn, _ = self.rnn(e)  # 再进 RNN
        e, _ = self.rnn(self.embedding(x))  # (B, L, hidden_dim)
        pooled = e.max(dim=1)[0]  # (B, hidden_dim)
        #pooled = self.bn(pooled)
        #pooled = self.dropout(pooled)
        pooled = self.dropout(self.bn(pooled))
        out = self.fc(pooled)  # (B, 5)
        return out


# ─── 5. 训练与评估 ──────────────────────────────────────────
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
        all_labels, all_preds, labels=[0, 1, 2, 3, 4], average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
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


def print_metrics(metrics, title="Metrics"):
    """打印评估指标"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"\n{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print(f"{'-'*50}")
    for i in range(5):
        print(f"{i:<10} {metrics['precision'][i]:<10.4f} {metrics['recall'][i]:<10.4f} "
              f"{metrics['f1'][i]:<10.4f} {metrics['support'][i]:<10}")
    print(f"{'='*60}\n")


def train():
    """训练主函数"""
    print(f"{'='*60}")
    print(f"开始训练: RNN 模型")
    print(f"{'='*60}")

    # 生成数据集
    print("\n生成数据集...")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    # 划分训练集验证集
    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]
    print(f"  训练集：{len(train_data)}，验证集：{len(val_data)}")

    # DataLoader
    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    # 初始化模型
    model = NiPositionRNN(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    # 训练循环
    print(f"{'Epoch':<8} {'Loss':<12} {'Val Acc':<12}")
    print(f"{'-'*32}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader)
        print(f"{epoch:<8} {avg_loss:<12.4f} {val_metrics['accuracy']:<12.4f}")

    # 最终评估
    final_metrics = evaluate(model, val_loader)
    print_metrics(final_metrics, "最终验证集指标")

    # 推理示例
    print("--- 推理示例 ---")
    model.eval()
    test_sents = []
    for pos in range(5):
        sent, _ = generate_sample(fixed_position=pos)
        test_sents.append(sent)

    with torch.no_grad():
        for sent in test_sents:
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)
            pred = torch.argmax(logits, dim=1).item()
            true_pos = sent.index('你')
            status = "✓" if pred == true_pos else "✗"
            print(f"{status} 文本: '{sent}' → 预测: {pred}, 真实: {true_pos}")
    print()


if __name__ == '__main__':
    train()
