# Simple Transformer Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a standard Transformer encoder layer with multi-head attention, residual connections, and LayerNorm in PyTorch, including embedding layer and usage example.

**Architecture:** Single file implementation containing MultiHeadAttention, FeedForward, TransformerLayer, and SimpleEmbedding classes, plus a usage example.

**Tech Stack:** PyTorch, Python

---

## Task 1: MultiHeadAttention

**Files:**
- Create: `week4/simple_transformer.py`

- [ ] **Step 1: Write the MultiHeadAttention class skeleton

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        # 定义线性层
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        new_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_size)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性映射
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)

        # 分头
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        # 缩放点积注意力
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_size)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)

        # 拼接回来
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        new_shape = attention_output.size()[:-2] + (self.hidden_size,)
        attention_output = attention_output.view(*new_shape)

        # 输出线性层
        output = self.output_linear(attention_output)
        return output
```

- [ ] **Step 2: Commit**

```bash
git add week4/simple_transformer.py
git commit -m "feat: add MultiHeadAttention class"
```

---

## Task 2: FeedForward

**Files:**
- Modify: `week4/simple_transformer.py`

- [ ] **Step 1: Add FeedForward class after MultiHeadAttention**

```python
class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x
```

- [ ] **Step 2: Commit**

```bash
git add week4/simple_transformer.py
git commit -m "feat: add FeedForward class"
```

---

## Task 3: TransformerLayer

**Files:**
- Modify: `week4/simple_transformer.py`

- [ ] **Step 1: Add TransformerLayer class after FeedForward**

```python
class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout_rate=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(hidden_size, num_heads)
        self.feed_forward = FeedForward(hidden_size, intermediate_size)

        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        # 第一个子层：自注意力 + 残差连接
        residual = x
        x = self.layer_norm1(x)
        x = self.self_attention(x, x, x, mask)
        x = self.dropout1(x)
        x = x + residual

        # 第二个子层：前馈网络 + 残差连接
        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout2(x)
        x = x + residual

        return x
```

- [ ] **Step 2: Commit**

```bash
git add week4/simple_transformer.py
git commit -m "feat: add TransformerLayer class"
```

---

## Task 4: SimpleEmbedding

**Files:**
- Modify: `week4/simple_transformer.py`

- [ ] **Step 1: Add SimpleEmbedding class after TransformerLayer**

```python
class SimpleEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_seq_len=512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()

        # 确保序列长度不超过最大长度
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"

        # token embedding
        token_emb = self.token_embedding(input_ids)

        # position embedding
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        position_emb = self.position_embedding(position_ids)

        # 相加
        return token_emb + position_emb
```

- [ ] **Step 2: Commit**

```bash
git add week4/simple_transformer.py
git commit -m "feat: add SimpleEmbedding class"
```

---

## Task 5: Usage Example

**Files:**
- Modify: `week4/simple_transformer.py`

- [ ] **Step 1: Add usage example at the end of the file**

```python
def main():
    # 配置参数
    vocab_size = 1000
    hidden_size = 768
    num_heads = 12
    intermediate_size = 3072  # 4 * hidden_size
    max_seq_len = 512
    dropout_rate = 0.1
    batch_size = 2
    seq_len = 10

    # 创建模型
    embedding = SimpleEmbedding(vocab_size, hidden_size, max_seq_len)
    transformer_layer = TransformerLayer(hidden_size, num_heads, intermediate_size, dropout_rate)

    # 构造输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 前向传播
    print("输入形状:", input_ids.shape)
    x = embedding(input_ids)
    print("Embedding输出形状:", x.shape)
    output = transformer_layer(x)
    print("Transformer层输出形状:", output.shape)

    # 验证输出形状正确
    assert output.shape == (batch_size, seq_len, hidden_size), f"Expected shape {(batch_size, seq_len, hidden_size)}, got {output.shape}"
    print("✓ 形状验证通过！")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add week4/simple_transformer.py
git commit -m "feat: add usage example"
```

---

## Task 6: Test Run

**Files:**
- Run: `week4/simple_transformer.py`

- [ ] **Step 1: Run the script to verify it works**

```bash
cd /Users/bytedance/PycharmProjects/NLPProject
python week4/simple_transformer.py
```

Expected output:
```
输入形状: (2, 10)
Embedding输出形状: (2, 10, 768)
Transformer层输出形状: (2, 10, 768)
✓ 形状验证通过！
```

- [ ] **Step 2: Commit (if any fixes needed)**

---

## Task 7: Optional - Add Transformer Encoder Stack

**Files:**
- Modify: `week4/simple_transformer.py`

- [ ] **Step 1: Add TransformerEncoder class**

```python
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_heads, intermediate_size,
                 num_layers, max_seq_len=512, dropout_rate=0.1):
        super().__init__()
        self.embedding = SimpleEmbedding(vocab_size, hidden_size, max_seq_len)
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads, intermediate_size, dropout_rate)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, mask=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.layer_norm(x)
        return x
```

- [ ] **Step 2: Update main() to demonstrate multi-layer usage**

```python
def main():
    # 配置参数
    vocab_size = 1000
    hidden_size = 768
    num_heads = 12
    intermediate_size = 3072
    max_seq_len = 512
    dropout_rate = 0.1
    batch_size = 2
    seq_len = 10
    num_layers = 3  # 3层transformer

    print("=== 单层Transformer示例 ===")
    # 创建模型
    embedding = SimpleEmbedding(vocab_size, hidden_size, max_seq_len)
    transformer_layer = TransformerLayer(hidden_size, num_heads, intermediate_size, dropout_rate)

    # 构造输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 前向传播
    print("输入形状:", input_ids.shape)
    x = embedding(input_ids)
    print("Embedding输出形状:", x.shape)
    output = transformer_layer(x)
    print("Transformer层输出形状:", output.shape)
    assert output.shape == (batch_size, seq_len, hidden_size)
    print("✓ 单层形状验证通过！\n")

    print("=== 多层TransformerEncoder示例 ===")
    encoder = TransformerEncoder(vocab_size, hidden_size, num_heads,
                                  intermediate_size, num_layers, max_seq_len, dropout_rate)
    encoder_output = encoder(input_ids)
    print("Encoder输出形状:", encoder_output.shape)
    assert encoder_output.shape == (batch_size, seq_len, hidden_size)
    print("✓ 多层形状验证通过！")
```

- [ ] **Step 3: Run and verify**

```bash
python week4/simple_transformer.py
```

- [ ] **Step 4: Commit**

```bash
git add week4/simple_transformer.py
git commit -m "feat: add TransformerEncoder with multiple layers"
```
