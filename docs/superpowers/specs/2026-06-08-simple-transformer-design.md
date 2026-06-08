# Simple Transformer Layer 设计文档

## 概述

使用PyTorch实现一个简易但完整的Transformer编码器层，用于学习理解Transformer的核心原理。

## 目标

- 实现标准版Transformer编码器层
- 包含多头注意力、残差连接、LayerNorm
- 提供简单的embedding层和使用示例
- 代码清晰，便于学习

## 架构设计

### 组件结构

```
simple_transformer.py
├── MultiHeadAttention      # 多头注意力机制
├── FeedForward             # 前馈网络
├── TransformerLayer        # 完整的Transformer层
├── SimpleEmbedding         # 简单的embedding层
└── usage_example           # 使用示例
```

### 数据流

```
输入序列 (batch_size, seq_len)
    ↓
SimpleEmbedding (token + position)
    ↓ (batch_size, seq_len, hidden_size)
TransformerLayer
    ├─ LayerNorm
    ├─ MultiHeadAttention
    ├─ 残差连接
    ├─ LayerNorm
    ├─ FeedForward
    └─ 残差连接
    ↓
输出 (batch_size, seq_len, hidden_size)
```

## 详细设计

### 1. MultiHeadAttention

**输入：**
- query, key, value: (batch_size, seq_len, hidden_size)
- mask: (batch_size, 1, seq_len, seq_len) 可选

**输出：**
- attention_output: (batch_size, seq_len, hidden_size)

**实现细节：**
```python
# 线性映射
Q = Linear(hidden_size, hidden_size)(query)
K = Linear(hidden_size, hidden_size)(key)
V = Linear(hidden_size, hidden_size)(value)

# 分头 (batch_size, num_heads, seq_len, head_size)
Q = reshape_and_transpose(Q)
K = reshape_and_transpose(K)
V = reshape_and_transpose(V)

# 缩放点积注意力
attention_scores = Q @ K.transpose(-2, -1) / sqrt(head_size)
attention_weights = softmax(attention_scores, dim=-1)
attention_output = attention_weights @ V

# 拼接并输出
output = Linear(hidden_size, hidden_size)(concat(attention_output))
```

### 2. FeedForward

**结构：**
```
Linear(hidden_size, intermediate_size)
    ↓
GELU激活
    ↓
Linear(intermediate_size, hidden_size)
```

### 3. TransformerLayer

**结构：**
```
x → LayerNorm → MultiHeadAttention → dropout → + → LayerNorm → FeedForward → dropout → + → output
└───────────────── residual ──────────────────┘   └────────────────── residual ──────────────────┘
```

### 4. SimpleEmbedding

**组成：**
- TokenEmbedding: (vocab_size, hidden_size)
- PositionEmbedding: (max_seq_len, hidden_size)
- 输出: token_emb + position_emb

## 配置参数

```python
hidden_size = 768
num_heads = 12
intermediate_size = 3072  # 4 * hidden_size
max_seq_len = 512
dropout_rate = 0.1
vocab_size = 1000  # 示例用
```

## 使用示例

```python
# 创建模型
embedding = SimpleEmbedding(vocab_size, hidden_size, max_seq_len)
transformer_layer = TransformerLayer(hidden_size, num_heads, intermediate_size, dropout_rate)

# 前向传播
input_ids = torch.randint(0, vocab_size, (2, 10))  # batch_size=2, seq_len=10
x = embedding(input_ids)
output = transformer_layer(x)

print(output.shape)  # (2, 10, 768)
```

## 文件位置

- 实现文件: `week4/simple_transformer.py`
