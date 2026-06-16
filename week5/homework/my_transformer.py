import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        # 确保hidden_size可以被num_heads整除
        assert self.head_size * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

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


class FeedForward(nn.Module):
    """前馈网络"""
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


class TransformerLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout_rate=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(hidden_size, num_heads)
        self.feed_forward = FeedForward(hidden_size, intermediate_size)

        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        # 第一个子层：自注意力 + 残差连接 (Pre-LN结构)
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


class PositionalEncoding(nn.Module):
    """位置编码 - 使用正弦余弦函数"""
    def __init__(self, embed_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 计算位置编码
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """完整的Transformer模型"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout):
        super().__init__()

        self.model_type = model_type
        self.embed_dim = embed_dim

        # 词表到向量映射
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # 位置编码
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout)

        # Transformer层
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=embed_dim,
                num_heads=8,  # 默认8头
                intermediate_size=hidden_dim,
                dropout_rate=dropout
            )
            for _ in range(num_layers)
        ])

        # 最后的LayerNorm
        self.final_layer_norm = nn.LayerNorm(embed_dim)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, mask=None):
        """
        Args:
            src: 输入序列 (batch_size, seq_len)
            mask: 注意力mask (可选)

        Returns:
            output: Transformer输出 (batch_size, seq_len, embed_dim)
        """
        # 词嵌入
        x = self.token_embedding(src)  # (batch_size, seq_len, embed_dim)

        # 乘以sqrt(d_model)，这是Transformer论文中的技巧
        x = x * math.sqrt(self.embed_dim)

        # 位置编码
        x = self.pos_encoder(x)

        # 通过Transformer层
        for layer in self.layers:
            x = layer(x, mask)

        # 最后的LayerNorm
        x = self.final_layer_norm(x)

        return x


def test_transformer():
    """测试Transformer层"""
    print("=== 测试Transformer层 ===")

    # 配置参数
    vocab_size = 10000
    embed_dim = 512
    hidden_dim = 2048
    num_layers = 6
    model_type = "encoder"
    dropout = 0.1
    batch_size = 2
    seq_len = 10

    # 创建模型
    model = TransformerModel(vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout)

    # 打印模型结构
    print(f"模型结构:\n{model}\n")

    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数数量: {total_params:,}\n")

    # 构造输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"输入形状: {input_ids.shape}")

    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(input_ids)

    print(f"输出形状: {output.shape}")

    # 验证输出形状
    assert output.shape == (batch_size, seq_len, embed_dim), \
        f"期望形状 {(batch_size, seq_len, embed_dim)}，实际形状 {output.shape}"

    print("✓ Transformer层测试通过！")

    # 测试训练模式
    model.train()
    output = model(input_ids)
    print("✓ 训练模式测试通过！")


if __name__ == "__main__":
    test_transformer()
