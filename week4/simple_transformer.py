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
