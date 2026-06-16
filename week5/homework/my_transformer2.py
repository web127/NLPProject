import numpy as np
import torch.nn as nn
import torch
import math


input_tensor = torch.randn(3,5,10)
print(input_tensor)
linear = nn.Linear(10, 5)
output = linear(input_tensor)
# 查看输出的形状
print(output.shape)

print(output)
#定义一个模型输入是batch_size,seq_len,hidden_size长度
class MyTransformerModel(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.q_linear = nn.Linear(hidden_size, hidden_size)  # 输出是batch_size,seq_len,hidden_size
        self.k_linear = nn.Linear(hidden_size, hidden_size)  # 输出是batch_size,seq_len,hidden_size
        self.v_linear = nn.Linear(hidden_size, hidden_size)  # 输出是batch_size,seq_len,hidden_size
        self.ffn = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.Tanh(),
                                 nn.Linear(hidden_size * 4, hidden_size))


    def forward(self,x):
        batch_size,seq_len,hidden_size = x.shape
        q=self.q_linear(x).view(batch_size,seq_len,self.num_heads,self.head_size).transpose(1,2)#输出形状是(B, H, L, D)
        k=self.k_linear(x).view(batch_size,seq_len,self.num_heads,self.head_size).transpose(1,2)#输出形状是(B, H, L, D)
        v=self.v_linear(x).view(batch_size,seq_len,self.num_heads,self.head_size).transpose(1,2)#输出形状是(B, H, L, D)
        # 多头注意力计算
        score = nn.functional.softmax((q@k.transpose(-2,-1))/math.sqrt(self.head_size), dim=-1)#输出形状是B,H,L,L
        out = score @ v #输出形状是B,H,L,D
        # 合并多头
        out = out.transpose(1,2).contiguous().view(batch_size,seq_len,-1) #输出形状为B,L,Hidden_size

        # 残差+归一化
        out1 = nn.functional.layer_norm(out+x, normalized_shape=(self.hidden_size,))#输出形状是batch_size,seq_len,hidden_size//num_heads
        out2 = self.ffn(out1)#输出形状是batch_size,seq_len,hidden_size//num_heads

        # 残差+归一化
        out_final = nn.functional.layer_norm(out2 + out1, normalized_shape=(self.hidden_size,))
        return out_final


