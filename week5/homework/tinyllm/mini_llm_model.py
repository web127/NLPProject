# 导入数学运算库
import math
# 导入命令行参数解析库
import argparse
# 导入文件路径匹配库
import glob
# 导入随机数库
import random
# 导入PyTorch深度学习框架
import torch
# 导入PyTorch神经网络模块
import torch.nn as nn
# 导入PyTorch函数式API
import torch.nn.functional as F


# 定义mini版LLM模型
class MiniLLMModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embedding_dim, num_heads, num_layers, dropout, max_seq_len=512):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers # Transformer层数
        self.max_seq_len = max_seq_len

       # Token嵌入层：将字符索引转换为向量
        self.char_embedding = CharEmbedding(vocab_size, embedding_dim,max_seq_len)

       # 必须这么写，才是多层
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        # 最终LayerNorm
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

        #输出的是batch_size, seq_len, vocab_size)
        self.out_vocab = nn.Linear(hidden_dim, vocab_size)#将状态映射到词表大小

        # 初始化权重
        self._init_weights()




    def forward(self, input_ids,mask=None):
        """前向传播
        Args:
            input_ids: 输入字符索引序列 (batch_size, seq_len)
            mask: 注意力掩码（可选）
        Returns:
            logits: 输出（PyTorch张量，形状为(batch_size, seq_len, vocab_size)）
        """
        # 分别获取批量大小和序列长度
        batch_size, seq_len = input_ids.size()
        # 确保序列长度不超过最大长度
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"

        # 使用简单的embedding模型
        x = self.char_embedding(input_ids)

        if mask is None: # 先创建因果mask，在注意力机制中使用
            mask =  self.get_causal_mask(input_ids.size(1), input_ids.device)

        # 调用num_layersm次多头自注意力层
        for layer in (self.layers):
            x = layer(x,mask)

        # LM头：预测下一个token
        x = self.final_layer_norm(x)

        # 语言模型头：将隐藏状态映射到词表大小
        logits = self.out_vocab(x)

        return logits

    def _init_weights(self): #todo 看下做什么
        """初始化模型权重"""
        initrange = 0.1  # 初始化范围
        # Token嵌入权重均匀初始化
        # 词嵌入 + 位置嵌入都初始化
        nn.init.uniform_(self.char_embedding.embedding.weight, -initrange, initrange)
        nn.init.uniform_(self.char_embedding.position_embedding.weight, -initrange, initrange)
        # LM头偏置初始化为0
        self.out_vocab.bias.data.zero_()
        # LM头权重均匀初始化
        self.out_vocab.weight.data.uniform_(-initrange, initrange)

    def get_causal_mask(self, seq_len, device):
        """生成因果mask，防止模型看到未来的token
        Args:
            seq_len: 序列长度
            device: 设备（cpu或cuda）
        Returns:
            mask: 因果掩码 (1, 1, seq_len, seq_len)
        """
        # 下三角矩阵：mask[i,j]=1表示位置i可以看到位置j
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=device))  # torch.ones(seq_len, seq_len)： 造一个 seq_len×seq_len 的全 1 方阵，
        # 添加两个维度（batch和head维度）
        return mask[None, None, :, :]



##定义embedding层
class CharEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim,max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids):
        """前向传播
        Args:
            input_ids: 输入文本（PyTorch张量，形状为(batch_size, seq_len)）
        Returns:
            embedding: 嵌入表示（PyTorch张量，形状为(batch_size, seq_len, embedding_dim)）
        """
        # 分别获取批量大小和序列长度
        batch_size, seq_len = input_ids.size()
        # 确保序列长度不超过最大长度
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"

        token_emb = self.embedding(input_ids)
        # position embedding
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        # 增加一维用于批量处理，扩展到(batch_size, seq_len)
        position_ids= position_ids.unsqueeze(0).expand(batch_size, seq_len)
        pos_emb= self.position_embedding(position_ids)

        return token_emb + pos_emb

# 定义Transformer层
# 输入的是embedding层的输出(batch_size, seq_len, hidden_size)
# 输出的是经过计算的序列，形状还是(batch_size, seq_len, hidden_size)
class TransformerLayer(nn.Module):
    def __init__(self,  hidden_size, num_heads, dropout_rate=0.1):
        super().__init__()
        self.self_attn = MultiSelfAttention(
           hidden_size,num_heads
        )

        # 定义FFN层
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        # 定义dropout层，堆叠时做dropout，防止过拟合
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)

    def forward(self, x,mask=None):
        """前向传播
              Args:
                  x: 输入张量,经过embedding层(batch_size, seq_len, embedding_dim)
                  mask: 注意力掩码
              Returns:
                  x: 输出张量
              """
        # 第一个子层：多头自注意力 + 残差连接
        residual = x

        x = self.layer_norm(x) # 注意是先预归一化
        x = self.self_attn(x,mask)
        # dropout是设置0.后加残差连接，保留特征信息
        x =self.dropout(x)
        x = x+residual
        # 第二个子层：前馈网络 + 残差连接
        residual = x
        x=self.layer_norm1(x)
        x= self.ffn(x)
        x=self.dropout1(x)
        x= x+residual
        return x #输出形状(batch_size, seq_len, hidden_size)

# 定义多头自注意力层
# 输入的是序列(batch_size, seq_len, hidden_size)
# 输出的是经过计算的序列，形状还是(batch_size, seq_len, hidden_size)
class MultiSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads  # 每个头的维度


        # 确保hidden_size能被num_heads整除
        assert self.head_size * num_heads == hidden_size, "hidden_size must be divisible by num_heads"


        self.q_w = nn.Linear(hidden_size, hidden_size)
        self.k_w = nn.Linear(hidden_size, hidden_size)
        self.v_w = nn.Linear(hidden_size, hidden_size)

        # 对多头注意力的结果做特征融合
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)



# 输出注意力分数
    #输入 x: (batch_size, seq_len, hidden_size)
    def forward(self, x, mask=None):

        # 动态获取维度
        batch_size, seq_len,hidden_size = x.shape
        head_size = hidden_size // self.num_heads

        #先进行线性变化，再进行重塑，输出形状(batch_size, num_heads, seq_len, head_size)
        q = self.q_w(x).reshape(batch_size, seq_len, self.num_heads,head_size).transpose(1, 2)# 输出形状(batch_size, num_heads, seq_len, head_size)
        k = self.k_w(x).reshape(batch_size, seq_len, self.num_heads, head_size).transpose(1, 2)
        v = self.v_w(x).reshape(batch_size, seq_len, self.num_heads, head_size).transpose(1, 2)


        # 计算注意力分数
        qk_dot = q@k.transpose(-2,-1) / math.sqrt(head_size)#输出形状(batch_size, num_heads, seq_len, seq_len)

        # 应用掩码（防止看到未来的token）
        if mask is not None:#
            qk_dot = qk_dot.masked_fill(mask == 0, -1e9)

        attention_scores = F.softmax(qk_dot, dim=-1)

        out=attention_scores@v#shape(batch_size, num_heads, seq_len, head_size)
        out = out.transpose_(1,2).contiguous().reshape(batch_size, seq_len, -1)

        # 对多头注意力的结果做特征融合
        out=self.out(out)
        return out
