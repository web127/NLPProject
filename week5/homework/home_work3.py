
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim

from torch.utils.data import Dataset


# 加载语料
def load_corpus(pattern="*.txt"):
    corpus = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            corpus.append(f.read())
    return "".join(corpus)



# 构建词汇表
def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


# 构建数据集
class CharDataset(Dataset):
    """字符级数据集类，继承自PyTorch的Dataset"""
    def __init__(self, text, char2idx, seq_len):
        """初始化数据集
        Args:
            text: 原始文本字符串
            char2idx: 字符到索引的映射字典
            seq_len: 序列长度
        """
        self.seq_len = seq_len  # 存储序列长度
        # 将文本转换为索引序列，过滤不在词表中的字符
        ids = [char2idx[c] for c in text if c in char2idx]
        # 转换为PyTorch张量（长整型）
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        """返回数据集的样本数量"""
        # 每个样本需要seq_len+1个字符（x占seq_len，y占seq_len）
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        """获取单个样本
        Args:
            idx: 样本索引
        Returns:
            x: 输入序列（从idx开始的seq_len个字符）
            y: 目标序列（从idx+1开始的seq_len个字符，即x右移一位）
        """
        x = self.data[idx: idx + self.seq_len]      # 输入序列
        y = self.data[idx + 1: idx + self.seq_len + 1]  # 目标序列（x向右移一位）
        return x, y

# 定义mini版LLM模型
class MiniLLMModel(nn.Module):
    def __init__(self, vocab_size,  dropout,hidden_size, num_heads, embedding_dim, num_layers, max_seq_len=512):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers # Transformer层数
        self.max_seq_len = max_seq_len

       # Token嵌入层：将字符索引转换为向量
        self.char_embedding = CharEmbedding(vocab_size, embedding_dim,max_seq_len)

       # 必须这么写，才是多层
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        # 最终LayerNorm
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

        self.out_vocab = nn.Linear(hidden_size, vocab_size)#将状态映射到词表大小

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





def train(model, optimizer, loss_fn, train_loader, epoch_num):
    model.train()
    seqlen = 64
    text = load_corpus("*.txt")
    char2idx, idx2char = build_vocab(text)
    epoch_num = 10

    data = CharDataset(text, char2idx, seqlen)
    train_data = train_loader.load_data(data, shuffle=True)
    test_data = train_loader.load_data(data, shuffle=False) #测试数据
    criterion = nn.CrossEntropyLoss()
    # 定义优化器，设置策略?
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epoch_num):
        for batch_idx,x,y in enumerate(train_data):
            logits=model(x)
            loss=criterion(logits,y)# 这里损失函数计算出来做什么
            optimizer.step()# 这里更新参数，都更新哪些参数，这么多层，包括embedding层
            optimizer.zero_grad()

 # 使用模型
def use_model(model, char2idx, idx2char, seqlen, max_seq_len=512):
    prompts="你好"
    ids=torch.tensor([char2idx[c] for c in prompts], dtype=torch.long).unsqueeze(0)

    count=0
    with torch.no_grad():
        while True:
          logit=model(ids)
        #增加采样策略，先使用最大概率的字符
          top_index = torch.argmax(logit, dim=-1, keepdim=True)
          new_wd = idx2char[top_index.item()]

          count=count+1
          if count>=100:
              return
          ids=torch.cat([ids,top_index], dim=1)


    #定义数据集
    #定义损失函数
    #定义优化器
    #state_dict



## 定义transformer模型
## 定义ll模型

## 训练模型
## 评估模型
## 测试模型
