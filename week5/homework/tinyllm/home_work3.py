
import glob
import torch
import torch.nn as nn
import math
import torch.optim as optim
import random
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
from week5.homework.tinyllm.mini_llm_model import MiniLLMModel

"""
基于Transformer的轻量级字符级LLM语言模型
堆叠6层Transformer，支持预训练和文本续写生成
用法:
    python home_work.py --mode train --epochs 20 --num_layers 6
    python home_work.py --mode generate --prompt "今天股市" --top_k 40 --top_p 1.0
"""

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

       # chunk=self.data[idx]
       # return chunk[:-1],chunk[1:]
        return x, y



# 训练模型
def train_model(args):
    """训练模型的主函数"""
    # 选择设备（GPU优先）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: TransformerLM({args.num_layers} layers)")

    # 加载语料
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    # 构建词表
    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    # 划分训练集和验证集（按行划分，保持句子完整性）
    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))  # 划分点
    train_text = "\n".join(lines[:split])  # 训练文本
    val_text = "\n".join(lines[split:])  # 验证文本

    # 创建数据集
    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds = CharDataset(val_text, char2idx, args.seq_len)

    # 创建DataLoader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=True)

    # 创建模型
    model = MiniLLMModel(vocab_size,args.hidden_dim, args.embed_dim,args.num_heads, args.num_layers, args.dropout).to(device)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")



    criterion = nn.CrossEntropyLoss()
    # 定义优化器，设置策略?
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # todo 看下scheduler的用法
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)  # 余弦退火调度器

    best_val_ppl = float("inf")  # 最佳验证困惑度
    for epoch in range(1, args.epochs + 1):
        # 训练
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        # 验证（禁用梯度计算）
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        # 更新学习率
        scheduler.step()

        # 检查是否是最佳模型
        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            # 保存最佳模型
            torch.save({
                "model_state": model.state_dict(),  # 模型权重
                "char2idx": char2idx,  # 字符到索引映射
                "idx2char": idx2char,  # 索引到字符映射
                "args": vars(args),  # 训练参数
            }, args.save)

        # 打印日志
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")
# 训练或评估一个epoch
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """运行一个epoch的训练或评估
       Args:
           model: 模型实例
           loader: DataLoader
           criterion: 损失函数
           optimizer: 优化器
           device: 设备（cpu或cuda）
           train: 是否为训练模式
       Returns:
           avg_loss: 平均损失
           ppl: 困惑度（perplexity）
       """
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    for x,y in loader:
        x = x.to(device)
        y = y.to(device)
        logits=model(x)
        #计算输出批次上每个序列位置的logit(batch_size,seq_len,vocab_size) 和 目标序列(batch_size,seq_len) 的损失
        loss=criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if train:
            optimizer.zero_grad()
            loss.backward()
            #这句话什么意思
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪（防止梯度爆炸）
            optimizer.step()
        # 累计损失和token数
        total_loss += loss.item()
        total_tokens += y.numel()

    # 计算平均损失和困惑度
    avg_loss = total_loss / total_tokens  # 平均损失（per-token）
    ppl = math.exp(avg_loss)  # 困惑度（指数形式）
    return avg_loss, ppl


# 使用模型
def generate_text(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载预训练模型,保存模型的时候存了这些参数
    checkpoint = torch.load(args.load, map_location=device)
    char2idx = checkpoint["char2idx"]  # 字符到索引映射
    idx2char = checkpoint["idx2char"]  # 索引到字符映射
    saved_args = argparse.Namespace(**checkpoint["args"])  # 训练时的参数

    model = MiniLLMModel(
        vocab_size = len(char2idx),
        hidden_dim= saved_args.hidden_dim,
        embedding_dim = saved_args.embedding_dim,
        num_layers = saved_args.num_layers,
        num_heads = saved_args.num_heads,
        dropout = saved_args.dropout,
    ).to(device)
    # 加载模型权重
    model.load_state_dict(checkpoint["model_state"])
    model.eval()  # 设置为评估模式

    # 处理提示词
    prompts="你好"
    print(f"提示词: {prompts}")
    # 将提示词转换为索引序列
    input_ids = []
    for c in prompts:
        if c in char2idx:
            input_ids.append(char2idx[c])
        else:
            print(f"警告: 字符 '{c}' 不在词表中，已跳过")
 # 如果提示词为空或所有字符都不在词表中，使用第一个字符
    if not input_ids:
        input_ids = [char2idx.get(list(char2idx.keys())[0], 0)]

    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # 处理停止字符
    stop_tokens = []
    if args.stop_chars:
        for c in args.stop_chars:
            if c in char2idx:
                stop_tokens.append(char2idx[c])
        if stop_tokens:
            print(f"提前停止字符: {args.stop_chars}")

    with torch.no_grad():
       output_ids = model_generate(
            model,
            input_tensor,
            max_new_tokens=args.max_len,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop_tokens=stop_tokens if stop_tokens else None
       )
       # 将索引转换为文本
    generated = output_ids[0].tolist() #取第一个样本的生成序列,生成的时候，批次是1
    result = "".join([idx2char.get(i, "") for i in generated])

       # 打印结果
    print(f"生成结果: {result}")
    print(f"生成长度: {len(generated) - len(input_ids)} 个字符")
    return result

def model_generate(model,input_ids,max_new_tokens,temperature,top_k,top_p,stop_tokens=None):
    """
            自回归生成文本，支持top-k和top-p采样
            Args:
                input_ids: 输入序列 (batch_size, seq_len)
                max_new_tokens: 最大生成token数
                temperature: 温度参数（控制随机性）
                top_k: top-k采样参数
                top_p: top-p（nucleus）采样参数
                stop_tokens: 提前停止的token列表（如换行符、句号等）
            Returns:
                input_ids: 包含生成内容的完整序列
            """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # 设置为评估模式
    original_len = input_ids.size(1)

    for _ in range(max_new_tokens):
        input_cond = input_ids
        logits= model(input_cond)
        logits = logits.select(dim=1, index=-1) /temperature  # 只取最后一个位置，应用温度缩放
        # Top-k采样
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, top_k)  # 取top-k个最高分
            logits[logits < v[:, [-1]]] = -float('Inf')  # 其他设为负无穷
        # Top-p采样（nucleus sampling）
        if top_p is not None and top_p < 1.0:
            probs = F.softmax(logits, dim=-1)  # 计算概率
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)  # 降序排序
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # 累积概率

            # 找到累积概率超过top_p的位置
            sorted_indices_to_remove = cumulative_probs > top_p
            # 保证至少保留一个token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # 将需要移除的位置映射回原始索引
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')  # 设为负无穷

        # Softmax归一化
        probs = F.softmax(logits, dim=-1)
         # 随机采样下一个token
        next_token = torch.multinomial(probs, num_samples=1)
         # 将新token追加到输入序列
        input_ids = torch.cat([input_ids, next_token], dim=1)
        # 检查是否遇到停止token
        if stop_tokens is not None and next_token.item() in stop_tokens:
            break
    return input_ids

# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       default="train", choices=["train", "generate"])

    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--embedding_dim",  type=int,   default=256)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--num_layers", type=int,   default=6)
    parser.add_argument("--num_heads",  type=int,   default=8)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--corpus",     default="../corpus.txt")
    parser.add_argument("--save",       default="transformer_lm.pt")
    parser.add_argument("--load",       default="transformer_lm.pt")

    parser.add_argument("--prompt",     default="今天股市")
    parser.add_argument("--max_len",    type=int,   default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k",      type=int,   default=None)
    parser.add_argument("--top_p",      type=float, default=0.9)
    parser.add_argument("--stop_chars", type=str,   default="\n。！？", help="遇到这些字符时提前停止生成")

    args = parser.parse_args()

    if args.mode == "train":
        train_model(args)
    else:
        generate_text(args)


if __name__ == "__main__":
    main()
