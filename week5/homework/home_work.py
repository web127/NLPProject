"""
基于Transformer的轻量级字符级LLM语言模型
堆叠6层Transformer，支持预训练和文本续写生成
用法:
    python home_work.py --mode train --epochs 20 --num_layers 6
    python home_work.py --mode generate --prompt "今天股市" --top_k 40 --top_p 1.0
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ─────────────────────────── Transformer组件 ───────────────────────────

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        assert self.head_size * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_size)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)

        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        new_shape = attention_output.size()[:-2] + (self.hidden_size,)
        attention_output = attention_output.view(*new_shape)

        output = self.output_linear(attention_output)
        return output


class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerLayer(nn.Module):
    """Transformer解码器层（带因果mask）"""
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout_rate=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(hidden_size, num_heads)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout_rate)

        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        residual = x
        x = self.layer_norm1(x)
        x = self.self_attention(x, x, x, mask)
        x = self.dropout1(x)
        x = x + residual

        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout2(x)
        x = x + residual

        return x


class PositionalEncoding(nn.Module):
    """正弦余弦位置编码"""
    def __init__(self, embed_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ─────────────────────────── Transformer语言模型 ───────────────────────────

class TransformerLM(nn.Module):
    """
    基于Transformer的字符级语言模型
    6层Transformer堆叠，支持自回归生成
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_heads, dropout):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout)

        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=embed_dim,
                num_heads=num_heads,
                intermediate_size=hidden_dim,
                dropout_rate=dropout
            )
            for _ in range(num_layers)
        ])

        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        self.lm_head.bias.data.zero_()
        self.lm_head.weight.data.uniform_(-initrange, initrange)

    def get_causal_mask(self, seq_len, device):
        """生成因果mask，防止看到未来的token"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask[None, None, :, :]

    def forward(self, src, mask=None):
        x = self.token_embedding(src)
        x = x * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)

        if mask is None:
            mask = self.get_causal_mask(src.size(1), src.device)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.final_layer_norm(x)
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None, stop_tokens=None):
        """
        自回归生成文本，支持top-k和top-p采样
        stop_tokens: 提前停止的token列表（如换行符、句号等）
        """
        self.eval()
        original_len = input_ids.size(1)

        for _ in range(max_new_tokens):
            input_cond = input_ids
            logits = self(input_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            if top_p is not None and top_p < 1.0:
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if stop_tokens is not None and next_token.item() in stop_tokens:
                break

        return input_ids


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: TransformerLM({args.num_layers} layers)")

    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        scheduler.step()

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


def generate_text(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.load, map_location=device)
    char2idx = checkpoint["char2idx"]
    idx2char = checkpoint["idx2char"]
    saved_args = argparse.Namespace(**checkpoint["args"])

    model = TransformerLM(
        vocab_size=len(char2idx),
        embed_dim=saved_args.embed_dim,
        hidden_dim=saved_args.hidden_dim,
        num_layers=saved_args.num_layers,
        num_heads=saved_args.num_heads,
        dropout=saved_args.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    prompt = args.prompt
    print(f"提示词: {prompt}")

    input_ids = []
    for c in prompt:
        if c in char2idx:
            input_ids.append(char2idx[c])
        else:
            print(f"警告: 字符 '{c}' 不在词表中，已跳过")

    if not input_ids:
        input_ids = [char2idx.get(list(char2idx.keys())[0], 0)]

    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    stop_tokens = []
    if args.stop_chars:
        for c in args.stop_chars:
            if c in char2idx:
                stop_tokens.append(char2idx[c])
        if stop_tokens:
            print(f"提前停止字符: {args.stop_chars}")

    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=args.max_len,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop_tokens=stop_tokens if stop_tokens else None
        )

    generated = output_ids[0].tolist()
    result = "".join([idx2char.get(i, "") for i in generated])

    print(f"生成结果: {result}")
    print(f"生成长度: {len(generated) - len(input_ids)} 个字符")
    return result


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       default="train", choices=["train", "generate"])

    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--embed_dim",  type=int,   default=256)
    parser.add_argument("--hidden_dim", type=int,   default=512)
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
