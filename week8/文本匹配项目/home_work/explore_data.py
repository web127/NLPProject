"""
AFQMC 数据集探索与可视化

这个脚本非常适合初学者建立“数据感”。
在真正训练模型之前，先回答几个问题：
  - 数据一共有多少条？
  - 正负样本是否均衡？
  - 句子通常有多长？
  - max_length 设 32、64 还是 128 更合理？
  - 数据本身有没有偏差，可能让模型学到“投机技巧”？

教学重点：
  1. 文本匹配数据的结构——sentence pair + binary label，与分类任务的本质区别
  2. 类别不均衡（~31% 正例）——实际业务中"不同义"的问题对比"同义"更多见
  3. 句子长度分布——BERT max_length 截断阈值的选择依据
  4. 正/负样本的长度差异——是否存在"长句倾向于不相似"的捷径（shortcut）
  5. Token 数 vs 字符数——BERT 中文字节对编码的粒度

建议的阅读顺序：
  1. main()：理解总流程
  2. print_stats()：看控制台统计都输出了什么
  3. 各个 plot_xxx()：看每张图分别在回答什么问题

使用方式：
  python explore_data.py
  python explore_data.py --data_dir ../data/bq_corpus --output_dir ../hm_outputs/figures

依赖：
  pip install matplotlib transformers
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from transformers import BertTokenizer

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "afqmc"
BERT_PATH  = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
OUTPUT_DIR = ROOT / "hw_outputs" / "figures"

# 中文字体（matplotlib 默认不支持中文）
_CN_FONT = None


def _find_chinese_font_path():
    """优先选择明确支持中文的字体，避免误选到不支持中文的字体。

    初学者常见疑问：为什么画图还要处理字体？
    因为 matplotlib 默认字体经常不包含中文字符，
    如果不处理，就会看到很多 “Glyph missing” 告警，或者图里中文显示成方块。
    """
    preferred_keywords = [
        "pingfang",
        "hiragino sans gb",
        "stheiti",
        "heiti",
        "arial unicode",
        "simhei",
        "microsoft yahei",
        "msyh",
        "simsun",
        "noto sans cjk",
        "noto serif cjk",
        "source han sans",
        "source han serif",
    ]
    for path in fm.findSystemFonts():
        lower = path.lower()
        if any(keyword in lower for keyword in preferred_keywords):
            return path
    return None


def _get_font():
    # 这里做了一个“小缓存”：第一次找到字体后先保存起来，
    # 后面再次画图时就不用重复扫描系统字体了。
    global _CN_FONT
    if _CN_FONT is None:
        font_path = _find_chinese_font_path()
        if font_path:
            _CN_FONT = fm.FontProperties(fname=font_path)
            font_name = _CN_FONT.get_name()
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
        else:
            _CN_FONT = fm.FontProperties()
    return _CN_FONT


def load_jsonl(path):
    """读取 JSONL 文件。

    例子：
      一行数据可能是：
      {"sentence1": "花呗可以提现吗", "sentence2": "花呗能提现吗", "label": 1}
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── 图 1：标签分布 ────────────────────────────────────────────────────────

def plot_label_distribution(splits_data, output_dir):
    """图 1：观察 train / validation / test 的标签分布。

    这张图主要帮助回答：
      “数据是不是均衡的？”
    如果明显不均衡，后续就要更加关注 F1，而不是只看 accuracy。
    """
    fig, axes = plt.subplots(1, len(splits_data), figsize=(10, 4))
    if len(splits_data) == 1:
        axes = [axes]

    fp = _get_font()
    for ax, (split_name, rows) in zip(axes, splits_data.items()):
        labels = [r["label"] for r in rows]
        cnt = Counter(labels)
        counts = [cnt.get(0, 0), cnt.get(1, 0)]
        # 这里不用直接传中文字符串给 bar，而是先用数字位置，
        # 再单独设置 xticklabels，这样布局更稳定。
        tick_labels = ["不相似 (0)", "相似 (1)"]
        bars = ax.bar(range(len(tick_labels)), counts,
                      color=["#F44336", "#2196F3"], width=0.5)
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    f"{c}\n({c/len(rows)*100:.1f}%)", ha="center", va="bottom",
                    fontproperties=fp, fontsize=9)
        ax.set_title(f"{split_name}（{len(rows):,} 条）", fontproperties=fp)
        ax.set_ylabel("数量", fontproperties=fp)
        ax.set_xticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels, fontproperties=fp)
        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle("AFQMC 标签分布（正例约 31%，负例约 69%）", fontproperties=fp,
                 fontsize=12, y=0.98)
    fig.subplots_adjust(top=0.82, bottom=0.2, wspace=0.3)
    save_path = output_dir / "label_distribution.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")


# ── 图 2：句子字符长度分布 ────────────────────────────────────────────────

def plot_char_length(rows, output_dir):
    """图 2：统计字符级长度分布。

    这张图最重要的用途是帮助决定 max_length。
    例如如果 95% 句子长度都不超过 24 个字符，
    那么把 max_length 设得特别大，可能只会浪费算力。
    """
    pos_rows = [r for r in rows if r["label"] == 1]
    neg_rows = [r for r in rows if r["label"] == 0]

    def lens(rs):
        # 一个样本有 sentence1 和 sentence2，
        # 所以这里把两边句子的长度都收集起来一起统计。
        return [len(r["sentence1"]) for r in rs] + [len(r["sentence2"]) for r in rs]

    pos_lens = lens(pos_rows)
    neg_lens = lens(neg_rows)

    fp = _get_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pos_lens, bins=40, alpha=0.6, label="正样本（相似）",
            color="#2196F3", density=True)
    ax.hist(neg_lens, bins=40, alpha=0.6, label="负样本（不相似）",
            color="#F44336", density=True)
    ax.axvline(32, color="black", linestyle="--", linewidth=1,
               label="max_length=32")
    ax.axvline(64, color="gray", linestyle="--", linewidth=1,
               label="max_length=64")
    ax.set_xlabel("句子字符长度", fontproperties=fp)
    ax.set_ylabel("密度", fontproperties=fp)
    ax.set_title("正/负样本句子长度分布（train）", fontproperties=fp)
    ax.legend(prop=fp)
    fig.tight_layout()

    save_path = output_dir / "char_length_distribution.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")

    all_lens = [len(r["sentence1"]) for r in rows] + [len(r["sentence2"]) for r in rows]
    print(f"  字符长度统计（train 全部句子）：")
    print(f"    均值={np.mean(all_lens):.1f}  中位数={np.median(all_lens):.0f}  "
          f"P95={np.percentile(all_lens, 95):.0f}  最长={max(all_lens)}")
    for threshold in [32, 48, 64, 96]:
        cover = sum(1 for l in all_lens if l <= threshold) / len(all_lens) * 100
        print(f"    max_length={threshold:3d} 覆盖率: {cover:.1f}%")


# ── 图 3：Token 数分布（BERT Tokenizer） ─────────────────────────────────

def plot_token_length(rows, tokenizer, output_dir):
    """图 3：统计 tokenizer 分词后的 token 长度。

    为什么不仅看字符数，还要看 token 数？
      因为 BERT 真正处理的不是“字符数”，而是“token 数”。
      对中文来说，字符数和 token 数通常接近，但并不总是完全相同。
    """
    print("  计算 Token 长度（需要 tokenize，稍慢...）")
    token_lens = []
    # 为了避免探索脚本过慢，这里只抽样前 5000 条训练数据。
    # 对数据分布探索来说，这样通常已经够用了。
    for r in rows[:5000]:
        t1 = len(tokenizer.tokenize(r["sentence1"]))
        t2 = len(tokenizer.tokenize(r["sentence2"]))
        token_lens.extend([t1, t2])

    fp = _get_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(token_lens, bins=40, color="#4CAF50", alpha=0.8, density=True)
    ax.axvline(np.mean(token_lens), color="red", linestyle="-",
               label=f"均值={np.mean(token_lens):.1f}")
    ax.axvline(np.percentile(token_lens, 95), color="orange", linestyle="--",
               label=f"P95={np.percentile(token_lens, 95):.0f}")
    ax.set_xlabel("单句 Token 数（不含 [CLS]/[SEP]）", fontproperties=fp)
    ax.set_ylabel("密度", fontproperties=fp)
    ax.set_title("单句 Token 数分布（train 前 5000 条）", fontproperties=fp)
    ax.legend(prop=fp)
    fig.tight_layout()

    save_path = output_dir / "token_length_distribution.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")
    print(f"  Token 长度：均值={np.mean(token_lens):.1f}  "
          f"P95={np.percentile(token_lens, 95):.0f}  最长={max(token_lens)}")


# ── 图 4：正/负样本长度差（捷径检测） ────────────────────────────────────

def plot_length_diff(rows, output_dir):
    """
    检测长度差是否可作为"判别捷径"：
    若正样本句子长度差 << 负样本，则模型可能学到"长度接近 → 相似"这个捷径。
    教学价值：启发学生思考数据集偏差对模型泛化的影响。

    例子：
      如果正样本大多“长度很接近”，负样本大多“长度差很大”，
      模型就可能偷懒，只根据长度差做判断，而不是认真理解语义。
    """
    pos_diffs = [abs(len(r["sentence1"]) - len(r["sentence2"]))
                 for r in rows if r["label"] == 1]
    neg_diffs = [abs(len(r["sentence1"]) - len(r["sentence2"]))
                 for r in rows if r["label"] == 0]

    fp = _get_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pos_diffs, bins=30, alpha=0.6, label=f"正样本 均值={np.mean(pos_diffs):.1f}",
            color="#2196F3", density=True)
    ax.hist(neg_diffs, bins=30, alpha=0.6, label=f"负样本 均值={np.mean(neg_diffs):.1f}",
            color="#F44336", density=True)
    ax.set_xlabel("|len(s1) - len(s2)| 字符数", fontproperties=fp)
    ax.set_ylabel("密度", fontproperties=fp)
    ax.set_title("正/负样本句子长度差分布（length bias 检测）", fontproperties=fp)
    ax.legend(prop=fp)
    fig.tight_layout()

    save_path = output_dir / "length_diff_distribution.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")
    print(f"  长度差：正样本均值={np.mean(pos_diffs):.1f}  负样本均值={np.mean(neg_diffs):.1f}")
    if np.mean(pos_diffs) < np.mean(neg_diffs) * 0.7:
        print("  ⚠️  正样本长度差明显更小，存在 length bias 风险")
    else:
        print("  ✓ 正/负样本长度差接近，无明显 length bias")


# ── 控制台统计输出 ────────────────────────────────────────────────────────

def print_stats(name, rows):
    """打印单个数据划分的基础统计信息。

    建议初学者特别关注：
      1. 正负样本比例
      2. 句长分布
      3. 示例样本是否符合直觉
    """
    labels  = [r["label"] for r in rows]
    cnt     = Counter(labels)
    s1_lens = [len(r["sentence1"]) for r in rows]
    s2_lens = [len(r["sentence2"]) for r in rows]
    all_lens = s1_lens + s2_lens

    print(f"\n{'='*50}")
    print(f"【{name}】共 {len(rows):,} 条")
    print(f"{'='*50}")

    n_pos = cnt.get(1, 0)
    n_neg = cnt.get(0, 0)
    n_unlabeled = sum(v for k, v in cnt.items() if k not in (0, 1))
    if n_unlabeled:
        # AFQMC 竞赛测试集标签为 -1（未公开），不计入正/负统计
        print(f"  标签未公开（CLUE 竞赛格式）: {n_unlabeled:>6,} 条  —— 仅供参考，不用于评估")
    else:
        print(f"  正样本（相似）  : {n_pos:>6,} ({n_pos/len(rows)*100:.1f}%)")
        print(f"  负样本（不相似）: {n_neg:>6,} ({n_neg/len(rows)*100:.1f}%)")
        print(f"  不均衡比 (neg/pos): {n_neg/max(n_pos, 1):.1f}x")
    print(f"  句子字符长度 — 均值={np.mean(all_lens):.1f}  中位数={np.median(all_lens):.0f}  "
          f"P95={np.percentile(all_lens, 95):.0f}  最长={max(all_lens)}")
    print(f"  示例正样本：")
    for r in [r for r in rows if r["label"] == 1][:2]:
        print(f"    ✓  {r['sentence1']!r}  ||  {r['sentence2']!r}")
    print(f"  示例负样本：")
    for r in [r for r in rows if r["label"] == 0][:2]:
        print(f"    ✗  {r['sentence1']!r}  ||  {r['sentence2']!r}")


# ── 主流程 ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="AFQMC 数据集探索")
    parser.add_argument("--data_dir",   default=str(DATA_DIR),   type=Path)
    parser.add_argument("--bert_path",  default=str(BERT_PATH),  type=str)
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR), type=Path)
    parser.add_argument("--skip_token", action="store_true", help="跳过 Token 长度分析（较慢）")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 第 1 步：读取 train / validation / test 三个划分。
    splits = {}
    for split in ["train", "validation", "test"]:
        path = args.data_dir / f"{split}.jsonl"
        if path.exists():
            splits[split] = load_jsonl(path)

    # 第 2 步：先打印控制台统计，帮助我们快速感知数据规模和样例。
    for name, rows in splits.items():
        print_stats(name, rows)

    train_rows = splits.get("train", [])
    if not train_rows:
        print("train.jsonl 不存在，请先运行 download_data.py")
        return

    print(f"\n{'='*50}")
    print("生成可视化图表...")

    # 第 3 步：依次生成图表。
    # 推荐你第一次运行后，按下面顺序看图：
    #   先看标签分布 -> 再看长度分布 -> 最后看 length bias 和 token 分布

    plot_label_distribution(splits, args.output_dir)
    plot_char_length(train_rows, args.output_dir)
    plot_length_diff(train_rows, args.output_dir)

    if not args.skip_token:
        # 只有在需要 token 统计时，才加载 tokenizer。
        # 这样可以让“只看基础图”的情况更快启动。
        tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        plot_token_length(train_rows, tokenizer, args.output_dir)

    print(f"\n所有图表已保存至 → {args.output_dir}")


if __name__ == "__main__":
    main()
