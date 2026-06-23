"""
peoples_daily 数据集探索与可视化

教学重点：
  1. BIO 标注格式的统计方法
  2. 各实体类型的分布差异（为什么类别不均衡是NER的难点）
  3. 文本长度分布（影响 BERT max_length 的选择）
  4. 实体长度分布（短实体 vs 长实体的识别难度差异）

使用方式：
  python -m homework_src.explore_data

输出：
  homework_outputs/figures/entity_distribution.png   各类实体频次分布
  homework_outputs/figures/text_length_distribution.png  文本长度分布
  homework_outputs/figures/entity_length_distribution.png 实体长度分布
"""

import os

# 解决部分 macOS / Intel OpenMP 环境下的重复加载报错。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import argparse
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体候选列表，避免中文标题乱码。
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
FIG_DIR = ROOT / "homework_outputs" / "figures"


def load_split(split: str) -> list:
    """加载指定划分的数据。"""
    path = DATA_DIR / f"{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_stats(records: list) -> dict:
    """从数据中收集统计信息。"""
    entity_type_counts = Counter()
    entity_lengths = []
    text_lengths = []
    entity_per_sentence = []
    entities_by_type = {}
    all_tags = []

    for row in records:
        tokens = row["tokens"]
        ner_tags = row["ner_tags"]

        text_lengths.append(len(tokens))
        all_tags.extend(ner_tags)

        current_entity_type = None
        current_entity_length = 0

        for tag in ner_tags:
            if tag.startswith("B-"):
                if current_entity_type is not None:
                    entity_type_counts[current_entity_type] += 1
                    entity_lengths.append(current_entity_length)
                    if current_entity_type not in entities_by_type:
                        entities_by_type[current_entity_type] = []

                current_entity_type = tag[2:]
                current_entity_length = 1
            elif tag.startswith("I-"):
                if current_entity_type == tag[2:]:
                    current_entity_length += 1
                else:
                    # 标签不一致，开始新实体
                    if current_entity_type is not None:
                        entity_type_counts[current_entity_type] += 1
                        entity_lengths.append(current_entity_length)
                    current_entity_type = tag[2:]
                    current_entity_length = 1
            else:
                if current_entity_type is not None:
                    entity_type_counts[current_entity_type] += 1
                    entity_lengths.append(current_entity_length)
                    current_entity_type = None
                    current_entity_length = 0

        if current_entity_type is not None:
            entity_type_counts[current_entity_type] += 1
            entity_lengths.append(current_entity_length)

    for row in records:
        ner_tags = row["ner_tags"]
        entity_count = sum(1 for tag in ner_tags if tag.startswith("B-"))
        entity_per_sentence.append(entity_count)

    return {
        "entity_type_counts": entity_type_counts,
        "entity_lengths": entity_lengths,
        "text_lengths": text_lengths,
        "entity_per_sentence": entity_per_sentence,
        "entities_by_type": entities_by_type,
        "all_tags": all_tags,
    }


def print_summary(stats_train: dict, stats_val: dict, stats_test: dict):
    """打印数据集统计摘要。"""
    print("=" * 70)
    print("peoples_daily 数据集统计摘要")
    print("=" * 70)

    for name, stats in [("训练集", stats_train), ("验证集", stats_val), ("测试集", stats_test)]:
        print(f"\n【{name}】")
        print(f"  样本数：{len(stats['text_lengths'])} 条")
        print(f"  文本平均长度：{sum(stats['text_lengths']) / len(stats['text_lengths']):.1f} 字")
        print(f"  文本最大长度：{max(stats['text_lengths'])} 字")
        print(f"  文本长度中位数：{sorted(stats['text_lengths'])[len(stats['text_lengths']) // 2]} 字")
        print(f"  平均实体数/句：{sum(stats['entity_per_sentence']) / len(stats['entity_per_sentence']):.2f}")
        print(f"  实体总数：{sum(stats['entity_type_counts'].values())}")
        if stats["entity_lengths"]:
            print(f"  平均实体长度：{sum(stats['entity_lengths']) / len(stats['entity_lengths']):.1f} 字")

    print("\n【各类实体频次（训练集）】")
    et_label = {
        "PER": "人名", "LOC": "地点", "ORG": "机构",
    }
    total = sum(stats_train["entity_type_counts"].values())
    for etype, cnt in sorted(stats_train["entity_type_counts"].items(), key=lambda x: -x[1]):
        cn = et_label.get(etype, etype)
        percentage = (cnt / total) * 100 if total > 0 else 0
        print(f"  {etype:8s} ({cn:6s}) : {cnt:5d} 条 ({percentage:.1f}%)")

    print("\n【标签分布（训练集）】")
    tag_counts = Counter(stats_train["all_tags"])
    total_tags = len(stats_train["all_tags"])
    for tag, cnt in sorted(tag_counts.items(), key=lambda x: -x[1]):
        percentage = (cnt / total_tags) * 100
        print(f"  {tag:10s} : {cnt:6d}  ({percentage:.1f}%)")


def plot_entity_distribution(stats_train: dict):
    """绘制实体类型分布柱状图。"""
    et_label = {
        "PER": "人名", "LOC": "地点", "ORG": "机构",
    }
    counts = stats_train["entity_type_counts"]
    labels = [f"{k}\n({et_label.get(k, k)})" for k in sorted(counts)]
    values = [counts[k] for k in sorted(counts)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color="#4C72B0", alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            str(v),
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_title("peoples_daily 各类实体频次分布（训练集）", fontsize=14)
    ax.set_ylabel("实体数量")
    ax.set_xlabel("实体类型")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "entity_distribution.png", dpi=120)
    print(f"  已保存 → {FIG_DIR / 'entity_distribution.png'}")
    plt.close()


def plot_text_length_distribution(stats_train: dict):
    """绘制文本长度分布直方图。"""
    lengths = stats_train["text_lengths"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(lengths, bins=50, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(x=64, color="red", linestyle="--", linewidth=1.5, label="max_length=64")
    ax.axvline(x=128, color="orange", linestyle="--", linewidth=1.5, label="max_length=128")
    ax.axvline(x=256, color="purple", linestyle="--", linewidth=1.5, label="max_length=256")
    p95 = sorted(lengths)[int(len(lengths) * 0.95)]
    ax.axvline(x=p95, color="green", linestyle="--", linewidth=1.5, label=f"P95={p95}")
    ax.set_title("文本长度分布（训练集）", fontsize=14)
    ax.set_xlabel("文本字符数")
    ax.set_ylabel("样本数")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "text_length_distribution.png", dpi=120)
    print(f"  已保存 → {FIG_DIR / 'text_length_distribution.png'}")
    # 打印 P95 对 max_length 的启发信息。
    print(f"  P95 文本长度={p95}，建议 max_length 参考此值设置")
    plt.close()


def plot_entity_length_distribution(stats_train: dict):
    """绘制实体长度分布柱状图。"""
    length_counter = Counter(stats_train["entity_lengths"])
    xs = sorted(length_counter.keys())
    ys = [length_counter[x] for x in xs]

    fig, ax = plt.subplots(figsize=(12, 5))
    # 只显示前20个长度
    ax.bar([str(x) for x in xs[:20]], ys[:20], color="#55A868", alpha=0.85, edgecolor="white")
    ax.set_title("实体长度分布（训练集，前20）", fontsize=14)
    ax.set_xlabel("实体字符数")
    ax.set_ylabel("出现次数")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "entity_length_distribution.png", dpi=120)
    print(f"  已保存 → {FIG_DIR / 'entity_length_distribution.png'}")
    plt.close()

    if stats_train["entity_lengths"]:
        avg_len = sum(stats_train["entity_lengths"]) / len(stats_train["entity_lengths"])
        print(f"  实体平均长度={avg_len:.1f}字")


def main():
    parse_args()

    print("正在加载数据集...")
    train_records = load_split("train")
    val_records = load_split("validation")
    test_records = load_split("test")

    print(f"  训练集: {len(train_records)} 条")
    print(f"  验证集: {len(val_records)} 条")
    print(f"  测试集: {len(test_records)} 条")

    stats_train = collect_stats(train_records)
    stats_val = collect_stats(val_records)
    stats_test = collect_stats(test_records)

    print_summary(stats_train, stats_val, stats_test)

    print("\n正在生成可视化图表...")
    plot_entity_distribution(stats_train)
    plot_text_length_distribution(stats_train)
    plot_entity_length_distribution(stats_train)

    print("\n探索完成！图表已保存到 homework_outputs/figures/")


def parse_args():
    parser = argparse.ArgumentParser(description="探索 peoples_daily 数据集")
    return parser.parse_args()


if __name__ == "__main__":
    main()
