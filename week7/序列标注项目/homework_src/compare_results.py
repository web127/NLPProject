"""
汇总所有方案的评估结果，打印对比表

使用方式：
  python -m homework_src.compare_results

前提：
  - homework_outputs/logs/eval_linear_test.json   （已运行 evaluate.py）
  - homework_outputs/logs/eval_crf_test.json      （已运行 evaluate.py --use_crf）
"""

import sys
from pathlib import Path

# 计算当前脚本所在项目的根目录。
ROOT = Path(__file__).parent.parent
# 把项目根目录加入模块搜索路径，便于直接运行当前脚本。
sys.path.insert(0, str(ROOT))

import json


# 统一定义评估日志目录。
LOG_DIR = ROOT / "homework_outputs" / "logs"


def load_json(path: Path) -> dict | None:
    # 如果结果文件不存在，则直接返回 None。
    if not path.exists():
        return None
    # 以 UTF-8 编码打开 JSON 文件。
    with open(path, "r", encoding="utf-8") as f:
        # 读取并解析 JSON 内容。
        return json.load(f)


def main():
    # 加载线性头模型的测试集评估结果。
    linear_res = load_json(LOG_DIR / "eval_linear_test.json")
    # 加载 CRF 模型的测试集评估结果。
    crf_res = load_json(LOG_DIR / "eval_crf_test.json")

    # 打印分隔线，增强输出可读性。
    print("\n" + "=" * 80)
    # 打印标题。
    print("BERT NER 作业 — 两方案汇总对比 (Peoples Daily 数据集)")
    # 打印分隔线。
    print("=" * 80)

    # 构造表头字符串。
    header = f"{'方案':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'非法序列':>10}"
    # 输出表头。
    print(header)
    # 输出横线分隔表头与内容。
    print("-" * 80)

    # 兼容缺失值的格式化函数。
    def fmt(val):
        # 有值时保留四位小数，否则显示 N/A。
        return f"{val:.4f}" if val is not None else "  N/A  "

    # 如果找到了线性头模型的结果文件。
    if linear_res:
        # 读取非法 BIO 序列总数。
        ill = linear_res["illegal_stats"]["total_illegal"]
        # 按列对齐打印线性头模型结果。
        print(
            f"{'BERT + Linear':<20} "
            f"{fmt(linear_res['precision']):>10} "
            f"{fmt(linear_res['recall']):>10} "
            f"{fmt(linear_res['f1']):>10} "
            f"{ill:>10d}"
        )
    else:
        # 若没有结果文件，则提示先执行评估脚本。
        print(f"{'BERT + Linear':<20} {'（未找到结果，请运行 evaluate.py）':>60}")

    # 如果找到了 CRF 模型的结果文件。
    if crf_res:
        # 读取非法 BIO 序列总数。
        ill = crf_res["illegal_stats"]["total_illegal"]
        # 按列对齐打印 CRF 模型结果。
        print(
            f"{'BERT + CRF':<20} "
            f"{fmt(crf_res['precision']):>10} "
            f"{fmt(crf_res['recall']):>10} "
            f"{fmt(crf_res['f1']):>10} "
            f"{ill:>10d}"
        )
    else:
        # 若没有结果文件，则提示先执行带 CRF 的评估命令。
        print(f"{'BERT + CRF':<20} {'（未找到结果，请运行 evaluate.py --use_crf）':>60}")

    # 输出总结区块分隔线。
    print("\n" + "=" * 80)
    # 打印总结标题。
    print("关键教学结论：")
    # 只有当两个模型结果都存在时，才计算差异。
    if linear_res and crf_res:
        # 计算 CRF 相比线性头的 F1 变化。
        f1_diff = crf_res["f1"] - linear_res["f1"]
        # 读取线性头的非法序列数。
        ill_linear = linear_res["illegal_stats"]["total_illegal"]
        # 读取 CRF 的非法序列数。
        ill_crf = crf_res["illegal_stats"]["total_illegal"]
        # 计算非法序列下降比例。
        ill_reduction = 100 * (1 - ill_crf / max(ill_linear, 1))

        # 打印 F1 提升或下降情况。
        print(f"  1. CRF vs Linear：F1 {'↑' if f1_diff >= 0 else '↓'}{abs(f1_diff):.4f}")
        # 打印非法序列数对比。
        print(f"  2. 非法序列对比：Linear {ill_linear} 条 → CRF {ill_crf} 条")
        # 如果 CRF 确实减少了非法序列，则进一步打印降幅。
        if ill_linear > ill_crf:
            print(f"     → 减少了 {ill_reduction:.1f}% 的非法序列！")
        # 打印方法层面的结论。
        print("  3. CRF 通过转移矩阵约束 + Viterbi 解码，保证序列合法性")
    print("=" * 80)


if __name__ == "__main__":
    main()
