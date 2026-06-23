import json
from pathlib import Path

DATA_DIR = Path("../data/peoples_daily")

print("检查验证集...")
with open(DATA_DIR / "validation.json", "r", encoding="utf-8") as f:
    val_data = json.load(f)
    print(f"验证集: {len(val_data)} 条")

print("\n检查测试集...")
with open(DATA_DIR / "test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)
    print(f"测试集: {len(test_data)} 条")

print("\n检查训练集（只读取前几KB看格式）...")
with open(DATA_DIR / "train.json", "r", encoding="utf-8") as f:
    # 先看开头
    start = f.read(500)
    print("训练集开头预览:")
    print(start[:200])
    # 估算条数
    f.seek(0)
    content = f.read(50000)  # 读50KB估算
    count = content.count('{"tokens"')
    print(f"\n训练集预估: 约 {count * 20} 条（基于50KB估算）")
