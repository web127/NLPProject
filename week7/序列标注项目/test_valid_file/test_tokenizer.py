"""
测试不同 tokenizer
"""
from transformers import BertTokenizer, BertTokenizerFast

print("测试 BertTokenizer...")
tokenizer_slow = BertTokenizer.from_pretrained("../pretrain_models/bert-base-chinese")
print(f"Slow tokenizer loaded: {type(tokenizer_slow)}")

print("\n测试 BertTokenizerFast...")
try:
    tokenizer_fast = BertTokenizerFast.from_pretrained("../pretrain_models/bert-base-chinese")
    print(f"Fast tokenizer loaded: {type(tokenizer_fast)}")

    # 测试
    chars = ["海", "钓", "比", "赛"]
    encoding = tokenizer_fast(chars, is_split_into_words=True, max_length=10, truncation=True, padding="max_length", return_tensors="pt")
    print(f"\nEncoding keys: {list(encoding.keys())}")
    print(f"word_ids: {encoding.word_ids(batch_index=0)}")
    print("Fast tokenizer 工作正常！")
except Exception as e:
    print(f"Fast tokenizer 加载失败: {e}")
