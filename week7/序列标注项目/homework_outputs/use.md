  使用方式

  # 训练
  python -m homework_src.train --train_samples 2000 --val_samples 500
  python -m homework_src.train --use_crf --train_samples 2000 --val_samples 500

  # 评估
  python -m homework_src.evaluate --test_samples 500
  python -m homework_src.evaluate --use_crf --test_samples 500
  
  # 最终交付总结                                                                                                                                                                       
    
  ✅ 代码完整 - 参考 src/ 实现的 homework_src/                                                                                                                                       ─
  ✅ 双模型训练 - BertNER（Linear）和 BertCRFNER（CRF）     t                                                                                                                      e
  ✅ Peoples Daily数据集 - 支持 BIO格式
  ✅ Entity-level评估 - 使用seqeval，与src一致
  ✅ 对比结果文件 - homework_outputs/comparison_results.txt
