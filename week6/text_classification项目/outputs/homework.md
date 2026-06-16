## bert+分类头
### 运行命令：python train.py --use_class_weight
### 打印日志：
使用设备: cpu
类别数: 15
DataLoader 构建完成
  train: 53360 条, 1668 batch
  val  : 10000 条, 313 batch
  test : 10000 条, 313 batch
模型参数量: 45.6M  (BERT: 45.6M, 分类头: 11.5K)
池化策略: cls
类别权重（用于加权 loss）：
   0 故事  : 3.202
   1 文化  : 0.872
   2 娱乐  : 0.715
   3 体育  : 0.891
   4 财经  : 0.684
   5 房产  : 1.688
   6 汽车  : 0.864
   7 教育  : 1.035
   8 科技  : 0.597
   9 军事  : 0.979
  10 旅游  : 1.056
  11 国际  : 0.733
  12 证券  : 13.842
  13 农业  : 1.233
  14 电竞  : 1.049
使用加权 CrossEntropyLoss
总训练步数: 5004, warmup: 500
Epoch 1/3 | train_loss=1.6240 train_acc=0.4543 | val_acc=0.5324 val_macro_f1=0.5214 | 953s                                                                                          
  ✓ 新最优模型已保存 → /Users/bytedance/PycharmProjects/NLPProject/week6/text_classification项目/outputs/checkpoints/best_cls_weighted.pt  (val_acc=0.5324)
Epoch 2/3 | train_loss=1.1955 train_acc=0.5633 | val_acc=0.5418 val_macro_f1=0.5354 | 970s                                                                                          
  ✓ 新最优模型已保存 → /Users/bytedance/PycharmProjects/NLPProject/week6/text_classification项目/outputs/checkpoints/best_cls_weighted.pt  (val_acc=0.5418)
                                                                                                                                                                                    
分类报告：
              precision    recall  f1-score   support

          故事       0.36      0.67      0.47       215
          文化       0.52      0.54      0.53       736
          娱乐       0.61      0.54      0.57       910
          体育       0.73      0.68      0.71       767
          财经       0.49      0.40      0.44       956
          房产       0.49      0.68      0.57       378
          汽车       0.67      0.62      0.64       791
          教育       0.53      0.59      0.56       646
          科技       0.55      0.45      0.50      1089
          军事       0.52      0.57      0.54       716
          旅游       0.47      0.51      0.49       693
          国际       0.50      0.42      0.46       905
          证券       0.29      0.73      0.42        45
          农业       0.47      0.57      0.51       494
          电竞       0.62      0.63      0.62       659

    accuracy                           0.54     10000
   macro avg       0.52      0.57      0.53     10000
weighted avg       0.55      0.54      0.54     10000

Epoch 3/3 | train_loss=1.0445 train_acc=0.6020 | val_acc=0.5415 val_macro_f1=0.5346 | 960s

训练完成。最优 val_acc=0.5418
训练日志 → /Users/bytedance/PycharmProjects/NLPProject/week6/text_classification项目/outputs/train_log_cls_weighted.json


### 运行命令：python train.py --use_class_weight --pool mean

### 打印日志：
使用设备: cpu
类别数: 15
DataLoader 构建完成
  train: 53360 条, 1668 batch
  val  : 10000 条, 313 batch
  test : 10000 条, 313 batch
模型参数量: 45.6M  (BERT: 45.6M, 分类头: 11.5K)
池化策略: mean
类别权重（用于加权 loss）：
   0 故事  : 3.202
   1 文化  : 0.872
   2 娱乐  : 0.715
   3 体育  : 0.891
   4 财经  : 0.684
   5 房产  : 1.688
   6 汽车  : 0.864
   7 教育  : 1.035
   8 科技  : 0.597
   9 军事  : 0.979
  10 旅游  : 1.056
  11 国际  : 0.733
  12 证券  : 13.842
  13 农业  : 1.233
  14 电竞  : 1.049
使用加权 CrossEntropyLoss
总训练步数: 5004, warmup: 500
Epoch 1/3 | train_loss=1.5345 train_acc=0.4780 | val_acc=0.5380 val_macro_f1=0.5259 | 949s                                                                                          
  ✓ 新最优模型已保存 → /Users/bytedance/PycharmProjects/NLPProject/week6/text_classification项目/outputs/checkpoints/best_mean_weighted.pt  (val_acc=0.5380)
Epoch 2/3 | train_loss=1.0865 train_acc=0.5951 | val_acc=0.5446 val_macro_f1=0.5326 | 904s                                                                                          
  ✓ 新最优模型已保存 → /Users/bytedance/PycharmProjects/NLPProject/week6/text_classification项目/outputs/checkpoints/best_mean_weighted.pt  (val_acc=0.5446)
                                                                                                                                                                                    
分类报告：
              precision    recall  f1-score   support

          故事       0.38      0.60      0.47       215
          文化       0.53      0.56      0.54       736
          娱乐       0.58      0.56      0.57       910
          体育       0.71      0.70      0.71       767
          财经       0.50      0.42      0.46       956
          房产       0.52      0.67      0.58       378
          汽车       0.67      0.60      0.64       791
          教育       0.54      0.59      0.57       646
          科技       0.54      0.47      0.50      1089
          军事       0.53      0.57      0.55       716
          旅游       0.49      0.48      0.48       693
          国际       0.51      0.45      0.48       905
          证券       0.33      0.62      0.43        45
          农业       0.46      0.57      0.51       494
          电竞       0.62      0.61      0.62       659

    accuracy                           0.55     10000
   macro avg       0.53      0.57      0.54     10000
weighted avg       0.55      0.55      0.55     10000

Epoch 3/3 | train_loss=0.8306 train_acc=0.6701 | val_acc=0.5473 val_macro_f1=0.5397 | 968s
  ✓ 新最优模型已保存 → /Users/bytedance/PycharmProjects/NLPProject/week6/text_classification项目/outputs/checkpoints/best_mean_weighted.pt  (val_acc=0.5473)

训练完成。最优 val_acc=0.5473
训练日志 → /Users/bytedance/PycharmProjects/NLPProject/week6/text_classification项目/outputs/train_log_mean_weighted.json

## 大模型lora微调
### 运行命令：python train_sft.py --model_path Qwen2.5-0.5B-Instruct 
### 日志：
加载 tokenizer: /Users/bytedance/PycharmProjects/NLPProject/week6/pretrain_models/Qwen2.5-0.5B-Instruct
加载 base model: /Users/bytedance/PycharmProjects/NLPProject/week6/pretrain_models/Qwen2.5-0.5B-Instruct
trainable params: 1,081,344 || all params: 495,114,112 || trainable%: 0.2184
总训练步数: 937（batch=4, grad_accum=4, epochs=3, lr=0.0002）

Epoch 1/3 | train_loss=0.7727  val_loss=0.6835 | 1546s                                                                                                                              
  ✓ 最优LoRA adapter已保存 → /Users/bytedance/PycharmProjects/NLPProject/week6/text_classification项目/outputs/hk_sft_adapter  (val_loss=0.6835)
Epoch 2/3 | train_loss=0.6389  val_loss=0.6338 | 1599s                                                                                                                              
  ✓ 最优LoRA adapter已保存 → /Users/bytedance/PycharmProjects/NLPProject/week6/text_classification项目/outputs/hk_sft_adapter  (val_loss=0.6338)
#### 获取大模型的输入：
prompt_text:<|im_start|>system
你是一个新闻标题分类助手。请将给定的新闻标题分类到以下类别之一，只输出类别名称，不要输出任何其他内容。
可选类别：故事、文化、娱乐、体育、财经、房产、汽车、教育、科技、军事、旅游、国际、证券、农业、电竞<|im_end|>
<|im_start|>user
新闻标题：对山水画怎么样描叙
类别：<|im_end|>
<|im_start|>assistant