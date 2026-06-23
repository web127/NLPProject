# homework_src 模型学习问答整理

> 用途：把这轮和 `homework_src` 模型理解有关、但**不包含“加注释/删注释”过程**的问答整理成复习材料。

---

## 1. 为什么 `homework_src/model.py` 比 `src/model.py` 复杂很多？

### 我的问题

为什么 `homework_src` 下的 `model.py` 这么复杂，比 `src` 的 `model.py` 复杂多了，看不太明白？

### 整理后的回答

核心原因不是“模型本体高级很多”，而是：

> `src/model.py` 更像**教学最简版**，
> `homework_src/model.py` 更像**把实际训练中的坑补齐后的增强版**。

具体说，复杂度主要不是来自线性头 `BertNER`，而是来自 `BertCRFNER` 这部分。

#### 1）线性头其实差不多

- `src/model.py` 的线性头：`src/model.py:36`
- `homework_src/model.py` 的线性头：`homework_src/model.py:41`

这两边主干都差不多：

1. 输入给 BERT
2. 拿到 `last_hidden_state`
3. 经过 `dropout`
4. 过 `Linear`
5. 用交叉熵算 loss

所以真正让人觉得“复杂”的，不是线性头，而是 CRF 版本。

#### 2）`homework_src/model.py` 多做了 BIO 合法性约束

在简化版 `src/model.py` 里，CRF 更像是“普通 CRF”：

- CRF 初始化：`src/model.py:96`
- 训练时直接喂 `emissions + labels + mask`：`src/model.py:122`
- 解码时直接 `self.crf.decode(...)`：`src/model.py:142`

但在增强版 `homework_src/model.py` 里，额外加入了 BIO 规则：

- 起始标签约束：`homework_src/model.py:165`
- 标签转移约束：`homework_src/model.py:177`
- 把非法转移置成极小值：`homework_src/model.py:208`

这部分的本质是：

> 不只是让 CRF“自己学会” BIO，
> 而是先把不合法路径直接封掉。

这就比最简版代码多出了一大块逻辑。

#### 3）增强版还认真处理了 `-100` 和子词对齐

在 `homework_src/dataset.py` 里，很多位置会被标成 `-100`，比如：

- `[CLS]` / `[SEP]`
- 非首子词
- padding

相关位置见：`homework_src/dataset.py:104`

线性头没问题，因为交叉熵支持：

- `ignore_index=-100`：`homework_src/model.py:87`

但 CRF 比较麻烦，因为它不喜欢原始序列里夹着很多“无效位置”。

因此增强版额外做了两件事：

1. 构建真正有效的位置掩码：`homework_src/model.py:240`
2. 把有效位置压缩成连续序列再交给 CRF：`homework_src/model.py:257`

这也是整份文件最绕的地方。

#### 4）增强版还让 decode 结果和 gold 严格对齐

在 `homework_src/model.py` 里，`decode()` 允许传入 `labels`：

- `homework_src/model.py:337`

这样它会只在真正有效的位置上解码，让预测序列长度和 gold 序列严格匹配，便于后面评估。

#### 5）一句话总结

可以这样理解：

- `src/model.py`：先讲清楚 **BERT + Linear / BERT + CRF 的原理**
- `homework_src/model.py`：把 **BIO 约束、无效位置、评估对齐** 这些工程问题补齐

所以 `homework_src/model.py` 更复杂，不是因为“理论更高深”，而是因为“工程处理更多”。

---

## 2. 作为 PyTorch 初学者，看这种代码时最容易卡在哪？

### 我的问题

我也是 PyTorch 初学者，有些代码写法也看不懂。

### 整理后的回答

如果你是第一次接触 PyTorch，这份代码里最容易卡住的，通常不是 NER 理论，而是 PyTorch 的常见表达方式。

### 重点先理解这几个概念

#### 1）`nn.Module` 是什么

- `BertNER`：`homework_src/model.py:41`
- `BertCRFNER`：`homework_src/model.py:97`

`nn.Module` 可以理解成：

> PyTorch 里“所有模型的统一模板”。

只要继承它，就能方便地：

- 管理参数
- `.to(device)`
- `.train()` / `.eval()`
- 保存和加载参数

#### 2）`__init__` 和 `forward` 的区别

- `__init__`：先把层定义好
- `forward`：输入真的进来以后，按顺序跑一遍

比如 `BertNER`：

- 定义结构：`homework_src/model.py:52`
- 真正前向计算：`homework_src/model.py:66`

可以记成：

> `__init__` = 搭积木
>
> `forward` = 积木开始工作

#### 3）为什么平时写 `model(...)` 而不是 `model.forward(...)`

因为在 PyTorch 里：

> 调用 `model(...)` 时，会自动去执行 `forward(...)`

所以训练代码里直接写：

```python
logits, loss = model(input_ids, attention_mask, token_type_ids, labels)
```

是标准写法。

#### 4）`self.xxx = ...` 是什么

比如：

- `self.bert`：`homework_src/model.py:56`
- `self.dropout`：`homework_src/model.py:60`
- `self.classifier`：`homework_src/model.py:62`
- `self.crf`：`homework_src/model.py:132`

意思是：

> 把这些层挂到模型对象自己身上，后面别的函数就都能访问。

#### 5）`outputs.last_hidden_state` 是什么

位置：`homework_src/model.py:73`、`homework_src/model.py:228`

BERT 不只返回一个值，而是返回一个结果对象。NER 最常用的是：

> `last_hidden_state`：每个 token 的上下文表示

#### 6）`logits` 是什么

位置：`homework_src/model.py:83`

`logits` 不是概率，而是：

> 模型给每个类别打的原始分数

训练时不需要手动 softmax，因为交叉熵内部会处理。

#### 7）为什么要 `view(-1, self.num_labels)`

位置：`homework_src/model.py:89`

原始输出是三维：

- `logits.shape = (B, L, C)`

而交叉熵更喜欢二维：

- `(样本数, 类别数)`

所以这里是把一个 batch 里的所有 token 摊平，一起算 loss。

#### 8）`ignore_index=-100` 是什么

位置：`homework_src/model.py:92`

意思是：

> 标签值为 `-100` 的位置，不参与 loss。

这是为了跳过：

- `[CLS]`
- `[SEP]`
- 非首子词
- padding

#### 9）`mask = attention_mask.bool()` 是什么

位置：`homework_src/model.py:250`、`homework_src/model.py:361`

就是把 0/1 转成 True/False，方便 CRF 使用。

#### 10）`@staticmethod` / `@classmethod` 怎么理解

位置：

- `@staticmethod`：`homework_src/model.py:152`、`homework_src/model.py:240`、`homework_src/model.py:257`
- `@classmethod`：`homework_src/model.py:165`、`homework_src/model.py:177`

粗略理解就够：

- `staticmethod`：和类有关，但不依赖具体对象
- `classmethod`：和类本身有关，会收到 `cls`

#### 11）`register_buffer(...)` 是什么

位置：`homework_src/model.py:137`

它的作用是：

> 给模型挂一个 tensor，
> 但它不是要训练的参数。

这里挂的是 BIO 规则掩码，不需要优化器更新。

#### 12）`masked_fill(...)` 在干嘛

位置：`homework_src/model.py:211`

它是在：

> 把不合法位置直接填成一个非常小的值

在这里等价于：

> 把不合法 BIO 转移直接禁掉。

#### 13）最难的一块其实是 `_pack_valid_positions()`

位置：`homework_src/model.py:257`

这段函数的核心作用只有一句话：

> 把所有真正有效的位置抽出来，压成连续序列，再交给 CRF。

你第一次读时不用逐行抠实现，先抓住这个目的就行。

### 给初学者的阅读顺序建议

#### 第一遍：先看最简单主干

- `_load_bert`：`homework_src/model.py:28`
- `BertNER.__init__`：`homework_src/model.py:52`
- `BertNER.forward`：`homework_src/model.py:66`
- `build_model`：`homework_src/model.py:367`

#### 第二遍：再看 CRF 主流程

- `BertCRFNER.__init__`：`homework_src/model.py:113`
- `_get_emissions`：`homework_src/model.py:222`
- `forward`：`homework_src/model.py:314`
- `decode`：`homework_src/model.py:337`

#### 第三遍：最后看辅助函数

- `_split_bio_tag`：`homework_src/model.py:152`
- `_build_allowed_start_mask`：`homework_src/model.py:165`
- `_build_allowed_transition_mask`：`homework_src/model.py:177`
- `_build_crf_mask`：`homework_src/model.py:240`
- `_pack_valid_positions`：`homework_src/model.py:257`

这样读会比从上到下硬啃轻松很多。

---

## 3. 生产项目里，BIO 硬规则到底是写死还是让模型自己学？

### 我的问题

生产项目，BIO 硬规则到底是写死还是让模型自己学？

### 整理后的回答

短答案：

> **生产里，BIO 这种“输出格式约束”通常更倾向于写死；**
> **实体边界、类别判断这些“语义问题”交给模型自己学。**

也就是：

- 规则负责保底
- 模型负责效果

### 为什么生产里更偏向写死 BIO 规则？

因为 BIO 合法性本质上不是“知识”，而是：

> 一种标注格式规范。

比如：

- 不能 `I-XXX` 开头
- `B-PER` 后面不能直接接 `I-ORG`
- `I-LOC` 前面必须是 `B-LOC` 或 `I-LOC`

这些并不是语义理解问题，而是只要采用 BIO，就天然成立的规则。

所以生产里常见的思路是：

> **能确定的规则，不要交给模型猜。**

### 生产里的常见分工

#### 写死的部分

- BIO 起始合法性
- BIO 转移合法性
- 解码时的结构约束
- 后处理修复非法标签

#### 模型学习的部分

- 这里是不是实体
- 是人名还是地名
- 实体边界到底在哪
- 模糊场景下应该怎么判断

### 你这个项目里的两种写法，正好对应两种工程思路

#### 1）更偏“让模型自己学”的版本

`src/model.py` 的 CRF 更接近这个思路：

- CRF 初始化：`src/model.py:96`
- forward：`src/model.py:122`
- decode：`src/model.py:142`

它更像是：

> “我相信 CRF 在训练过程中能学会大部分 BIO 规律。”

优点：

- 代码简单
- 教学直观

缺点：

- 不保证 100% 合法
- 数据少、训练不够、分布变化时可能 still 出错

#### 2）更偏“把规则写死”的版本

`homework_src/model.py` 明显更偏生产保守思路：

- 起始约束：`homework_src/model.py:165`
- 转移约束：`homework_src/model.py:177`
- 应用约束：`homework_src/model.py:208`

相当于：

> 先把不合法路径封掉，模型只能在合法路径里选最优解。

### 生产里推荐的原则

更实用的表述是：

> **让模型学习分数，让规则控制合法性。**

这是非常典型的工程分层：

- 统计模型：处理模糊问题
- 显式规则：处理确定约束

### 如果坚持用 BIO + token classification，常见生产方案有两个

#### 方案 A：线性头 + 后处理修复

训练时用普通线性头，推理后如果发现：

- `I-PER` 开头
- `B-PER -> I-ORG`

就用规则修掉。

优点：

- 简单
- 部署轻
- 工程成本低

缺点：

- 修规则有时比较粗糙

#### 方案 B：CRF + 硬约束

就是你 `homework_src/model.py` 更接近的方案。

优点：

- 解码阶段天然更规范
- 非法 BIO 更少甚至可以消除

缺点：

- 代码更复杂
- 训练和推理都比线性头重一点

### 还有一种现实情况：很多生产系统根本不再执着于 BIO

现代工业 NER 很多会直接用：

- span classification
- GlobalPointer
- start-end pointer

因为这些方法直接预测实体区间，而不是逐 token 输出 BIO，很多结构性麻烦就少了。

### 最后一句话记忆

> **模型负责聪明，规则负责别犯低级错误。**

BIO 非法标签，通常就属于“低级错误”，生产里一般不愿完全交给模型自由发挥。

---

## 4. 复习时建议你重点记住的三句话

### 1）关于 `homework_src/model.py` 为什么复杂

> 它复杂，主要不是理论更难，而是工程补丁更多。

### 2）关于 PyTorch 初学者怎么读这份代码

> 先看模型主干，再看 CRF 主流程，最后看辅助函数。

### 3）关于生产里 BIO 规则怎么处理

> 能确定的格式规则尽量写死，语义判断交给模型学。

---

## 5. 后续继续学习时可以顺着追问的方向

如果后面继续复习，可以继续顺着这几个问题往下学：

1. `Linear + 后处理` 和 `CRF + 硬约束` 到底怎么选？
2. `attention_mask`、`labels != -100`、`CRF mask` 三者到底是什么关系？
3. `view()`、`reshape()`、`squeeze()` 在 PyTorch 里分别什么时候用？
4. 为什么 span-based NER 在生产里越来越常见？
5. `state_dict()`、`register_buffer()`、`nn.Parameter` 的区别到底是什么？

如果你愿意，后面可以继续把这些问题也一条条整理进这个文档。
