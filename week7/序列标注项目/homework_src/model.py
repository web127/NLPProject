"""
BertNER（线性头）和 BertCRFNER（CRF头）两个模型

教学重点：
  1. 线性头（BertNER）：每个 token 独立预测标签
     - 问题：softmax 的独立预测忽略标签间的依赖关系
     - 可能产生非法序列：B-name 后接 I-company，I-name 开头等

  2. CRF 层（BertCRFNER）：加入转移矩阵，全局最优解码
     - 转移矩阵学习"什么标签之后可以接什么标签"
     - 若未显式加入 BIO 硬约束，CRF 只能减少非法序列，不能理论保证为 0
     - 代价：训练时需要前向-后向算法，比线性头慢约 20~30%

  3. 两者区别的量化：evaluate.py 会统计非法序列数

依赖：
  pip install pytorch-crf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertModel
from torchcrf import CRF


def _load_bert(bert_path: str) -> BertModel:
    # 先记录 transformers 当前的日志级别，避免永久修改全局设置。
    prev = transformers.logging.get_verbosity()
    # 临时只显示错误日志，减少加载预训练模型时的控制台噪声。
    transformers.logging.set_verbosity_error()
    # 从给定路径加载 BERT 主干网络。
    bert = BertModel.from_pretrained(bert_path)
    # 恢复原来的日志级别。
    transformers.logging.set_verbosity(prev)
    # 返回加载好的 BERT 模型。
    return bert


class BertNER(nn.Module):
    """BERT + 线性分类头，逐 token 独立预测 BIO 标签。

    前向过程：
      input_ids → BertModel → last_hidden_state (B, L, 768)
               → Dropout → Linear(768, num_labels) → logits (B, L, num_labels)

    损失：CrossEntropy，ignore_index=-100 跳过特殊token和非首子词
    预测：argmax(logits, dim=-1)
    """

    def __init__(self, bert_path: str, num_labels: int, dropout: float = 0.1):
        # 初始化父类 nn.Module。
        super().__init__()
        # 加载预训练 BERT 编码器。
        self.bert = _load_bert(bert_path)
        # 读取 BERT 隐层维度，通常中文 BERT 为 768。
        hidden_size = self.bert.config.hidden_size
        # 定义 dropout 层，用于缓解过拟合。
        self.dropout = nn.Dropout(dropout)
        # 定义逐 token 线性分类头。
        self.classifier = nn.Linear(hidden_size, num_labels)
        # 记录标签总数，供 loss reshape 时使用。
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # 将输入送入 BERT，获取上下文表示。
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        # 取最后一层隐状态，形状为 (batch, seq_len, hidden_size)。
        seq_output = outputs.last_hidden_state  # (B, L, H)
        # 先做 dropout，再映射到标签维度，得到每个位置的分类分数。
        logits = self.classifier(self.dropout(seq_output))  # (B, L, num_labels)

        # 默认无监督时不计算损失。
        loss = None
        if labels is not None:
            # 将三维 logits 和二维 labels 拉平成 token 级样本来计算损失。
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss


class BertCRFNER(nn.Module):
    """BERT + CRF 层，全局最优序列解码。

    与 BertNER 的区别：
      - Linear 输出称为 emissions（发射分数），不直接 argmax
      - CRF 在 emissions 上叠加转移矩阵，用 Viterbi 找全局最优序列
      - 损失：负对数似然（CRF 内部计算前向-后向）
      - 解码：self.crf.decode() 返回全局分数最高的标签序列

    CRF 的转移偏好（通过训练学习）：
      - 倾向于学习合理的 BIO 转移
      - 但若未手工加入约束矩阵，并不严格保证 100% 合法
    """

    NEG_INF = -10000.0

    def __init__(
        self,
        bert_path: str,
        num_labels: int,
        dropout: float = 0.1,
        id2label: dict[int, str] | None = None,
    ):
        # 初始化父类 nn.Module。
        super().__init__()

        # 加载预训练 BERT 编码器。
        self.bert = _load_bert(bert_path)
        # 读取 BERT 隐层维度。
        hidden_size = self.bert.config.hidden_size
        # 定义 dropout 层。
        self.dropout = nn.Dropout(dropout)
        # 定义发射分数线性层，将隐表示映射到标签空间。
        self.classifier = nn.Linear(hidden_size, num_labels)
        # 定义 CRF 层，并设置 batch_first=True 以匹配 (B, L, C) 输入格式。
        self.crf = CRF(num_labels, batch_first=True)
        # 记录标签数。
        self.num_labels = num_labels
        # 保存 id 到标签字符串的映射，若未提供则退化为数字字符串。
        self.id2label = id2label or {i: str(i) for i in range(num_labels)}
        # 注册合法起始标签掩码为 buffer，避免被优化器更新。
        self.register_buffer(
            "allowed_start_mask",
            self._build_allowed_start_mask(self.id2label),
            persistent=False,
        )
        # 注册合法标签转移掩码为 buffer。
        self.register_buffer(
            "allowed_transition_mask",
            self._build_allowed_transition_mask(self.id2label),
            persistent=False,
        )
        # 初始化时就把 BIO 约束写入 CRF 参数中。
        self._apply_bio_constraints()

    @staticmethod
    def _split_bio_tag(tag: str) -> tuple[str, str | None]:
        # O 标签没有实体类型，单独处理。
        if tag == "O":
            return "O", None
        # 若标签里不包含连字符，则直接返回原标签前缀。
        if "-" not in tag:
            return tag, None
        # 按第一个连字符拆成前缀和实体类型。
        prefix, ent_type = tag.split("-", 1)
        # 返回 BIO 前缀和实体类型。
        return prefix, ent_type

    #allowed_start = [True, True, False, True, False]
    #                O     B-PER  I-PER  B-LOC  I-LOC
    @classmethod
    def _build_allowed_start_mask(cls, id2label: dict[int, str]) -> torch.Tensor:
        # 用列表保存每个标签是否允许作为序列起点。
        allowed = []
        for idx in range(len(id2label)):
            # 拆分出标签前缀。
            prefix, _ = cls._split_bio_tag(id2label[idx])
            # 起始位置不允许直接以 I- 开头。
            allowed.append(prefix != "I")
        # 转成布尔张量返回。
        return torch.tensor(allowed, dtype=torch.bool)

    # allowed = torch.tensor([
    #     #      O    B-PER  I-PER  B-LOC  I-LOC
    #     [True, True, False, True, False],  # O → ?
    #     [True, True, True, True, False],  # B-PER → ?
    #     [True, True, True, True, False],  # I-PER → ?
    #     [True, True, False, True, True],  # B-LOC → ?
    #     [True, True, False, True, True],  # I-LOC → ?
    # ])
    @classmethod
    def _build_allowed_transition_mask(cls, id2label: dict[int, str]) -> torch.Tensor:
        # 获取标签总数。
        num_labels = len(id2label)
        # 初始化所有转移均不允许的布尔矩阵。
        allowed = torch.zeros((num_labels, num_labels), dtype=torch.bool)

        for from_id in range(num_labels):
            # 解析起点标签的 BIO 前缀与实体类型。
            from_prefix, from_type = cls._split_bio_tag(id2label[from_id])
            for to_id in range(num_labels):
                # 解析终点标签的 BIO 前缀与实体类型。
                to_prefix, to_type = cls._split_bio_tag(id2label[to_id])

                # 先默认当前转移不合法。
                is_allowed = False
                if to_prefix == "O":
                    is_allowed = True
                elif to_prefix == "B":
                    is_allowed = True
                # 若目标是 I-，则要求来源必须是同类型的 B- 或 I-。
                elif to_prefix == "I":
                    if from_prefix in {"B", "I"} and from_type == to_type:
                        is_allowed = True

                # 写入当前 from -> to 的合法性结果。
                allowed[from_id, to_id] = is_allowed

        # 返回合法转移掩码矩阵。
        return allowed

    def _apply_bio_constraints(self):
        # 约束修改无需梯度参与。
        with torch.no_grad():
            # 把不允许作为起始标签的 start transition 置为极小值。
            self.crf.start_transitions.data = self.crf.start_transitions.data.masked_fill(
                ~self.allowed_start_mask,
                self.NEG_INF,
            )
            # 把所有非法标签转移的分数置为极小值。
            self.crf.transitions.data = self.crf.transitions.data.masked_fill(
                ~self.allowed_transition_mask,
                self.NEG_INF,
            )

    def _get_emissions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        # 调用 BERT 提取上下文特征。
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        # 取最后隐层表示作为序列特征。
        seq_output = outputs.last_hidden_state
        # 经过 dropout 和线性层得到 CRF 所需的发射分数。
        return self.classifier(self.dropout(seq_output))  # (B, L, num_labels)

    @staticmethod
    def _build_crf_mask(
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """构建 CRF 有效位置掩码。

        训练/评估时优先只保留有真实标签的位置（labels != -100），
        避免把 [CLS]/[SEP] 和被忽略的非首子词纳入 CRF 转移学习。
        """
        # 先根据 attention_mask 保留真实 token 位置。
        mask = attention_mask.bool()
        # 如果给出了 labels，则进一步去掉 labels 为 -100 的位置。
        if labels is not None:
            mask = mask & labels.ne(-100)
        return mask

    @staticmethod
    def _pack_valid_positions(
        emissions: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """将有效标签位置压缩成从第 0 步开始的连续序列。

        `torchcrf` 要求每条序列的第一个时间步 mask 必须为 True，
        因此不能直接把 `labels != -100` 的原始位置掩码传入。
        这里把有效位置抽取并左对齐，再交给 CRF。
        """
        # 先得到每个位置是否参与 CRF 的有效掩码。
        valid_mask = BertCRFNER._build_crf_mask(attention_mask, labels)
        # 解析 batch 大小与标签维度。
        batch_size, _, num_labels = emissions.size()
        # 统计每条样本的有效长度。
        lengths = valid_mask.sum(dim=1)
        # 找到当前 batch 中最长的有效序列长度。
        max_valid_len = int(lengths.max().item())

        # 创建压缩后的发射分数张量。
        packed_emissions = emissions.new_zeros((batch_size, max_valid_len, num_labels))
        # 创建压缩后的 mask 张量。
        packed_mask = torch.zeros(
            (batch_size, max_valid_len), dtype=torch.bool, device=emissions.device
        )
        # 默认不创建标签张量，只有 labels 不为空时才构造。
        packed_labels = None
        # 如果调用方传入了真实标签，则同步压缩标签。
        if labels is not None:
            packed_labels = labels.new_zeros((batch_size, max_valid_len))

        for i in range(batch_size):
            # 找到当前样本所有有效位置的索引。
            valid_indices = valid_mask[i].nonzero(as_tuple=False).squeeze(-1)
            # 统计当前样本的有效长度。
            valid_len = int(valid_indices.numel())
            # 若当前样本没有有效位置，则跳过。
            if valid_len == 0:
                continue

            # 把有效位置的发射分数复制到左对齐的新张量中。
            packed_emissions[i, :valid_len] = emissions[i, valid_indices]
            # 对应前 valid_len 个位置标记为 True。
            packed_mask[i, :valid_len] = True
            if packed_labels is not None:
                # 取出当前样本有效位置上的标签。
                valid_labels = labels[i, valid_indices].clone()
                # 把可能残留的 -100 替换成 0，避免 CRF 接收非法标签值。
                valid_labels[valid_labels == -100] = 0
                # 写入压缩标签张量。
                packed_labels[i, :valid_len] = valid_labels

        # 返回压缩后的发射分数、掩码以及可选标签。
        return packed_emissions, packed_mask, packed_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # 每次前向前都再次施加 BIO 约束，防止训练更新后破坏限制。
        self._apply_bio_constraints()
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)

        loss = None
        if labels is not None:
            # 将有效位置压缩成连续序列，满足 torchcrf 的 mask 约束。
            packed_emissions, packed_mask, packed_labels = self._pack_valid_positions(
                emissions, attention_mask, labels
            )
            # crf() 返回对数似然（正值），取负得到损失
            loss = -self.crf(packed_emissions, packed_labels, mask=packed_mask, reduction="mean")

        # 返回原始发射分数以及可选损失。
        return emissions, loss

    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """Viterbi 解码。

        若提供 labels，则只在 labels != -100 的有效标签位置解码，
        返回长度与有效标签数一致，便于与 gold 直接对齐。
        """
        # 解码前先重新施加一次 BIO 约束。
        self._apply_bio_constraints()
        # 计算当前 batch 的发射分数。
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)
        if labels is not None:
            # 压缩有效位置，以便 CRF 返回与 gold 对齐的序列。
            packed_emissions, packed_mask, _ = self._pack_valid_positions(
                emissions, attention_mask, labels
            )
            # 在压缩后的连续序列上执行 Viterbi 解码。
            return self.crf.decode(packed_emissions, mask=packed_mask)

        # 若未提供 labels，则直接以 attention_mask 作为有效位置掩码。
        mask = attention_mask.bool()
        # 返回完整序列上的解码结果。
        return self.crf.decode(emissions, mask=mask)


def build_model(
    use_crf: bool,
    bert_path: str,
    num_labels: int,
    dropout: float = 0.1,
    id2label: dict[int, str] | None = None,
) -> nn.Module:
    """模型工厂函数。"""
    # 根据 use_crf 选择具体模型类别。
    model_cls = BertCRFNER if use_crf else BertNER
    if use_crf:
        model = model_cls(
            bert_path=bert_path,
            num_labels=num_labels,
            dropout=dropout,
            id2label=id2label,
        )
    else:
        # 线性头模型只需基础参数即可构造。
        model = model_cls(bert_path=bert_path, num_labels=num_labels, dropout=dropout)

    # 统计模型参数总量。
    total_params = sum(p.numel() for p in model.parameters())
    # 统计可训练参数量。
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_name = "BERT + CRF" if use_crf else "BERT + Linear"
    print(f"模型：{model_name}")
    print(f"  标签数：{num_labels}")
    print(f"  参数总量：{total_params / 1e6:.1f}M")
    print(f"  可训练参数：{trainable_params / 1e6:.1f}M")
    return model
