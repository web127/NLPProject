1. 交叉熵损失函数的原理，为什么是-y*log(y_pred),y_pred为什么要经过softmax计算
<details>
<summary>点击查看答案</summary>
答：交叉熵衡量真实标签和预测概率之间的差异，公式为 -y*log(y_pred)。
</details>

2. 损失函数有什么特征
<details>
<summary>点击查看答案</summary>
答：损失函数的特征是：
- 损失函数的值必须大于等于0
- 损失函数的值越小，模型的预测越准确
- 损失函数的导数必须大于等于0
</details>

3. softmax,sigmod,交叉熵之间的区别 
<details>
<summary>点击查看答案</summary>
- 答：softmax将向量转化为总和为1，0-1的概率分布，Sigmoid 就是 Softmax 在二分类时的简化形式！,交叉熵是多分类损失函数，内部已经做了softmax</details>

4. 神经网络核心步骤
<details>
<summary>点击查看答案</summary>
答：神经网络的核心步骤是：
核心步骤：
0.前向传播
1.计算损失
2.计算梯度
3.更新权重
</details>
