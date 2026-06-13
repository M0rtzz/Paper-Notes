---
title: >-
  [论文解读] Rethinking Long-tailed Dataset Distillation: A Uni-Level Framework with Unbiased Recovery and Relabeling
description: >-
  [AAAI 2026][模型压缩][数据集蒸馏] 提出首个面向长尾分布的单层(uni-level)数据集蒸馏框架，通过专家模型去偏、BN统计量公平校准和置信度引导初始化三大策略，在CIFAR-100-LT上提升15.6%、Tiny-ImageNet-LT上提升11.8%，全面超越DAMED。
tags:
  - "AAAI 2026"
  - "模型压缩"
  - "数据集蒸馏"
  - "长尾分布"
  - "单层优化"
  - "BN统计量校准"
  - "无偏恢复"
---

# Rethinking Long-tailed Dataset Distillation: A Uni-Level Framework with Unbiased Recovery and Relabeling

**会议**: AAAI 2026 Oral  
**arXiv**: [2511.18858](https://arxiv.org/abs/2511.18858)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 数据集蒸馏, 长尾分布, 单层优化, BN统计量校准, 无偏恢复

## 一句话总结
提出首个面向长尾分布的单层(uni-level)数据集蒸馏框架，通过专家模型去偏、BN统计量公平校准和置信度引导初始化三大策略，在CIFAR-100-LT上提升15.6%、Tiny-ImageNet-LT上提升11.8%，全面超越DAMED。

## 研究背景与动机

数据集蒸馏(Dataset Distillation, DD)旨在合成一个极小但具有代表性的数据集，使在此数据集上训练的模型达到近似在完整数据上训练的效果，对资源受限场景至关重要。

**长尾分布下的核心挑战**：现实场景中类别不平衡普遍存在（少数头类样本丰富，大量尾类样本稀少），但现有DD方法（DREAM、DATM、EDC等）都假设数据均匀分布，导致：

**合成集被头类主导**：尾类表示不足

**BN统计量偏倚**：不平衡的类别频率破坏了BN的均值和方差估计

**训练不稳定**：中频类接收到不稳定或不充分的梯度反馈

**DAMED的局限（唯一先前工作）**：DAMED是唯一明确针对长尾DD的工作，但存在三个根本问题：

**尾类表示不足**：依赖在长尾数据上训练但未去偏的特征提取专家，尾类表示质量差

**轨迹匹配的无意权衡**：双层优化中，中频类受到不稳定梯度影响

**计算开销大**：双层轨迹优化需要大量GPU内存和时间

**核心矛盾**：轨迹匹配(trajectory matching)方法天然与去偏策略冲突——去偏操作（如重加权、logit校正）会改变专家的优化路径，破坏轨迹匹配的前提；而事后去偏又不可行。

**切入角度**：放弃轨迹匹配范式，转向单层统计对齐框架(uni-level statistical alignment)。通过两个互补组件实现无偏蒸馏：无偏合成图像恢复（通过Observer模型）+ 无偏软标签重标注（通过Teacher模型），配合三个专门策略实现。

## 方法详解

### 整体框架

Pipeline分为准备阶段和蒸馏阶段：
1. 训练去偏的Observer模型（用于BN统计量对齐）和Teacher模型（用于软标签生成）
2. 冻结Observer模型，在整个训练集上进行公平的BN统计量校准
3. 使用Teacher模型引导的置信度多轮初始化策略生成合成图像初始集
4. 通过BN统计量对齐恢复合成图像，然后用Teacher模型重标注软标签
5. 在蒸馏集上训练学生模型评估质量

### 关键设计

#### 1. **专家模型去偏（Observer + Teacher）**
- **功能**：消除Observer和Teacher模型中的类别不平衡偏差
- **混合一致性损失**（增强鲁棒性）：
$$\mathcal{L}_{robust} = -\sum_{i=1}^{2} \cos(\mathbf{z}_i, \text{sg}(\mathbf{p}_{\bar{i}}))$$
  对两个混合标签增强视图进行对称对齐，$\text{sg}(\cdot)$ 为停止梯度算子，确保单侧对齐
- **类别去偏损失**（重平衡类别监督）：
$$\mathcal{L}_{debias} = \alpha \sum_{k=0}^{C-1} \frac{-(r_k)^{-q} y_k \log p_k}{\sum_j (r_j)^{-q}} - \beta \sum_{k=0}^{C-1} y_k \log p_k$$
  其中 $\alpha = (t/T)^2$, $\beta = 1-\alpha$，动态调度使焦点逐渐转向少数类
- **设计动机**：Observer偏倚→BN统计量偏→合成图像偏；Teacher偏倚→软标签不准→语义指导失效。必须在源头去偏

#### 2. **公平BN统计量校准**
- **功能**：消除BN统计量中的样本内偏差(intra-class bias)和类间偏差(inter-class bias)
- **动态动量校准**：冻结Observer模型参数，在整个训练集上做一次前向传播。对每个BN层、每个类别，使用动态调整的动量更新：
$$\mu_{l,t}^c = (1 - \alpha_t^c) \cdot \mu_{l,t-1}^c + \alpha_t^c \cdot \hat{\mu}_{l,t}^c, \quad \alpha_t^c = \frac{B_t^c}{N_{t-1}^c + B_t^c}$$
  其中 $B_t^c$ 是当前batch中类c的样本数，$N_{t-1}^c$ 是之前的累积计数
- **全局均衡平均**：
$$\mu_l(\mathcal{D}; \theta_R) = \frac{1}{C} \sum_{c=0}^{C-1} \mu_{l,T}^c(\mathcal{D}; \theta_R)$$
- **设计动机**：标准BN用固定动量的指数移动平均，最近的batch主导统计量而早期被遗忘。在长尾设置下，尾类每个样本都有高代表性价值，必须平等贡献。动态动量确保每个样本对最终统计量的贡献相等

#### 3. **置信度引导的多轮初始化**
- **功能**：为合成数据集提供多样且高质量的初始化
- **核心思路**：对每张真实图像生成多个增强变体（如随机裁剪），使用Teacher模型的负交叉熵评分。采用多轮选择策略：每轮每张图像最多贡献一个未使用的最高置信增强，确保样本级多样性
- **尾类处理**：当尾类真实样本不足时，插入零初始化的占位符以保持跨类的结构一致性
- **设计动机**：随机初始化收敛差；直接采样真实图像在尾类不可行（尾类样本太少）。多轮选择+置信度引导兼顾质量和多样性

### 损失函数 / 训练策略

- 专家模型训练：$\mathcal{L} = \gamma_1 \mathcal{L}_{robust} + \gamma_2 \mathcal{L}_{debias}$
- 统计对齐损失：$\mathcal{L}(\mathcal{S}) = \sum_{l=1}^{L} \mathbf{D}_l^\mu(\mathcal{S}, \mathcal{D}; \theta_R) + \mathbf{D}_l^\sigma(\mathcal{S}, \mathcal{D}; \theta_R)$
- 学生模型训练（评估）：$\mathcal{L}_{match} = \kappa_1 \cdot \mathcal{L}_{CE}(s(x_s^i), y_s^i) + \kappa_2 \cdot \|\tilde{\mathbf{y}}_s^i - s(x_s^i)\|_2^2$
- 学生网络：CIFAR用depth-3 ConvNet，Tiny-ImageNet用depth-4 ConvNet，ImageNet-LT额外用ResNet-50
- 评估训练1000 epochs，实验重复5次，主要在单张RTX 3090上进行

## 实验关键数据

### 主实验

| 数据集 | IF | IPC | DAMED | 本文 | 提升 |
|--------|-----|-----|-------|------|------|
| CIFAR-10-LT | 100 | 10 | 53.4% | **62.7%** | +9.3% |
| CIFAR-10-LT | 100 | 50 | 64.0% | **68.8%** | +4.8% |
| CIFAR-100-LT | 10 | 10 | 31.5% | **47.1%** | **+15.6%** |
| CIFAR-100-LT | 50 | 10 | 29.8% | **42.1%** | +12.3% |
| Tiny-ImageNet-LT | 10 | 10 | 26.0% | **37.8%** | **+11.8%** |
| ImageNet-LT | 5 | 10 | 20.8% | **24.7%** | +3.9% |
| ImageNet-LT | 10 | 10 | 20.3% | **23.5%** | +3.2% |

### 极端设置（IPC=1，每类仅1张合成图像）

| 数据集 | DAMED | 本文 | 提升 |
|--------|-------|------|------|
| CIFAR-10-LT (IF100) | 24.1% | **44.8%** | +20.7% |
| CIFAR-100-LT (IF50) | 7.8% | **31.8%** | +24.0% |
| Tiny-ImageNet-LT (IF100) | 6.0% | **20.1%** | +14.1% |

### 消融实验（CIFAR-100-LT, IF=50）

| 配置 | IPC=10 | IPC=20 | IPC=50 |
|------|--------|--------|--------|
| 无模型去偏 | 31.7 | 32.3 | 32.8 |
| 无统计量校准 | 40.9 | 41.8 | 42.1 |
| 无自适应初始化 | 40.8 | - | - |
| **完整方法** | **42.1** | **43.4** | **44.2** |

### 计算效率对比（运行时间）

| 方法 | CIFAR-10-LT (IF100) | CIFAR-100-LT (IF50) |
|------|-----|-----|
| DAMED: 专家训练 | 31388s | 26269s |
| DAMED: 蒸馏合成 | 30141s | 29328s |
| 本文: 专家训练 | **2395s** | **2183s** |
| 本文: 蒸馏合成 | **118s** | **273s** |

总耗时不到DAMED的1/20。

### 关键发现

1. **模型去偏是最关键组件**：移除后性能下降最大（42.1→31.7），因为偏倚专家直接限制了蒸馏的性能上限
2. **极端不平衡下优势更大**：IF=256（ImageNet-LT, ResNet-50）下本文48.2% vs DAMED 17.2%，提升31%
3. **跨架构泛化强**：在ConvNet-3/VGG-11/ResNet-18/AlexNet上均显著超越DAMED，且各架构间性能方差更小
4. **尾类精度大幅提升**：class-wise分析显示本文方法在头类和尾类上均优于DAMED，尤其尾类提升显著
5. **内存恒定**：不同IPC下GPU内存保持不变（约3.1GB），而DAMED内存随IPC线性增长

## 亮点与洞察

1. **范式转变**：从双层轨迹匹配→单层统计对齐，从根本上解决了去偏与轨迹匹配之间的矛盾
2. **动态动量的数学优雅**：$\alpha_t^c = B_t^c / (N_{t-1}^c + B_t^c)$ 这一简单公式自然保证了每个样本无论出现顺序如何都平等贡献，同时消除了intra-class和inter-class偏差
3. **20倍加速+内存恒定**：计算效率的巨大提升使方法具有很强的实用性
4. **IPC=1的极端场景**：在每类仅1张图像时仍能大幅超越基线，证明了去偏策略在信息极度匮乏时的价值
5. **超越平衡数据集蒸馏**：在IF=256时甚至超过了在完整平衡ImageNet-1K上蒸馏的某些方法

## 局限与展望

1. 当前仅在图像分类任务上验证，可扩展到检测、分割等任务
2. Teacher和Observer用相同去偏策略，可探索差异化的去偏方案
3. 置信度初始化依赖增强策略的质量
4. 可扩展到联邦学习或多域数据集蒸馏场景
5. 未探索生成模型辅助的初始化方式

## 相关工作与启发

- **DAMED (CVPR 2025)**：唯一先前长尾DD工作，用轨迹匹配+频率感知偏移，但继承了偏倚专家的表示偏差
- **EDC (NeurIPS 2024)**：单层优化的数据集蒸馏，在平衡设置下效果好，但缺乏去偏策略
- **RDED (CVPR 2024)**：基于真实图像增强的单层方法，在不平衡下尾类采样困难
- **SRe2L (CVPR 2023)**：通过特征匹配替代轨迹匹配降低内存开销
- **UniMix**：类别感知的Mixup增强策略，为长尾识别中的尾类增强提供启发

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Rectifying Soft-Label Entangled Bias in Long-Tailed Dataset Distillation](../../NeurIPS2025/model_compression/rectifying_soft-label_entangled_bias_in_long-tailed_dataset_distillation.md)
- [\[CVPR 2025\] Distilling Long-tailed Datasets](../../CVPR2025/model_compression/distilling_long-tailed_datasets.md)
- [\[ICLR 2026\] Dataset Color Quantization: A Training-Oriented Framework for Dataset-Level Compression](../../ICLR2026/model_compression/dataset_color_quantization_a_training-oriented_framework_for_dataset-level_compr.md)
- [\[AAAI 2026\] TGDD: Trajectory Guided Dataset Distillation with Balanced Distribution](tgdd_trajectory_guided_dataset_distillation_with_balanced_distribution.md)
- [\[AAAI 2026\] Post Training Quantization for Efficient Dataset Condensation](post_training_quantization_for_efficient_dataset_condensation.md)

</div>

<!-- RELATED:END -->
