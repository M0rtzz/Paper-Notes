---
title: >-
  [论文解读] Test-Time Canonicalization by Foundation Models for Robust Perception
description: >-
  [ICML 2025][自监督学习][测试时优化] 提出 FoCal 框架，在推理阶段利用 CLIP 和 Stable Diffusion 的视觉先验，通过"变换-排序"策略将输入图像变换为最具视觉典型性的版本，无需重训练即可提升模型对视角、光照、旋转等变换的鲁棒性。
tags:
  - ICML 2025
  - 自监督学习
  - 测试时优化
  - 典范化
  - 基础模型
  - 鲁棒感知
  - 能量函数
---

# Test-Time Canonicalization by Foundation Models for Robust Perception

**会议**: ICML 2025  
**arXiv**: [2507.10375](https://arxiv.org/abs/2507.10375)  
**代码**: [https://github.com/sutkarsh/focal](https://github.com/sutkarsh/focal)  
**领域**: 自监督学习  
**关键词**: 测试时优化, 典范化, 基础模型, 鲁棒感知, 能量函数

## 一句话总结

提出 FoCal 框架，在推理阶段利用 CLIP 和 Stable Diffusion 的视觉先验，通过"变换-排序"策略将输入图像变换为最具视觉典型性的版本，无需重训练即可提升模型对视角、光照、旋转等变换的鲁棒性。

## 研究背景与动机

现实场景中，机器人和自动驾驶系统需要在多变的视角、光照和环境下稳定感知物体。然而即便是 CLIP、SAM 这样的大规模基础模型也存在脆弱性——CLIP 在非常规视角下会误分类，SAM 在侧向物体上分割失败。这种脆弱性源于训练数据中的**摄影师偏差 (photographer's bias)**：互联网图片过度集中于正面/正立姿态和理想光照条件。

现有应对方案有两类固有缺陷：

- **数据增强 (DA)**：需预定义变换类型，对稀有类效果差，可能过度正则化伤害部分类别
- **等变网络**：将数学对称性硬编码进架构，但无法扩展到 3D 视角变换等复杂现实变换

两类方法的根本问题是将不变性在训练时固化，无法适应训练分布之外的新变换。本文受人类"心理旋转 (mental rotation)"启发——人类识别不熟悉物体时会先将其心理旋转到典型视角——提出在推理时动态实现不变性。

## 方法详解

### 整体框架

FoCal (Foundation-model guided Canonicalization) 采用**"变换 (Vary) + 排序 (Rank)"**两阶段策略：

1. **Vary 阶段**：对输入图像生成一组候选变换版本（如不同旋转角度的图像、不同视角的 3D 渲染）
2. **Rank 阶段**：用基础模型构建的能量函数对所有候选打分，选择能量最低（最"典型"）的版本作为典范形式
3. 将典范化后的图像送入下游模型（CLIP 分类、SAM 分割等）进行推理

核心优化目标：

$$t^* = \arg\min_{t \in \mathcal{T}} E_{\text{FoCal}}(t(\mathbf{x}))$$

$$\mathbf{y} = f(t^*(\mathbf{x}))$$

其中 $\mathcal{T}$ 是变换集合，$E_{\text{FoCal}}$ 是能量函数，$f$ 是下游任务模型。

### 关键设计

**1. 典范化的理论基础**

基于 Kaba et al. (2022) 的形式化：定义典范化函数 $h(\mathbf{x}) = \arg\min_{t \in \mathcal{T}} E(t(\mathbf{x}))$，在温和条件下可证明该函数满足不变性/等变性。关键洞察是：一张图像的所有变换版本定义了自然图像分布的一个"切片"，在此切片中，某些版本在真实世界数据中出现频率更高。基础模型隐式学到了这种分布先验。

**2. CLIP 能量函数**

将 CLIP 视为能量模型，定义无条件能量为均值和最大值 logit 的组合：

$$E_{\text{CLIP}}(\mathbf{x}; \alpha, \beta) = (\alpha \cdot \text{mean} - \beta \cdot \text{max})_{c \in \{1,...,|C|\}} f_\theta(\mathbf{x})[c]$$

其中 $\alpha, \beta$ 为超参数，使用 CLIP ViT-H-14，以图文嵌入的余弦相似度作为 logit。CLIP 能量侧重语义：选择最接近某预定义类别的图像。

**3. 扩散模型能量函数**

基于 Stable Diffusion 2 提取能量：

$$E_{\text{diff}}(\mathbf{x}) = \frac{1}{T} \sum_{t=1}^{T} \mathbb{E}_{\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ \| \epsilon - \epsilon_\theta(\mathbf{x}_t, t) \|^2 \right]$$

实际只需 5-10 个去噪步骤即可足够。扩散能量充当通用的外观先验。

**4. 联合能量函数**

两种能量加权组合：

$$E_{\text{FoCal}}(t(\mathbf{x})) = \gamma_1 \cdot E_{\text{CLIP}}(t(\mathbf{x})) + \gamma_2 \cdot E_{\text{diff}}(t(\mathbf{x}))$$

**5. 不同变换的候选生成**

- **2D 旋转**：直接枚举 $C_8$（8 个离散旋转角度）
- **3D 视角**：用 TRELLIS 生成模型在球面上每隔 30° 渲染多视角图像（共 60 个候选）
- **颜色/对比度**：在 log-chrominance 空间和 gamma 空间中采样
- **日/夜变换**：在 Stable Diffusion 潜在空间中插值

### 损失函数 / 训练策略

FoCal 是**完全无训练**的框架，不涉及任何梯度更新或微调。其"训练策略"体现在：

- **贝叶斯优化 (BO)**：对连续/高维变换空间（如颜色 2D、主动视觉 6D），使用高斯过程 (GP) + RBF 核 + Expected Improvement 采集函数，通常 50-100 次评估即可找到良好解，避免暴力搜索
- **超参数选择**：$\alpha=1, \beta=0.5$ 适用于大多数分类场景；分割任务使用 $\gamma_1=0.54, \gamma_2=0.67$；通过 BO 在小验证集上调参
- **假设条件**：(1) 变换集合中至少存在一个分布内图像；(2) 基础模型对分布内图像赋予更低能量；(3) 下游模型在分布内表现最优

## 实验关键数据

### 主实验

**3D 视角鲁棒性（Objaverse-LVIS & CO3D）**

| 数据集 | 指标 | 本文 (FoCal) | 之前方法 | 提升 |
|---|---|---|---|---|
| Objaverse-LVIS（最差视角） | 分类准确率 | 62.0% | 12.0% (OV-Seg) | +50.0% |
| Objaverse-LVIS（整体 Top-10） | 平均准确率 | 84.5% | 76.4% (TTA-10) | +8.1% |
| CO3D (t=0.3) | 分类准确率 | 49.5% | 45.9% (TRELLIS) | +3.6% |
| CO3D (t=0.5) | 分类准确率 | 55.3% | 53.4% (TRELLIS) | +1.9% |

**2D 旋转（vs PRLC，在 PRLC 训练设定下）**

| 数据集 | 架构 | FoCal 旋转准确率 | PRLC 旋转准确率 | 提升 |
|---|---|---|---|---|
| CIFAR10 | ResNet-50 | 95.6% | 95.1% | +0.5% |
| CIFAR10 | ViT | 96.0% | 94.8% | +1.2% |
| CIFAR100 | ResNet-50 | 82.2% | 81.8% | +0.4% |
| CIFAR100 | ViT | 84.4% | 82.2% | +2.2% |
| ImageNet (ViT) | ViT | 71.9% | 60.5% | +11.4% |

### 消融实验

| 能量配置 | 姿态准确率 | 姿态误差 | 说明 |
|---|---|---|---|
| 仅 CLIP 能量 | 68.9% | 37.1° | 语义先验不足以精确定位 |
| 仅扩散能量 | 82.7% | 22.6° | 外观先验更有效 |
| CLIP + 扩散（完整 FoCal） | 89.5% | 13.5° | 两者互补，误差降低 64% |

| 方法 | CIFAR10 | CIFAR100 | STL10 | 说明 |
|---|---|---|---|---|
| 无矫正 | 65.4 | 50.6 | 93.4 | 基线 |
| FoCal (Ours) | 93.7 | 76.2 | 97.5 | 大幅提升 |
| TTA | 82.8 | 61.7 | 96.6 | FoCal 优于 TTA 10-15% |

### 关键发现

1. **零样本超越有监督典范化器**：FoCal 在 PRLC 的所有训练设定（6 个数据集×架构组合）上匹配或超越 PRLC，尽管完全不需要训练
2. **跨数据集泛化能力强**：PRLC 跨数据集迁移时姿态准确率下降 12-18%，FoCal 波动 <3%
3. **分割任务同样有效**：在 COCO 上与 PRLC 的 mAP 持平（65.9），同时姿态准确率提升 2.1%
4. **日/夜变换**：仅用 "street" 一个类别的 CLIP 能量，91% 概率选择白天图像
5. **主动视觉**：在 6-DoF 虚拟场景中，摄像机自然聚焦到显著物体并保持正立角度

## 亮点与洞察

- **范式创新**：将不变性从训练时硬编码转变为推理时优化，类比 LLM 的测试时计算缩放 (test-time compute scaling)
- **理论优雅**：利用 Kaba et al. 的能量最小化框架，理论上保证不变性/等变性，而不要求能量函数本身满足等变性
- **"切片"直觉**：变换族定义了自然图像分布的一个切片，基础模型的能量函数可以在切片上找到最可能的点——这一视角统一了旋转/颜色/视角等不同变换
- **实用性**：完全即插即用，无需修改任何下游模型架构或重新训练

## 局限与展望

1. **计算开销大**：需要对每个候选变换评估 CLIP + SD 能量，2D 旋转约 56× 推理成本；3D 视角包含 TRELLIS 生成约 13.3 秒/样本。可通过 System-1/2 策略（先判断是否需要典范化）缓解
2. **变换选择需人工指定**：当前需要人工决定使用哪种变换生成器（旋转？视角？颜色？），未来应自动检测
3. **非可逆变换的局限**：理论框架要求变换可逆，但 3D 视角变换不可逆，仅能提供近似不变性
4. **颜色校正不及专用方法**：在 RCC 数据集上中值角误差 6.4°，远不及 Barron & Tsai 的 1.3°，因为 FoCal 优化的是"视觉典型性"而非"色彩中性"
5. **并行化需求**：虽然理论上所有候选评估可并行，但实际需要大量 GPU 显存

## 相关工作与启发

- **Kaba et al. (2022)** & **PRLC (Mondal et al., 2023)**：学习型典范化的理论和实践基础，但需要针对特定数据集/变换训练，泛化能力有限
- **Grathwohl et al. (2020)**：将分类器解读为能量模型的理论基础，FoCal 直接应用此框架
- **Graikos et al. (2022)**：扩散模型作为即插即用先验，FoCal 将其用于能量函数构建
- **Test-time compute scaling (Snell et al., 2024; Zaremba et al., 2025)**：FoCal 可视为视觉领域的测试时计算缩放——生成多个候选，用学习到的排序器选择最佳
- **启发**：该工作开创性地将测试时搜索/优化引入视觉鲁棒性问题，可能催生更多"推理时自适应"的视觉方法

## 评分

| 维度 | 评分 (1-5) | 说明 |
|---|---|---|
| 新颖性 | ⭐⭐⭐⭐⭐ | 将不变性从训练时约束转为推理时优化，范式层面的创新 |
| 理论深度 | ⭐⭐⭐⭐ | 能量最小化框架有严格理论保证，但近似不变性部分较弱 |
| 实验充分性 | ⭐⭐⭐⭐⭐ | 覆盖 5 类变换、4+ 数据集、3 个下游模型，消融充分 |
| 实用价值 | ⭐⭐⭐⭐ | 零训练即插即用，但计算开销是实际部署的瓶颈 |
| 写作质量 | ⭐⭐⭐⭐⭐ | 动机清晰，图示优秀，直觉与理论结合好 |
| 综合评分 | ⭐⭐⭐⭐⭐ | 视觉鲁棒性领域的开创性工作，有望成为有影响力的论文 |

<!-- RELATED:START -->

## 相关论文

- [Test-Time Training Provably Improves Transformers as In-Context Learners](test-time_training_provably_improves_transformers_as_in-context_learners.md)
- [Robust Tabular Foundation Models](../../AAAI2026/self_supervised/robust_tabular_foundation_models.md)
- [Towards Benchmarking Foundation Models for Tabular Data With Text](towards_benchmarking_foundation_models_for_tabular_data_with_text.md)
- [Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting](../../CVPR2026/self_supervised/redepth_anything_test-time_depth_refinement_via_self-supervised_re-lighting.md)
- [What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models](what_has_a_foundation_model_found_using_inductive_bias_to_probe_for_world_models.md)

<!-- RELATED:END -->
