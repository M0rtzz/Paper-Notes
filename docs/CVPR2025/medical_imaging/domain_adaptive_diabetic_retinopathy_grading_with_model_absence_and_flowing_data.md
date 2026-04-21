---
title: >-
  [论文解读] Domain Adaptive Diabetic Retinopathy Grading with Model Absence and Flowing Data
description: >-
  [CVPR 2025][医学图像][领域自适应] 本文提出 GUES（Generative Unadversarial Examples）方法，在无法访问源模型参数和标签、目标数据以流式到达的极端在线无模型领域自适应（OMG-DA）场景下，通过 VAE 生成个性化非对抗性扰动并以显著性图作为伪监督，提升冻结源模型在目标域上的糖尿病视网膜病变（DR）分级性能。
tags:
  - CVPR 2025
  - 医学图像
  - 领域自适应
  - 糖尿病视网膜病变
  - 模型不可见
  - 非对抗样本
  - 在线适应
---

# Domain Adaptive Diabetic Retinopathy Grading with Model Absence and Flowing Data

**会议**: CVPR 2025  
**arXiv**: [2412.01203](https://arxiv.org/abs/2412.01203)  
**代码**: 无  
**领域**: 医学图像  
**关键词**: 领域自适应、糖尿病视网膜病变、模型不可见、非对抗样本、在线适应

## 一句话总结

本文提出 GUES（Generative Unadversarial Examples）方法，在无法访问源模型参数和标签、目标数据以流式到达的极端在线无模型领域自适应（OMG-DA）场景下，通过 VAE 生成个性化非对抗性扰动并以显著性图作为伪监督，提升冻结源模型在目标域上的糖尿病视网膜病变（DR）分级性能。

## 研究背景与动机

**领域现状**：糖尿病视网膜病变（DR）是全球主要致盲原因之一，深度学习在自动化 DR 分级中取得了显著进展。然而，不同设备、种族、时间等因素导致的域偏移严重影响了模型在真实临床场景中的泛化能力。现有领域自适应方法包括 UDA（需要源域数据）、DG（需要多个源域标注数据）和 SFDA（需要源模型参数）。

**现有痛点**：在实际临床环境中，存在三个同时出现的约束：(1) **模型不可见**——源模型的架构和参数不可访问，以防止模型攻击（如成员推断攻击）；(2) **流式数据**——患者数据逐批到达而非预先收集完毕，无法进行离线训练；(3) **源数据隐私**——源域训练数据不可访问。现有 SFDA 方法依赖完全访问模型参数进行适应，无法满足模型不可见的要求。

**核心矛盾**：没有源模型参数、没有标注数据、数据流式到达——在这三重约束下，如何提升模型在新域上的预测能力？传统方法至少需要访问模型参数（SFDA/TTA），而 OMG-DA 场景中模型被当作完全黑箱。

**本文目标** (1) 定义并解决 OMG-DA（Online Model-aGnostic Domain Adaptation）这一新问题设置；(2) 在不访问任何模型信息的前提下，通过修改输入数据的分布来适应目标域；(3) 在 DR 分级任务上验证方法的有效性。

**切入角度**：既然无法修改模型，就修改数据——从"模型适应"转向"数据适应"。作者将传统的非对抗学习（unadversarial learning）从迭代优化形式重新推导为生成式形式，证明扰动可以通过一个生成函数直接输出，而无需模型梯度。再用 VAE 实例化这个生成函数，用显著性图作为伪监督。

**核心 idea**：用 VAE 学习生成个性化非对抗扰动，以显著性图为伪标签，在模型完全不可见的条件下从数据端实现领域自适应。

## 方法详解

### 整体框架

GUES 的输入是无标签的目标域流式数据 $x_t$。VAE 编码器将 $x_t$ 映射为潜在变量 $z$（近似初始噪声对图像的偏导数 $\partial\delta_0/\partial x$），解码器生成个性化扰动 $\delta_t$。将 $\delta_t$ 与原始输入通过旁路连接组合，得到非对抗样本 $\hat{x}_t = x_t + \delta_t$。训练时以 $x_t$ 的显著性图 $g_t$ 作为重建监督。推理时，将生成的 $\hat{x}_t$ 送入冻结的源模型进行预测。

### 关键设计

1. **生成式非对抗学习理论（定理 1）**:

    - 功能：将传统迭代式非对抗扰动优化重新表述为生成函数
    - 核心思路：传统非对抗学习通过迭代 $\delta_{k+1} = \delta_k + \alpha \cdot \text{sign}(\nabla_x L(f_\theta(x+\delta_k), y))$ 求解最优扰动，需要模型参数 $\theta$、损失函数 $L$ 和标签 $y$。作者证明该迭代过程等价于 $\delta_k = \delta_0 + V \cdot F_\Phi(\partial\delta_0/\partial x)$，其中 $F_\Phi$ 是一个可学习的生成函数，$\partial\delta_0/\partial x$ 是潜在输入变量。这将问题转化为学习该生成函数，不再需要模型参数和标签。
    - 设计动机：OMG-DA 场景中无法获得模型梯度和标签，因此无法使用传统迭代式非对抗学习。生成式重构使得问题可以用标准的生成模型训练来解决。

2. **VAE 模型实例化**:

    - 功能：实现生成函数 $F_\Phi(\partial\delta_0/\partial x)$ 的学习
    - 核心思路：需要解决两个子问题——(A) 识别未知的潜在输入 $\partial\delta_0/\partial x$，(B) 选择伪监督信号。对于 (A)，由于 $\partial\delta_0/\partial x$ 同时依赖随机噪声 $\delta_0$ 和输入 $x$，用 VAE 的编码器+重参数化技巧将其建模为以 $x$ 为条件的高斯分布 $\mathcal{N}(\mu(x), \sigma^2(x))$。解码器则实现 $F_\Phi$。
    - 设计动机：VAE 天然具备从输入中采样潜在变量并生成与输入同尺寸输出的能力，与生成式非对抗学习的理论框架完美匹配。

3. **显著性图作为伪监督（定理 2）**:

    - 功能：在无标签条件下提供扰动的训练信号
    - 核心思路：选择细粒度显著性图 $s = G(x)$ 作为伪扰动标签。理论上证明了 $\partial\delta_0/\partial x \leq U \cdot s$，即显著性图提供了潜在输入变量的上界。实际上，DR 图像的显著性图能有效定位出血点、软硬渗出物等病灶区域（与 Grad-CAM 的激活类似），对扰动方向具有指导意义。
    - 设计动机：显著性图是纯基于图像像素计算的（中心-周围差异），无需模型和标签，且 (1) 能定位 DR 相关病灶，(2) 为潜在变量提供理论上界，是满足 OMG-DA 约束的理想选择。

### 损失函数 / 训练策略

总损失为 $L_{GUES} = \alpha L_{KL} + \beta L_{MSE}$。$L_{KL}$ 是潜在空间的 KL 散度正则化，确保 $z$ 接近标准正态分布；$L_{MSE}$ 是生成的非对抗样本 $\hat{x}_t$ 与显著性图 $g_t$ 之间的重建损失。训练在流式数据上在线进行，每批数据到达即更新 VAE 参数。

## 实验关键数据

### 主实验

在 4 个 DR 数据集（APTOS、DDR、DeepDR、MD2）上构建 12 个跨域任务。与 Source（直接用源模型）、SFDA 方法（SHOT、NRC、CoWA）、TTA 方法（TENT、DDA）对比。

| 迁移任务 | 评价 | Source | GUES | SHOT (SFDA) | TENT (TTA) |
|---------|------|--------|------|-------------|------------|
| DDR→APTOS | ACC/QWK/AVG | 65.6/72.9/69.3 | **76.0/81.8/78.9** | 77.0/84.2/80.6 | 66.3/73.2/69.7 |
| DDR→DeepDR | AVG | 52.8 | **62.5** | 66.9 | 53.1 |
| APTOS→DDR | AVG | 59.9 | **60.8** | 67.9 | 59.2 |

GUES 作为 OMG-DA 方法（无需访问模型参数），性能与需要完全访问模型的 SFDA 方法相比有竞争力，在部分任务上已接近 SHOT。与同等约束的 TTA 方法相比优势明显。

### 组合实验（GUES + 已有方法）

| 方法 | DDR→APTOS AVG | APTOS→DDR AVG | 平均 |
|------|--------------|--------------|------|
| SHOT | 80.6 | 67.9 | 74.1 |
| SHOT + GUES | **83.0** | **70.2** | 76.6 |
| TENT | 69.7 | 59.2 | 64.5 |
| TENT + GUES | **73.4** | **61.8** | 67.6 |

GUES 作为数据预处理模块，可以叠加到已有适应方法上进一步提升性能。

### 关键发现
- 在最严格的 OMG-DA 设置下（无模型、无标签、流式数据），GUES 仍能有效提升源模型在 12 个迁移任务中 10 个以上的性能
- GUES 对 batch size 鲁棒：即使 batch=1，也能保持有效的适应
- GUES 可作为插件与 SFDA/TTA 方法组合使用，进一步提升性能
- 显著性图作为伪监督在 DR 任务中特别有效，因为它天然高亮病灶区域

## 亮点与洞察
- **从"适应模型"到"适应数据"的范式转变**：在模型不可见的约束下，转而修改输入数据分布，这一思路在医疗隐私场景下极具实用价值
- **生成式重构非对抗学习**的理论贡献（定理 1）将迭代优化转化为生成函数，摆脱了对模型梯度的依赖，是核心的理论支撑
- **显著性图的双重角色**——既是病灶定位工具，又是潜在变量的理论上界——这一联系的建立（定理 2）非常巧妙
- OMG-DA 作为新问题设置，比 SFDA 更贴近真实临床部署场景（模型通常作为加密 API 提供），有实际落地意义

## 局限与展望
- GUES 的性能在部分任务上仍明显弱于需要完全访问模型的 SFDA 方法（如 SHOT），因为数据端适应能力天然弱于模型端适应
- 显著性图作为伪监督仅在 DR 这种病灶区域较明显的任务上验证，推广到其他医学任务（如肿瘤分级）需要进一步验证
- VAE 的生成质量可能受限于流式数据的多样性，早期 batch 训练不足时扰动质量可能较差
- 文中仅在分类任务上验证，检测和分割任务是否适用有待探索

## 相关工作与启发
- **vs SHOT/NRC (SFDA)**: 这些方法需要完全访问模型参数进行熵最小化/对比学习，GUES 则完全不需要模型参数，仅修改输入数据，约束更强但适用范围更广
- **vs TENT (TTA)**: TENT 在测试时更新模型的 BN 层，仍需模型参数。GUES 在同等模型不可见条件下显著优于 TENT
- **vs 传统非对抗学习 [Salman et al.]**: 传统方法需要模型梯度和标签生成类别级扰动，GUES 生成个体级扰动且无需任何模型信息

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ OMG-DA 问题设置贴合临床需求且无先例，生成式非对抗学习理论推导有深度
- 实验充分度: ⭐⭐⭐⭐ 12 个跨域任务覆盖全面，但消融实验可更详细
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，理论推导严谨，但部分符号表达偏复杂
- 价值: ⭐⭐⭐⭐ 在医疗隐私保护场景下的适应方案有实际价值，但需更多场景验证

<!-- RELATED:START -->

## 相关论文

- [Domain-Adaptive Transformer for Data-Efficient Glioma Segmentation in Sub-Saharan MRI](../../NeurIPS2025/medical_imaging/domain-adaptive_transformer_for_data-efficient_glioma_segmentation_in_sub-sahara.md)
- [Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning](residual_sodap_residual_self-organizing_domain-adaptive_prompting_with_structura.md)
- [Adaptive Domain Shift in Diffusion Models for Cross-Modality Image Translation](../../ICLR2026/medical_imaging/adaptive_domain_shift_in_diffusion_models_for_cross-modality_image_translation.md)
- [LangDAug: Langevin Data Augmentation for Multi-Source Domain Generalization in Medical Imaging](../../ICML2025/medical_imaging/langdaug_langevin_data_augmentation_for_multi-source_domain_generalization_in_me.md)
- [EchoONE: Segmenting Multiple Echocardiography Planes in One Model](echoone_segmenting_multiple_echocardiography_planes_in_one_model.md)

<!-- RELATED:END -->
