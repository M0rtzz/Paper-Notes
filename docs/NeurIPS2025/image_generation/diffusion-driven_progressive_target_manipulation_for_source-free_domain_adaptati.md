---
title: >-
  [论文解读] Diffusion-Driven Progressive Target Manipulation for Source-Free Domain Adaptation
description: >-
  [NeurIPS 2025][图像生成][域适应] 提出 DPTM 框架，利用潜在扩散模型对不可信目标样本进行语义变换，生成伪目标域并通过渐进式重建机制迭代缩小与真实目标域的差距，在大域偏移场景下比现有 SFDA SOTA 提升高达 18.6%。 Source-Free Domain Adaptation (SFDA) 要求…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "域适应"
  - "扩散模型"
  - "伪目标域生成"
  - "渐进式优化"
  - "语义变换"
---

# Diffusion-Driven Progressive Target Manipulation for Source-Free Domain Adaptation

**会议**: NeurIPS 2025  
**arXiv**: [2510.25279](https://arxiv.org/abs/2510.25279)  
**代码**: 无  
**领域**: 图像生成 / 扩散模型 / 域适应  
**关键词**: Source-Free Domain Adaptation, 扩散模型, 伪目标域生成, 渐进式优化, 语义变换

## 一句话总结

提出 DPTM 框架，利用潜在扩散模型对不可信目标样本进行语义变换，生成伪目标域并通过渐进式重建机制迭代缩小与真实目标域的差距，在大域偏移场景下比现有 SFDA SOTA 提升高达 18.6%。

## 研究背景与动机

Source-Free Domain Adaptation (SFDA) 要求在仅有预训练源模型和无标注目标数据的条件下实现域适应。现有方法分为两类，均受限于源-目标域差异：

- **非生成式方法**：依赖源模型产生的伪标签，但在大域偏移场景下伪标签大量不可靠（如 Office-Home 上 Ar→Pr 仅约 60%），性能极不稳定
- **生成式方法**：生成伪源域再转化为标准 UDA 问题，但生成过程引入无关域特征，反而扩大源-目标域差距

关键洞察：两类方法的根本瓶颈都是源-目标域偏移。作者提出一种全新范式——直接生成伪目标域而非伪源域，从源头消除域差距的根本限制。

## 方法详解

### 整体框架

DPTM 由三个核心组件构成，通过 R 次渐进迭代执行：

1. **信任集/非信任集划分**：基于预测不确定性将目标数据分为可信赖子集 V 和不可信赖子集 U
2. **非信任集操纵策略**：用扩散模型将 U 中样本语义变换到新分配类别，同时保持目标域分布特征
3. **渐进式重建机制**：迭代减小伪目标域与真实目标域的差距

### 关键设计

#### 信任集/非信任集划分

使用目标模型预测的熵 H 作为不确定性度量，设置阈值 E 划分：
- H ≤ E 的样本加入信任集 V，直接用伪标签监督训练
- H > E 的样本加入非信任集 U，进行扩散操纵处理

为非信任集每个样本均匀分配新类别标签（ŷ_l = l mod ⌊|U|/C⌋），确保类别均衡，丢弃尾部残余样本。

#### Target-guided Initialization（目标域引导初始化）

基于扩散模型采样起点对生成结果有重大影响的发现，利用 FFT 分频构造起点：
- 从原始非信任样本 x_l^u 提取**低频分量** F_x^L = FFT(x) ⊙ H（包含域特征如风格、纹理）
- 从语义中性的高斯噪声 I_G 提取**高频分量** F_IG^H = FFT(I_G) ⊙ (1-H)（避免原样本语义泄漏）
- 组合后 IFFT 得到语义中性但保留目标域特征的伪图像 x̃
- 编码到潜空间 ẑ₀ = E(x̃)，加 T 步 DDPM 前向噪声得到采样起点 z_T

#### Semantic Feature Injection（语义特征注入）

在去噪每个时间步 t 执行 zigzag 自反射操作：
- 先去噪 z_t → z_{t-1}
- 通过 DDIM inversion 将分配标签 ŷ_l 的语义注入 z_{t-1} → z̃_t
- 使用引导尺度 γ₂ 的 classifier-free guidance 确保语义对齐
- **仅提取 z̃_t 的高频分量**（含语义信息），丢弃低频（可能含域噪声伪影）

#### Domain-specific Feature Preservation（域特征保留）

在每个时间步组合两部分频率分量：
- **高频**：来自语义注入的 F_{z̃_t}^H（目标类别语义）
- **低频**：从 ẑ₀ 加 t 步噪声后提取 F_{ẑ₀,t}^L（目标域分布特征）
- IFFT 合成增强潜变量 z̃'_t，同时保有语义和域特征

#### 渐进式重建机制

进行 R 次迭代优化：
- 第 r 次迭代后用更新的目标模型重新划分 V^(r+1) 和 U^(r+1)
- 随模型改善 |V^(r+1)| > |V^(r)|，非信任集逐渐缩小
- 操纵量减少→域差距减小→模型进一步改善，形成正反馈循环

### 损失函数 / 训练策略

- 信任集与操纵后的非信任集合并构成伪目标域 D_p = V ∪ U^m
- 在伪目标域上标准交叉熵监督微调目标模型
- 扩散模型：预训练 SD v1.5，512×512，20 步去噪
- 超参数：γ₁=5.5, γ₂=0, E=0.01, R=10
- ResNet-50/101 作为适应模型，SGD 优化器，训练 15K-20K 步

## 实验关键数据

### 主实验

**Office-31（ResNet-50）**

| 方法 | D→A | W→A | Avg |
|------|------|------|------|
| ProDe (ICLR25) | 79.8 | 79.0 | 89.9 |
| DM-SFDA | 82.7 | 83.5 | 93.7 |
| **DPTM (Ours)** | **92.0** | **91.7** | **95.8** |

困难任务 D→A 和 W→A 分别提升 **+9.3%** 和 **+8.2%**。

**Office-Home（ResNet-50，12 个 DA 任务）**

| 方法 | Ar→Cl | Pr→Cl | Rw→Cl | Avg |
|------|-------|-------|-------|------|
| ProDe (ICLR25) | 64.0 | 65.4 | 65.5 | 81.1 |
| DM-SFDA | 68.5 | 69.6 | 68.5 | 79.5 |
| **DPTM** | **86.7** | **86.4** | **87.1** | **91.2** |

比 ProDe 平均提升 **+10.1%**，困难 →Cl 任务提升 20%+。

**DomainNet-126（ResNet-50，12 个任务）**

| 方法 | C→P | Avg |
|------|------|------|
| CPGA | 61.2 | 67.6 |
| ProDe | 79.3 | 81.5 |
| **DPTM** | **85.6** | **85.2** |

### 消融实验

**阈值 E 的影响（Office-Home, R=10）**

| E | Avg |
|------|------|
| 0.001 | 80.7 |
| 0.005 | 86.7 |
| 0.01 | 91.2 |

较大 E 将更多样本归入非信任集操纵，效果更好。

**操纵策略组件消融**：缺少任何一个（Target-guided Init / Semantic Injection / Domain Preservation）都导致语义对齐或域保持失败。SD15 和 SDXL 性能相当（均 75.6% Avg），SD15 更高效。

### 关键发现

- 大域偏移（如 Rw→Cl）提升最为显著，验证伪目标域策略对域差距的根本性解决
- 渐进式重建中信任集不断扩大，体现自我改善正反馈
- 频域分离有效解耦语义和域特征，三个操纵组件缺一不可

## 亮点与洞察

1. **范式创新**：首次提出生成伪目标域的 SFDA 策略，从根本上突破域偏移瓶颈
2. **频域设计精妙**：FFT 低频=域特征、高频=语义特征的分离贯穿整个方法
3. **渐进式自我改善**：非信任集随迭代缩小的正反馈，无需额外数据即可持续优化
4. **困难场景表现突出**：在其他方法通常失败的大域偏移场景实现最大增益（18.6%）

## 局限与展望

- 依赖预训练扩散模型（SD v1.5），计算和存储开销大
- 阈值 E 和迭代次数 R 需要调优，不同数据集最优设置不同
- 仅在分类任务验证，可扩展到检测、分割等下游任务
- 扩散生成的 512×512 分辨率可能限制在更高分辨率任务上的表现

## 相关工作与启发

- 与 DM-SFDA 的核心区别：DM-SFDA 生成伪源域，DPTM 操纵目标数据生成伪目标域
- 频域操作灵感来自 FreeInit/FlexiEdit 对扩散起点低频影响的研究
- Zigzag inversion（bai2024zigzag）用于在采样轨迹中注入语义
- 渐进式训练与课程学习相关但方向不同——是数据集在缩小而非难度在变化

## 评分

- 新颖性：⭐⭐⭐⭐⭐（伪目标域生成范式全新，频域分离原创性强）
- 技术深度：⭐⭐⭐⭐⭐（扩散模型+频域操作+渐进式优化深度融合）
- 实验充分性：⭐⭐⭐⭐⭐（4 个基准、21 个对比方法、详尽消融）
- 实用性：⭐⭐⭐⭐（效果显著但计算开销较高）
- 表达清晰度：⭐⭐⭐⭐（方法复杂但分模块表述清晰）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] A Data-Driven Prism: Multi-View Source Separation with Diffusion Model Priors](a_data-driven_prism_multi-view_source_separation_with_diffusion_model_priors.md)
- [\[CVPR 2025\] Everything to the Synthetic: Diffusion-driven Test-time Adaptation via Synthetic-Domain Alignment](../../CVPR2025/image_generation/everything_to_the_synthetic_diffusion-driven_test-time_adaptation_via_synthetic-.md)
- [\[NeurIPS 2025\] Vicinity-Guided Discriminative Latent Diffusion for Privacy-Preserving Domain Adaptation](vicinity-guided_discriminative_latent_diffusion_for_privacy-preserving_domain_ad.md)
- [\[NeurIPS 2025\] Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities](progressive_inference-time_annealing_of_diffusion_models_for_sampling_from_boltz.md)
- [\[NeurIPS 2025\] Track, Inpaint, Resplat: Subject-driven 3D and 4D Generation with Progressive Texture Infilling](track_inpaint_resplat_subject-driven_3d_and_4d_generation_with_progressive_textu.md)

</div>

<!-- RELATED:END -->
