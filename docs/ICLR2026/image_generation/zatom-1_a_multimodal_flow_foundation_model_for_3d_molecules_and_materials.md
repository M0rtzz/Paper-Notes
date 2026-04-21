---
title: >-
  [论文解读] Zatom-1: A Multimodal Flow Foundation Model for 3D Molecules and Materials
description: >-
  [ICLR 2026][图像生成][基础模型] Zatom-1是首个端到端全开源的基础模型，通过多模态流匹配(multimodal flow matching)统一了3D分子和材料的生成建模与属性预测，使用标准Transformer架构在欧几里得空间直接建模离散原子类型和连续3D几何，实现了跨化学域的正迁移学习。
tags:
  - ICLR 2026
  - 图像生成
  - 基础模型
  - 流匹配
  - 3D分子
  - 3D材料
  - 多模态生成
  - 属性预测
---

# Zatom-1: A Multimodal Flow Foundation Model for 3D Molecules and Materials

**会议**: ICLR 2026  
**arXiv**: [2602.22251](https://arxiv.org/abs/2602.22251)  
**代码**: 开源（fully open-source）  
**领域**: 图像生成 / 科学机器学习  
**关键词**: 基础模型, 流匹配, 3D分子, 3D材料, 多模态生成, 属性预测

## 一句话总结
Zatom-1是首个端到端全开源的基础模型，通过多模态流匹配(multimodal flow matching)统一了3D分子和材料的生成建模与属性预测，使用标准Transformer架构在欧几里得空间直接建模离散原子类型和连续3D几何，实现了跨化学域的正迁移学习。

## 研究背景与动机

**领域现状**：AI驱动的化学建模已取得重大突破（AlphaFold等），但现有方法通常针对单一领域（分子或材料）和单一任务（生成或预测）进行优化，限制了表示共享和迁移学习。

**现有痛点**：(1) 分子和材料的生成模型分开训练，无法利用跨域数据的互补信息。(2) 稀疏图神经网络架构和手工设计的生成先验限制了扩展性和推理速度。(3) 生成和预测任务使用不同模型，无法共享表示。

**核心矛盾**：3D化学系统包含离散（原子类型）和连续（3D坐标、晶格参数）多种模态，如何在统一框架下同时建模这些模态？如何让生成预训练为下游预测任务提供好的初始化？

**本文目标**：构建一个统一的基础模型，能同时进行3D分子和材料的生成建模和表示学习，实现跨域正迁移。

**切入角度**：将生成建模视为化学表示学习的理想预训练任务，采用标准Transformer架构 + 多模态流匹配，在环境全原子空间 $\mathbb{R}^3$ 中直接建模。

**核心 idea**：用多模态流匹配在统一的Transformer中联合建模离散原子类型和连续3D几何，预训练后可微调用于多任务属性预测。

## 方法详解

### 整体框架
Zatom-1采用两阶段训练：(1) 多模态流预训练用于3D分子和材料生成；(2) 多任务微调用于能量、力和属性预测。核心架构是Trunk-based Flow Transformer (TFT)，由标准Transformer编码器 + 交叉注意力解码器组成。

### 关键设计

1. **统一的五模态表示**:

    - 功能：将分子和材料统一表示为五种模态
    - 核心思路：原子类型 $\bm{A} \in \mathbb{Z}^{1 \times N}$（离散）、3D坐标 $\bm{X} \in \mathbb{R}^{3 \times N}$（连续）、分数坐标 $\bm{F} \in [0,1)^{3 \times N}$、晶格长度 $\bm{L}_{\text{len}} \in \mathbb{R}^{3 \times 1}$、晶格角度 $\bm{L}_{\text{ang}} \in \mathbb{R}^{3 \times 1}$。对分子掩蔽晶格相关输入，对材料掩蔽3D坐标
    - 设计动机：统一的输入格式使单个模型可以同时处理周期性材料和非周期性分子

2. **多模态流匹配训练**:

    - 功能：同时训练离散和连续模态的生成
    - 核心思路：连续模态使用欧几里得CFM：$\bm{X}_t = t \cdot \bm{X} + (1-t) \cdot \epsilon$，$L_{\text{metric}}(\bm{X}) = \mathbb{E}_{\epsilon,t}\left[\frac{1}{N}\|\bm{X}' - \bm{X}\|_2^2\right]$；离散模态使用Discrete CFM：$\bm{A}_t \sim \text{Cat}(t \cdot \delta(\bm{A}) + (1-t) \cdot \delta(\frac{1}{\#\text{atom types}}))$，$L_{\text{discrete}}(\bm{A}) = \mathbb{E}_t\left[-\sum_i a_i \log(a_i')\right]$
    - 设计动机：端点formulation在科学应用中表现更好，不需要预训练的自编码器（如latent diffusion），大幅提升训练和推理速度

3. **Trunk-based Flow Transformer (TFT)**:

    - 功能：统一的编码器-解码器架构
    - 核心思路：$L$层Transformer编码器提取共享表示 $\bm{Z}$，然后通过残差交叉注意力解码器分别预测各模态的去噪输出。编码器第$K$层的表示用于下游预测任务（属性、能量、力），第$L$层用于生成
    - 设计动机：标准Transformer架构（QK Normalization, Flash Attention, SwiGLU FFN）实现可预测的参数缩放性能

4. **采样策略（带随机性的SDE采样）**:

    - 功能：在ODE采样基础上加入可控随机性
    - 核心思路：对连续模态使用SDE采样：$\mathbf{z}_t \leftarrow (\mathbf{v}_t + \mathbf{s}_t + d\mathbf{W}_t)\Delta t$，其中 $d\mathbf{W}_t \leftarrow \sqrt{2\gamma_g g(t)}\mathcal{N}(0,I)$；对离散模态使用categorical采样
    - 设计动机：随机性提升生成多样性和采样质量

### 损失函数 / 训练策略
总损失 $L_{\text{total}} = L_{\text{metric}}(\bm{X}) + L_{\text{metric}}(\bm{F}) + L_{\text{metric}}(\bm{L}_{\text{len}}) + L_{\text{metric}}(\bm{L}_{\text{ang}}) + \lambda_{\text{discrete}} \cdot L_{\text{discrete}}(\bm{A})$，其中 $\lambda_{\text{discrete}} = 0.1$。时间采样 $t \sim \text{Beta}(1.8, 1)$，损失缩放 $\beta(t) = \min\{100, \frac{1}{(1-t)^2}\}$。训练时随机旋转和平移输入数据用于数据增强。模型规模300M参数，单A100生成10,000样本不到4分钟。

## 实验关键数据

### 主实验（材料生成 - MP20数据集）

| 方法 | Match Rate ↑ | RMSD ↓ | 推理时间 |
|------|-------------|--------|---------|
| DiffCSP | - | - | 慢 |
| FlowMM (500M) | 基准 | 基准 | ~50分钟/10K |
| **Zatom-1 (300M)** | **竞争力** | **竞争力** | **<4分钟/10K** |

### 分子生成（QM9 + GEOM-Drugs）

| 方法 | QM9 Stability ↑ | GEOM-Drugs Validity ↑ |
|------|-----------------|----------------------|
| EDM | 基准 | 基准 |
| **Zatom-1** | **SOTA** | **SOTA** |

### 属性预测（QM9 Multi-task）

Zatom-1在QM9多任务属性预测上达到SOTA，证明生成预训练可以为预测任务提供有效的表示初始化。

### 消融实验

| 配置 | 说明 |
|------|------|
| 仅分子预训练 → 分子属性预测 | 基准 |
| 分子+材料联合预训练 → 分子属性预测 | **更好**（跨域正迁移） |
| 无预训练 → 属性预测 | 明显更差 |

### 关键发现
- **跨域正迁移**：在预训练中加入材料数据可以提升分子属性预测精度，这是首次在化学基础模型中观察到的正迁移
- **12.5x推理速度**：相比500M参数的latent diffusion基线（FlowMM），300M的Zatom-1实现12.5倍加速
- **3x训练节省**：相比latent diffusion方法，因无需预训练自编码器，GPU训练小时减少3倍
- **可预测的缩放**：随着模型参数从50M增至300M，生成和预测性能均可预测地提升

## 亮点与洞察
- **范式统一**：首次在单一模型中统一了分子/材料的生成和预测，证明了"生成即预训练"在化学领域的可行性。这一思路可推广到蛋白质、晶体等其他科学领域
- **去latent化设计**：直接在环境空间建模，避免了latent diffusion需要的自编码器，大幅简化pipeline且提速。这与视觉领域"latent优于pixel"的趋势形成有趣对比
- **标准化架构**：使用标准Transformer而非领域特定的等变GNN，说明足够大的数据增强+标准架构可以替代手工等变性设计
- **O(3)等变变体Platom-1**：还实验了等变版本的Platonic Transformer变体，展示了灵活性

## 局限与展望
- 当前只支持最多约200原子的系统（受token长度限制），无法处理蛋白质等大分子
- 材料生成仍主要在MP20数据集上评估，高温高压等极端条件的泛化未知
- 预训练数据的组合（分子vs材料的比例）对下游性能的影响需进一步研究
- 可以探索与大语言模型结合，实现文本条件的分子设计

## 相关工作与启发
- **vs FlowMM**: FlowMM也用flow matching做材料生成，但使用稀疏GNN且不支持属性预测。Zatom-1用标准Transformer + 统一框架实现13x推理加速
- **vs EDM/GeoLDM**: 这些分子生成方法不支持材料，也缺乏预测能力。Zatom-1在QM9上超越它们
- **vs AlphaFold3**: AlphaFold3启发了"用标准Transformer学习科学数据"的思路，Zatom-1将其推广到化学生成

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个统一分子/材料生成+预测的基础模型，跨域正迁移是重要发现
- 实验充分度: ⭐⭐⭐⭐ 多数据集多任务评估，有缩放实验和消融分析
- 写作质量: ⭐⭐⭐⭐ 思路清晰，动机充分，但部分细节在附录中
- 价值: ⭐⭐⭐⭐⭐ 对科学AI的基础模型研究有重要推动，开源贡献大

<!-- RELATED:START -->

## 相关论文

- [All-atom Diffusion Transformers: Unified Generative Modelling of Molecules and Materials](../../ICML2025/image_generation/all-atom_diffusion_transformers_unified_generative_modelling_of_molecules_and_ma.md)
- [CoD: A Diffusion Foundation Model for Image Compression](../../CVPR2026/image_generation/cod_a_diffusion_foundation_model_for_image_compression.md)
- [RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation](rmflow_refined_mean_flow_by_a_noise-injection_step_for_multimodal_generation.md)
- [Unified Multi-Modal Interactive & Reactive 3D Motion Generation via Rectified Flow](unified_multi-modal_interactive_reactive_3d_motion_generation_via_rectified_flow.md)
- [AlignTok: Aligning Visual Foundation Encoders to Tokenizers for Diffusion Models](aligntok_aligning_visual_foundation_encoders_to_tokenizers_for_diffusion_models.md)

<!-- RELATED:END -->
