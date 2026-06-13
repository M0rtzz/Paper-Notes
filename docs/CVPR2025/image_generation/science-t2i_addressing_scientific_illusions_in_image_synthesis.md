---
title: >-
  [论文解读] Science-T2I: Addressing Scientific Illusions in Image Synthesis
description: >-
  [CVPR 2025][图像生成][文本生成图像] Science-T2I 构建了涵盖 16 个科学领域的 20k+ 对抗图像对基准，揭示当前图像生成模型在隐式科学推理上的系统性缺陷（所有模型得分低于 50/100），并提出 SciScore 奖励模型和两阶段对齐框架（SFT+OFT），将 FLUX.1[dev] 的科学推理能力提升超过 50%。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "文本生成图像"
  - "科学推理"
  - "奖励模型"
  - "对齐"
  - "基准测试"
---

# Science-T2I: Addressing Scientific Illusions in Image Synthesis

**会议**: CVPR 2025  
**arXiv**: [2504.13129](https://arxiv.org/abs/2504.13129)  
**代码**: [https://github.com/Jialuo-Li/Science-T2I](https://github.com/Jialuo-Li/Science-T2I)  
**领域**: 图像生成  
**关键词**: 文本生成图像, 科学推理, 奖励模型, 对齐, 基准测试

## 一句话总结

Science-T2I 构建了涵盖 16 个科学领域的 20k+ 对抗图像对基准，揭示当前图像生成模型在隐式科学推理上的系统性缺陷（所有模型得分低于 50/100），并提出 SciScore 奖励模型和两阶段对齐框架（SFT+OFT），将 FLUX.1[dev] 的科学推理能力提升超过 50%。

## 研究背景与动机

**领域现状**：当前文本到图像（T2I）生成模型（FLUX、SDXL 等）在视觉保真度上取得了显著进步，能生成高分辨率、高美感的图像。评估指标（FID 等）也持续改善。

**现有痛点**：这些模型生成的图像虽然视觉上逼真，但在科学层面常常是不合理的。例如，给定"一个未成熟的苹果"，模型往往生成红色苹果（基于视觉原型记忆）而非绿色苹果（基于科学知识）。这暴露了"视觉逼真"和"物理/科学正确"之间的根本性鸿沟。

**核心矛盾**：模型的训练数据很少将科学概念与其正确的视觉表现配对，标准评估协议也不检测模型是否理解 prompt 背后的科学原理。问题不在于模型无法渲染正确的场景（显式 prompt 得分高出约 35 分），而在于无法从隐式科学线索推理出正确的视觉结果。

**本文目标**：（1）构建系统的科学图像合成基准；（2）开发能捕捉细粒度科学现象的奖励模型；（3）提出有效的对齐框架将科学知识注入生成模型。

**切入角度**：通过"隐式-显式-浅层"三层 prompt 结构解耦模型的组合渲染能力和科学推理能力——显式 prompt 衡量渲染上限，隐式 prompt 衡量推理能力，浅层 prompt 提供硬负例。

**核心 idea**：用专家标注的对抗图像对微调 CLIP-H 得到 SciScore 奖励模型，再用 SFT+基于 SciScore 的在线微调（OFT with 主体遮罩）将科学推理能力注入 FLUX。

## 方法详解

### 整体框架

整个工作分为三部分：（1）Science-T2I 数据集构建（20k+ 训练对 + 454 测试 prompt）；（2）SciScore 奖励模型训练；（3）两阶段对齐框架（SFT → OFT）。输入为隐式科学 prompt，输出为科学正确的生成图像。

### 关键设计

1. **三层 Prompt 结构（IP/EP/SP）**:

    - 功能：系统地解耦模型的科学推理能力与组合渲染能力
    - 核心思路：对每个科学任务构建三元组 prompt。Implicit Prompt (IP) 包含需要科学推理的术语（如"未成熟的苹果"）；Explicit Prompt (EP) 直接描述正确视觉结果（如"绿色苹果"）；Superficial Prompt (SP) 给出表层关联的错误结果（如"红色苹果"）。IP 测试推理能力，EP 建立渲染上限，SP 提供偏好训练的困难负例
    - 设计动机：之前的工作无法区分模型是"不会画"还是"不知道该画什么"。三层结构明确回答了这个问题——实验显示显式 prompt 比隐式高约 35 分，证明瓶颈在推理而非渲染

2. **SciScore 奖励模型**:

    - 功能：评估生成图像是否正确反映 prompt 隐含的科学原理，超越 GPT-4o 和人类专家
    - 核心思路：基于 CLIP-H 微调，训练目标包含两个互补损失。IPA（隐式 prompt 对齐）通过 KL 散度最小化，让隐式 prompt 的嵌入靠近显式图像而非浅层图像：$\mathcal{L}_{IPA} = KL(p_{txt} || \hat{p}_{txt})$。IEE（图像编码器增强）在图像侧加入偏好损失，增强对细粒度科学细节（如微妙的颜色和分层模式）的敏感度。总损失 $\mathcal{L} = \mathcal{L}_{IPA} + \lambda \mathcal{L}_{IEE}$，$\lambda=0.25$ 取得最佳平衡
    - 设计动机：原始 CLIP 倾向于将隐式 prompt 嵌入到浅层对应物附近而非显式对应物附近，因为表面级共现模式主导了科学语义。需要专门的微调来纠正这种偏差

3. **两阶段对齐框架（SFT + Masked OFT）**:

    - 功能：将科学知识注入生成模型，提升隐式推理能力
    - 核心思路：第一阶段在 Science-T2I 训练集上对 FLUX.1[dev] 做监督微调（SFT with LoRA，2000步），教会模型"科学正确的图像长什么样"。第二阶段用 SciScore 作为奖励信号做在线微调（OFT），采用 DPO 目标函数。关键创新在于主体遮罩策略：用 GroundingDINO 定位科学主体区域，只在该区域内反传梯度，避免无关背景引入噪声
    - 设计动机：标准后训练（PPO/DPO）在预训练分布内优化，但模型从未接触过科学现象的图像，纯偏好优化无法教会它不知道的东西。SFT 先提供知识基础，OFT 再优化隐式推理能力。不加遮罩时训练不稳定，因为首选和拒绝图像通常只在科学相关区域不同

### 损失函数 / 训练策略

SFT 阶段使用 Flow Matching 目标函数 $L_{SFT} = \mathbb{E}\|v_\theta(z,t) - u_t(z|\epsilon)\|_2^2$。OFT 阶段将 Flow Matching 的确定性 ODE 解释为 SDE，得到高斯策略 $\pi_\theta(a_t|s_t) = \mathcal{N}(a_t; \mu_\theta(s_t), \sigma_t^2 I)$，然后用 DPO 对轨迹做偏好优化，并融入主体遮罩。SFT 用 LoRA 微调 2000 步；OFT 每轮采样 32 个 prompt，每个生成两张图，约训练 100 步。

## 实验关键数据

### 主实验

| 模型 | 物理 | 化学 | 生物 | 总分 |
|------|------|------|------|------|
| FLUX.2[dev] (最佳) | **53.19** | **53.55** | **32.50** | **47.80** |
| Z-Image | 26.53 | 32.98 | 22.22 | 26.73 |
| SDXL | 16.11 | 20.92 | 25.56 | 19.60 |
| 显式 vs 隐式 prompt 平均差距 | - | - | - | ~35分 |

SciScore 分类准确率（Science-T2I-S / Science-T2I-C）：

| 评估器 | S-Simple | S-Complex |
|--------|----------|-----------|
| SciScore | **93.14** | **91.19** |
| 人类专家 | 87.01 | 86.02 |
| GPT-4o mini + CoT | 74.97 | 77.16 |
| CLIP-H | 54.69 | 59.47 |

### 消融实验

| 方法 | Science-T2I-S | RI | Science-T2I-C | RI |
|------|--------------|-----|--------------|-----|
| FLUX.1[dev] 基线 | 23.56 | - | 27.26 | - |
| + SFT | ~27 | ~37% | ~29 | ~23% |
| + SFT + OFT (Full) | **28.52** | **53.39%** | **30.11** | **38.31%** |

### 关键发现

- 所有 18 个 T2I 模型在隐式科学 prompt 下得分均低于 50/100，生物领域最难（无模型超过 33%）
- 显式 prompt 比隐式高约 35 分，直接证明瓶颈在科学推理而非视觉渲染
- Z-Image 视觉质量顶级但科学得分仅 26.73，说明视觉保真度和科学推理能力不相关
- SciScore 的失败案例几乎全部集中在主体导向任务（ST），因为需要特定主体的知识（如哪种金属产生什么颜色的火焰）
- SFT 是必要的前置步骤——不做 SFT 直接 OFT 无法提升 SciScore
- 主体遮罩对 OFT 训练稳定性至关重要，不加遮罩时性能不稳定甚至停滞

## 亮点与洞察

- **三层 prompt 结构的诊断能力**：IP/EP/SP 的设计非常巧妙，将一个模糊的"模型不够好"的问题精确定位为"推理能力缺失"。这个方法论可以迁移到任何需要区分"知道 vs 能做到"的评估场景
- **SciScore 超越人类专家**：一个微调的 CLIP 模型在科学判别上超越了有科学学位的人类评估者，说明对抗训练数据的质量可以弥补模型先天知识的不足
- **主体遮罩的 OFT 策略**：用 GroundingDINO 定位科学主体区域做局部梯度更新，避免了图像全局优化带来的噪声——这个策略可推广到任何需要细粒度控制的 RLHF/DPO 微调

## 局限与展望

- 训练集规模有限（20k 对），可能无法覆盖所有科学领域的长尾知识
- SciScore 在主体导向任务上仍有明显不足，对未见过的主体缺乏先验
- 当前框架以 FLUX 为基础，对其他架构的迁移性有待验证
- 科学正确性的评估本身依赖 LMM（Qwen3.5-27B），引入了评估器偏差
- 更深层的物理推理（如流体动力学、复杂光学现象）可能需要更强的训练信号

## 相关工作与启发

- **vs PhyBench**: PhyBench 也评估物理推理但仅关注物理领域；本文扩展到物理+化学+生物 16 个子领域，且提出了完整的对齐解决方案
- **vs Commonsense-T2I**: 关注常识推理，但常识文化依赖且缺乏明确标准；本文的科学知识提供了无歧义的 ground truth
- **vs ImageReward/HPSv2**: 这些奖励模型优化美感偏好；SciScore 优化科学正确性，问题设定完全不同

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统性地定义和解决 T2I 的科学推理问题，三层 prompt 设计和两阶段对齐框架都很新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 评估了 18 个模型，对比 VLM/LMM/人类，消融全面，定性定量兼备
- 写作质量: ⭐⭐⭐⭐⭐ 论述逻辑严密，问题-诊断-解决的叙事链非常清晰
- 价值: ⭐⭐⭐⭐⭐ 揭示了当前 T2I 模型的根本性缺陷，数据集+奖励模型+对齐框架对社区贡献很大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] The Art of Deception: Color Visual Illusions and Diffusion Models](the_art_of_deception_color_visual_illusions_and_diffusion_models.md)
- [\[CVPR 2025\] Exploring Sparse MoE in GANs for Text-conditioned Image Synthesis](exploring_sparse_moe_in_gans_for_text-conditioned_image_synthesis.md)
- [\[ICCV 2025\] Addressing Text Embedding Leakage in Diffusion-Based Image Editing](../../ICCV2025/image_generation/addressing_text_embedding_leakage_in_diffusion-based_image_editing.md)
- [\[CVPR 2025\] Multi-focal Conditioned Latent Diffusion for Person Image Synthesis](multi-focal_conditioned_latent_diffusion_for_person_image_synthesis.md)
- [\[CVPR 2025\] Noise Diffusion for Enhancing Semantic Faithfulness in Text-to-Image Synthesis](noise_diffusion_for_enhancing_semantic_faithfulness_in_text-to-image_synthesis.md)

</div>

<!-- RELATED:END -->
