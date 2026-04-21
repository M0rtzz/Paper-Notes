---
title: >-
  [论文解读] Attribution as Retrieval: Model-Agnostic AI-Generated Image Attribution
description: >-
  [CVPR 2026][图像生成][AIGC归因] 将 AI 生成图像归因从分类范式转为实例检索范式，提出 LIDA 框架：利用 RGB 低位平面提取生成器特有指纹作为输入，通过在真实图像上无监督预训练 + 少样本适配实现开放集归因，在 GenImage 和 WildFake 上以 1-shot 设置即取得 40.4%/77.5% 的平均 Rank-1 准确率，大幅超越现有方法。
tags:
  - CVPR 2026
  - 图像生成
  - AIGC归因
  - 实例检索
  - 低位平面指纹
  - 少样本学习
  - Deepfake检测
---

# Attribution as Retrieval: Model-Agnostic AI-Generated Image Attribution

**会议**: CVPR 2026  
**arXiv**: [2603.10583](https://arxiv.org/abs/2603.10583)  
**代码**: [有](https://github.com/hongsong-wang/LIDA)  
**领域**: 扩散模型/图像生成  
**关键词**: AIGC归因, 实例检索, 低位平面指纹, 少样本学习, Deepfake检测

## 一句话总结
将 AI 生成图像归因从分类范式转为实例检索范式，提出 LIDA 框架：利用 RGB 低位平面提取生成器特有指纹作为输入，通过在真实图像上无监督预训练 + 少样本适配实现开放集归因，在 GenImage 和 WildFake 上以 1-shot 设置即取得 40.4%/77.5% 的平均 Rank-1 准确率，大幅超越现有方法。

## 研究背景与动机
AIGC 技术快速发展，图像生成器不断涌现（SD、FLUX、Midjourney 等），对数字媒体的真实性和溯源提出了严峻挑战。

现有归因方法分两大类：(1) **生成式水印**——在图像生成过程中嵌入隐形水印（如 Tree-Ring、Gaussian Shading），准确率高但需要完全访问并修改生成模型，不能泛化到不同生成器；(2) **AI 生成图像归因**——独立于生成步骤，但现有方法几乎都将其视为分类问题。闭集方法假设所有生成器在训练时已知，无法适应新出现的模型；开集方法虽然尝试处理未知生成器，但仍依赖大量标注或未标注的 AI 生成图像进行训练，灵活性不足。

核心矛盾：生成模型快速迭代 vs 归因方法需要大量再训练。需要一个**不依赖生成模型、仅需极少注册样本就能适配新生成器**的通用框架。本文的关键 insight：将归因重定义为检索问题，只需训练一个好的特征编码器，新生成器只需往注册数据库中添加几张样例即可。

## 方法详解

### 整体框架
LIDA 包含三个模块：(1) 低位指纹生成 → (2) 无监督预训练 → (3) 少样本归因适配。维护一个注册的 AI 生成图像数据库 $\mathcal{D}$，归因时通过特征相似性检索最近邻来确定生成器来源。

### 关键设计
1. **Low-Bit Fingerprint Generation（低位指纹生成）**:

    - 功能：从 RGB 图像提取生成器特有的结构化噪声指纹
    - 核心思路：对每个通道进行位平面分解 $\mathbf{x}_c = \sum_{k=0}^{7} 2^k \cdot \mathbf{b}_c^k$，取最低 3 个位平面组合并二值化：
    $\tilde{\mathbf{x}}_c = 255 \cdot \text{sgn}(\sum_{k=0}^{2} 2^k \cdot \mathbf{b}_c^k)$
      得到一个丢弃了大部分图像内容、仅保留生成器特有结构噪声的指纹图像
    - 设计动机：PCA 可视化显示，低位指纹让不同生成器的特征明显聚类分离，而 RGB 图像中不同生成器的特征几乎不可区分。低位平面包含生成过程中无意嵌入的模型特有 artifact

2. **Unsupervised Pre-Training（无监督预训练）**:

    - 功能：在大规模真实图像（ImageNet）的低位指纹上预训练编码器，学习通用噪声结构表示
    - 核心思路：以 ResNet-50 为骨干（去除低层下采样以保留空间信息），使用 ImageNet 分类作为代理任务：
    $\mathcal{L}_P = -\sum_{b=1}^{B}\sum_{c=1}^{C} s_b^c \log q_b^c$
    - 设计动机：在真实图像指纹上预训练提供鲁棒的权重初始化，学到的内在噪声结构可迁移到 AI 生成图像取证任务

3. **Few-Shot Attribution Adaptation（少样本归因适配）**:

    - 功能：利用注册数据库中少量样本微调编码器，使其能区分不同生成器
    - 核心思路：两个损失函数联合训练
        - **图像归因损失**（center loss，避免破坏预训练特征空间结构）：
       $$\mathcal{L}_A = \sum_{i=1}^{m} \|x_i - c_{y_i}\|_2^2$$
       类中心 $c_j$ 在 mini-batch 中动态更新
        - **Deepfake 检测损失**（基于真实原型的对比损失）：
       $$\mathcal{L}_D = -\frac{1}{N_r}\sum_{i=1}^{N_r}\log\sigma(\frac{\text{sim}(x_i^r, p_r)}{\tau}) - \frac{1}{N_f}\sum_{j=1}^{N_f}\log(1-\sigma(\frac{\text{sim}(x_i^f, p_r)}{\tau}))$$
      总损失：$\mathcal{L} = \mathcal{L}_A + \lambda \mathcal{L}_D$
    - 设计动机：不用交叉熵而用 center loss 是因为 CE 会破坏预训练学到的特征空间结构；两阶段归因策略（先检测是否 AI 生成，再归因到具体生成器）更符合实际工作流

### 损失函数 / 训练策略
- 预训练：ImageNet 低位指纹上的分类任务
- 适配：center loss（归因）+ 原型对比损失（检测），仅需少量（1/5/10-shot）AI 生成样本
- 推理时两阶段：先 Deepfake 检测 → 再检索归因

## 实验关键数据

### 主实验（GenImage 跨生成器归因）

| Shot | 方法 | Avg Rank-1 | Avg mAP |
|------|------|-----------|---------|
| 1-shot | ResNet | 17.4 | 37.5 |
| 1-shot | DIRE | 14.3 | 34.8 |
| 1-shot | ESSP | 17.0 | 36.0 |
| 1-shot | **Ours** | **40.4** | **61.5** |
| 10-shot | ResNet | 21.4 | 22.4 |
| 10-shot | DIRE | 17.2 | 28.8 |
| 10-shot | ESSP | 22.4 | 23.0 |
| 10-shot | **Ours** | **54.0** | **51.6** |

WildFake 跨架构归因（1-shot）：Ours 平均 Rank-1 77.5%，mAP 87.7%，远超次优的 37.4%/60.4%。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| RGB 输入 vs 低位指纹 | PCA 可视化 | RGB 特征中不同生成器几乎不可分；低位指纹自然聚类 |
| 无监督预训练 | 基础 | ImageNet 上的分类代理任务提供鲁棒初始化 |
| Center loss vs CE | 归因质量 | Center loss 保留预训练空间结构，CE 会破坏 |
| 两阶段归因 | 先检测再归因 | 比端到端分类更可靠 |

### 关键发现
- BigGAN 在 1-shot 下即达到 97-100% Rank-1，表明其生成指纹极为独特
- Stable Diffusion v4/v5 版本间指纹相似，互相混淆率较高
- 低位平面指纹的计算代价极低（仅位运算），几乎不增加推理开销
- 1-shot 即可达到可用精度，展现了检索范式在极少样本下的实际价值

## 亮点与洞察
- 从分类到检索的范式转换极其自然且有效——新生成器只需注册几张图像，无需重新训练模型
- 低位平面指纹是一个简洁而强大的先验：几乎零成本提取，却能显著放大不同生成器间的特征差异
- 不使用 CE 而选择 center loss 来保护预训练特征空间的洞察值得借鉴
- 两阶段归因（先检测后归因）符合现实工作流，且各阶段可独立优化

## 局限与展望
- SD v1.4/v1.5 等同族模型之间的区分能力仍然有限，需要更精细的指纹
- 低位平面对图像压缩（JPEG）的鲁棒性未充分讨论，因为压缩会破坏低位信息
- 注册数据库需要人工维护，如何自动发现和注册新生成器是开放问题
- 仅在图像级别评估，对部分区域生成（如 inpainting）的归因未涉及

## 相关工作与启发
- 与 Tree-Ring、Gaussian Shading 等水印方法的本质区别：水印需要修改生成模型，本方法完全后验
- 与相机指纹技术（PRNU）的思路类似：将"相机指纹"概念迁移到"生成器指纹"
- 低位平面分析可与频域方法结合，进一步提升鉴别力
- 检索范式可推广到 AI 生成视频、音频的溯源

## 评分
- 新颖性: ⭐⭐⭐⭐ 从分类到检索的范式转换清晰有力，低位指纹是有效的先验，但单个技术点不算全新
- 实验充分度: ⭐⭐⭐⭐⭐ 两个大规模数据集、跨生成器/跨架构、1/5/10-shot 全面评估
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，pipeline 设计简洁明了
- 价值: ⭐⭐⭐⭐ 检索范式的可扩展性和低样本需求对 AIGC 安全有实际应用价值

<!-- RELATED:START -->

## 相关论文

- [AEDR: Training-Free AI-Generated Image Attribution via Autoencoder Double-Reconstruction](../../AAAI2026/image_generation/aedr_training-free_ai-generated_image_attribution_via_autoen.md)
- [Diversity over Uniformity: Rethinking Representation in Generated Image Detection](diversity_over_uniformity_rethinking_representation_in_generated_image_detection.md)
- [Diffusion Probe: Generated Image Result Prediction Using CNN Probes](diffusion_probe_generated_image_result_prediction_using_cnn_probes.md)
- [CoD: A Diffusion Foundation Model for Image Compression](cod_a_diffusion_foundation_model_for_image_compression.md)
- [Imagine Before Concentration: Diffusion-Guided Registers Enhance Partially Relevant Video Retrieval](imagine_before_concentration_diffusion-guided_registers_enhance_partially_releva.md)

<!-- RELATED:END -->
