---
title: >-
  [论文解读] RGB-to-Polarization Estimation: A New Task and Benchmark Study
description: >-
  [NeurIPS 2025][偏振估计] 本文首次定义从标准RGB图像估计偏振分量（S₁/S₂/S₃）的新任务，构建涵盖修复型与生成型方法的首个系统基准，发现预训练MAE在像素精度上综合最优（PSNR 24.74），修复型方法整体显著优于扩散生成型方法，且预训练权重迁移是关键优势。
tags:
  - NeurIPS 2025
  - 偏振估计
  - Stokes参数
  - RGB图像
  - benchmark
  - 扩散模型
---

# RGB-to-Polarization Estimation: A New Task and Benchmark Study

**会议**: NeurIPS 2025  
**arXiv**: [2505.13050](https://arxiv.org/abs/2505.13050)  
**代码**: 待发布  
**领域**: 计算机视觉 / 偏振成像  
**关键词**: 偏振估计, Stokes参数, RGB图像, benchmark, 扩散模型

## 一句话总结

本文首次定义从标准RGB图像估计偏振分量（S₁/S₂/S₃）的新任务，构建涵盖修复型与生成型方法的首个系统基准，发现预训练MAE在像素精度上综合最优（PSNR 24.74），修复型方法整体显著优于扩散生成型方法，且预训练权重迁移是关键优势。

## 研究背景与动机

**领域现状**：偏振图像蕴含标准RGB无法捕捉的丰富物理信息——双折射、表面应力、粗糙度等材料属性——在反射分离、材料分类、阴影去除、三维重建等CV任务中具有重要价值。然而偏振信息的获取依赖专用硬件（偏振相机或旋转偏振片），成本高昂且操作不便。

**现有痛点**：偏振成像的硬件门槛严重限制了偏振信息的普及应用。现有偏振数据集较少，且没有任何工作尝试过从RGB直接估计偏振信息，这一方向完全空白。

**核心矛盾**：RGB图像广泛可得但缺失偏振信息，同一RGB外观可能对应多种偏振状态（RGB只编码强度和颜色，不编码光的矢量性质），这使得估计本质上是一个ill-posed问题。

**本文目标** 能否用神经网络直接从RGB输入推断偏振Stokes分量？哪类深度学习方法更适合这个任务？预训练与从头训练的差异如何？

**切入角度**：RGB图像本身对应Stokes向量的总强度分量S₀，与其余S₁/S₂/S₃存在内在的物理关联（如表面法线、材料属性在RGB中有隐含线索），预训练模型可能已学到与偏振相关的视觉先验。

**核心 idea**：将RGB-to-Polarization作为新任务正式定义，用Stokes参数作为统一表示，构建首个系统基准来评估多种深度学习方法。

## 方法详解

### 整体框架

任务被定义为逐像素预测：输入RGB图像 $\mathbf{I}_{\text{RGB}} \in \mathbb{R}^{H \times W \times 3}$（对应总强度S₀），输出 $\mathbf{S} = [S_1, S_2, S_3] \in \mathbb{R}^{H \times W \times 9}$，其中每个Stokes分量为3通道图像。所有图像缩放至256×256，归一化到[0,1]。评估使用PSNR、SSIM和LPIPS三个互补指标，对每个分量独立计算后取平均。

### 关键设计

1. **Stokes参数表示体系**:

    - 功能：提供完整描述光偏振状态的统一数学框架
    - 核心思路：$S_1 = S_0 \cos(2\psi)\cos(2\chi)$（水平/垂直线偏振），$S_2 = S_0 \sin(2\psi)\cos(2\chi)$（±45°线偏振），$S_3 = S_0 \sin(2\chi)$（圆偏振）。衍生指标包括偏振度DoP、线偏振度DoLP、圆偏振比CoP和线偏振角AoLP，提供更直观的物理可视化
    - 设计动机：各分量携带不同材料敏感信息——高S₁指示光滑介电或金属面，高S₂指示双折射或纤维材料，高S₃指示散射或光学活性介质。这使得分通道评估变得有意义

2. **修复型基线方案**:

    - 功能：将偏振估计建模为图像修复/翻译任务
    - 核心思路：Restormer（4级层次结构，4/6/6/8个Transformer块）和Uformer（4层编码器+4层解码器）修改输出为9通道，L1损失训练。MAE使用预训练权重初始化编码器（24层）和解码器（8层），通道复制扩展投影层到9通道，端到端微调
    - 设计动机：修复型架构擅长像素级精确预测，MAE的自监督预训练提供了强大的视觉先验表示

3. **生成型基线方案**:

    - 功能：将偏振估计建模为条件图像生成任务
    - 核心思路：WDiff（6级U-Net）和DiT（10个Transformer块）作为从头训练的条件扩散模型，RGB为条件输入直接生成9通道输出。RealFill和Img2ImgTurbo基于预训练Stable Diffusion微调，为每个Stokes分量训练独立模型（保持3通道输出结构），使用LoRA进行轻量微调
    - 设计动机：测试扩散模型的生成能力能否捕捉偏振信息的复杂分布，预训练扩散模型是否能迁移视觉先验

### 损失函数 / 训练策略

- 修复模型统一使用L1损失监督预测与真实Stokes分量的差异
- 扩散模型使用标准去噪损失
- 数据基于Jeon等人的大规模RGB-偏振数据集，前1000对训练，后200对测试

## 实验关键数据

### 主实验

| 方法 | 类型 | Avg PSNR↑ | Avg SSIM↑ | Avg LPIPS↓ |
|------|------|-----------|-----------|------------|
| MAE | 修复（预训练） | **24.74** | **0.8876** | 0.2684 |
| Uformer | 修复（从头） | 24.34 | 0.8722 | **0.2267** |
| Restormer | 修复（从头） | 23.98 | 0.8730 | 0.2369 |
| DiT | 生成（从头） | 23.33 | 0.8409 | 0.2447 |
| Img2ImgTurbo | 生成（预训练） | 23.33 | 0.8725 | 0.3542 |
| RealFill | 生成（预训练） | 21.81 | 0.8025 | 0.2654 |
| WDiff | 生成（从头） | 13.11 | 0.6822 | 0.3772 |

### 消融实验

跨数据集泛化测试（模型在Jeon数据集上训练）：

| 方法 | Jeon PSNR | Qiu PSNR | Kurita PSNR |
|------|-----------|----------|-------------|
| MAE | 24.74 | 15.02 | 18.81 |
| Uformer | 24.34 | 14.68 | 18.74 |
| Img2ImgTurbo | 23.33 | 15.65 | 18.78 |
| DiT | 23.33 | 14.74 | 17.86 |

### 关键发现

- **S₁最难估计**：三个分量中S₁的PSNR一致最低（MAE在S₁上仅22.73 vs S₃上25.94），因为S₁对表面朝向和材料属性最敏感，自然场景中信噪比更低
- **修复型 > 生成型**：修复型方法在像素级和结构指标上全面优于生成型。生成型方法在结构一致性上存在困难，虽然Img2ImgTurbo的PSNR较高但视觉保真度不如DiT
- **预训练权重至关重要**：MAE利用大规模RGB预训练权重迁移达到最优PSNR/SSIM，Img2ImgTurbo利用Stable Diffusion先验优于从头训练的WDiff/DiT
- **跨数据集性能大幅下降**：所有方法在域外数据上掉点严重（PSNR降低约6-10），泛化能力有限

## 亮点与洞察

- **首次定义RGB-to-Polarization任务**：将偏振估计从依赖专用硬件转向计算方法，潜在地让任何RGB相机都能获取偏振信息，这是一个有实际应用价值的新方向
- **修复 vs 生成的互补性**：Uformer在DoLP重建上更好，DiT在AoLP上反而更好，暗示两类方法捕捉了不同维度的偏振特征，融合方案可能更优
- **物理先验的缺失**：所有方法都是纯数据驱动的，没有利用Stokes参数间的物理约束（如 $S_1^2 + S_2^2 + S_3^2 \leq S_0^2$），加入这类物理约束可能显著提升性能

## 局限与展望

- **ill-posed本质未充分讨论**：同一RGB可能对应多种偏振状态，当前方法无法给出置信度估计，这在下游应用中是关键缺陷
- **数据规模有限**：仅1000张训练图像，限制了深度模型的学习能力，尤其是扩散模型通常需要更大数据
- **评估指标单一**：PSNR/SSIM/LPIPS是通用图像质量指标，缺少针对偏振物理属性的评估（如DoLP/AoLP精度、材料分类精度等下游任务评估）
- **跨域泛化差**：所有模型在域外数据上掉点严重，实用价值受限
- 改进方向：加入物理约束损失、使用不确定性估计、扩大数据规模、设计偏振特异性架构

## 相关工作与启发

- **vs 传统偏振成像**：传统方法依赖硬件（偏振相机/旋转偏振片），本文提出计算替代方案，大幅降低获取门槛
- **vs 图像翻译任务**（如深度估计、法线估计）：偏振估计的独特挑战在于一对多映射的歧义性和物理约束的复杂性
- **vs MAE在下游任务的应用**：进一步验证了自监督预训练ViT在低层视觉任务上的强大迁移能力

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次定义新任务+构建基准，虽然方法层面没有新设计
- 实验充分度: ⭐⭐⭐⭐ 覆盖多种方法和多数据集，但缺少下游任务验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，Stokes参数的物理背景介绍详尽
- 价值: ⭐⭐⭐⭐ 开辟新方向并提供基准资源，有实际应用前景

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] HouseLayout3D: A Benchmark and Training-Free Baseline for 3D Layout Estimation in the Wild](houselayout3d_a_benchmark_and_training-free_baseline_for_3d_layout_estimation_in.md)
- [\[CVPR 2025\] PolarFree: Polarization-based Reflection-Free Imaging](../../CVPR2025/llm_evaluation/polarfree_polarization-based_reflection-free_imaging.md)
- [\[ACL 2025\] Navigating Rifts in Human-LLM Grounding: Study and Benchmark](../../ACL2025/llm_evaluation/navigating_rifts_in_human-llm_grounding_study_and_benchmark.md)
- [\[NeurIPS 2025\] Exploiting Task Relationships in Continual Learning via Transferability-Aware Task Embeddings](exploiting_task_relationships_in_continual_learning_via_transferability-aware_ta.md)
- [\[ICML 2025\] LLM-SRBench: A New Benchmark for Scientific Equation Discovery with LLMs](../../ICML2025/llm_evaluation/llm-srbench_a_new_benchmark_for_scientific_equation_discovery_with_large_languag.md)

<!-- RELATED:END -->
