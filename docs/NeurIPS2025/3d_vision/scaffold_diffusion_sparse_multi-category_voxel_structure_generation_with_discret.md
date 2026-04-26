---
title: >-
  [论文解读] Scaffold Diffusion: Sparse Multi-Category Voxel Structure Generation with Discrete Diffusion
description: >-
  [NeurIPS 2025][3D视觉][离散扩散模型] 提出Scaffold Diffusion，将稀疏多类别3D体素视为token序列，使用Masked Diffusion Language Model（MDLM）配合3D正弦位置编码，在条件占用图上生成空间连贯的多类别体素结构，在极端稀疏（>98%背景）的Minecraft房屋数据集上显著优于自回归和传统离散扩散baseline。
tags:
  - NeurIPS 2025
  - 3D视觉
  - 离散扩散模型
  - 体素生成
  - 稀疏结构
  - 3D生成
  - Minecraft
---

# Scaffold Diffusion: Sparse Multi-Category Voxel Structure Generation with Discrete Diffusion

**会议**: NeurIPS 2025  
**arXiv**: [2509.00062](https://arxiv.org/abs/2509.00062)  
**代码**: https://scaffold.deepexploration.org/ (在线Demo)  
**领域**: 3D视觉  
**关键词**: 离散扩散模型, 体素生成, 稀疏结构, 3D生成, Minecraft

## 一句话总结

提出Scaffold Diffusion，将稀疏多类别3D体素视为token序列，使用Masked Diffusion Language Model（MDLM）配合3D正弦位置编码，在条件占用图上生成空间连贯的多类别体素结构，在极端稀疏（>98%背景）的Minecraft房屋数据集上显著优于自回归和传统离散扩散baseline。

## 研究背景与动机

稀疏多类别3D体素结构在计算机视觉、机器人、游戏娱乐、环境模拟等领域有广泛应用。然而，生成这类结构面临两个独特挑战：

**内存问题**：体素结构的内存消耗随分辨率呈立方增长（O(n³)），当分辨率增大时迅速触及内存瓶颈

**类别严重不平衡**：由于稀疏性，超过98%的体素是空背景，导致生成模型容易被背景类主导，无法准确生成前景结构

现有工作中，针对二值体素和一般3D结构的生成模型已有较多研究（如PointVoxelDiffusion、XCube等），但多类别稀疏体素生成的研究非常有限。Lee et al.尝试用多项式扩散直接在完整体素图上操作，但受到稀疏性导致的类别不平衡困扰。自回归模型在这种极端稀疏数据上也表现不佳。

本文的核心idea是：**只对占用体素建模**，将其提取为token序列，用离散扩散语言模型（MDLM）生成类别标签，同时引入3D位置编码保持空间连贯性。这样天然地避免了背景类的主导问题和立方级内存问题。

## 方法详解

### 整体框架

Scaffold Diffusion的pipeline分为两步：给定一个布尔占用图O ∈ Z^{D×D×D}（标识哪些位置有体素），提取所有k个占用体素的位置{(x_i, y_i, z_i)}，将对应的类别标签编码为长度L≥k的token序列（可能附加padding token），送入MDLM生成各位置的体素类别。生成完成后，根据位置信息重建完整体素图。

### 关键设计

1. **仅建模占用体素（Sparse Formulation）**: 核心设计决策——不在完整D³体素图上操作，而是只提取占用体素的token序列。这同时解决了内存问题（序列长度L=1024远小于D³=32768或262144）和类别不平衡问题（不再有大量空背景token）。与Lee et al.在全体素图上操作的方法形成鲜明对比。

2. **MDLM作为生成模型**: 采用Masked Diffusion Language Model（吸收态离散扩散），这是D3PM框架的一个特例。前向过程将每个token以概率β_t转变为[MASK]，反向过程学习从mask中恢复原始类别。使用简化的连续时间ELBO目标函数训练。相比多项式扩散（Lee et al.），MDLM提供了更简洁的训练目标和更好的生成质量。

3. **3D正弦位置编码**: 将每个体素的(x,y,z)位置编码为3D正弦嵌入（而非可学习位置编码），注入到DiT backbone中。这使模型能感知token的空间位置，生成空间连贯的结构。消融实验证明这比学习式位置编码效果显著更好（NLL: 0.58 vs 3.369）。

4. **DiT Backbone**: 使用Diffusion Transformer架构，12个block，12个head，序列长度1024。使用log-linear噪声调度和cached updates加速推理。EMA β=0.9999。

### 损失函数 / 训练策略

使用MDLM的连续时间ELBO目标函数：L = E_q ∫ (α'_t / (1-α_t)) · log⟨x_θ(z_t,t), x⟩ dt。使用AdamW优化器（lr=3e-4, β1=0.9, β2=0.999, ε=1e-8），constant warm-up 2500步，最大训练100万步。单卡RTX 5090，每个实验不到12小时。

## 实验关键数据

### 主实验

本文主要以定性评估为主（因为3D生成质量评判本质上是定性任务），但也提供了定量指标：

| 方法 | NLL ↓ | Perplexity ↓ | 定性表现 |
|------|-------|-------------|---------|
| Scaffold Diffusion | **0.58** | **1.787** | 真实、连贯、多样的房屋结构 |
| Lee et al. (VQ-VAE + 多项式扩散) | - | - | 背景token过度表示，结构不完整 |
| 自回归Baseline | - | - | 仅含少数block类型，结构坍塌 |
| DiT-3D (全体素图) | - | - | 即使加反频率加权也无法生成合理占用结构 |

### 消融实验

| 配置 | NLL ↓ | Perplexity ↓ | 说明 |
|------|-------|-------------|------|
| 学习式位置编码 | 3.369 | 29.05 | 无法学习正确的空间关系 |
| 3D正弦位置编码 | **0.58** | **1.787** | 性能提升近6倍，位置感知至关重要 |

### 关键发现

- **稀疏token化是关键**：在全体素图上操作的方法（Lee et al.、DiT-3D）无论是否使用反频率加权损失，都受制于>98%背景token的支配，生成质量差。仅对占用体素建模从根本上解决了这个问题
- **离散扩散显著优于自回归**：自回归baseline使用相同backbone但采用next-token prediction目标，生成的结构被少数block类型主导或出现不合理放置。这表明扩散模型在需要全局空间连贯性的任务上优势明显
- **3D正弦位置编码 vs 学习式编码**：差异巨大（NLL从3.369降到0.58），说明在数据量有限（1432个样本）时，先验式的位置编码远优于需要学习的编码
- **多样性生成**：同一个占用图可以生成多种不同但合理的block材质配置

## 亮点与洞察

- 将体素生成问题优雅地转化为离散序列生成，利用语言模型领域的成熟工具
- "仅建模占用体素"的思路简单但极其有效，一举解决内存和类别不平衡两大难题
- 展示了离散扩散模型可以超越其天然的序列领域，扩展到3D空间结构生成
- 在极端稀疏数据（98.3%背景）上仍能生成高质量结构
- 提供交互式在线demo供读者自行评估，这一做法值得推广

## 局限与展望

- 需要预先给定布尔占用图作为条件，完全端到端生成（先生成占用图再填充类别）留作未来工作
- 受限于离散扩散模型的序列长度限制(L=1024)，无法生成非常大的结构
- 仅在Minecraft房屋数据集上验证，未在更通用的3D场景生成任务上测试
- 数据集较小（1432个样本），泛化性有待进一步验证
- 缺乏标准的定量评估指标，主要依赖定性评估

## 相关工作与启发

- Lee et al.首先将离散扩散用于3D分类数据，但在全体素图上操作受限于稀疏性
- XCube用层次化latent扩散处理稀疏二值体素图，但不处理多类别
- VoxelCNN提供了3D-Craft数据集但仅做自回归的局部扩展，未做完整结构生成
- MDLM（Sahoo et al.）为文本生成提供了简化的离散扩散训练目标，本文将其成功推广到3D

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] ShapeShifter: 3D Variations Using Multiscale and Sparse Point-Voxel Diffusion](../../CVPR2025/3d_vision/shapeshifter_3d_variations_using_multiscale_and_sparse_point-voxel_diffusion.md)
- [\[NeurIPS 2025\] WildCAT3D: Appearance-Aware Multi-View Diffusion in the Wild](wildcat3d_appearance-aware_multi-view_diffusion_in_the_wild.md)
- [\[NeurIPS 2025\] More Than Generation: Unifying Generation and Depth Estimation via Text-to-Image Diffusion Models](more_than_generation_unifying_generation_and_depth_estimation_via_text-to-image_.md)
- [\[NeurIPS 2025\] TRIM: Scalable 3D Gaussian Diffusion Inference with Temporal and Spatial Trimming](trim_scalable_3d_gaussian_diffusion_inference_with_temporal_and_spatial_trimming.md)
- [\[NeurIPS 2025\] GauDP: Reinventing Multi-Agent Collaboration through Gaussian-Image Synergy in Diffusion Policies](gaudp_reinventing_multi-agent_collaboration_through_gaussian-image_synergy_in_di.md)

<!-- RELATED:END -->
