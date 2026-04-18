---
title: >-
  [论文解读] ComboVerse: Compositional 3D Assets Creation Using Spatially-Aware Diffusion Guidance
description: >-
  [ECCV 2024][3D视觉][组合式3D生成] 提出ComboVerse，一个组合式3D资产生成框架：先将包含多个物体的输入图像分解并独立重建为单物体3D模型，再通过空间感知的Score Distillation Sampling (SSDS)引导物体的位置、缩放和旋转参数优化，实现高质量多物体组合3D资产创建，在CLIP Score和人类评估中均显著优于现有方法。
tags:
  - ECCV 2024
  - 3D视觉
  - 组合式3D生成
  - 多物体3D重建
  - 扩散模型引导
  - 空间感知SDS
  - 单图3D重建
---

# ComboVerse: Compositional 3D Assets Creation Using Spatially-Aware Diffusion Guidance

**会议**: ECCV 2024  
**arXiv**: [2403.12409](https://arxiv.org/abs/2403.12409)  
**代码**: https://cyw-3d.github.io/ComboVerse/ (项目页面)  
**领域**: 3D视觉  
**关键词**: 组合式3D生成, 多物体3D重建, 扩散模型引导, 空间感知SDS, 单图3D重建

## 一句话总结

提出ComboVerse，一个组合式3D资产生成框架：先将包含多个物体的输入图像分解并独立重建为单物体3D模型，再通过空间感知的Score Distillation Sampling (SSDS)引导物体的位置、缩放和旋转参数优化，实现高质量多物体组合3D资产创建，在CLIP Score和人类评估中均显著优于现有方法。

## 研究背景与动机

从单张图像生成高质量3D资产是AR/VR、游戏、影视等领域的核心需求。近年来基于feed-forward的单图3D生成模型取得了显著进展，但面临一个系统性问题：

**"多物体差距"（Multi-Object Gap）**——当前主流模型在处理单物体时效果很好，但面对包含多个物体的复杂场景时性能急剧下降。通过深入分析，三个根本原因被揭示：

**相机设置偏差（Camera Setting Bias）**：大多数模型假设物体归一化大小并居中，多物体场景中小物体或偏离中心的物体重建质量显著下降

**数据集偏差（Dataset Bias）**：训练集Objaverse以单物体资产为主，几乎不含遮挡情况，导致模型无法泛化到多物体组合和遮挡场景，生成结果出现"融合"现象

**泄漏模式（Leaking Pattern）**：同时生成多物体时，一个物体的几何和外观会"泄漏"到另一个物体（如虎的背面染上猫头鹰的颜色）

核心洞察：既然现有方法在单物体重建上效果好，为何不先独立重建每个物体，再自动组合？这正是专业3D艺术家的工作流程——先建模各个物体，再整合为一个场景。

## 方法详解

### 整体框架

ComboVerse分为两个阶段：
1. **单物体重建阶段**：分解输入图像中的各个物体，去除遮挡，独立进行单图3D重建
2. **多物体组合阶段**：固定各物体的几何和纹理，仅优化它们的缩放 $s_i$、旋转 $r_i$ 和平移 $t_i$ 参数，通过空间感知SDS损失和参考视图损失引导空间布局

### 关键设计

1. **物体分解与修复（Components Decomposition & Object Inpainting）**：

    - 使用SAM根据2D边界框分割每个物体：$O_i, M_i = \text{SAM}(I, b_i)$
    - **遮挡修复策略**：
        - 将物体背景替换为随机噪声（避免修复时产生白/黑边框）：$I_i = O_i + noise \cdot (\sim M_i)$
        - 构建边界框感知mask：$m_i = (\sim M_i) \cap b_i$，标记需要修复的区域
        - 使用Stable Diffusion配合文本提示"a complete 3D model"进行修复
    - **设计动机**：噪声背景+边界框感知mask+文本引导三者缺一不可，消融实验证实每个组件都有必要

2. **空间感知Score Distillation Sampling (SSDS)**：

    - **标准SDS的局限**：当图像内容已匹配文本prompt时，SDS不会推动位置调整——它优先匹配内容而非空间关系
    - **SSDS核心思路**：在UNet的cross-attention中，增强描述空间关系的token（如"sitting on"、"riding"、"front"）的注意力权重
    $M := \begin{cases} c \cdot M_j & \text{if } j = j^\star \\ M_j & \text{otherwise} \end{cases}$
   其中 $c > 1$ 为放大常数（实验中 $c=25$），$j^\star$ 是空间关系token的索引
    - SSDS梯度：
    $\nabla_\theta \mathcal{L}_{\text{SSDS}}(\phi^\star, x) = \mathbb{E}_{t,\epsilon}[w(t)(\hat{\epsilon}_{\phi^\star}(x_t;y,t) - \epsilon)\frac{\partial x}{\partial \theta}]$
    - 时间步采样范围：[800, 900]（高噪声水平），因为这些步骤对空间布局影响最大
    - **空间token提取**：可由LLM自动提取或用户指定

3. **物体组合优化（Combine the Objects）**：

    - **粗初始化**：
        - 缩放：$s_i = \max\{W_{b_i}/W_I, H_{b_i}/H_I\}$，基于边界框与图像尺寸比
        - 平移：x/y由边界框中心坐标决定，z由单目深度估计均值决定
        - 旋转：初始化为(0,0,0)
    - **精细优化**：使用SSDS作为新视角监督 + 参考视图重建损失
    $\mathcal{L}_{\text{Ref}} = \lambda_{\text{RGB}}|\hat{I}_{\text{RGB}} - I_{\text{RGB}}| + \lambda_A|\hat{I}_A - I_A|$
    - 总损失为 $\mathcal{L}_{\text{Ref}} + \mathcal{L}_{\text{SSDS}}$ 的加权和

### 损失函数 / 训练策略

- 渲染引擎：PyTorch3D
- 优化器：Adam
- z方向平移学习率：0.01，其他参数：0.001
- 损失权重：$\lambda_{\text{Ref}} = 1$，$\lambda_{\text{SSDS}} = 1$，$\lambda_{\text{RGB}} = 1000$，$\lambda_A = 1000$
- 注意力放大常数 $c = 25$
- Stable Diffusion修复设置：guidance scale=7.5，inference steps=30
- 每次迭代渲染10个视角
- 每个物体mesh降采样至50000面

## 实验关键数据

### 主实验

**定量对比（100张测试图像）**：

| 方法 | CLIP-Score↑ | GPT-3DScore↑ |
|------|------------|-------------|
| SyncDreamer | 81.47% | 13.54% |
| OpenLRM | 83.65% | 53.12% |
| Wonder3D | 85.57% | 56.25% |
| **ComboVerse** | **86.58%** | **65.63%** |

- 用户研究：990份回答来自22位评估者，ComboVerse在几何和纹理质量上持续优于所有对比方法

### 消融实验

**SSDS消融（注意力放大效果）**：

| 引导方式 | CLIP B/16 Color↑ | CLIP B/16 Geo↑ | ResNet50 Color↑ | ResNet50 Geo↑ |
|---------|-----------------|---------------|----------------|--------------|
| 无引导 (Base) | 86.62% | 75.24% | 80.35% | 74.19% |
| Depth Loss | 84.57% | 78.42% | 81.69% | 75.83% |
| 标准SDS | 84.16% | 78.25% | 84.08% | 74.66% |
| SSDS (均匀噪声) | 85.33% | 78.49% | 85.55% | 75.85% |
| SSDS (低噪声) | 84.86% | 79.03% | 84.42% | 75.44% |
| **SSDS (full)** | **89.01%** | **79.66%** | **86.60%** | **78.10%** |

**物体修复消融**：
- 去除噪声背景 → 修复结果出现黑色边框
- 使用背景mask替换边界框感知mask → 修复结果产生多余部分
- 去除文本引导 → 修复质量下降

### 关键发现

- SSDS在高噪声区间[800,900]效果最佳，符合扩散模型去噪过程中早期步骤决定全局布局的直觉
- 标准SDS甚至在某些指标上不如无引导的Base，说明SDS在位置调整任务上确实存在固有缺陷
- 空间token注意力放大系数 $c=25$ 效果显著，表明扩散模型确实编码了空间关系知识，只是在标准SDS中被内容匹配淹没
- 方法可扩展到>2个物体的场景重建（实验展示了4物体场景）

## 亮点与洞察

1. **"分而治之"的组合式范式**：模仿人类3D艺术家的工作流程，将复杂问题分解为已解决的子问题+自动组合，是务实且可扩展的思路
2. **"Multi-Object Gap"分析深入**：从Camera Setting Bias、Dataset Bias、Leaking Pattern三个维度系统分析了现有方法的失败模式，为后续研究提供了方向
3. **SSDS的简洁有效**：仅通过注意力权重的简单缩放即可显著改善空间布局引导，实现代价极低但效果显著
4. **只优化空间参数不优化几何纹理**：大幅加速了优化过程，同时避免了SDS带来的几何/纹理退化

## 局限与展望

1. 适用于2-5个物体的场景，更复杂的场景仍面临挑战
2. 组合阶段不优化几何和纹理，最终质量受限于backbone单图3D模型的能力
3. 深度-尺寸歧义：单视角图像中的深度估计不准确可能导致不合理的初始化
4. 可探索在组合阶段利用物理约束（如重力、接触）进一步改善空间关系
5. 物体间交互的建模（如变形、接触面融合）尚未考虑
6. 可结合多视角扩散模型（如Zero123++）替代单目深度估计提升组合精度

## 相关工作与启发

- 与Set-the-Scene等基于文本的组合3D生成对比：图像比文本更精确地描述空间关系，因此对组合质量要求更高
- SSDS思路可推广到其他需要空间精确对齐的SDS应用（如文本到3D场景生成）
- 注意力图操作的方法（来源于Attend-and-Excite）在3D生成中有广泛适用性
- 单物体3D生成模型的进步（如更强的backbone）将直接提升ComboVerse的表现

## 评分

- 新颖性: ⭐⭐⭐⭐ (组合式范式和SSDS设计新颖，但各组件复用了现有方法)
- 实验充分度: ⭐⭐⭐⭐ (自建benchmark、多基线对比、用户研究、消融详细，但benchmark规模较小)
- 写作质量: ⭐⭐⭐⭐ (分析清晰，图文并茂，motivation阐述充分)
- 价值: ⭐⭐⭐⭐ (填补了多物体3D生成的空白，组合式思路有实际应用前景)
