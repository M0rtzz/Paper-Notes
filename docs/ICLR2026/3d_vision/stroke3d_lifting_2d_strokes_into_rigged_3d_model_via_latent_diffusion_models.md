---
title: >-
  [论文解读] Stroke3D: Lifting 2D Strokes into Rigged 3D Model via Latent Diffusion Models
description: >-
  [ICLR 2026][3D视觉][3D生成] Stroke3D 首次实现从用户绘制的2D笔画和文本提示直接生成绑骨3D网格模型，采用骨骼优先的两阶段流水线：先用图VAE+图DiT生成可控3D骨骼，再通过TextuRig数据集增强和SKA-DPO优化生成高质量网格。
tags:
  - ICLR 2026
  - 3D视觉
  - 3D生成
  - 骨骼生成
  - 图扩散
  - 绑骨
  - DPO
---

# Stroke3D: Lifting 2D Strokes into Rigged 3D Model via Latent Diffusion Models

**会议**: ICLR 2026  
**arXiv**: [2602.09713](https://arxiv.org/abs/2602.09713)  
**代码**: [https://whalesong-zrs.github.io/Stroke3D_project_page/](https://whalesong-zrs.github.io/Stroke3D_project_page/)  
**领域**: 3D视觉  
**关键词**: 3D生成, 骨骼生成, 图扩散, 绑骨, DPO

## 一句话总结

Stroke3D 首次实现从用户绘制的2D笔画和文本提示直接生成绑骨3D网格模型，采用骨骼优先的两阶段流水线：先用图VAE+图DiT生成可控3D骨骼，再通过TextuRig数据集增强和SKA-DPO优化生成高质量网格。

## 研究背景与动机

绑骨3D资产是3D变形和动画的基础，广泛应用于AR/VR、机器人仿真和影视行业。现有方法面临两大关键限制：

**难以生成可动画几何体**：大量3D生成方法（MVDream、CLAY等）仅产生静态几何，缺少动画所需的骨骼层次结构。SKDream等条件于骨骼的方法受限于高质量配对数据集的稀缺

**骨骼创建缺乏结构控制**：现有骨骼生成方法（MagicArticulate、UniRig）采用端到端的网格到骨骼范式，缺少显式结构约束，导致骨骼在不需要的位置出现而在关键位置缺失

核心创新在于**骨骼驱动工作流**：与先生成网格再绑骨的传统方法不同，Stroke3D 先从2D笔画生成骨骼，再以骨骼为条件生成网格。

## 方法详解

### 整体框架

Stroke3D 分两个主要阶段：(1) 可控骨骼生成——Sk-VAE编码骨骼图结构到潜空间，Sk-DiT在潜空间中生成骨骼嵌入；(2) 增强网格合成——用TextuRig数据集增强训练数据，再用SKA-DPO优化骨骼-网格对齐。

### 关键设计

1. **骨骼图VAE (Sk-VAE)**

   将3D骨骼表示为无向图 $\mathcal{G} = (\mathbf{X}, \mathbf{E})$，$\mathbf{X} \in \mathbb{R}^{N \times 3}$ 为关节坐标，$\mathbf{E}$ 为拓扑边。编码器由GCN和TransformerConv组成，将图结构编码到连续潜空间。训练采用 $L_2$ 重建损失加轻量KL散度正则化（$kl\_\beta = 1 \times 10^{-8}$），确保潜空间平滑。

2. **骨骼图DiT (Sk-DiT)**

   基于DiT架构，用TransformerConv替代标准自注意力以适配图结构数据，并加入跨注意力整合CLIP编码的文本嵌入。2D笔画通过特征映射后与噪声潜表示拼接提供结构引导。训练时通过对3D骨骼的2D投影施加扰动来模拟手绘笔画：

   $$\mathcal{L}_{\text{Sk-DiT}} = \mathbb{E}_{\mathbf{z}_0, t, \epsilon, \mathbf{J}_{xy}, \mathbf{E}, \mathbf{c}_{\text{text}}} \left[\|\epsilon_\phi(\mathbf{z}_t, t, \mathbf{J}_{xy}, \mathbf{E}, \mathbf{c}_{\text{text}}) - \epsilon\|_2^2\right]$$

3. **TextuRig 数据集**

   针对Objaverse-XL中绑骨模型缺少纹理的问题，开发专门的处理流程：筛选含纹理贴图或顶点颜色的模型，用Gemini重新生成描述性标注。最终增加6,800个高质量样本到SKDream的24,000训练数据中。

4. **SKA-DPO (骨骼-网格对齐偏好优化)**

   用参考模型为每个骨骼-文本对生成一对候选多视角图像，通过SKA Score评估骨骼-网格对齐质量，选出优胜/劣势样本构建偏好数据集，再用DiffusionDPO目标微调：

   $$\mathcal{L}(\theta) = -\mathbb{E} \log\sigma\big(-\beta(\|\epsilon^{win} - \epsilon_\theta(x_t^{win}, t)\|_2^2 - \|\epsilon^{win} - \epsilon_{\text{ref}}(x_t^{win}, t)\|_2^2 - (\text{lose项}))\big)$$

### 损失函数 / 训练策略

- Sk-VAE：$L_2$ 重建 + 极轻KL正则（$10^{-8}$）
- Sk-DiT：标准扩散降噪损失 + 分类器无关引导（CFG）
- 网格生成：先SFT增强（TextuRig），再SKA-DPO对齐优化
- Sk-VAE和Sk-DiT各训练500K迭代，SKDream SFT 9K步，DPO 1K步

## 实验关键数据

### 主实验

| 数据集/指标 | 本文 (Stroke3D) | MagicArticulate | UniRig | SKDream |
|--------|------|----------|------|------|
| CD-J2J (All)↓ | **0.048** | 0.052 | 0.063 | 0.111 |
| CD-J2B (All)↓ | **0.039** | 0.041 | 0.051 | 0.092 |
| CD-B2B (All)↓ | **0.034** | 0.034 | 0.041 | 0.083 |
| SKA MeanInst.↑ | **87.83** | - | - | 80.43 |
| SKA MeanClass↑ | **84.36** | - | - | 74.38 |

### 消融实验

| 配置 | MeanInst.↑ | MeanClass↑ | 说明 |
|------|---------|---------|------|
| SKDream baseline | 80.43 | 74.38 | 原始基线 |
| +TextuRig (SFT) | 82.37 | 76.84 | 数据增强+1.9 |
| +SKA-DPO | 85.57 | 81.12 | DPO+5.1 |
| +TextuRig & SKA-DPO | **87.83** | **84.36** | 二者互补+7.4 |

### 关键发现

- 结构条件（2D笔画）对模型收敛速率至关重要，无结构条件训练在大规模数据上难以收敛
- 骨骼生成对输入稀疏性具有鲁棒性，删除少于5个关节时CD评分保持稳定
- SKA-DPO偏好分数的margin为0.1时达到最优平衡
- 生成的骨骼-网格对可直接通过Blender自动蒙皮进行动画，结构完整性良好

## 亮点与洞察

1. **骨骼优先范式**：颠覆了传统先网格后绑骨的工作流，赋予用户直接的结构控制能力
2. **2D到3D的巧妙桥接**：通过画布工具让用户以点击连接的方式创建拓扑同构的2D输入，优雅地解决了2D-3D领域差距
3. **RL引入3D生成**：将DPO从语言/图像模型引入3D网格生成，以骨骼-网格对齐作为奖励信号
4. **模块化设计**：骨骼生成和网格合成解耦，各自可独立改进

## 局限与展望

- 骨骼关节数限制在0-30个，复杂骨骼结构可能受限
- 仅从正交投影的2D视角提供输入，多视角引导可能提升质量
- TextuRig数据集规模（6.8K）仍较小，更大规模数据可能进一步提升
- 自动蒙皮质量依赖Blender工具，端到端蒙皮是未来方向

## 相关工作与启发

- MagicArticulate和UniRig代表自回归骨骼生成趋势，但缺乏显式结构控制
- SKDream的MCF骨骼+条件生成提供了基础，Stroke3D在此上进行了数据和优化层面的显著增强
- DiffusionDPO (Wallace et al., 2024) 的偏好优化思路被巧妙适配到3D领域

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次从2D笔画生成绑骨3D网格，骨骼优先管线具有开创性
- 实验充分度: ⭐⭐⭐⭐ 骨骼和网格分别在标准基准上评估，消融充分，但缺少用户研究
- 写作质量: ⭐⭐⭐⭐ 方法阐述清晰，图表信息量大，但部分章节略冗长
- 价值: ⭐⭐⭐⭐ 降低了3D动画资产创建门槛，但实际艺术家采纳仍需验证
