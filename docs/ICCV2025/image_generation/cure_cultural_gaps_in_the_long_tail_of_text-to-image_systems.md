---
title: >-
  [论文解读] CURE: Cultural Gaps in the Long Tail of Text-to-Image Systems
description: >-
  [ICCV 2025][图像生成][文本到图像生成] 提出 CURE 基准与评分套件，利用**属性规范的边际效用**（Marginal Information Attribution）作为人类判断的代理指标，系统评估 T2I 系统在全球文化长尾分布上的代表性能力。
tags:
  - ICCV 2025
  - 图像生成
  - 文本到图像生成
  - 文化代表性
  - 基准评测
  - 边际信息归因
  - 长尾偏差
---

# CURE: Cultural Gaps in the Long Tail of Text-to-Image Systems

**会议**: ICCV 2025  
**arXiv**: [2506.08071](https://arxiv.org/abs/2506.08071)  
**代码**: [https://aniketrege.github.io/cure/](https://aniketrege.github.io/cure/)  
**领域**: 图像生成 / T2I 公平性  
**关键词**: 文本到图像生成, 文化代表性, 基准评测, 边际信息归因, 长尾偏差

## 一句话总结

提出 CURE 基准与评分套件，利用**属性规范的边际效用**（Marginal Information Attribution）作为人类判断的代理指标，系统评估 T2I 系统在全球文化长尾分布上的代表性能力。

## 研究背景与动机

当前主流 T2I 系统（如 Stable Diffusion、FLUX、DALL-E 3）基于网络爬取数据训练，数据分布严重偏向美欧文化，导致全球南方文化的文物生成质量低下（幻觉、细节错误）。现有文化偏差评估方法面临三大问题：

**人工评估不可扩展**：大规模用户研究成本高昂且难以复现

**代理评分器与人类判断相关性弱**：基于 CLIP 相似度或真实图像对比的传统指标无法准确反映文化代表性

**生成纠缠（Generative Entanglement）**：评分器与 T2I 系统共享预训练数据（如 LAION-2B），导致评估结果偏高且具有误导性

作者以陶器生成为例：生成"ceramic diyas"（印度）效果好，但"jebena"（埃塞俄比亚）和"amphora of Hermonax"（希腊）则严重失真。这种现象本质上是训练数据长尾分布的直接反映。

## 方法详解

### 整体框架

CURE 包含两个核心组件：**数据集**和**评分器套件**，通过"逐步增加提示信息"来评估 T2I 系统对文化知识的内化程度。

### 关键设计

#### 1. CURE 数据集构建

从 Wikimedia 知识图谱自动构建分层类别体系：
- **6 个文化轴**（s）：食物、艺术、时尚、建筑、庆典、人物
- **32 个文化类别**（c）：如 dumpling、flatbread、pottery 等
- **300 个文化文物**（n）：如 banku（加纳饺子）、modak（印度甜点）
- **64 个国家/地区**（r）

属性层次为 $s \to c \to n, r$，通过系统遍历 Wikimedia 的父子节点实现民主化、可扩展的数据集构建。

#### 2. 边际信息归因（MIA）评分器

核心假设：如果 T2I 系统已充分学习某文化文物的知识，那么**增加提示中的属性信息**（如从仅指定名称到同时指定类别、地区）不应显著改变生成质量。

具体通过三类评分器实现：

**感知相似性（PS）评分器**：
$$\phi_{PS}(n) = sim(I(n), I(c))$$
比较仅按名称生成的图像 $I(n)$ 与按类别生成的图像 $I(c)$ 之间的相似度。如果两者相似，说明 T2I 系统已学会 $n \to c$ 的文化关联。

**图像-文本对齐（ITA）评分器**：
$$\phi_{ITA}(a) = \frac{sim(I(n), P(n)) + sim(I(n), P(a))}{2}$$
同时评估视觉正确性和文化关联性，避免直接查询区域相关性的低效问题。

**多样性（DIV）评分器**：
$$\phi_{DIV} = LPIPS(n, \{n,c\}, \{n,r\}, \{n,c,r\})$$
计算不同属性规范下生成图像间的 LPIPS 成对差异，衡量信息增加对多样性的影响。

#### 3. 用户研究设计

在 Prolific 平台雇佣匹配国籍的众包工人，评估三项 Likert 量表：文化代表性、感知相似性、类别似然性。关键创新：**要求工人明确认同其国籍文化**，而非简单假设。

### 损失函数 / 训练策略

本文为评测工作，不涉及模型训练。核心评估指标为 Spearman 秩相关系数 $\rho$，衡量自动评分器与人类金标准判断的单调关系。

## 实验关键数据

### 主实验（感知相似性评分器与人类判断的 Spearman 相关性）

| 编码器 | 评分器 | FLUX.1 $\phi^*_{CURE}$ | FLUX.1 $\phi^*_{PS}$ | SD 3.5 $\phi^*_{CURE}$ | SD 3.5 $\phi^*_{PS}$ |
|--------|--------|:---:|:---:|:---:|:---:|
| SigLIP 2 | $\phi_{GT}(n)$↑ | 0.25 | 0.44 | 0.27 | 0.45 |
| SigLIP 2 | $\phi_{PS}(n)$↑ | 0.18 | 0.32 | 0.22 | 0.38 |
| SigLIP 2 | $\Delta\phi_{PS}(\{n,c\})$↓ | -0.16 | -0.31 | -0.21 | -0.37 |
| DINOv2 | $\phi_{GT}(n)$↑ | 0.17 | 0.40 | 0.25 | 0.46 |
| DINOv2 | $\Delta\phi_{PS}(\{n,c\})$↓ | -0.19 | -0.32 | -0.21 | -0.35 |

### 消融实验（ITA 评分器对比，SigLIP 2 backbone）

| 评分器 | FLUX.1 $\phi^*_{CURE}$ | FLUX.1 $\phi^*_{GT}$ | SD 3.5 $\phi^*_{CURE}$ | SD 3.5 $\phi^*_{GT}$ |
|--------|:---:|:---:|:---:|:---:|
| Khanuja et al. | 0.13 | 0.08 | 0.05 | 0.04 |
| sim(I(n), P(n)) | 0.24 | 0.35 | 0.18 | 0.31 |
| $\phi_{ITA}(\{c,r\})$ (Ours) | **0.27** | **0.38** | **0.23** | **0.34** |
| PickScore | 0.20 | 0.29 | 0.23 | 0.37 |
| Gemini 2.0 Flash | 0.23 | 0.41 | 0.27 | 0.37 |

### 关键发现

1. **MIA 评分器在无需真实图像的情况下**接近甚至匹配需要 GT 图像的基线性能
2. 所有量化评分器与人类判断的最高相关性仅 $\rho=0.51$，说明**当前视觉编码器仍远不足以替代人类文化判断**
3. T2I 系统质量越高（ELO 越高），多样性越高但文化准确性反而下降（**事实性-多样性权衡**）
4. Gemini 2.0 Flash 作为 MLLM 评判者表现不错但会**幻觉全球南方文化细节**

## 亮点与洞察

- **数据集设计的巧思**：利用 Wikimedia 知识图谱的层次结构实现自动化构建，任何人都可以添加新文化类别
- **边际信息归因的核心洞察**：不直接评估生成质量，而是观察"给更多信息后质量变化多少"——这是一个优雅的间接评估策略
- 揭示了**生成纠缠**问题：使用与 T2I 系统共享训练数据的 VLM 评估会系统性高估性能
- 评测覆盖了 64 个国家、6 个 T2I 系统、多种编码器/VLM，实验规模扎实

## 局限与展望

- 以地理（国家）作为文化代理过于粗糙，未考虑宗教、语言等维度
- PS 评分器对低质量 T2I 系统（如 SD 1.5）效果差，依赖 T2I 系统本身的覆盖质量
- 无法处理语义歧义（如"damper"既是澳大利亚面包又是机械设备）
- 用户研究仅使用英语问卷，可能引入语言偏差

## 相关工作与启发

- 与 CulturalBench、CCC-Bench 等文化评测工作互补，但 CURE 首次系统化提出基于边际效用的评分框架
- MIA 思路可推广到其他生成模型（视频、3D）的文化偏差评估
- 启发了"利用信息量变化间接评估模型知识"的通用评测范式

## 评分

- 新颖性: ⭐⭐⭐⭐ （MIA 评分框架新颖且有理论基础）
- 实验充分度: ⭐⭐⭐⭐⭐ （6 个 T2I 系统 + 大规模用户研究 + 多维度消融）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，图表丰富）
- 价值: ⭐⭐⭐⭐ （为 T2I 文化公平性提供了实用评测工具）

<!-- RELATED:START -->

## 相关论文

- [Fix-CLIP: Dual-Branch Hierarchical Contrastive Learning via Synthetic Captions for Better Understanding of Long Text](fix-clip_dual-branch_hierarchical_contrastive_learning_via_synthetic_captions_fo.md)
- [Long-Context State-Space Video World Models](long-context_state-space_video_world_models.md)
- [InfiniDreamer: Arbitrarily Long Human Motion Generation via Segment Score Distillation](infinidreamer_arbitrarily_long_human_motion_generation_via_segment_score_distill.md)
- [CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models](compass_enhancing_spatial_understanding_in_text-to-image_diffusion_models.md)
- [LongT2IBench: A Benchmark for Evaluating Long Text-to-Image Generation with Graph-structured Annotations](../../AAAI2026/image_generation/longt2ibench_a_benchmark_for_evaluating_long_text-to-image_generation_with_graph.md)

<!-- RELATED:END -->
