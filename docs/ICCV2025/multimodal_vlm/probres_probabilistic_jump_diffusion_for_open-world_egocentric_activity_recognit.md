---
title: >-
  [论文解读] ProbRes: Probabilistic Jump Diffusion for Open-World Egocentric Activity Recognition
description: >-
  [ICCV 2025][多模态][开放世界活动识别] 提出 ProbRes 框架，通过基于跳跃扩散的概率残差搜索策略，结合 ConceptNet 常识先验与 VLM 似然估计，在开放世界第一人称活动识别中高效导航大规模搜索空间，大幅减少 VLM 查询次数的同时提升识别准确率。
tags:
  - ICCV 2025
  - 多模态
  - 开放世界活动识别
  - 第一人称视角
  - 跳跃扩散
  - 结构化搜索
  - VLM
---

# ProbRes: Probabilistic Jump Diffusion for Open-World Egocentric Activity Recognition

**会议**: ICCV 2025  
**arXiv**: [2504.03948](https://arxiv.org/abs/2504.03948)  
**代码**: 待公开  
**领域**: 多模态VLM  
**关键词**: 开放世界活动识别, 第一人称视角, 跳跃扩散, 结构化搜索, VLM  

## 一句话总结

提出 ProbRes 框架，通过基于跳跃扩散的概率残差搜索策略，结合 ConceptNet 常识先验与 VLM 似然估计，在开放世界第一人称活动识别中高效导航大规模搜索空间，大幅减少 VLM 查询次数的同时提升识别准确率。

## 研究背景与动机

开放世界第一人称活动识别要求模型从庞大的、无约束的可能性空间中推断正在进行的活动，这与传统的闭集分类有根本不同。主要挑战包括：

**搜索空间爆炸**：活动由动作（action）和物体（object）组合而成，组合数随开放程度成指数增长，穷举枚举在计算上不可行

**VLM 的局限性**：虽然 CLIP、LAVILA 等 VLM 具有强大的零样本泛化能力，但它们依赖穷举枚举进行推理，对大规模开放世界推理效率低下

**开放程度定义模糊**：现有研究对"开放世界"缺乏统一定义，不同因素（物体、动作、领域）独立贡献不确定性，比较方法和评估泛化能力困难

作者首先提出了一个清晰的**开放程度层级分类**（L0-L3）：
- **L0**：所有活动类别预定义（传统零样本）
- **L1**：原子概念（动作/物体）已知，但组合未定义
- **L2**：领域已知（如"烹饪"），搜索空间无约束
- **L3**：完全无先验知识，需从零构建搜索空间

## 方法详解

### 整体框架

ProbRes 是一个基于跳跃扩散的自适应搜索框架，核心思想是在 VLM 的文本嵌入空间中高效导航，通过结构化先验引导搜索。框架分为三个阶段：

1. **探索阶段（Exploration）**：先验驱动的采样发现候选活动
2. **利用阶段（Exploitation）**：基于 VLM 似然的精细搜索
3. **残差精化（Residual Refinement）**：概念分解与重排序

### 搜索空间构建

**先验空间构建**：利用 ConceptNet 知识图谱估计动作-物体对的先验概率。将 ConceptNet 建模为有向图，通过最短路径的边权重之和（带指数衰减因子 $\lambda^i$）计算语义得分：

$$f(a_{\text{action}}, a_{\text{object}}) \leftarrow f(a_{\text{action}}, a_{\text{object}}) \cdot R(a_{\text{action}}, a_{\text{object}})$$

其中 $R(\cdot)$ 是关系调整权重，对负面关系（NotCapableOf）惩罚、正面关系（UsedFor）加强。先验归一化后形成概率分布。

**搜索空间排序**：基于 VLM 嵌入 $\phi(a)$ 的欧氏距离 $d(a_i, a_j) = \|\phi(a_i) - \phi(a_j)\|$ 对活动排序，选择锚点 $a_{\text{ref}}$ 按距离排序，确保语义相近的活动在搜索空间中毗邻。

### 自适应搜索：跳跃扩散

**探索阶段**采用先验引导采样：

$$P_{\text{explore}}(a) = \frac{\lambda P_{\text{prior}}(a) + (1-\lambda) \frac{1}{|\mathcal{S}|}}{\sum_{a'} \lambda P_{\text{prior}}(a') + (1-\lambda)}$$

$\lambda \in [0,1]$ 控制先验引导与均匀随机采样的权衡。$\lambda \approx 1$ 时强先验引导，$\lambda \approx 0$ 时为随机游走。

**利用阶段**转为似然驱动：

$$P_{\text{guided}}(a) = \frac{P_{\text{prior}}(a) \cdot P_{\text{likelihood}}(v|a)}{\sum_{a'} P_{\text{prior}}(a') \cdot P_{\text{likelihood}}(v|a')}$$

通过贝叶斯后验逐步收敛到高似然区域。

### 概念分解与重排序

从精化集合 $\mathcal{A}_{\text{refine}}$ 中取 top-k 候选，将活动分解为动作和物体组件，分别计算对齐得分：

$$S_{\text{final}}(a) = P_{\text{likelihood}}(v|a) + \lambda_a S_a + \lambda_o S_o$$

其中 $S_a = v^T \phi(a_{\text{action}})$，$S_o = v^T \phi(a_{\text{object}})$。这种层级重排序确保语义一致的组件获得更高置信度。

### 损失函数

ProbRes 是推理时框架，不涉及训练损失。核心优化目标为：

$$a^* = \arg\max_{a \in \mathcal{S}} [P_{\text{search}}(a) + \lambda_a S_a + \lambda_o S_o]$$

### 关键实现细节

- VLM 骨干：EGOVLP 和 LAVILA
- 先验来源：ConceptNet
- L2/L3 的搜索空间由 Gemini 2.0 Flash 生成
- $\lambda = 0.5$，$T = 3000$（小数据集）/ $1000$（大数据集）
- $\lambda_a, \lambda_o \in [0.3, 0.7]$
- 推理时间：约 2 秒/视频（RTX 3090）

## 实验

### 主实验结果

| 方法 | GTEA Gaze VLM调用 | GTEA Gaze WUPS | GTEA Gaze+ VLM调用 | GTEA Gaze+ WUPS | EK100 VLM调用 | EK100 WUPS |
|------|------|------|------|------|------|------|
| ALGO+LAVILA | N/A | 49.42 | N/A | 53.38 | N/A | 34.47 |
| LAVILA | 380 | 51.31 | 405 | 53.27 | 29100 | 43.52 |
| LAVILA+ProbRes | **110** | **53.34** | **175** | **53.82** | **3000** | **43.55** |
| EGOVLP | 380 | 46.77 | 405 | 51.48 | 29100 | 39.84 |
| EGOVLP+ProbRes | **110** | **49.25** | **178** | **53.98** | **3000** | **40.53** |

**关键发现**：ProbRes 在 GTEA Gaze 上将 VLM 查询从 380 降至 110（减少 71%），EK100 上从 29100 降至 3000（减少 90%），同时保持或提升准确率。

### L2/L3 开放程度评估

| 设置 | 方法 | VLM调用 | GTEA活动WUPS | EK100活动WUPS |
|------|------|------|------|------|
| L2 | LAVILA | 37191 | 38.34 | 30.64 |
| L2 | LAVILA+ProbRes | **1500** | **43.28** | **31.92** |
| L3 | LAVILA | 195714 | 46.06 | 31.06 |
| L3 | LAVILA+ProbRes | **5000** | **47.71** | **32.58** |

在 L3 最开放设置下，VLM 查询从 195714 降至 5000，性能提升 1.65 WUPS。

### 消融实验

| 消融项 | WUPS变化 | 说明 |
|------|------|------|
| 去除ConceptNet先验 | 显著下降 | L3设置下影响最大，非结构化搜索空间增加不确定性 |
| 去除重排序 | Exact Match下降 | 重排序对区分语义相似但错误的预测至关重要 |
| $\lambda=0$（纯似然） | 性能下降 | 过早收敛到高置信但错误的活动 |
| $\lambda \approx 0.6$（最优） | 最佳 | 探索与利用的最佳平衡点 |

## 亮点与洞察

1. **效率-准确率双赢**：ProbRes 的核心贡献在于论证了结构化搜索可以替代穷举枚举，用 10% 的 VLM 查询实现更好的性能
2. **层级分类法（L0-L3）**：系统化定义了开放世界识别的不同挑战层级，为该领域提供了统一的评估框架
3. **先验知识的关键作用**：实验证明在搜索空间无约束时，ConceptNet 先验比随机搜索高效得多
4. **概念分解的有效性**：将活动分解为动作和物体组件分别评估，有效缓解了 VLM 短语级嵌入的噪声问题

## 局限性

1. 依赖 VLM 的预训练偏差，可能导致搜索轨迹不够高效
2. VLM 文本嵌入缺乏语义组织，导致搜索中出现不相关跳跃
3. L2/L3 设置中 LLM 生成的搜索空间需要精心筛选，否则可能过于宽泛或嘈杂
4. ConceptNet 的覆盖范围有限，对于高度特化的领域可能不足

## 相关工作

- **零样本活动识别**：KGL (2022) 使用 ConceptNet 知识图谱，ALGO (2024) 通过可供性先验迭代精化
- **VLM 骨干**：EGOVLP、LAVILA、EgoVLPv2 等专注于自中心视频-文本嵌入
- **开放世界检测**：Open World DETR、UMB 等在目标检测中处理新类偏差

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 总体推荐 | 8/10 |

<!-- RELATED:START -->

## 相关论文

- [Visual Intention Grounding for Egocentric Assistants](visual_intention_grounding_for_egocentric_assistants.md)
- [DisenQ: Disentangling Q-Former for Activity-Biometrics](disenq_disentangling_q-former_for_activity-biometrics.md)
- [WearVQA: A Visual Question Answering Benchmark for Wearables in Egocentric Authentic Real-world scenarios](../../NeurIPS2025/multimodal_vlm/wearvqa_a_visual_question_answering_benchmark_for_wearables_in_egocentric_authen.md)
- [Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy](dita_scaling_diffusion_transformer_for_generalist_visionlang.md)
- [CrafText Benchmark: Advancing Instruction Following in Complex Multimodal Open-Ended World](../../ACL2025/multimodal_vlm/craftext_benchmark_advancing_instruction_following_in_complex_multimodal_open-en.md)

<!-- RELATED:END -->
