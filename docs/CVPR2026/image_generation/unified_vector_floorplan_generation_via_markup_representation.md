---
title: >-
  [论文解读] Unified Vector Floorplan Generation via Markup Representation
description: >-
  [CVPR 2026][图像生成][户型图生成] 本文提出 Floorplan Markup Language (FML) 标记语言，将房间、门等户型元素编码为结构化 token 序列，用一个 LLaMA 风格的 Transformer 模型（FMLM）统一解决无条件/边界条件/图条件/补全等多种户型图生成任务，FID 指标比 HouseDiffusion 低 80%+。
tags:
  - CVPR 2026
  - 图像生成
  - 户型图生成
  - 标记语言
  - 自回归序列模型
  - 约束解码
  - 向量化表示
---

# Unified Vector Floorplan Generation via Markup Representation

**会议**: CVPR 2026  
**arXiv**: [2604.04859](https://arxiv.org/abs/2604.04859)  
**代码**: https://mapooon.github.io/FMLPage  
**领域**: 图像生成  
**关键词**: 户型图生成、标记语言、自回归序列模型、约束解码、向量化表示

## 一句话总结

本文提出 Floorplan Markup Language (FML) 标记语言，将房间、门等户型元素编码为结构化 token 序列，用一个 LLaMA 风格的 Transformer 模型（FMLM）统一解决无条件/边界条件/图条件/补全等多种户型图生成任务，FID 指标比 HouseDiffusion 低 80%+。

## 研究背景与动机

1. **领域现状**：户型图自动生成是建筑设计和房地产行业的核心需求。现有方法按条件类型分治——边界条件用 Graph2Plan，邻接图条件用 HouseGAN++/HouseDiffusion，但每种任务需要专用模型。
2. **现有痛点**：(1) 不同生成任务用不同架构，无法统一；(2) 基于扩散模型的方法（GSDiff）生成的是栅格图，后处理转换为矢量格式会引入误差；(3) GAN 方法容易模式崩溃且生成多样性受限。
3. **核心矛盾**：户型图本质是结构化的矢量数据（房间多边形+门的位置+连接关系），但现有方法要么在像素空间工作（丢失结构信息），要么需要针对性的图神经网络。
4. **本文目标**：设计一种统一表示，将所有户型图生成任务转化为同一种序列预测问题。
5. **切入角度**：受 NLP 中标记语言（HTML/XML）的启发——用语法规则定义的 token 序列天然适合表示结构化信息，且可以直接用自回归 Transformer 建模。
6. **核心 idea**：定义 FML 语法将户型图编码为"标签+坐标+索引+类型"的 token 序列，用约束解码保证生成结果的语法合法性。

## 方法详解

### 整体框架

输入条件（可选：边界点序列/邻接图/部分户型）→ 编码为 FML 条件段 → FMLM 自回归生成 FML 序列 → 约束解码确保语法合法 → 解析 FML 得到矢量户型图（含房间多边形、门位置）。

### 关键设计

1. **Floorplan Markup Language (FML)**

    - 功能：将户型图的所有元素编码为线性 token 序列
    - 核心思路：定义四种 token 类型——标签（如 `<room>`, `<door>`）、坐标（1D 编码 $z = x + y \times W$，$W=256$）、房间索引、房间类型。语法规则为 `<sequence> → <condition> → <floorplan> → rooms → doors → front_door → </sequence>`。房间按索引降序排列
    - 设计动机：1D 坐标编码避免了 2D 位置的高维稀疏问题；降序排列经消融验证 FID 从 94.57 降至 25.50；标签 token 提供结构监督信号

2. **FMLM 模型架构**

    - 功能：自回归生成 FML token 序列
    - 核心思路：LLaMA-3 风格 Transformer，24 层，512 维隐藏状态，32 注意力头。坐标 token 使用正弦位置编码+可学习投影，标签/索引/类型使用可学习嵌入。输出头是统一线性层 $W \in \mathbb{R}^{(C_{tag}+C_{coord}+C_{index}+C_{type}) \times C}$
    - 设计动机：统一输出头允许模型自动学习不同 token 类型的生成时机，无需手动切换解码模式

3. **约束解码（Constrained Decoding）**

    - 功能：推理时保证生成的 FML 序列语法合法
    - 核心思路：硬约束包括：门恰好有 2 个顶点、房间多边形不能与已有房间重叠、门必须位于房间边界上。这些规则在解码时直接 mask 不合法 token 的概率
    - 设计动机：自回归模型可能生成语法错误的序列（如 3 顶点的门），约束解码以零额外计算成本保证 100% 合法输出

### 损失函数 / 训练策略

标准交叉熵损失在 FML 序列的非结构标签 token 上计算。训练时使用房间排列增强（随机打乱房间顺序），学习排列等变性——消融显示排列增强使 FID 从 24.36 降至 14.17。

## 实验关键数据

### 主实验

| 任务 | 方法 | FID↓ | GED↓ | IoU↑ |
|------|------|------|------|------|
| 无条件 | GSDiff | 15.02 | - | - |
| 无条件 | **FMLM** | **7.22** | - | - |
| 边界条件 | Graph2Plan | 34.20 | - | 95.87% |
| 边界条件 | **FMLM** | **6.51** | - | **97.86%** |
| 图条件(ALL) | HouseGAN++ | 48.44 | 2.57 | - |
| 图条件(ALL) | HouseDiffusion | 29.31 | 1.55 | - |
| 图条件(ALL) | **FMLM** | **3.41** | **1.21** | - |
| 边界+图(ALL) | Graph2Plan | 22.87 | 3.43 | 92.96% |
| 边界+图(ALL) | **FMLM** | **14.17** | **1.24** | **97.59%** |

### 消融实验

| 配置 | FID↓ | GED↓ | IoU↑ | 说明 |
|------|------|------|------|------|
| Full + 排列增强 | 14.17 | 1.24 | 97.59% | 完整模型 |
| w/o 排列增强 | 24.36 | 2.35 | 95.82% | FID 涨 72% |
| 升序索引 | 94.57 | - | - | FID 极差 |
| 降序索引 | 25.50 | - | - | 降序远优于升序 |

### 关键发现

- 房间排列增强是性能的关键——去掉后 FID 从 14.17 涨到 24.36（+72%），说明模型需要学习排列等变性才能有效泛化
- FMLM 在所有条件设定下都大幅超越 GAN 和扩散模型方法
- 约束解码保证了 100% 语法合法的生成结果，而 HouseDiffusion 等方法的后处理步骤无法保证这一点
- 8 房间场景性能略有下降（FID 从 3.41 升至 4.64），因为训练样本较少

## 亮点与洞察

- **标记语言表示的精妙**：通过定义语法规则将结构化生成问题优雅转化为序列预测，这种思路可迁移到其他结构化生成任务（如电路版图、分子结构）
- **约束解码的零开销保证**：在推理时通过 mask 非法 token 实现硬约束，不增加计算成本但消除了所有非法输出——这比后处理修正更可靠
- **统一多任务**：同一个模型同时处理无条件/边界/图/补全四种任务，消除了之前"每个任务一个模型"的冗余

## 局限与展望

- 仅支持单层户型图，多层建筑需要扩展 FML 语法
- 8 房间以上场景训练数据不足，效果下降
- 坐标量化到 256×256 网格可能丢失精度，更高分辨率会增加词表大小
- 与 LLM 结合（用自然语言描述需求→生成户型）是有前景的方向

## 相关工作与启发

- **vs HouseDiffusion**: 扩散方法在连续空间建模，需要后处理矢量化，而 FMLM 直接在离散 token 空间生成矢量结果，更精确
- **vs Graph2Plan**: 需要 GNN 编码邻接图为条件，架构复杂。FMLM 将邻接关系直接序列化为 FML 条件段，不需要额外编码器
- **vs GSDiff**: 栅格化扩散方法 FID 15.02，FMLM 7.22，差距主要来自矢量表示的结构先验

## 评分

- 新颖性: ⭐⭐⭐⭐ 标记语言表示是新颖的视角，但自回归生成本身不算新
- 实验充分度: ⭐⭐⭐⭐⭐ 四种条件设定全面对比+消融+多房间数量分析
- 写作质量: ⭐⭐⭐⭐ 清晰流畅，FML 语法定义严谨
- 价值: ⭐⭐⭐⭐ 对建筑设计领域有直接应用价值，标记语言思路有可迁移性

<!-- RELATED:START -->

## 相关论文

- [ExpPortrait: Expressive Portrait Generation via Personalized Representation](expportrait_expressive_portrait_generation_via_personalized_representation.md)
- [Purrception: Variational Flow Matching for Vector-Quantized Image Generation](../../ICLR2026/image_generation/purrception_variational_flow_matching_for_vector-quantized_image_generation.md)
- [BiGain: Unified Token Compression for Joint Generation and Classification](bigain_token_compression.md)
- [CoLoGen: Progressive Learning of Concept-Localization Duality for Unified Image Generation](cologen_progressive_learning_of_concept-localization_duality_for_unified_image_g.md)
- [Diversity over Uniformity: Rethinking Representation in Generated Image Detection](diversity_over_uniformity_rethinking_representation_in_generated_image_detection.md)

<!-- RELATED:END -->
