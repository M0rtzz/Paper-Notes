---
title: >-
  [论文解读] COHO: Context-Sensitive City-Scale Hierarchical Urban Layout Generation
description: >-
  [ECCV2024][自监督学习][Urban Layout Generation] 提出基于图掩码自编码器 (GMAE) 的城市级 2.5D 布局生成方法，通过规范图表示捕获建筑-街区-社区的多层语义上下文，结合优先级调度的迭代采样，在 330 个美国城市上实现了兼具真实感、语义一致性和正确性的大规模城市布局生成。
tags:
  - ECCV2024
  - 自监督学习
  - Urban Layout Generation
  - Graph Masked Autoencoder
  - Context Sensitivity
  - 量化
  - Scheduled Iterative Sampling
---

# COHO: Context-Sensitive City-Scale Hierarchical Urban Layout Generation

**会议**: ECCV2024  
**arXiv**: [2407.11294](https://arxiv.org/abs/2407.11294)  
**代码**: [Arking1995/COHO](https://github.com/Arking1995/COHO)  
**领域**: self_supervised  
**关键词**: Urban Layout Generation, Graph Masked Autoencoder, Context Sensitivity, Vector Quantization, Scheduled Iterative Sampling

## 一句话总结

提出基于图掩码自编码器 (GMAE) 的城市级 2.5D 布局生成方法，通过规范图表示捕获建筑-街区-社区的多层语义上下文，结合优先级调度的迭代采样，在 330 个美国城市上实现了兼具真实感、语义一致性和正确性的大规模城市布局生成。

## 背景与动机

大规模城市布局生成在计算机视觉、城市规划、数字孪生和游戏设计等领域有广泛需求。一个城市通常包含 1,000 到 50,000 个街区，每个街区的配置差异显著，但相邻街区之间往往共享某种风格或有目的性的多街区排列。

现有方法的不足：

- **程序化生成**（如 Parish & Müller 等）需要手工编写规则，灵活性差
- **基于像素的深度学习**方法（InfiniCity、CityDreamer、CityGEN）基于图像 patch 训练，无法学习全局特征及其相互依赖关系，语义和真实感有限，通常还需要后处理
- **基于图的布局生成**方法（LayoutTransformer、VTN、LayoutDM、GlobalMapper）将每个街区视为独立单元，忽略了城市布局的上下文敏感性——相邻街区之间的风格一致性和语义关系未被建模

关键观察：城市中建筑、街区和社区的布局不是孤立的，它们之间存在多层语义依赖关系。不考虑上下文的生成会导致相邻街区之间出现不自然的过度多样性或过度相似性。

## 核心问题

如何在保持大规模可扩展性的前提下，实现上下文敏感的城市级 2.5D 布局生成？具体来说，需要解决三个子问题：

1. **表示问题**：如何用统一的数据结构紧凑表示城市中任意形状的建筑、街区、社区和道路？
2. **上下文敏感性**：如何在生成时显式考虑多层结构之间的相互依赖关系？
3. **优先级问题**：在城市级大规模生成中，如何确定各部分的生成顺序以提高整体质量？

## 方法详解

### 1. 规范图表示 (Canonical Graph Representation)

将整个城市表示为图 $G = \{B, E\}$：

- **节点** $b_i$：每个城市街区，包含两组特征：
    - $s_i$（4维）：街区的形状和位置特征——长宽比、总面积、面积与凸包面积之比、到城市中心的相对距离
    - $q_i$（512维）：从预训练量化码本 $C$ 中获得的离散向量，层次化地编码街区内所有建筑的布局、形状和高度
    - 两者拼接为 516 维节点特征 $[s_i, q_i]$
- **边** $e_{ij}$：连接空间相邻的街区，特征 $d_{ij}$ 为相邻街区质心之间的距离
- **子图**：任意连通子图对应一个社区 (community)

### 2. 建筑布局量化 (Building Layout Quantization)

512 维向量 $q_i$ 的获取过程：

1. 训练一个基于 GAT 骨干的街区级变分自编码器 (BVAE)，使用图结构表示单个街区内多栋建筑的形状、位置和高度
2. 使用预训练好的 BVAE 编码器，得到所有建筑布局的 512 维潜在向量
3. 对每个维度的分布做等百分位量化，分为 $L=20$ 个 bin
4. 每个潜在值替换为其所在 bin 的序号，最终得到量化索引向量 $q_i = [c_1, ..., c_{512}]$，$c \in \{1,2,...,20\}$

注意：该码本在 BVAE 训练后定义，非动态可训练（不同于 VQVAE/VQGAN）。

### 3. 图掩码自编码器 (GMAE)

自监督训练流程：

- **掩码策略**：对建筑布局特征 $Q$ 随机掩码，掩码比率 $m$ 从截断高斯分布 $\mathcal{N}(0.55, 0.25)$ 中采样，范围 $[0.5, 1.0]$；街区形状特征 $S$ 和边特征 $E$ 保留
- **编码器** $f_E$：GAT（3层，即 D=3 跳消息传递），通过邻域消息传递获得上下文感知的节点特征 $F$
- **解码器** $f_D$：简单 MLP，从 $F$ 重建量化建筑布局特征 $Q'$
- **损失函数**：对掩码节点使用交叉熵损失 $L_{recon} = -\sum_{i=1}^{L} [Q_i \log(Q'_i)]^{mask}$
- **社区采样**：训练时从城市图中随机采样半径 500 米的子图作为 batch，覆盖合理数量的街区

### 4. 优先级调度的迭代生成 (Priority-based Scheduled Generation)

推理时采用 $T=12$ 次迭代采样：

1. 每次迭代中，GMAE 预测所有剩余掩码节点的建筑布局特征
2. 按预测置信度排序，接受置信度最高的节点
3. 接受比例由余弦调度函数控制：$\beta(t) = 1 - \cos(t/T)$
    - 初始迭代保守（接受少量高置信度节点），后期加速
    - 效果：重要的、有代表性的街区先被生成，引导邻近街区的风格
4. 支持任意比例的先验约束 $[0, 100\%]$，可实现从零生成、补全和精化

## 实验关键数据

### 数据集

- 330 个美国城市（人口 > 10 万），共 833,473 个街区、17,663,607 栋建筑
- 数据来源：OpenStreetMap、Microsoft Building Footprints、TIGER 数据集
- 划分：训练 70%、验证 20%、测试 10%

### 定量对比（100 个社区、3700 个街区）

| 方法 | CTS↓ | WD-5D↓ | WD-CO↓ | Overlap↓ | FID↓ | KID↓ | LPIPS↓ |
|------|-------|--------|--------|----------|------|------|--------|
| SDXL | - | - | - | - | 120.24 | 0.079 | 0.48 |
| VTN | -1.14 | 3.18 | 5.81 | 1.24 | 69.14 | 0.047 | 0.32 |
| LayoutDM | -2.20 | 2.92 | 12.50 | 4.56 | 66.77 | 0.040 | 0.39 |
| GlobalMapper | 0.62 | 4.77 | 4.14 | 2.52 | 49.55 | 0.024 | 0.34 |
| **COHO** | **0.21** | **2.28** | **1.91** | **1.27** | **23.63** | **0.005** | **0.20** |

- FID 从次优的 49.55 降至 23.63（提升 52%），KID 从 0.024 降至 0.005
- CTS（上下文分数）最接近 0，表明邻域风格一致性最佳
- 训练时间：BVAE 12h + GMAE 15h（单张 A5000 GPU）；推理几乎即时

### 关键消融实验

- **BVAE 骨干**：GAT >> GCN > GraphSAGE >> LayoutVAE，GAT 在所有几何误差指标上最优
- **量化方法**：逐维量化 (DIM-Q) >> 可训练 VQ >> KMeans-Q；$L=20$ 为最佳折中
- **GMAE 编码器深度**：D=3 最优，D=1 上下文不足，D=4 过度相似且计算负担增加
- **掩码策略**：动态高掩码率 $[0.5, 1.0]$ 最优，固定小掩码率 0.15 导致过度相似
- **调度函数**：余弦函数 + T=12 最优；对数函数最差，单步生成 (T=1) 质量极差

## 亮点

1. **上下文敏感的图表示**：首次将城市多层语义（建筑→街区→社区→城市）统一编码为规范图，支持大规模生成
2. **自监督学习范式巧妙**：借鉴 BERT/MAE/MaskGIT 的掩码-重建思想，迁移到城市图生成任务，无需配对数据
3. **优先级调度策略有理有据**：先生成高置信度的代表性街区再扩散，符合城市规划中"锚点→填充"的直觉
4. **实验规模大且系统**：330 个城市、83 万街区、1766 万建筑，消融实验非常全面
5. **开源数据集**：发布了大规模城市布局数据集，对社区后续研究有重要价值

## 局限与展望

- 无法处理非简单多边形的建筑（如带中庭的建筑）和极不规则的凹形街区（如袋形道路 cul-de-sac）
- 道路网络与上下文结构之间的关系未被显式建模——道路网络作为给定输入而非生成目标
- 量化过程可能丢失细粒度的建筑细节（L=20 仅 20 级量化）
- 目前仅在美国城市上训练和评估，欧洲/亚洲城市的泛化性未知
- 未来方向：结合大规模照片真实感多视图场景合成、城市级 3D 建模，以及自动驾驶合成数据生成

## 与相关工作的对比

| 维度 | COHO (本文) | GlobalMapper | LayoutDM | VTN | SDXL |
|------|-------------|--------------|----------|-----|------|
| 表示 | 图（节点=街区） | 图（单街区） | Token 序列 | Token 序列 | 像素 |
| 上下文感知 | 多层图消息传递 | 无 | 无 | 无 | 无 |
| 优先级生成 | 余弦调度迭代 | 单步 | 单步 | 自回归 | 扩散 |
| 可扩展性 | 城市级 | 单街区 | 单街区 | 单街区 | Patch 级 |
| 任意形状 | 支持 | 支持 | 有限 | 有限 | 不支持 |
| FID | 23.63 | 49.55 | 66.77 | 69.14 | 120.24 |

与 MaskGIT/MAGE 的联系：COHO 将图像生成中掩码+迭代解码的范式迁移到了图结构的城市布局生成，核心思想一脉相承，但在节点特征量化、图消息传递和调度策略上做了针对性设计。

## 启发与关联

- **图掩码自编码器用于结构化生成**：掩码+重建的自监督范式不仅适用于图像/文本，也可迁移到图结构的大规模场景生成，值得在其他结构化生成任务（如室内场景、分子设计）中探索
- **层次化量化表示**：先用任务特定的 VAE 将底层细节量化为离散 token，再在更高层做生成，是处理复杂多层结构的有效策略
- **优先级调度**：先生成高置信度/高重要性的元素再逐步填充，比均匀并行或简单自回归更适合具有空间依赖的生成任务
- **评估指标 CTS**：提出的上下文分数 (Context Score) 可作为衡量邻域风格一致性的通用评估工具

## 评分
- 新颖性: 8/10 — 首次将上下文敏感性引入大规模城市布局生成，图掩码自编码器+优先级调度的组合新颖
- 实验充分度: 9/10 — 330 城市大规模实验，消融实验系统覆盖模型选择、量化策略、掩码策略和调度函数
- 写作质量: 8/10 — 结构清晰，三个观察→三个设计的逻辑线明确，图示质量高
- 价值: 7/10 — 大规模城市布局生成相对小众，但开源数据集和方法有实际应用价值

<!-- RELATED:START -->

## 相关论文

- [Few-Shot Implicit Function Generation via Equivariance](../../CVPR2025/self_supervised/few-shot_implicit_function_generation_via_equivariance.md)
- [SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery](../../NeurIPS2025/self_supervised/seal_semantic-aware_hierarchical_learning_for_generalized_category_discovery.md)
- [Towards Reliable and Holistic Visual In-Context Learning Prompt Selection](../../NeurIPS2025/self_supervised/towards_reliable_and_holistic_visual_in-context_learning_prompt_selection.md)
- [Improving Region Representation Learning from Urban Imagery with Noisy Long-Caption Supervision](../../AAAI2026/self_supervised/improving_region_representation_learning_from_urban_imagery_with_noisy_long-capt.md)
- [Test-Time Training Provably Improves Transformers as In-Context Learners](../../ICML2025/self_supervised/test-time_training_provably_improves_transformers_as_in-context_learners.md)

<!-- RELATED:END -->
