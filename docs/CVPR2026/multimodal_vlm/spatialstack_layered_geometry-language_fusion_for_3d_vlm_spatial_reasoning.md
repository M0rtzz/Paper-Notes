---
title: >-
  [论文解读] SpatialStack: Layered Geometry-Language Fusion for 3D VLM Spatial Reasoning
description: >-
  [CVPR2026][多模态][3D空间推理] 提出SpatialStack框架，将多视图几何编码器（VGGT）的多层级几何特征逐层注入LLM解码器的不同层（而非仅融合最后一层），通过浅层→细粒度空间感知、深层→高层语义推理的层级对齐，在多个3D空间推理基准上达到开源SOTA。
tags:
  - CVPR2026
  - 多模态
  - 3D空间推理
  - 几何-语言融合
  - 层级特征融合
  - VLM
  - VGGT
---

# SpatialStack: Layered Geometry-Language Fusion for 3D VLM Spatial Reasoning

**会议**: CVPR2026  
**arXiv**: [2603.27437](https://arxiv.org/abs/2603.27437)  
**代码**: [https://spatial-stack.github.io/](https://spatial-stack.github.io/)  
**领域**: 多模态VLM  
**关键词**: 3D空间推理, 几何-语言融合, 层级特征融合, VLM, VGGT

## 一句话总结
提出SpatialStack框架，将多视图几何编码器（VGGT）的多层级几何特征逐层注入LLM解码器的不同层（而非仅融合最后一层），通过浅层→细粒度空间感知、深层→高层语义推理的层级对齐，在多个3D空间推理基准上达到开源SOTA。

## 研究背景与动机
大型视觉语言模型（VLM）在3D空间推理方面仍有明显短板——无法可靠地编码3D几何结构和空间关系。现有方法如Spatial-MLLM、VG-LLM、VLM-3R虽然将端到端几何编码器（DUST3R/VGGT等）集成到VLM中，但**仅融合几何编码器最后一层的特征**与视觉编码器特征。

**核心矛盾**：几何编码器（如VGGT）采用DPT架构，从不同Transformer层显式提取多层级表示来恢复详细几何信息。仅取最后一层会丢弃中间层的丰富层级几何线索——浅层保留尖锐的局部结构和几何边界，深层产生过度同质化的激活。实验验证了这一发现：注入浅层几何特征有利于低级感知任务（深度估计、距离比较），注入深层特征有利于高级推理任务（跨视图关系推理）。

**关键发现**：简单将多层几何特征拼接后注入视觉通路（naive multi-layer fusion）反而导致**特征干扰而非协同**，性能不如单层融合。这揭示了真正的挑战在于**融合策略**，而非仅提取多层级特征。

**切入角度**：将几何特征的融合从视觉编码器端转移到LLM解码器端，通过层级对齐实现浅层几何→LLM浅层、深层几何→LLM深层的渐进式融合。

## 方法详解

### 整体框架
SpatialStack是一个通用的层级融合框架，核心思路：将几何编码器（VGGT）多个层的输出，经过独立的投影器对齐后，以加性残差的方式分别注入到LLM解码器的不同层中。整体pipeline为：
1. **视觉编码器**：处理K帧输入图像，经spatial merger得到视觉token $\tilde{\mathbf{V}}$
2. **几何编码器**（VGGT，冻结）：同一组图像通过多视图几何Transformer提取多层级几何特征
3. **层级融合**：从VGGT第11/17/23层提取patch token，经独立投影器对齐后注入LLM第0/1/2层
4. **LLM解码器**：处理融合后的多模态序列，输出答案

### 关键设计

1. **层级几何-语言融合（Layered Geometry-Language Fusion）**：从VGGT的第$l_i$层（$l_i \in \{11, 17, 23\}$）提取patch token $\mathbf{Z}_{l_i} \in \mathbb{R}^{(KN) \times D_{\text{geo}}}$，通过层特定的geometry token merger投影对齐：
   $$\mathbf{G}_{l_i} = \mathcal{M}_{\text{geo}}^{(l_i)}(\mathbf{Z}_{l_i}), \quad \mathbf{G}_{l_i} \in \mathbb{R}^{N' \times D_{\text{lang}}}$$
   然后以加性残差注入LLM对应层：
   $$\mathbf{H}^{(j)'} = \mathbf{H}^{(j)} + \mathbf{G}_{l_j}, \quad j \in \{0, 1, 2\}$$
   
   设计动机：浅层几何特征保留局部精细结构→注入LLM浅层增强底层感知；深层几何特征编码全局语义→注入LLM深层支撑高级推理。这种对齐比将所有层特征混合注入视觉通路更有效。

2. **Geometry Token Merger**：每个注入层有独立的投影器 $\mathcal{M}_{\text{geo}}^{(l_i)}$，负责将几何特征的空间分辨率和嵌入维度与LLM hidden state对齐。类似视觉编码器的spatial merger，对每2×2邻近patch分组后投影。层独立设计避免了不同抽象层级特征间的干扰。

3. **训练策略**：冻结视觉编码器和VGGT几何编码器，仅训练geometry token merger和LLM解码器。使用标准next-token交叉熵损失，无额外辅助目标。空间先验纯粹通过统一的指令微调自然涌现。

### 损失函数 / 训练策略
- 损失：标准交叉熵 $\mathcal{L}_{\text{ce}} = -\sum_{i=1}^{|o|} \log P_\theta(o^{(i)} | o^{(<i)}, q, \mathcal{C})$
- 基座模型：Qwen2.5-VL / Qwen3.5，几何编码器VGGT
- Batch size 64，学习率 $1 \times 10^{-5}$，AdamW，warmup ratio 0.03，cosine schedule
- 训练数据：SPAR、LLaVA-Hound、ScanNet、VSI-590K子集

## 实验关键数据

### 主实验（VSI-Bench）

| 方法 | 排名 | 平均 | Obj.Count | Abs.Dist | Rel.Dist | Rel.Dir | Route Plan | Appr.Order |
|------|------|------|-----------|----------|----------|---------|------------|------------|
| GPT-4o | - | 34.0 | 46.2 | 5.3 | 37.0 | 41.3 | 31.5 | 28.5 |
| Gemini-2.5 Pro | - | 51.5 | 43.8 | 34.9 | 61.1 | 47.8 | 45.9 | 71.3 |
| SpatialStack-4B (Qwen2.5) | 2 | 60.9 | 69.2 | 45.4 | 57.9 | 68.4 | 40.2 | 79.6 |
| **SpatialStack-5B (Qwen3.5)** | **1** | **67.5** | 71.0 | **55.6** | **67.3** | **84.1** | 41.2 | **83.5** |
| Cambrian-S-3B | 3 | 57.3 | 70.7 | 40.6 | 64.8 | 61.9 | 27.3 | 78.8 |
| VG-LLM-4B | 5 | 47.3 | 66.0 | 37.8 | 44.6 | 45.6 | 33.5 | 36.4 |

### 跨基准对比

| 方法 | VSI-Bench | SPAR-Bench | BLINK-Spatial | CV-Bench | Overall |
|------|-----------|------------|---------------|----------|---------|
| Qwen3.5 (fine-tuned) | 64.76 | 68.75 | **56.10** | 84.49 | 68.52 |
| GVF-L23 (VG-LLM) | 66.36 | 70.83 | 51.91 | 84.64 | 68.43 |
| GVF-L11/17/23 (naive multi) | 65.15 | 71.20 | 51.28 | 84.33 | 67.99 |
| **SpatialStack** | **67.52** | **71.39** | 52.12 | **85.53** | **69.14** |

### 消融实验

| 配置 | 低级任务 Avg | 高级任务 Avg | 说明 |
|------|-------------|-------------|------|
| 单层注入 L11 | **66.11** | 64.48 | 浅层最利于低级感知 |
| 单层注入 L23 | 64.33 | **66.36** | 深层最利于高级推理 |
| Naive多层融合(视觉端) | 64.69 | 65.15 | 特征干扰，两头不讨好 |
| SpatialStack (LLM端层级融合) | 65.89* | 67.52 | 兼顾两类任务 |

### 融合顺序消融

| 方法 | VSI-Bench | SPAR-Bench | BLINK-Spatial | CV-Bench | Overall |
|------|-----------|------------|---------------|----------|---------|
| SpatialStack (正序) | **67.52** | 71.39 | 52.12 | **85.53** | **69.14** |
| SpatialStack (反序) | 67.22 | **71.97** | 50.08 | 84.82 | 68.52 |
| Vision Fusion | 64.27 | 69.68 | **56.45** | 83.11 | 68.38 |

### 关键发现
- **层级对应关系**：VGGT浅层→精细局部几何，深层→全局语义结构；这与LLM解码器的层级功能天然对应
- **Naive多层融合失败**：将多层几何特征混合注入视觉通路导致特征干扰，不如单层融合（这是本文的核心motivation）
- **SpatialStack显著优于同base model方法**：在Qwen2.5上，SpatialStack-4B (60.9) vs VG-LLM-4B (47.3) vs Cambrian-S-3B (57.3)
- 融合顺序matters：正序（shallow-to-shallow）优于反序，验证了层级对齐假设
- 通用能力不退化：在MMBench、Video-MME上与base model持平，无灾难性遗忘
- **Route Planning零样本泛化**：训练数据中无路径规划数据，但SpatialStack-5B在该任务上仍达84.1（超越所有开源模型），展示强零样本迁移

## 亮点与洞察
- **"从哪里融合"比"融合什么"更重要**：论文系统性地论证了将几何特征从视觉编码器端迁移到LLM解码器端的必要性，这一发现对多模态模型的架构设计有普遍指导意义
- **层级对应性的实证分析**：通过定性（相似度热图）和定量（低/高级任务性能）双重验证，建立了几何编码器层-LLM解码器层的最优对应关系
- **加性残差注入的简洁性**：融合方式仅为 $H' = H + G$，无需cross-attention或门控机制，极度简洁且有效
- **模型无关框架**：SpatialStack可应用于任意开源VLM（论文在Qwen2.5和Qwen3.5上均验证），具有很好的通用性
- **DeepStack启发**：将DeepStack在视觉token层级堆叠LLM的思路迁移到几何token，是一种elegant的cross-pollination

## 局限与展望
- BLINK-Spatial上SpatialStack不如base model Qwen3.5（52.12 vs 56.10），几何注入在某些细粒度视觉感知上可能引入干扰
- 仅选择了3个VGGT层（11/17/23），更细粒度的层选择策略（如可学习门控）未探索
- 当前仅验证了VGGT作为几何编码器，对DUST3R/CUT3R等其他几何编码器的兼容性未测试
- 加性残差可能不是最优融合方式——自适应权重或cross-attention融合可能进一步提升
- 训练数据来自室内场景为主，室外/动态场景的泛化性有待验证

## 相关工作与启发
- **vs VG-LLM**：VG-LLM仅融合VGGT最后一层到视觉端，SpatialStack通过层级LLM端融合在VSI-Bench上从47.3提升到60.9（同base model）
- **vs Cambrian-S**：Cambrian-S额外引入自监督空间学习，SpatialStack通过架构创新无需额外训练范式即可超越（60.9 vs 57.3）
- **vs DeepStack**：SpatialStack将DeepStack的视觉token堆叠思路扩展到几何token，是一种自然的推广
- **vs Spatial-MLLM**：双编码器架构但仅做单层融合，SpatialStack的层级融合带来显著提升

## 评分
- 新颖性: ⭐⭐⭐⭐ 层级几何-语言融合是首创，核心insight（从哪里融合比融合什么更重要）有深度
- 实验充分度: ⭐⭐⭐⭐⭐ 4个空间推理基准+通用能力测试+多维度消融（层选择、融合顺序、视觉vs语言端融合）
- 写作质量: ⭐⭐⭐⭐⭐ 从定性分析→定量验证→方法设计→实验验证的逻辑链非常完整流畅
- 价值: ⭐⭐⭐⭐ 建立了视觉-语言-几何融合的新范式，对3D空间推理及更广泛的模态融合有重要参考价值

<!-- RELATED:START -->

## 相关论文

- [HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models](hispatial_taming_hierarchical_3d_spatial_understanding_in_vision-language_models.md)
- [Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](continual_learning_with_visionlanguage_models_via.md)
- [VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm](vlm-pruner_buffering_for_spatial_sparsity_in_an_efficient_vlm_centrifugal_token_.md)
- [SpatialTraceGen: High-Fidelity Traces for Efficient VLM Spatial Reasoning Distillation](../../NeurIPS2025/multimodal_vlm/spatialtracegen_high-fidelity_traces_for_efficient_vlm_spatial_reasoning_distill.md)
- [SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards](../../NeurIPS2025/multimodal_vlm/spatialthinker_reinforcing_3d_reasoning_in_multimodal_llms_via_spatial_rewards.md)

<!-- RELATED:END -->
