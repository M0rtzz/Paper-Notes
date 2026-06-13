---
title: >-
  [论文解读] RSVG-ZeroOV: Exploring a Training-Free Framework for Zero-Shot Open-Vocabulary Visual Grounding in Remote Sensing Images
description: >-
  [AAAI 2026][语义分割][遥感视觉定位] 提出 RSVG-ZeroOV，一个免训练框架，通过"概览-聚焦-进化"三阶段策略融合 VLM 的交叉注意力图和扩散模型的自注意力图，实现零样本开放词汇遥感视觉定位。 遥感视觉定位（RSVG）旨在根据自由形式的自然语言描述在遥感图像中定位目标对象。这一任务在城市规划、环境监测…
tags:
  - "AAAI 2026"
  - "语义分割"
  - "遥感视觉定位"
  - "零样本"
  - "开放词汇"
  - "扩散模型"
  - "免训练"
---

# RSVG-ZeroOV: Exploring a Training-Free Framework for Zero-Shot Open-Vocabulary Visual Grounding in Remote Sensing Images

**会议**: AAAI 2026  
**arXiv**: [2509.18711](https://arxiv.org/abs/2509.18711)  
**代码**: 无  
**领域**: 分割  
**关键词**: 遥感视觉定位, 零样本, 开放词汇, 扩散模型, 免训练

## 一句话总结

提出 RSVG-ZeroOV，一个免训练框架，通过"概览-聚焦-进化"三阶段策略融合 VLM 的交叉注意力图和扩散模型的自注意力图，实现零样本开放词汇遥感视觉定位。

## 研究背景与动机

遥感视觉定位（RSVG）旨在根据自由形式的自然语言描述在遥感图像中定位目标对象。这一任务在城市规划、环境监测等场景中有重要应用——例如定位"河边最高的建筑"或"操场旁边的工厂"。

现有方法面临三大局限性：

**闭集词汇约束**：现有 RSVG 方法局限于预定义类别（如"道路"、"农田"），无法处理开放世界中的自由文本描述。实际场景中的目标（如"临时路边停车区"）可能无法用简单类别名称表示，需要模型理解视觉属性、空间关系和功能角色。

**依赖昂贵监督**：少数尝试利用基础模型进行开放词汇 RSVG 的方法高度依赖高质量标注数据和耗时的微调过程，扩展性差。

**基础模型的互补性未被利用**：VLM 擅长高级语义理解但空间感知弱，扩散模型（DM）具有优秀的结构先验但缺乏语言理解。两者的注意力图在目标感知上存在互补关系，但此前未被有效整合。

作者通过系统性探索实验（Guidelines 1-3），总结出三条经验准则：（1）通用 VLM 比遥感特化 VLM 泛化能力更强；（2）DM 的自注意力编码了更优的目标结构先验；（3）交叉注意力和自注意力是互补的，融合后能持续提升性能。这三条准则直接指导了框架设计。

## 方法详解

### 整体框架

RSVG-ZeroOV 遵循**"概览-聚焦-进化"（Overview-Focus-Evolve）**三阶段策略：

1. **概览阶段**：利用冻结的 VLM 提取交叉注意力图，捕获文本查询与视觉区域的语义关联
2. **聚焦阶段**：利用冻结的扩散模型提取自注意力图作为结构先验，通过注意力交互模块填补 VLM 遗漏的形状信息
3. **进化阶段**：引入注意力进化模块，抑制无关激活，生成纯净的分割掩码

整个框架**完全免训练**，仅使用预训练的冻结模型进行推理。

### 关键设计

#### 1. **概览阶段——VLM 交叉注意力提取**

- 使用冻结的 Qwen2.5-VL 作为 VLM，输入遥感图像和文本查询
- 从 VLM 的所有 Transformer 头中提取注意力权重 $\mathcal{W}^{(t)} \in \mathbb{R}^{H \times 1 \times N}$
- 提取图像-文本相关的注意力段 $\mathcal{W}^{(t)}_{p:p'}$，并在所有注意力头和自回归步上取平均：

$$\mathcal{A}_C = \frac{1}{T}\sum_{t=1}^{T}\left(\frac{1}{H}\sum_{h=1}^{H}\mathcal{W}^{(t)}_{p:p'}\right)$$

- **核心发现**：交叉注意力图存在两个问题——（i）注意力集中在目标边界/角落而非完整区域；（ii）注意力分散，常包含无关区域

**设计动机**：VLM 的高级语义集中特性使注意力倾向于关键特征点；理解复杂文本表达需要从多个视觉区域聚合上下文线索，导致注意力分散。

#### 2. **聚焦阶段——DM 自注意力交互**

- 使用 Stable Diffusion V1.4 的 U-Net 提取多尺度自注意力图，融合为统一结构先验：

$$\mathcal{A}_S = \frac{1}{L}\sum_{l \in L}\mathcal{A}^l_S$$

- 通过**余弦相似度**计算交叉注意力和自注意力的关联：

$$\mathcal{A}_{CS} = \frac{\mathcal{A}_C \cdot \mathcal{A}_S}{\|\mathcal{A}_C\|_2 \|\mathcal{A}_S\|_2}$$

**设计动机**：DM 的自注意力对目标结构（形状、完整轮廓）的感知远优于 VLM（后者注意力分散）和 SAM（后者过度关注背景）。余弦相似度策略优于锚点法、乘法和指数法（Tab. 3），因为它生成的初始掩码语义一致性更好。

**为什么选 DM 而非 SAM？** 实验证明（Tab. 2），DM 的自注意力生成最连贯的结构表示——注意力均匀密集地分布在整个目标范围内，而 SAM 虽然边界锐利，但纯视觉设计常导致过度关注周围背景。

#### 3. **进化阶段——注意力进化模块**

- 从交叉注意力图 $\mathcal{A}_C$ 中选取 Top-K 个最高响应像素作为种子点：

$$\mathcal{S} = \text{TopK}(\mathcal{A}_C, K)$$

- 从每个种子进行**深度优先搜索（DFS）**递归扩展区域，像素 $(u,v)$ 被纳入当且仅当：

$$\mathcal{A}_{CS}[u,v] \geq \tau \text{ 且 } (u,v) \in \text{DFS}(\mathcal{S})$$

- 最终对进化后的注意力图二值化得到分割掩码：$\mathbf{M}(i,j) = \mathbb{1}[\mathcal{A}_E(i,j) > \alpha]$

**设计动机**：DFS 从高置信种子出发进行区域生长，只保留与种子连通且响应高于阈值的像素，有效抑制背景中的散射激活，生成纯净掩码。

#### 4. **可选精炼阶段**

使用 SAM 的 box prompt 后处理进一步提升掩码质量，实验证明 box prompt 效果最好。

### 损失函数 / 训练策略

无训练框架，无需损失函数。超参数：K=7（种子选择），τ=0.3（响应阈值），α=0.4（二值化阈值），扩散20步DDIM采样。在单卡 RTX-4090 上即可推理。

## 实验关键数据

### 主实验

**RRSIS-D 数据集（Test，带精炼）**：

| 方法 | 类型 | RSREC Pr@0.5 | RSREC mIoU | RSRES Pr@0.5 | RSRES mIoU |
|------|------|-------------|------------|-------------|------------|
| QueryMatch | 弱监督 | 16.22 | 17.21 | 15.54 | 15.73 |
| DiffSegmenter (w/ VLM) | 零样本 | 25.11 | 28.50 | 19.42 | 23.73 |
| DiffPNG (w/ VLM) | 零样本 | 21.29 | 24.89 | 17.64 | 20.99 |
| OV-VG | 零样本 | 16.20 | 21.62 | - | - |
| **RSVG-ZeroOV** | **零样本** | **31.39** | **34.49** | **27.39** | **28.35** |

**RISBench 数据集（Test，带精炼）**：

| 方法 | RSREC Pr@0.5 | RSREC mIoU | RSRES Pr@0.5 | RSRES mIoU |
|------|-------------|------------|-------------|------------|
| GroundVLP | 19.91 | 19.19 | 15.82 | 15.58 |
| OV-VG | 22.40 | 22.85 | 17.75 | 16.17 |
| **RSVG-ZeroOV** | **38.90** | **38.87** | **31.03** | **31.84** |

### 消融实验

| 配置 | RSREC Pr@0.5 | RSREC mIoU | RSRES Pr@0.5 | RSRES mIoU | 说明 |
|------|-------------|------------|-------------|------------|------|
| w/o VLM | 16.22 | 18.82 | 11.43 | 15.81 | 去掉VLM损失巨大 |
| w/o DM | 21.49 | 26.26 | 1.18 | 6.15 | 无DM则RSRES几乎失效 |
| w/o Evolve | 22.63 | 26.65 | 10.26 | 20.56 | 进化模块必要 |
| O-F-E（本文） | **30.15** | **32.92** | **12.84** | **21.85** | 最优顺序 |
| O-E-F | 27.34 | 29.51 | 7.18 | 15.89 | 顺序影响显著 |

**自注意力图分辨率消融**：

| 分辨率 | RSREC mIoU | RSRES mIoU | 说明 |
|--------|------------|------------|------|
| 32 | 31.97 | 21.11 | 单分辨率 |
| 64 | 30.76 | 20.13 | 单分辨率 |
| [32, 64]（本文） | **32.92** | **21.85** | 多尺度最优 |
| [16, 32, 64] | 30.51 | 20.36 | 过多反而下降 |

**交互策略对比（Tab. 3）**：

| 策略 | RSREC Pr@0.5 | RSRES mIoU | 说明 |
|------|-------------|------------|------|
| 锚点法 + Evolve | 28.73 | 16.58 | 过度简化 |
| 乘法 + Evolve | 29.26 | 20.75 | 中等 |
| 指数法 + Evolve | 27.38 | 14.00 | 过度放大 |
| **余弦相似度** + Evolve | **30.15** | **21.85** | 最优 |

### 关键发现

- **VLM 和 DM 缺一不可**：去掉 VLM 导致 RSREC mIoU 下降 14.10%；去掉 DM 导致 RSRES 从 21.85% 暴跌至 6.15%
- **O-F-E 顺序优于 O-E-F**：先聚焦（嵌入结构先验）再进化（区域生长）效果更好，反之结构信息被过早裁剪
- **通用 VLM > 遥感特化 VLM**：Qwen2.5-VL（通用）在零样本 RSREC 上达 28.66% Pr@0.5，优于 GeoChat（遥感特化，23.93%）
- **DM 自注意力 > VLM/SAM 自注意力**：DM 在 RSREC/RSRES 上分别达 30.15%/12.84% Pr@0.5，大幅领先其他自注意力来源
- 多尺度自注意力 [32, 64] 同时保持高分辨率细节和上下文语义

## 亮点与洞察

1. **首个零样本遥感视觉定位框架**：完全免训练即可在遥感场景工作，实用价值极高
2. **系统性探索实验总结的三条准则**非常有价值：每条准则都有实验支撑，为后续研究提供了清晰的指导
3. **DFS 区域生长**的进化策略简洁有效，无需学习参数即可抑制散射噪声
4. **多模型注意力互补**的发现可推广：VLM 提供语义但缺结构，DM 提供结构但缺语义，融合互补

## 局限与展望

- 零样本性能绝对值仍然有限（RSRES mIoU 仅约 28%），与全监督方法差距较大
- 扩散模型推理耗时（20 步 DDIM），整体推理速度受限
- DFS 区域生长的超参数（K, τ, α）需要手动调节，鲁棒性有限
- 仅在遥感 RSVG 上验证，未测试自然图像的泛化能力
- 对复杂空间关系描述（如多目标关系推理）的处理能力未明确评估

## 相关工作与启发

- VLM + DM 注意力融合的思路可推广到其他跨领域零样本分割任务
- DFS 区域生长可作为通用的注意力图后处理模块
- 三条准则为遥感基础模型的选用和组合提供了重要参考
- Overview-Focus-Evolve 的递进式流水线设计可启发更多多模态感知方法

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （首个零样本开放词汇遥感视觉定位框架，VLM+DM 注意力融合原创性强）
- 实验充分度: ⭐⭐⭐⭐⭐ （两个数据集、系统性探索实验、全面消融、多种基线对比）
- 写作质量: ⭐⭐⭐⭐⭐ （准则引导式论述清晰流畅，图表高质量）
- 价值: ⭐⭐⭐⭐ （免训练框架实用性强，但绝对性能仍有提升空间）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition](../../NeurIPS2025/segmentation/instructsam_a_training-free_framework_for_instruction-oriented_remote_sensing_ob.md)
- [\[CVPR 2026\] Direct Segmentation without Logits Optimization for Training-Free Open-Vocabulary Semantic Segmentation](../../CVPR2026/segmentation/direct_segmentation_without_logits_optimization_for_training-free_open-vocabular.md)
- [\[AAAI 2026\] Target Refocusing via Attention Redistribution for Open-Vocabulary Semantic Segmentation: An Explainability Perspective](target_refocusing_via_attention_redistribution_for_open-vocabulary_semantic_segm.md)
- [\[AAAI 2026\] LWGANet: Addressing Spatial and Channel Redundancy in Remote Sensing Visual Tasks with Light-Weight Grouped Attention](lwganet_addressing_spatial_and_channel_redundancy_in_remote_sensing_visual_tasks.md)
- [\[AAAI 2026\] S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing](s5_scalable_semi-supervised_semantic_segmentation_in_remote_sensing.md)

</div>

<!-- RELATED:END -->
