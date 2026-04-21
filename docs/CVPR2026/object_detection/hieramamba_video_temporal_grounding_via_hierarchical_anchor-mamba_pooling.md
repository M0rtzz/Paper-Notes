---
title: >-
  [论文解读] HieraMamba: Video Temporal Grounding via Hierarchical Anchor-Mamba Pooling
description: >-
  [CVPR 2026][目标检测][视频时间定位] HieraMamba 提出了基于 Mamba 的层次化视频时间定位架构，核心是 Anchor-MambaPooling（AMP）模块，用 Mamba 的选择性扫描将视频特征逐层压缩为多尺度锚点 token，配合 anchor-conditioned 和 segment-pooled 对比损失增强层次表示的紧凑性和判别性，在 Ego4D-NLQ、MAD 和 TACoS 上达到 SOTA。
tags:
  - CVPR 2026
  - 目标检测
  - 视频时间定位
  - 状态空间模型
  - Mamba
  - 层次化表示
  - 对比学习
---

# HieraMamba: Video Temporal Grounding via Hierarchical Anchor-Mamba Pooling

**会议**: CVPR 2026  
**arXiv**: [2510.23043](https://arxiv.org/abs/2510.23043)  
**代码**: https://vision.cs.utexas.edu/projects/hieramamba  
**领域**: 视频理解  
**关键词**: 视频时间定位, 状态空间模型, Mamba, 层次化表示, 对比学习

## 一句话总结

HieraMamba 提出了基于 Mamba 的层次化视频时间定位架构，核心是 Anchor-MambaPooling（AMP）模块，用 Mamba 的选择性扫描将视频特征逐层压缩为多尺度锚点 token，配合 anchor-conditioned 和 segment-pooled 对比损失增强层次表示的紧凑性和判别性，在 Ego4D-NLQ、MAD 和 TACoS 上达到 SOTA。

## 研究背景与动机

1. **领域现状**：视频时间定位（Video Temporal Grounding）任务要求根据自然语言查询在未裁剪视频中定位起止时间。从预定义动作定位发展到自由文本查询，该任务支持 VQA、自动视频编辑等应用。现有方法如 ActionFormer、SnAG、DeCafNet 等已引入多尺度特征金字塔。

2. **现有痛点**：长视频（几分钟到数小时）带来两个交织的挑战：(a) **时间保真度问题**——许多方法通过固定长度池化、朴素下采样或固定窗口来降低计算成本，但这些操作丢弃了关键的时间线索或在窗口边界处割裂了时间结构；(b) **多粒度问题**——不同查询需要不同时间粒度（"侦探在图书馆做了什么"需要粗粒度理解，"侦探何时从书架抽出纸条"需要细粒度定位），单分辨率方法难以兼顾。

3. **核心矛盾**：Transformer 的二次方注意力成本是导致下采样和窗口化启发式方法的根源——为了处理长序列不得不牺牲时间分辨率。现有的多尺度模型（SnAG、DeCafNet、OSGNet）虽然引入了多尺度，但多尺度仍通过均匀下采样或粗池化产生，缺乏内容感知的压缩。

4. **本文目标** (1) 如何在线性时间复杂度内处理全长视频序列，避免下采样/窗口化？(2) 如何构建内容感知的多尺度层次表示，而非简单的下采样金字塔？(3) 如何确保层次锚点既紧凑（忠实汇总局部信息）又具判别性（与其他事件可区分）？

5. **切入角度**：人类的情景记忆天然具有层次结构——从房间的整体布局到手指的精确运动，可以无缝在时间尺度间切换。作者识别出 Transformer 的二次注意力是导致时间降采样的根源，提出用 Mamba 的线性时间选择性扫描替代，实现全分辨率的长程建模。

6. **核心 idea**：用 Mamba 的选择性扫描构建内容感知的层次化锚点压缩（而非朴素下采样），通过堆叠 AMP 模块形成从细到粗的多尺度时间金字塔，以线性复杂度实现精确的长视频时间定位。

## 方法详解

### 整体框架

输入：冻结的视频骨干（如 EgoVLP）提取的 clip 级特征 $V \in \mathbb{R}^{L_V \times D_v}$ 和冻结文本编码器（如 CLIP）提取的查询嵌入 $Q \in \mathbb{R}^{L_Q \times D_q}$。视频编码器是 $L$ 个 AMP 模块的层次堆叠，逐层产生精炼特征 $\tilde{V}^{(l)}$ 和下一层的锚点 $A^{(l+1)}$，形成特征金字塔 $\mathcal{V}_{\text{pyr}} = \{\tilde{V}^{(0)}, \ldots, \tilde{V}^{(L-1)}\}$。金字塔与文本嵌入通过跨模态注意力融合后，由轻量解码器回归起止时间 $(t_s, t_e)$。

### 关键设计

1. **Anchor-MambaPooling (AMP) 模块**:

    - 功能：在每一层同时完成当前分辨率的特征精炼和到下一层的内容感知压缩
    - 核心思路：三步流程——(a) **锚点生成与交错**：每隔 $s$ 帧初始化一个锚点 token（通过局部窗口池化），然后将锚点插入到它汇总的帧之前，形成交错序列 $\hat{V} = [a_0, v_0, \ldots, v_{s-1}, a_1, v_s, \ldots] \in \mathbb{R}^{(L_0+L_1) \times D_v}$；(b) **全局编码**：用 Hydra（双向 Mamba 扫描）处理交错序列，实现线性复杂度的全局上下文建模——正向扫描让锚点从前序帧接收信息，反向扫描让锚点从后续帧接收信息；(c) **局部编码**：用窄窗口 Transformer（窗口大小 5）补充短程细粒度注意力模式。最终输出精炼的当前层特征 $\tilde{V}^{(l)}$ 和压缩的下一层锚点 $A^{(l+1)}$
    - 设计动机：交错设计使锚点和帧特征共享同一个 Mamba 扫描，锚点能广播粗粒度上下文给邻近帧，帧特征能提供细节信息精炼锚点——双向信息流。这与传统特征金字塔的关键区别在于：AMP 通过 token 级压缩而非朴素下采样来产生多尺度表示，实现内容感知抽象

2. **设计细节：门控融合与解耦**:

    - 功能：控制信息在层次间的传播质量
    - 核心思路：全局编码、局部编码和 FFN 之间用 RMS 归一化 + 残差连接。阶段之间用可学习的 sigmoid 门控 $\boldsymbol{\sigma}$ 替代无条件的残差加法，提供内容自适应的信息传播控制
    - 设计动机：Mamba 捕获全局结构、窄窗口 Transformer 捕获局部模式——两者角色显式解耦，避免了混合架构中角色模糊的问题。门控确保只有显著信息沿层次上传

3. **Anchor-Conditioned Contrastive (ACC) 损失**:

    - 功能：自监督目标，确保锚点紧凑且有区分度
    - 核心思路：在每一层，将每个锚点 $a_i^{(l+1)}$ 与它汇总的 $s$ 个帧 token（正样本 $\mathcal{P}_i^{(l)}$）拉近，与远处锚点（负样本 $\mathcal{N}_i^{(l)}$，有时间间隔以避免惩罚相邻锚点）推远：$\mathcal{L}_{\text{acc}}(a_i^{(l+1)}) = -\log \frac{\sum_{p \in \mathcal{P}_i^{(l)}} \exp(a_i^{(l+1)} \cdot p / \tau)}{\sum_{c \in \mathcal{P}_i^{(l)} \cup \mathcal{N}_i^{(l)}} \exp(a_i^{(l+1)} \cdot c / \tau)}$
    - 设计动机：紧凑性要求锚点忠实汇总其局部窗口（与窗口内帧对齐），区分度要求不同锚点代表不同事件（与远处锚点分离）。多正样本设计避免了单正样本可能导致的信息损失

4. **Segment-Pooled Contrastive (SPC) 损失**:

    - 功能：有监督目标，将 GT 段落的表示与周围非目标内容区分
    - 核心思路：在每一层，将 GT 段落 $[t_{\text{start}}, t_{\text{end}})$ 内的帧 token 池化为段落原型 $z_{\text{seg}}^{(l)}$，以段落内的帧作为正样本、段落外的帧作为负样本进行对比。池化后的原型作为正锚点而非单帧，避免段落内不同子动作（如"伸手→抓取→收回"）被强制对齐到同一表示
    - 设计动机：ACC 提供结构级一致性（层次内自监督），SPC 提供语义级对齐（与查询标注对齐）。两者互补：ACC 确保锚点质量，SPC 确保锚点与查询语义匹配

### 训练策略

总对比损失 $\mathcal{L}_{\text{contrast}} = \lambda_{\text{ACC}} \mathcal{L}_{\text{ACC}} + \lambda_{\text{SPC}} \mathcal{L}_{\text{SPC}}$，与标准的时间定位任务损失（边界回归 + 分类）联合优化。

## 实验关键数据

### 主实验

在 Ego4D-NLQ（使用 EgoVLP 特征）上的结果：

| 方法 | R@1 IoU=0.3 | R@1 IoU=0.5 | R@5 IoU=0.3 | R@5 IoU=0.5 | Avg. |
|------|------------|------------|------------|------------|------|
| SnAG | 15.72 | 10.78 | 38.39 | 27.44 | 23.08 |
| DeCafNet | 18.10 | 12.55 | 38.85 | 28.27 | 24.44 |
| RGNet | 18.28 | 12.04 | 34.02 | 22.89 | 21.81 |
| OSGNet | 16.13 | 11.28 | 36.78 | 25.63 | 22.46 |
| **HieraMamba** | **18.81** | **13.04** | **40.82** | **29.96** | **25.66** |

在 MAD 和 TACoS 上也达到 SOTA（论文报告了详细数据）。

### 方法特性对比

| 方法 | 朴素下采样 | 固定池化 | 二次方代价 | 滑动窗口 | Ego4D Avg.R |
|------|-----------|---------|-----------|---------|------------|
| 2D-TAN | ✓ | ✓ | ✓ | — | 6.46 |
| CONE | — | — | ✓ | ✓ | 17.67 |
| SnAG | ✓ | — | — | — | 23.08 |
| DeCafNet | ✓ | — | — | — | 24.44 |
| **HieraMamba** | **—** | **—** | **—** | **—** | **25.66** |

HieraMamba 是唯一一个同时避免了所有四种不良特性的方法。

### 关键发现

- 避免下采样和窗口化的好处在长视频上尤为明显——HieraMamba 在 Ego4D（8 分钟平均）和 MAD（数小时电影）上提升最大
- ACC 和 SPC 损失的贡献互补——ACC 主要提升层次内的锚点质量和一致性，SPC 主要提升与查询的语义对齐（消融实验见附录）
- Mamba + 窄窗口 Transformer 的全局-局部解耦效果优于纯 Mamba 或纯 Transformer
- 门控机制（sigmoid gate）优于无条件残差连接，说明内容自适应的信息传播对层次化模型很重要

## 亮点与洞察

- **AMP 的交错设计极为巧妙**：将锚点 token 插入到帧序列中一起参与 Mamba 扫描，既让锚点自然获得全局上下文总结能力（Mamba 的状态压缩），又让帧特征从锚点获得邻域摘要信息——一次扫描完成双向信息流，计算代价仅为序列长度的线性增长
- **"避免所有不良特性"的方法论**：通过系统性地分析现有方法的四种限制（下采样、固定池化、二次代价、滑动窗口），设计出一个同时规避所有问题的架构，体现了优雅的工程设计思路
- **ACC 损失的多正样本设计**：传统对比学习用单正样本，但在时间定位中一个锚点需要忠实代理多帧内容，多正样本 InfoNCE 自然地适配了这种需求

## 局限与展望

- 依赖冻结的视频骨干（EgoVLP/InternVideo），如果骨干提取的 clip 特征质量不足，后续层次化建模也无法弥补
- AMP 的步长 $s$ 是固定的超参数，对不同时间尺度的查询可能需要不同的步长——自适应步长值得探索
- Mamba 的单向因果结构需要通过双向 Hydra 补偿，这增加了复杂度——是否有更原生的双向 SSM 设计
- 论文未讨论推理速度——虽然理论上是线性复杂度，但 AMP 的交错、双向扫描和多层堆叠的实际速度需要验证
- 对比损失中的温度 $\tau$ 和负样本选择策略的敏感性分析不够充分

## 相关工作与启发

- **vs ActionFormer**: ActionFormer 首先引入时间特征金字塔，但通过 stride pooling 构建——信息有损。HieraMamba 用 Mamba 扫描替代池化，实现内容感知压缩
- **vs SnAG / DeCafNet / OSGNet**: 都是近期强基线，但仍然依赖均匀下采样来构建多尺度。HieraMamba 证明了通过 learned token 压缩打败了这些方法
- **vs CONE / RGNet**: 使用固定窗口的滑动窗口方法，SnAG 窗口边界破坏时间连续性。HieraMamba 用 Mamba 的全局状态避免了这个问题
- 这篇工作启发了一个方向：SSM 不仅可以作为 Transformer 的效率替代，还能作为学习层次压缩的工具

## 评分

- 新颖性: ⭐⭐⭐⭐ AMP 模块的交错扫描设计和双对比损失都有明确新意
- 实验充分度: ⭐⭐⭐⭐ 三个基准 SOTA，方法特性对比系统全面
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，对比表格设计巧妙，图示直观易懂
- 价值: ⭐⭐⭐⭐ 为长视频时间定位提供了清晰的范式——线性复杂度 + 层次化内容压缩

<!-- RELATED:START -->

## 相关论文

- [SHINE: Saliency-aware HIerarchical NEgative Ranking for Compositional Temporal Grounding](../../ECCV2024/object_detection/shine_saliency-aware_hierarchical_negative_ranking_for_compositional_temporal_gr.md)
- [Sparse-Dense Side-Tuner for Efficient Video Temporal Grounding](../../ICCV2025/object_detection/sparse-dense_side-tuner_for_efficient_video_temporal_grounding.md)
- [TubeRMC: Tube-conditioned Reconstruction with Mutual Constraints for Weakly-supervised Spatio-Temporal Video Grounding](../../AAAI2026/object_detection/tubermc_tube-conditioned_reconstruction_with_mutual_constraints_for_weakly-super.md)
- [DA-Mamba: Learning Domain-Aware State Space Model for Global-Local Alignment in Domain Adaptive Object Detection](da-mamba_learning_domain-aware_state_space_model_for_global-local_alignment_in_d.md)
- [Drift-Resilient Temporal Priors for Visual Tracking](drift-resilient_temporal_priors_for_visual_tracking.md)

<!-- RELATED:END -->
