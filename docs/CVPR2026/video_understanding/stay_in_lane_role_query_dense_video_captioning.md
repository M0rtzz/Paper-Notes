---
title: >-
  [论文解读] Stay in your Lane: Role Specific Queries with Overlap Suppression Loss for Dense Video Captioning
description: >-
  [CVPR 2026][视频理解][密集视频描述] ROS-DVC为DETR-based密集视频描述设计角色专用查询（定位和描述独立初始化）、跨任务对比对齐损失和重叠抑制损失三个互补组件，无需预训练或LLM即在YouCook2上CIDEr达39.18，超越使用GPT-2的DDVC。
tags:
  - CVPR 2026
  - 视频理解
  - 密集视频描述
  - 角色专用查询
  - 重叠抑制
  - DETR
  - 跨任务对比对齐
---

# Stay in your Lane: Role Specific Queries with Overlap Suppression Loss for Dense Video Captioning

**会议**: CVPR 2026  
**arXiv**: [2603.11439](https://arxiv.org/abs/2603.11439)  
**代码**: [github.com/MMAI-Konkuk/ROS-DVC](https://github.com/MMAI-Konkuk/ROS-DVC)  
**领域**: 视频理解  
**关键词**: 密集视频描述, 角色专用查询, 重叠抑制, DETR, 跨任务对比对齐

## 一句话总结

ROS-DVC为DETR-based密集视频描述设计角色专用查询（定位和描述独立初始化）、跨任务对比对齐损失和重叠抑制损失三个互补组件，无需预训练或LLM即在YouCook2上CIDEr达39.18，超越使用GPT-2的DDVC。

## 研究背景与动机

**领域现状**：密集视频描述（DVC）需要同时定位视频中的多个事件时间段并为每段生成自然语言描述。PDVC开创性地将DETR架构引入DVC，实现端到端联合优化。后续方法（CM2、MCCL、E2DVC、DDVC）在此基础上持续改进。

**现有痛点**：DETR-based DVC使用共享的可学习查询同时驱动定位和描述，产生两个严重问题：(1) **多任务干扰**——定位需要广泛的时间上下文来预测精确边界，描述需要密集关注关键帧的语义细节，共享查询的注意力分布模糊不清（Figure 1b上方）；(2) **预测重叠**——多个查询捕获高度重叠的时间段，产生冗余描述（Figure 1a）。

**核心矛盾**：查询角色的统一性与子任务需求的差异性之间的根本冲突——单个查询无法同时最优化时间定位和语义描述两个方向。

**本文目标** 让查询"各司其职"——定位查询专注时间边界，描述查询专注语义内容，同时减少预测的时间重叠。

**切入角度**：从查询初始化和损失函数两个层面入手——物理分离查询空间+对比对齐保持一致性+重叠惩罚减少冗余。

**核心 idea**：将DETR查询分为独立初始化的定位和描述两组，用对比损失桥接两组查询的语义一致性，用IoU惩罚抑制预测重叠。

## 方法详解

### 整体框架

视频帧 → 预训练CLIP ViT-L/14提取特征 → Transformer编码器生成帧级特征 → DETR解码器（接收两组独立初始化的角色专用查询）→ 定位查询经Hungarian匹配后输出事件时间段，描述查询经CTCA对齐后输出事件描述文本。定位查询还受OSL约束以减少重叠。

### 关键设计

1. **角色专用查询初始化（Role-Specific Query Initialization）**:
    - 功能：将标准DETR的单一查询集分为定位查询 $\{q_{\text{loc}}^j\}_{j=1}^K$ 和描述查询 $\{q_{\text{cap}}^j\}_{j=1}^K$，各自从独立的嵌入空间初始化
    - 核心思路：两组查询在解码器cross-attention层中共享视觉定位（引用相同视觉位置的reference point），但保持各自表示空间的独立性。定位查询学习广泛关注时间上下文以预测边界，描述查询学习密集关注关键帧以捕获语义。与DDVC不同（用MLP从定位查询派生描述查询），本文是真正的物理分离
    - 设计动机：完全独立的嵌入空间允许每组查询被各自的目标函数独立优化，避免梯度方向冲突。Figure 1b可视化证实分离后的注意力分布确实呈现差异化模式

2. **跨任务对比对齐损失（CTCA Loss）**:
    - 功能：确保对应位置的定位查询和描述查询指向同一事件的语义内容
    - 核心思路：Hungarian匹配后，对于匹配到GT的索引集 $\mathcal{M}$，将 $(q_{\text{cap}}^j, q_{\text{loc}}^j)$ 视为正对，$(q_{\text{cap}}^j, q_{\text{loc}}^{j'})$ 为负对。损失函数 $\mathcal{L}_{\text{CTCA}}=-\sum_{j\in\mathcal{M}}\log\frac{\exp(\text{sim}(\tilde{q}_{\text{cap}}^j,\tilde{q}_{\text{loc}}^j)/\tau)}{\sum_{j'}\exp(\text{sim}(\tilde{q}_{\text{cap}}^j,\tilde{q}_{\text{loc}}^{j'})/\tau)}$
    - 设计动机：查询空间分离后语义一致性不再自动保证，CTCA通过对比学习显式桥接定位和描述查询，使定位查询也获得语义感知能力

3. **重叠抑制损失（Overlap Suppression Loss, OSL）**:
    - 功能：惩罚预测事件之间的过度时间重叠，减少冗余预测
    - 核心思路：基于预测边界 $B_i,B_j$ 的成对时间IoU $P_o(i,j)$，引入GT对齐权重 $\alpha=\gamma\cdot P_g+(1-\gamma)\cdot(1-P_g)$（$\gamma\leq0.5$），最终损失 $L_{\text{OSL}}=-\alpha\cdot\log(\beta-P_o)$。与GT高度匹配的预测受较小惩罚（$P_g$ 大 → $\alpha$ 小），避免误抑制真正的连续事件
    - 设计动机：直接在训练时优化比NMS后处理更有效；GT调制惩罚区分了"与GT对应的合理重叠"和"冗余的无效重叠"

### 损失函数 / 训练策略

总损失 $\mathcal{L}_{\text{total}}=\lambda_{\text{giou}}\mathcal{L}_{\text{giou}}+\lambda_{\text{cls}}\mathcal{L}_{\text{cls}}+\lambda_{\text{cap}}\mathcal{L}_{\text{cap}}+\lambda_{\text{ec}}\mathcal{L}_{\text{ec}}+\lambda_{\text{CTCA}}\mathcal{L}_{\text{CTCA}}+\lambda_{\text{OSL}}\mathcal{L}_{\text{OSL}}+\lambda_{\text{CG}}\mathcal{L}_{\text{CG}}$。其中 $\mathcal{L}_{\text{CG}}$ 为Concept Guider的辅助交叉熵损失（推理时不用）。超参数 $\gamma=0.25$, $\beta=1.0$, $N_C=30$。2层deformable transformer解码器，YouCook2用50组查询，ActivityNet用10组。

## 实验关键数据

### 主实验

| 数据集 | 指标 | ROS-DVC | DDVC(GPT-2) | MCCL | E2DVC | PDVC |
|--------|------|---------|-------------|------|-------|------|
| YouCook2 | CIDEr↑ | **39.18** | 38.75 | 36.09 | 34.26 | 29.69 |
| YouCook2 | SODA_c↑ | **7.06** | 6.68 | 5.21 | 5.39 | 4.92 |
| YouCook2 | BLEU4↑ | **2.10** | 1.92 | 2.04 | 1.68 | 1.40 |
| ActivityNet | CIDEr↑ | **35.04** | — | 34.92 | 33.63 | 29.97 |
| ActivityNet | SODA_c↑ | **6.45** | — | 6.16 | 6.13 | 5.92 |

| 数据集 | 定位指标 | ROS-DVC | E2DVC | PDVC |
|--------|---------|---------|-------|------|
| YouCook2 | Recall↑ | **29.34** | 24.36 | 22.89 |
| YouCook2 | F1↑ | **32.03** | 28.64 | 26.81 |
| ActivityNet | Recall↑ | **55.35** | 54.67 | 53.27 |

### 消融实验

| 配置 | CIDEr | 说明 |
|------|-------|------|
| 基线 (E2DVC) | 34.26 | 共享查询 |
| + 角色分离 | 36.14 (+1.88) | 查询解耦本身有效 |
| + 角色分离 + CTCA | 37.92 (+3.66) | 跨任务对齐保持语义一致 |
| + 角色分离 + CTCA + OSL | **39.18 (+4.92)** | 重叠抑制进一步减少冗余 |
| OSL无GT调制 (固定惩罚) | ~38.4 | GT调制避免误抑制合理重叠 |

### 关键发现

- 三个组件贡献递增（+1.88, +1.78, +1.26），组合效果最优（+4.92）
- 角色分离比共享查询+CTCA（软约束）更有效，说明表示空间的物理分离优于软对齐
- Recall与Precision近乎匹配——事件计数器预测的事件数更接近真实值
- 无需LLM即超越使用GPT-2的DDVC（CIDEr +0.43），证明方法的轻量高效

## 亮点与洞察

- "让查询各做各的"思路朴素而有效——从DETR查询设计层面解决DVC的多任务干扰问题
- OSL的GT调制设计精巧——用 $\alpha$ 自适应惩罚强度，区分合理重叠与冗余重叠
- Concept Guider是无开销的辅助增强——训练时用MLP预测事件概念向量enriches查询表示，推理时去掉
- 不依赖外部记忆库或LLM，方法轻量且可迁移

## 局限与展望

- 仅在YouCook2和ActivityNet上验证，未测试更长或更复杂的视频场景
- 角色分离将查询参数翻倍（2K vs K），大查询集时可能有额外开销
- CTCA用全局对比可能对极短或极长事件的区分不够敏感，可探索时间感知的对比策略
- 未与最新的LLM-based DVC方法（如用LLaMA的）充分对比
- Concept Guider的概念词表($N_C=30$)固定，对域外视频的泛化有待验证

## 相关工作与启发

- **vs PDVC**: 共享查询的DVC先驱，ROS-DVC在其基础上解耦查询+加损失约束，CIDEr提升+9.49
- **vs DDVC**: 用GPT-2做描述生成达CIDEr 38.75，ROS-DVC无需LLM即超越（39.18），证明查询设计比模型容量更重要
- **vs E2DVC**: 改进的端到端DVC基线，ROS-DVC在其上CIDEr +4.92
- **vs MCCL**: 使用外部记忆库增强描述多样性，ROS-DVC不需要额外记忆即达更高CIDEr

## 评分

- 新颖性: ⭐⭐⭐⭐ 查询角色分离+OSL GT调制组合设计新颖，三个组件互补递增
- 实验充分度: ⭐⭐⭐⭐ 两个标准数据集、逐组件递增消融、多基线多指标对比
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，"Stay in your Lane"标题贴切，方法图示直观
- 价值: ⭐⭐⭐⭐ 对DVC有直接实用改进，查询角色分离思路可迁移到其他DETR多任务架构

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] SAIL: Similarity-Aware Guidance and Inter-Caption Augmentation-based Learning for Weakly-Supervised Dense Video Captioning](sail_similarity-aware_guidance_and_inter-caption_augmentation-based_learning_for.md)
- [\[CVPR 2026\] FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking](fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)
- [\[CVPR 2026\] VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning](videochatm1_collaborative_policy_planning_for_vide.md)
- [\[CVPR 2026\] CVA: Context-aware Video-text Alignment for Video Temporal Grounding](cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)
- [\[CVPR 2026\] STORM: End-to-End Referring Multi-Object Tracking in Videos](storm_referring_multi_object_tracking.md)

<!-- RELATED:END -->
