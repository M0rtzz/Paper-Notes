---
title: >-
  [论文解读] Unified Multimodal Visual Tracking with Dual Mixture-of-Experts
description: >-
  [ICML 2026][视频理解][视觉跟踪] OneTrackerV2 把 RGB / RGB+D / RGB+T / RGB+E / RGB+N 五种跟踪任务统一在一个网络里端到端训练…
tags:
  - "ICML 2026"
  - "视频理解"
  - "视觉跟踪"
  - "RGB+X"
  - "Mixture-of-Experts"
  - "特征解耦"
  - "模态缺失鲁棒"
---

# Unified Multimodal Visual Tracking with Dual Mixture-of-Experts

**会议**: ICML 2026  
**arXiv**: [2605.03716](https://arxiv.org/abs/2605.03716)  
**代码**: 无  
**领域**: 视频理解 / 多模态视觉跟踪 / Mixture-of-Experts  
**关键词**: 视觉跟踪、RGB+X、Mixture-of-Experts、特征解耦、模态缺失鲁棒

## 一句话总结
OneTrackerV2 把 RGB / RGB+D / RGB+T / RGB+E / RGB+N 五种跟踪任务统一在一个网络里端到端训练，靠 Meta Merger 做模态融合、Dual MoE 把"时空匹配"与"模态融合"两类异质特征显式拆到 T-MoE 与 M-MoE，并用 dissimilarity loss + router clustering 保证它们不塌成同一子空间。

## 研究背景与动机
**领域现状**：视觉目标跟踪按输入模态分成 RGB 与 RGB+X（X=Depth/Thermal/Event/Language）。主流路线有三条：(a) 每个 X 任务独立设计架构与训练；(b) OneTracker 这类对预训练 RGB 跟踪器做 fine-tuning 适配；(c) SUTrack 这类初步统一模型，把多模态 token 拼起来用同一 backbone。

**现有痛点**：(1) 多步训练，pretrained → finetune 通常收敛到次优；(2) 缺统一架构，仍要手工设计任务分支；(3) 共享架构里参数仍按任务分组，不是真的"unified params"；(4) 一旦推理时某种模态缺失，性能崩塌；(5) 特征冲突 — 简单 token 拼接迫使同一参数空间同时学时空运动匹配与模态特异 pattern，互相干扰。

**核心矛盾**：跟踪本质同时需要两类截然不同的能力：时空匹配（template ↔ search 跨帧运动）与模态融合（RGB ↔ X 的互补线索）。把它们塞进单一 backbone / 单 MoE 会形成 zero-sum 的参数竞争。

**本文目标**：(1) 单步端到端、共享参数、共享架构；(2) 把模态融合做成模态无关、对缺失鲁棒的"meta embedding"；(3) 用结构解耦解决时空匹配与模态融合的特征冲突；(4) 容量可扩展但推理成本不爆炸。

**切入角度**：用 learnable meta embedding 做模态中央枢纽；引入 Dual MoE，让两组专家分别承担时空与模态任务，通过显式 decoupling loss 强制它们正交。

**核心 idea**：Meta Merger + Dual MoE = 一个网络、一次训练、一套参数处理 5 种跟踪任务，并对模态缺失与模型压缩都保持鲁棒。

## 方法详解

### 整体框架
输入 template 与 search 区域，每个区域包含 RGB 与某个 X 模态帧（RGB-only 任务把 X 帧用 RGB 自身代替）。两路通过共享 patch embedding 得到 $F_{rgb},F_x$，经 Meta Merger 用 learnable meta embedding $F_{meta}$ 做空间 + 通道 attention + 中心化卷积融合，得到模态无关 token 序列。该序列送入 Vision Transformer backbone，其中每个 block 用 Dual MoE 替换 FFN：每个 token 同时通过 shared expert、T-MoE（top-$k$）与 M-MoE（top-$k$）三路计算并相加。最后接 SUTrack 风格的 classification + IoU + L1 检测头输出 bbox。整套架构提供 B224 / B384 / L224 / L384 四个版本，参数 80M–271M、推理 FPS 23.4–72.4。

### 关键设计

1. **Meta Merger：模态无关中央枢纽**：

    - 功能：把异质的 RGB 与 X 模态特征压缩到统一空间，同时提供对模态缺失的天然鲁棒性。
    - 核心思路：先对 $F_{rgb}$ 与 $F_x$ 各自计算 $W^{spatial}=\sigma(\mathrm{Conv}(F^{avg})+\mathrm{Conv}(F^{max}))$ 与 $W^{channel}=\sigma(\mathrm{Linear}(F^{avg})+\mathrm{Linear}(F^{max}))$ 增强；再引入可学习 $F_{meta}$，通过 $F_{meta}'=\mathrm{Conv}(\mathrm{Conv}(F_{meta}+F'_{rgb})+\mathrm{Conv}(F_{meta}+F'_x)+F_{meta})$ 让 meta embedding 充当跨模态中介，最终输出全局对齐的 token；当 X 缺失时退化为只与 RGB 交互，融合管道无需修改。
    - 设计动机：相比 SUTrack 直接 token 拼接，meta embedding 既避免多分支带来的双倍计算，又用一个全局变量做"模态翻译官"，自然适配任意模态组合。

2. **Dual MoE：T-MoE 与 M-MoE 显式解耦**：

    - 功能：把时空匹配与模态融合分到两组独立专家，避免同一参数空间承担异质目标。
    - 核心思路：对每个 token $x$，DMoE 输出 $y=E_{shared}(x)+\sum_{i\in S^T_k}\hat g_i^T(x)E_i^T(x)+\sum_{i\in S^M_k}\hat g_i^M(x)E_i^M(x)$，其中 $S^T_k,S^M_k$ 为 top-$k$ 专家集合，$\hat g$ 是 renormalize 后的 softmax 权重；每个 expert 实现为"投影到秩 $r$ → 非线性 → 投影回 $d$"，capacity 高但成本可控。再加上 expert decoupling loss $\mathcal L_{dis}=(\cos(y^T,y^M))^2$ 强制两路输出正交。
    - 设计动机：跟踪需要 temporal consistency，因此 T-MoE 一旦被 push away 出 M-MoE 的子空间，就会自然吸引到 motion 特征；M-MoE 则吸收模态特异信号。表 4 显示 D-MoE 显著优于单 MoE，证明拆解必要。

3. **Multimodal Router Cluster：让 M-MoE 真按模态分工**：

    - 功能：让 M-MoE 的路由 logits 在同模态样本间保持高相似度、跨模态低相似度，从而真正学到模态特异的 expert 选择策略。
    - 核心思路：用 batch 内路由相似度矩阵 $S_{ij}=\langle g^M(x_i),g^M(x_j)\rangle$ 配合 margin $\delta$，构造 $\mathcal L_{same}=\frac{1}{|M_{same}|}\sum_{(i,j)\in M_{same}}\max(0,(1/K+\delta)-S_{ij})$ 与 $\mathcal L_{diff}=\frac{1}{|M_{diff}|}\sum_{(i,j)\in M_{diff}}\max(0,S_{ij}-(\delta-1/K))$，合成 $\mathcal L_{cluster}=\mathcal L_{same}+\mathcal L_{diff}$。
    - 设计动机：仅靠 $\mathcal L_{dis}$ 能让 T/M 输出正交但不保证 M-MoE 内部按模态分簇；router cluster 损失提供模态级别的层次化偏好，使得某些 expert 专责 Depth、某些专责 Thermal 等。

### 损失函数 / 训练策略
总损失 $\mathcal L=\mathcal L_{class}+\lambda_G\mathcal L_{IoU}+\lambda_{L_1}\mathcal L_{L_1}+\mathcal L_{task}+\lambda_{dis}\mathcal L_{dis}+\lambda_{cluster}\mathcal L_{cluster}+\lambda_{balance}\mathcal L_{balance}$，默认 $\lambda_G\!=\!2,\lambda_{L_1}\!=\!5,\lambda_{dis}\!=\!0.1,\lambda_{cluster}\!=\!1$；$\mathcal L_{balance}$ 用于约束 MoE 负载均衡。整网一次端到端训练，无 pretrain → finetune 多阶段。

## 实验关键数据

### 主实验

| 任务 / 基准 | 指标 | OneTrackerV2-L384 | SUTrack-L384 (强 baseline) | 说明 |
|-------------|------|--------------------|----------------------------|------|
| LaSOT | AUC | 76.1 | 75.2 | 长时单目标，统一架构仍领先 |
| LaSOT_ext | AUC | 55.2 | 53.6 | OOD class 上提升明显 |
| TrackingNet | AUC / P | 88.6 / 89.0 | 87.7 / 88.7 | 大规模在线跟踪 |
| GOT-10k | AO | 81.3 | 81.5 | 同档，但用统一参数 |
| UAV123 | AUC | 71.0 | 70.4 | 无人机视角 |
| 模型规格 | Params (M) / FLOPs (G) / FPS | 80.2 / 23.8 / 72.4 (B224) | — | DMoE 几乎不增加成本 |

### 消融实验

| 设计 | 关键观察 | 解读 |
|------|----------|------|
| 完整 OneTrackerV2 | 在 5 任务 12 基准全 SOTA | 单模型即可统一 RGB + RGB+X |
| 去 Dual MoE / 用 single MoE | 显著掉点（Table 4 中 D-MoE 优于 single MoE） | 异质目标必须显式解耦 |
| 去 $\mathcal L_{dis}$ | T-MoE / M-MoE 输出相似度上升、性能下降 | 正交约束是 decoupling 关键 |
| 去 router cluster | M-MoE 退化为通用 FFN，跨模态泛化变差 | 模态特异 expert 选择能力丧失 |
| 缺失模态推理 | 性能仍稳定，远好于 SUTrack | Meta Merger 提供模态鲁棒 |
| 模型压缩 | 压缩后仍保留主要精度 | DMoE 结构性冗余允许稀疏化 |

### 关键发现
- T-MoE 的 expert 选择模式与目标运动强度高度相关（Fig. 5），证明它确实学到 motion-related 特征；M-MoE 不同 expert 对不同 X 模态有明显偏好，证明 router cluster 有效。
- 单 MoE 试图同时承担两种任务时会塌缩成生成性强但区分弱的特征提取器；解耦后两组 expert 各司其职，性能 + 鲁棒性同时上升。
- 在模型压缩与模态缺失这两个工程关键场景，OneTrackerV2 优势显著扩大，说明 unified + decoupled 设计天然具备鲁棒性 budget。

## 亮点与洞察
- 把"特征冲突"显式作为优化目标：用 $\cos^2$ dissimilarity 这一最简单的正交化损失就让 dual MoE 各自专精，是非常 ROI 高的设计。
- Router cluster 提供模态层级的 inductive bias：把"路由相似度"作为可观察变量直接施加 margin 损失，比 expert capacity loss 更精确地约束路由行为。
- Meta embedding 作为"模态中介"对模态缺失天然鲁棒——它是一个广泛适用的设计模式（可迁移到 RGB+X 检测 / 分割 / 多模态推理）。
- 单阶段训练 + 共享参数 + 12 个基准 SOTA，是 multimodal tracking 当前最有"工业落地味"的方案之一。

## 局限与展望
- 仍依赖 ImageNet 风格 ViT backbone；对纯事件 / 雷达 / 点云这类与 RGB 域差距更大的模态是否仍能 plug-and-play 未充分讨论；
- DMoE 把 FFN 替换成多专家，虽然 FLOPs 增加有限，但显存与训练时间显著增加，对小团队不友好；
- 文章用 dissimilarity 与 router cluster 两个手动权重，缺乏自动调度（如根据任务难度动态调权重）；
- 多模态训练数据仍按任务汇总，未充分讨论跨任务正负迁移。

## 相关工作与启发
- **vs SUTrack (Chen et al. 2025)**：SUTrack naive token concat，模态缺失场景崩；OneTrackerV2 用 Meta Merger 中央枢纽 + DMoE 显式解耦，全面超越。
- **vs OneTracker (Hong et al. 2024)**：原作走 pretrain → finetune 路径，仍依任务分组参数；本作真正 unified params + 一次训练。
- **vs MoE 跟踪器 (Tan et al. 2025, Cai et al. 2025)**：他们用 MoE 仅做容量扩展或域适配；本作把 MoE 用作"任务解耦的结构容器"，是 MoE 在跟踪中较新颖的用法。
- **vs OneTracker / SUTrack 的多模态融合**：本作的 Meta Merger 是一个泛用模块，可迁移到任意需要"主模态 + 辅助模态"的检测 / 分割任务。

## 评分
- 新颖性: ⭐⭐⭐⭐ Dual MoE + router cluster 把"特征冲突"问题做成结构解 — 在跟踪里算清新颖。
- 实验充分度: ⭐⭐⭐⭐⭐ 5 任务 12 基准 + 4 个模型规格 + 模型压缩 + 模态缺失 + 多消融，覆盖度极高。
- 写作质量: ⭐⭐⭐⭐ 图示清晰、损失公式整齐，能看清每个设计为何存在。
- 价值: ⭐⭐⭐⭐ 是当下 multimodal tracking 最具实用价值的 unified baseline，结构上的模态枢纽 + dual MoE 模式可外推到其他多模态视觉任务。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss](../../ICLR2026/video_understanding/coupling_experts_and_routers_in_mixture-of-experts_via_an_auxiliary_loss.md)
- [\[ICML 2026\] RELO: Reinforcement Learning to Localize for Visual Object Tracking](relo_reinforcement_learning_to_localize_for_visual_object_tracking.md)
- [\[CVPR 2026\] UTPTrack: Towards Simple and Unified Token Pruning for Visual Tracking](../../CVPR2026/video_understanding/utptrack_towards_simple_and_unified_token_pruning_for_visual_tracking.md)
- [\[ECCV 2024\] Occluded Gait Recognition with Mixture of Experts: An Action Detection Perspective](../../ECCV2024/video_understanding/occluded_gait_recognition_with_mixture_of_experts_an_action_detection_perspectiv.md)
- [\[CVPR 2026\] Drift-Resilient Temporal Priors for Visual Tracking](../../CVPR2026/video_understanding/drift-resilient_temporal_priors_for_visual_tracking.md)

</div>

<!-- RELATED:END -->
