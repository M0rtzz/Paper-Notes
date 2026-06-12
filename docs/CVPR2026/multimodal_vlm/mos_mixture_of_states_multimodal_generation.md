---
title: >-
  [论文解读] Mixture of States (MoS): Routing Token-Level Dynamics for Multimodal Generation
description: >-
  [CVPR 2026][多模态VLM][多模态融合] 提出Mixture of States (MoS)——一种新的多模态扩散模型融合范式，用可学习的token级路由器将理解塔(冻结LLM/VLM)的任意层hidden state动态路由到生成塔(DiT)的任意层…
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "多模态融合"
  - "状态路由"
  - "T2I/图像编辑"
  - "Transformer"
  - "token级动态"
---

# Mixture of States (MoS): Routing Token-Level Dynamics for Multimodal Generation

**会议**: CVPR 2026  
**arXiv**: [2511.12207](https://arxiv.org/abs/2511.12207)  
**代码**: 无（但基于开源组件）  
**领域**: 图像生成 / 多模态融合 / 扩散模型  
**关键词**: 多模态融合, 状态路由, T2I/图像编辑, 非对称Transformer, token级动态  

## 一句话总结
提出Mixture of States (MoS)——一种新的多模态扩散模型融合范式，用可学习的token级路由器将理解塔(冻结LLM/VLM)的任意层hidden state动态路由到生成塔(DiT)的任意层，以3-5B参数在图像生成和编辑上匹配或超越20B的Qwen-Image。

## 背景与动机
多模态扩散模型的核心挑战是文本/视觉信号的有效对齐。现有融合方式各有缺陷：(1) Cross-Attention仅用最终层特征,信息有限；(2) Self-Attention将文本视觉token拼接处理,计算$O(n^2)$昂贵；(3) MoT(Mixture-of-Transformers)层对层共享KV,要求两个塔结构对称、深度相同,极不灵活。三个关键设计原则被忽视：层选择应自适应而非固定、条件信号应随去噪时刻动态变化、条件信号应token级个性化。

## 核心问题
能否设计一种灵活的跨模态融合机制，允许理解塔和生成塔完全非对称（不同深度、不同宽度），且融合方式动态适应输入内容和去噪进度？

## 方法详解

### 整体框架
MoS 想解决的是多模态扩散模型里文本/视觉信号怎么灵活融合的问题。它用双塔设计：理解塔 $\mathcal{U}$（冻结的 PLM-8B/InternVL-14B）一次前向处理文本/图像条件、输出全部 $m$ 层 hidden state，生成塔 $\mathcal{G}$（从头训练的 3B/5B DiT）进行扩散去噪。两塔之间插一个轻量路由器 $\mathcal{R}$（仅 100M 参数、2 个 Transformer 块），它同时接收理解塔各层状态、噪声潜变量 $z_t$ 和去噪时刻 $t$，为每个 context token 预测一个路由矩阵，动态决定理解塔哪些层的 hidden state 被聚合后送进生成塔的哪一层。训练时理解塔冻结、只训生成塔与路由器，且每个去噪步都重新路由一次，形成"随去噪进度变化"的动态融合。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 420}}}%%
flowchart TD
    C["文本 / 图像条件 c"] --> U["理解塔 U（冻结 PLM-8B / InternVL-14B）<br/>一次前向 → 全部 m 层 hidden state"]
    U --> R
    T["噪声潜变量 z_t ＋ 去噪时刻 t"] --> R
    subgraph R["路由器 R（约 100M，2 个 Transformer 块）"]
        direction TB
        R1["Token 级稀疏路由<br/>每 token 预测 m×n 路由矩阵，列向 softmax 取 top-k=2"]
        R2["时刻敏感路由<br/>z_t、t 作为输入，路由随去噪进度变化"]
        R3["ε-greedy 探索训练<br/>训练时以 ε 概率随机选层防锁死"]
    end
    R --> G["生成塔 G（3B / 5B DiT）<br/>第 j 块拼接路由来的 hidden state 做扩散去噪"]
    G -->|每个去噪步 t 重复| T
    G --> OUT["输出图像"]
```

### 关键设计

**1. Token 级稀疏路由：让每个 token 自己决定该读谁的哪几层**

现有融合要么只用最终层特征（信息有限），要么全塔拼接（成本高），都是静态的、对所有 token 一视同仁。MoS 让每个 context token 独立预测一个 logit 矩阵 $\mathcal{W} \in \mathbb{R}^{m \times n}$（$m$=理解塔层数, $n$=生成塔层数），$w_{ij}$ 表示把理解塔第 $i$ 层路由到生成塔第 $j$ 层的权重，softmax 归一化后 top-k（$k=2$）只传最相关的两层。消融证实 token 级路由明显优于 sample 级（FID 20.17 vs 21.66），因为不同 token 确实需要不同层的特征；$k=2$ 最优——$k=1$ 过于局部，$k \geq 3$ 反而稀释信息。

**2. 时刻敏感的路由：让条件信号随去噪进度变**

"一个 embedding 处理所有去噪步"是旧范式的通病。路由器同时接收文本 prompt、噪声潜变量 $z_t$ 和去噪时刻 $t$，消融证实三者都不可或缺（FID：仅 prompt 21.12 → +latent 21.89 → **+timestep 20.15**）。可视化显示路由模式随去噪推进变化：早期稀疏地选特定层，后期趋向平均权重——恰好与扩散模型"先结构后细节"的去噪节奏一致。

**3. $\epsilon$-greedy 探索训练：防止路由器锁死在局部最优**

top-k 选择容易让路由器过早收敛到某几层上。MoS 以 $\epsilon=0.05$ 的概率随机选层（而非 top-k），给路由一点探索空间。消融显示 $\epsilon$-greedy 不但加速收敛且最终性能更好。

### 损失函数 / 训练策略
Rectified flow matching 标准训练：$\mathbb{E}[\|v_t - \mathcal{G}(z_t, t, \mathcal{R}(\cdot))\|^2]$。四阶段渐进训练：512²(1400 A100-days) → 1024²(等量) → 美学微调(100 A100-days) → 2048²超分(80 A100-days)。总计~3000 A100-days——远低于SD1.5的6250 A100-days。

## 实验关键数据

| 方法 | 参数 | 融合类型 | GenEval↑ | DPG↑ | oneIG↑ | ImgEdit↑ |
|--------|------|------|------|----------|------|------|
| FLUX.1[Dev] | 12B | Self-Attn | 0.66 | 83.84 | 0.43 | — |
| SANA-1.5 | 4.8B | Cross-Attn | 0.81 | 84.70 | 0.33 | — |
| Bagel | 14B | MoT | 0.88 | — | 0.36 | 3.20 |
| Qwen-Image | **20B** | Self-Attn | 0.87 | 88.32 | 0.54 | 4.27 |
| **MoS-S** | **3B** | MoS | 0.89 | 86.33 | 0.50 | 4.17 |
| **MoS-L** | **5B** | MoS | **0.90** | **87.01** | **0.52** | **4.33** |

MoS-L(5B)在GenEval 0.90、ImgEdit 4.33上甚至超越Qwen-Image(20B)——参数量仅1/4。

### 消融实验要点
- **MoS > MoT > Cross-Attn**: FID 17.77 vs 21.66(手工), GenEval 0.79 vs 0.74(Cross-Attn)
- **非对称塔的优势**: 理解塔可独立scaling(8B→14B提升一致),MoT无法做到
- **路由器开销极低**: 仅0.008s/iter,几乎可忽略
- **总延迟更低**: MoS < Qwen-Image ≈ Bagel(因为理解塔仅执行一次)
- **编辑任务同理有效**: 双塔各取reference image的不同粒度信息(语义 vs 像素)

## 亮点
- **MoS突破了MoT的对称约束** —— 允许完全异构的理解/生成塔自由组合,这对实际部署极有价值
- "理解塔冻结+仅训练生成塔"策略大幅降低训练成本——3000 A100-days创建SOTA级模型
- Token级时刻敏感路由是对扩散模型融合方式的范式转变——不再是"一个embedding处理所有去噪步骤"
- 路由器可视化提供了跨模态交互的可解释性窗口——不同token/不同时刻确实需要不同层的特征
- 5B > 20B的效率故事非常compelling——计算效率是产业界最关注的

## 局限与展望
- 目前仅支持理解塔→生成塔的单向路由,双向MoS可能更强
- 未探索RLHF/GRPO等人类偏好对齐技术
- 小物体生成仍有瑕疵（视觉artifact）
- 未探索与量化/蒸馏/特征缓存等效率技术的组合
- 仅验证了图像生成/编辑,视频生成的MoS有待开发

## 与相关工作的对比
- **vs MoT (Bagel/LMFusion)**：MoT要求对称塔+层层对应,限制灵活性。MoS通过路由器实现任意层到任意层的稀疏连接,且3B就超越14B的Bagel
- **vs Cross-Attention (SANA/PixArt)**：Cross-Attn仅用最终层embedding,静态且信息量有限。MoS动态选择所有层的hidden state
- **vs Self-Attention (FLUX/SD3)**：Self-Attn计算昂贵且也是静态embedding。MoS计算更少(生成塔更小)且动态适应
- **vs Qwen-Image (20B)**：Qwen-Image性能强但参数量4×。MoS-L(5B)匹配或超越其性能

## 启发与关联
- MoS的"异构塔+路由器"架构可直接推广到**视频生成**——理解塔处理文本/首帧,路由器随时间步和帧位置动态调整
- 与LinVideo正交互补——LinVideo将softmax替换为linear attention加速,MoS减少整体参数量,二者可组合
- Token级路由的思想可以启发**VLM推理**中的跨模态交互——目前VLM也是固定层融合,动态路由可能提升效率

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ MoS作为新的融合范式,突破对称约束且token/timestep级路由均为原创
- 实验充分度: ⭐⭐⭐⭐⭐ 全面消融(路由输入/输出/架构/稀疏度/scaling)、多任务(生成+编辑)、多基准(GenEval/DPG/WISE/oneIG/ImgEdit/GEdit)
- 写作质量: ⭐⭐⭐⭐⭐ 三大设计原则→MoS设计→系统消融→SOTA结果的逻辑链完美
- 价值: ⭐⭐⭐⭐⭐ 5B=20B的效率故事+可解释的路由+范式创新=对图像生成领域影响巨大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World](thinking_in_dynamics_how_multimodal_large_language_models_perceive_track_and_rea.md)
- [\[ICML 2025\] RollingQ: Reviving the Cooperation Dynamics in Multimodal Transformer](../../ICML2025/multimodal_vlm/rollingq_reviving_the_cooperation_dynamics_in_multimodal_transformer.md)
- [\[CVPR 2026\] Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation](where_mllms_attend_and_what_they_rely_on_explaining_autoregressive_token_generat.md)
- [\[CVPR 2026\] AVR: Adaptive VLM Routing for Computer Use Agents](adaptive_vision-language_model_routing_for_computer_use_agents.md)
- [\[CVPR 2026\] VL-RouterBench: A Benchmark for Vision-Language Model Routing](vl-routerbench_a_benchmark_for_vision-language_model_routing.md)

</div>

<!-- RELATED:END -->
