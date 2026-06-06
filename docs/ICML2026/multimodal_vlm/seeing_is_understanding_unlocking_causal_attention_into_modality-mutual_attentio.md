---
title: >-
  [论文解读] Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs
description: >-
  [ICML 2026][多模态VLM][MLLM] 作者把 decoder-only MLLM 里的因果注意力掩码改一个"洞"，让排在前面的图像 token 反过来去看后面的文本问题 token——这一行掩码修改不加任何参数、不改训练数据…
tags:
  - "ICML 2026"
  - "多模态VLM"
  - "MLLM"
  - "注意力机制"
  - "跨模态对齐"
  - "幻觉缓解"
  - "因果掩码"
---

# Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs

**会议**: ICML 2026  
**arXiv**: [2503.02597](https://arxiv.org/abs/2503.02597)  
**代码**: https://github.com/sony/aki  
**领域**: 多模态VLM  
**关键词**: MLLM, 注意力机制, 跨模态对齐, 幻觉缓解, 因果掩码  

## 一句话总结
作者把 decoder-only MLLM 里的因果注意力掩码改一个"洞"，让排在前面的图像 token 反过来去看后面的文本问题 token——这一行掩码修改不加任何参数、不改训练数据，在 3 个 LLM backbone 与 12 个多模态基准上平均涨 6.2 个点。

## 研究背景与动机
**领域现状**：主流 MLLM（LLaVA-1.5、BLIP-3、Cambrian、MM-1.5 等）共享一个三段式骨架——视觉编码器→视觉语言连接器（VL-connector）→ decoder-only LLM。输入序列按 $S=[V, T_Q]$ 排列：先放 $|V|$ 个图像 token，再放 $|T_Q|$ 个文本问题 token，最后自回归地解码答案 $T_R$。

**现有痛点**：MLLM 在视觉中心任务（计数、空间关系、细节识别）上仍频繁出现物体幻觉。论文 Fig. 1 给的例子里，GPT-4o、Molmo、DeepSeek-VL2-Small 面对一张"周六 8 点到 20 点限时 2 小时停车"的复杂指示牌全部识错。

**核心矛盾**：以前的缓解工作走的是数据扩量（Molmo 加 clock/pointing/counting 数据）或换 VL-connector（abstractor、spatial vision aggregator）这两条路，前者要海量标注预算，后者至今没有公认最优解（McKinzie et al. 2024 实证表明没有一个 connector 在所有 benchmark 上都赢）。但论文指出真正的瓶颈在 LLM 本身的因果注意力——原本为单模态自回归设计的下三角掩码，让排在前面的图像永远看不到后面的文本，所以无论用户问"几只车"还是"什么颜色"，图像表示都是同一份静态特征，对话内容对图像理解毫无反向影响。

**本文目标**：让"前面的模态（图像）"能反向感知"后面的模态（文本问题）"，同时不增加参数、不破坏自回归生成、不打破现有 SFT pipeline。

**切入角度**：作者先做一个 sanity check——把训练顺序在 [图像, 文本] 与 [文本, 图像] 之间交替（Dual-Order Training, DOT），结果确实有提升，证明"让前模态见到后模态"这个方向是对的。但 DOT 把训练时间翻倍，且模态数变成 $n$ 时成本是 $n!$ 级的。这促使他们绕开训练顺序，直接动注意力掩码。

**核心 idea**：把因果掩码 $M$ 改造为 $M'$，只在 SFT 阶段额外解锁"图像 token → 文本问题 token"这一矩形区域，其他位置保持原样。

## 方法详解

### 整体框架
方法仍走标准两阶段流水线（PT + SFT），骨架沿用 Cha et al. 2024 的设计：视觉编码器 $f_V$（CLIP 类）提取图像特征，VL-connector $p_V$ 投影到文本空间，文本 embedder $f_T$ 产出查询 embedding，最终 LLM $f_L$ 在 $H_V \in \mathbb{R}^{|V|\times d}$ 与 $H_{T_Q} \in \mathbb{R}^{|T_Q|\times d}$ 上自回归生成 $T_R = f_L(H_V, H_{T_Q})$。预训练阶段（用 Blip3-kale 做 captioning）冻视觉编码器、放开 VL-connector 和 LLM，原因是这一阶段没有具体用户问题，谈不上"图像看哪个问题"；SFT 阶段同样冻视觉编码器，并在这里启用 MMA 掩码。生成阶段对已产出的回答 $T_R$ 仍走标准因果注意力——MMA 只作用于输入 $S$，存进 KV cache 后整段对话沿用。

### 关键设计

1. **Dual-Order Training (DOT) — 直觉基线**:

    - 功能：通过显式喂模型两种输入顺序 $[V, T_Q]$ 与 $[T_Q, V]$，让 LLM 在某些位置上能学到"前模态依赖后模态"的依存关系。
    - 核心思路：采用 tandem 训练——每一阶段先按 T&I 顺序训一遍，再按 I&T 顺序训一遍，保证推理时与 I&T 对齐。形式化为 $[T_{Q_{PT}}, V_{PT}] \to [V_{PT}, T_{Q_{PT}}] \to [T_{Q_{SFT}}, V_{SFT}] \to [V_{SFT}, T_{Q_{SFT}}]$。
    - 设计动机：作为"为什么需要解锁跨模态注意力"的诊断实验。论文用 DOT 证明这个方向确实涨点（LLaMA-3.2-3B 上 LLaVA-W 由 38.6 涨到 43.8，CV-Bench2D 由 37.5 涨到 46.7），但代价是每阶段训练时间翻倍且随模态数阶乘膨胀，因此必须找一个"不动训练数据"的替代方案。

2. **Modality-Mutual Attention (MMA) — 掩码级解锁**:

    - 功能：把因果掩码 $M$ 改成 $M'$，仅在"图像位置 $i$、文本问题位置 $j$"的矩形区域追加一条通路，使图像 token 能 attend 到后续的文本问题 token。
    - 核心思路：标准因果注意力 $\text{Attention}_{causal} = \text{softmax}((QK^T + M)/\sqrt{d})$，其中 $M_{ij}=0$ 若 $j \le i$ 否则 $-\infty$。MMA 把 $M'_{ij}$ 改为：当 $j \le i$（保因果）或 $1 \le i \le |V|$ 且 $|V|+1 \le j \le |V|+|T_Q|$（图像可看问题）时为 0，其余仍为 $-\infty$。本质上等于在原本全 $-\infty$ 的右上角挖出一个 $|V|\times|T_Q|$ 矩形通路，softmax 仍可正常归一化。
    - 设计动机：因果掩码原本是为单模态文本生成而生的，但在多模态对话里"图像静态、问题动态"的实际语义里反而成了瓶颈——同一张图配不同问题理应得到不同的图像中间表示。MMA 解锁后的图像 token 可以根据"几只车""什么颜色""哪个标志在右边"分别聚焦不同区域，相当于把"问题驱动的视觉编码"放进 LLM 内部。

3. **零参数、零额外算量的工程实现**:

    - 功能：在不引入任何可训练参数、不增加 FLOPs、不改 PT 阶段、不改 SFT 数据的前提下落地 MMA。
    - 核心思路：MMA 仅是把若干 $-\infty$ 位换成 0，注意力矩阵总元素数仍为 $(|V|+|T_Q|)^2$，softmax 计算量不变，KV cache 结构不变；生成阶段的 $T_R$ 沿用原因果掩码以保证自回归一致性。对交错多模态输入，论文给了广义条件 $M'_{ij}=0$ 当 $j\le i$ 或 $\phi(i)\ne\phi(j)$（$\phi$ 把 token 映射到模态），等价于"不同模态间默认相互可见、同模态内仍因果"。
    - 设计动机：要让这个方法成为可被现有 MLLM 训练栈一行代码接入的改动，而不是又一个需要重新设计连接器/数据集的方案。仅改 attention mask 这个工程克制度，是论文能"+6.2%"还无人质疑成本的关键。

### 损失函数 / 训练策略
两阶段标准 pipeline：PT 用 Blip3-kale 做 captioning，SFT 混合 VQAv2/VSR/GQA/OCRVQA（开放式 VQA）、ScienceQA/A-OKVQA（选择题）、RefCOCO/RefCOCO+/RefCOCOg/VisualGenome（referring expression）以及 LLaVA-150k（指令跟随）。两阶段都冻视觉编码器、放开 VL-connector 和 LLM，LLM 全参数更新。MMA 仅在 SFT 阶段启用；PT 阶段不启用，理由是 captioning 数据里没有"用户具体问题"这一概念，谈不上"图像该看哪个问题"。

## 实验关键数据

### 主实验
作者在 3 个 LLM backbone（LLaMA-3.2-3B-Instruct、Phi-3.5-Mini-Instruct，另一为论文中第三 backbone）×12 个 benchmark 上对比 MMA 与 (a) 传统 I&T 因果、(b) 反序 T&I 因果、(c) DOT。下表展示 LLaMA-3.2-3B 上 SFT 阶段的代表性结果（POPE 衡量幻觉率，CV-Bench 偏视觉中心，MMEP/MMEC 偏感知）。

| 配置（LLaMA-3.2-3B） | MMEP | MMB | LLaVA-W | POPE | RealWorldQA | CV-Bench2D | CV-Bench3D |
|---------------------|-----:|----:|--------:|-----:|------------:|-----------:|-----------:|
| (w/o T&I)SFT 基线 | 1134.2 | 51.3 | 38.6 | 73.5 | 37.8 | 37.5 | 50.7 |
| (w/o I&T)SFT 反序 | 1128.1 | 51.6 | 34.1 | 72.7 | 35.6 | 39.1 | 51.4 |
| DOT（双序训练） | 1219.5 | 46.9 | 43.8 | 77.0 | 42.0 | 46.7 | 52.9 |
| **MMA（本文）** | 论文报告 12 benchmark 平均 +6.2%（跨 3 backbones） | | | | | | |

跨 3 backbone × 12 benchmark 平均 +6.2%，在 vision-centric（CV-Bench、RealWorldQA）与幻觉（POPE）类基准上增益最显著，说明"问题驱动的图像表示"对需要根据指令动态聚焦的任务效果最大。

### 消融与机制分析

| 配置 | 关键现象 | 说明 |
|------|---------|------|
| 因果掩码（baseline） | 同图不同问图像表示恒定 | 论文 Sec. 3.3 提出的核心瓶颈 |
| 反序输入 T&I | 仅提升有限/偶有下降 | 验证"换顺序"无法根治，问题在掩码 |
| DOT 双序训练 | 多数 benchmark 涨点但训练 2× | 证明跨模态可见性方向正确 |
| **MMA（解锁矩形）** | +6.2% 平均、训练成本不变 | 等价改进，且零参数零算量 |
| PT 阶段也加 MMA | 论文选择不加 | captioning 无具体用户问题，加 MMA 语义不成立 |

### 关键发现
- 反序输入（T&I）在多数 benchmark 上几乎没有改善，说明"图像看不到文本"才是真正瓶颈而非"顺序选错了"——把模态交换并不会让文本回头看图像之外的东西。
- DOT 与 MMA 都体现"放松跨模态可见性"的收益，但 MMA 用一行掩码改动达成同等甚至更好的效果，是对 DOT 的彻底替代。
- 收益在 vision-centric 任务（CV-Bench、RealWorldQA、POPE）上最大；在偏知识或纯文本推理（MMMU 等）上较小，符合"图像表示因问而变"的预期机制。

## 亮点与洞察
- 这是一种"教科书级"的极小改动获得显著收益的案例——只动注意力掩码一个矩形，就把 MLLM 的根本性架构限制揭出来并修复，几乎所有现有 MLLM 训练栈都能一行集成。
- 把 DOT 作为诊断基线非常有说服力：用一个"贵但合理"的 baseline 证明方向，再用一个"便宜得多"的方法去替代，论证逻辑闭环。
- 给出的广义条件 $\phi(i)\ne\phi(j) \Rightarrow M'_{ij}=0$ 自然推广到 $\ge 2$ 模态（音频、视频等）的交错输入，思路对未来 Any-to-Any 模型有直接迁移价值。
- 揭示了"为什么 VL-connector 调参怎么都收敛不到最优"的潜在原因——问题不在连接器，而在 LLM 内部那个被忽略的因果掩码假设。

## 局限与展望
- MMA 只在 SFT 阶段启用，作者明确说 PT 不加是因为没有"用户问题"。但 captioning 也是文本驱动，能否在 PT 阶段构造合适的 prompt 让 MMA 生效未深入。
- 实验集中在 single-image + single-turn QA；多轮对话或多图输入下 $\phi$ 的具体分组（多张图算同模态还是分开？）尚未细化。
- 没有对比解锁 image→answer tokens 的方案——目前仅解锁 image→question tokens；答案生成期间是否也应让图像能"看"到部分已生成的答案是开放问题。
- 12 benchmark 已经覆盖较广，但缺少 video-language 与 audio-language benchmark 的验证，方法的"广义性"目前还是理论上的。

## 相关工作与启发
- **vs Concentric Causal Attention (CCA, Xing et al. 2024)**: CCA 假设图像中心区域更重要，通过位置编码与掩码偏置缓解幻觉；但 CCA 仍未打通跨模态信息流。MMA 直接让图像看到文本问题，绕过"哪块图重要"这一启发式假设。
- **vs Mixed Attention (Xie et al. 2025)**: Mixed Attention 在统一理解+生成的多模态 Transformer 里把图像 token 之间改成 full attention，但图像→文本仍封死；MMA 反过来打通"图像→文本问题"这条线，更贴合 understanding 场景的需求。
- **vs Cambrian 的 Spatial Vision Aggregator / Honeybee 的 abstractor**: 这些都在 VL-connector 上下功夫；MMA 主张瓶颈不在 connector 而在 LLM 内部掩码，提供另一条与 connector 改进正交且可叠加的路径。
- **vs Molmo 等数据-centric 方法**: 数据扩量缓解幻觉但成本高；MMA 完全不动数据、不动 connector，是数据稀缺场景的廉价首选。

## 评分
- 新颖性: ⭐⭐⭐⭐ 改动极小但角度独到——指出 decoder-only LLM 的因果掩码假设在多模态场景下的根本不合理性，思路罕见。
- 实验充分度: ⭐⭐⭐⭐ 3 backbone × 12 benchmark 大规模验证，且配 DOT、反序等诊断 baseline，论证链条完整。
- 写作质量: ⭐⭐⭐⭐ 从 sanity check（DOT）到正式方法（MMA）层层递进，把"为什么不简单换顺序就行"讲得很透。
- 价值: ⭐⭐⭐⭐⭐ 零参数、零算量、即插即用，可立即接入任何现有 MLLM 训练栈，工业落地价值非常高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding](../../CVPR2025/multimodal_vlm/seeing_far_and_clearly_mitigating_hallucinations_in_mllms_with_attention_causal_.md)
- [\[ICML 2026\] Smoothing Slot Attention Iterations and Recurrences](smoothing_slot_attention_iterations_and_recurrences.md)
- [\[ICML 2026\] Large Vision-Language Models Get Lost in Attention](large_vision-language_models_get_lost_in_attention.md)
- [\[ICML 2025\] MODA: MOdular Duplex Attention for Multimodal Perception, Cognition, and Emotion Understanding](../../ICML2025/multimodal_vlm/moda_modular_duplex_attention_for_multimodal_perception_cognition_and_emotion_un.md)
- [\[ICML 2026\] Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal ICL](hyper-icl_attention_calibration_with_hyperbolic_anchor_distillation_for_multimod.md)

</div>

<!-- RELATED:END -->
