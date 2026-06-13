---
title: >-
  [论文解读] Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs
description: >-
  [ICML 2026][多模态VLM][MLLM] 作者把 decoder-only MLLM 里的因果注意力掩码改一个"洞"，让排在前面的图像 token 反过来去看后面的文本问题 token——这一行掩码修改不加任何参数、不改训练数据，在 3 个 LLM backbone 与 12 个多模态基准上平均涨 6.2 个点。
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

**1. Dual-Order Training（DOT）：先用一个"贵但合理"的基线，证明让前模态看到后模态确实有用**

在正式动掩码之前，作者得先回答一个前置问题：图像看不到文本，真的是瓶颈吗？DOT 就是为此设计的诊断实验。它的做法很直接——每个阶段把输入顺序训两遍，先按 $[T_Q, V]$（文本在前）训一遍，再按 $[V, T_Q]$（图像在前）训一遍，整条 pipeline 写成 $[T_{Q_{PT}}, V_{PT}] \to [V_{PT}, T_{Q_{PT}}] \to [T_{Q_{SFT}}, V_{SFT}] \to [V_{SFT}, T_{Q_{SFT}}]$，最后一段保持与推理时的 I&T 顺序对齐。当文本排在图像前面时，因果掩码自然允许后面的图像 token 回看前面的文本，于是模型确实学到了"前模态依赖后模态"的依存关系。结果也支持这个方向：LLaMA-3.2-3B 上 LLaVA-W 从 38.6 涨到 43.8，CV-Bench2D 从 37.5 涨到 46.7。但 DOT 的代价是每阶段训练时间翻倍，而且模态数变成 $n$ 时枚举所有顺序是 $n!$ 级的开销——方向对了，但太贵，必须换一个不碰训练数据的实现。

**2. Modality-Mutual Attention（MMA）：在因果掩码的右上角挖一个矩形洞，让图像直接看到后面的问题**

DOT 的代价全在"训两遍"上，而它真正想要的效果其实只是"图像能 attend 到文本问题"。MMA 把这件事直接做进掩码里，连训练数据都不用动。标准因果注意力是 $\text{Attention}_{causal} = \text{softmax}((QK^T + M)/\sqrt{d})$，其中 $M_{ij}=0$ 当 $j \le i$、否则为 $-\infty$，也就是经典的下三角。MMA 把掩码改成 $M'$：当 $j \le i$（保留因果）或者 $1 \le i \le |V|$ 且 $|V|+1 \le j \le |V|+|T_Q|$（图像位置看问题位置）时取 0，其余仍是 $-\infty$。几何上就是在原本全 $-\infty$ 的右上角挖出一个 $|V|\times|T_Q|$ 的矩形通路，softmax 依旧能正常归一化。这一改之所以有效，是因为它纠正了一个被沿用已久的错误假设：因果掩码是为单模态文本自回归设计的，但多模态对话里图像是静态的、问题是动态的，同一张图配不同问题理应得到不同的视觉表示。打通之后，图像 token 就能根据"几只车""什么颜色""哪个标志在右边"分别聚焦不同区域，相当于把"问题驱动的视觉编码"直接搬进了 LLM 内部，而不再依赖前面那个一锤定音的静态特征。

**3. 零参数、零算量落地，并自然推广到多模态**

MMA 之所以能被任何现有 MLLM 训练栈一行接入，是因为它的改动小到几乎不留痕迹：只是把右上角若干 $-\infty$ 位换成 0，注意力矩阵元素数仍是 $(|V|+|T_Q|)^2$，softmax 算量不变、KV cache 结构不变，没有引入任何可训练参数，也不增加 FLOPs。生成阶段产出的回答 $T_R$ 仍走原始因果掩码，保证自回归一致性。对于图文交错的多模态输入，论文给了一个更一般的条件：$M'_{ij}=0$ 当 $j\le i$ 或 $\phi(i)\ne\phi(j)$，其中 $\phi$ 把 token 映射到它所属的模态——也就是"不同模态之间默认互相可见、同模态内部仍保持因果"。这个广义形式让方法天然延伸到音频、视频等更多模态，而不必为每种模态重新设计连接器或数据集；正是这种工程上的克制，使得它在拿到 +6.2% 增益的同时几乎没人质疑它的成本。

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

- [\[ICML 2026\] Smoothing Slot Attention Iterations and Recurrences](smoothing_slot_attention_iterations_and_recurrences.md)
- [\[ICML 2026\] Large Vision-Language Models Get Lost in Attention](large_vision-language_models_get_lost_in_attention.md)
- [\[ICML 2025\] MODA: MOdular Duplex Attention for Multimodal Perception, Cognition, and Emotion Understanding](../../ICML2025/multimodal_vlm/moda_modular_duplex_attention_for_multimodal_perception_cognition_and_emotion_un.md)
- [\[ICML 2026\] Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal ICL](hyper-icl_attention_calibration_with_hyperbolic_anchor_distillation_for_multimod.md)
- [\[ICLR 2026\] Constructive Distortion: Improving MLLMs with Attention-Guided Image Warping](../../ICLR2026/multimodal_vlm/constructive_distortion_improving_mllms_with_attention-guided_image_warping.md)

</div>

<!-- RELATED:END -->
