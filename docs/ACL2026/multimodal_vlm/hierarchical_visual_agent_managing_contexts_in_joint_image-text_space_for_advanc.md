---
title: >-
  [论文解读] HierVA: Hierarchical Visual Agent — Managing Contexts in Joint Image-Text Space for Advanced Chart Reasoning
description: >-
  [ACL 2026][多模态VLM][chart QA] HierVA 用 "manager–worker" 双层多模态 agent，把图表推理过程中的图像和文本 context 都按"获取–限定–蒸馏"的纪律管理起来…
tags:
  - "ACL 2026"
  - "多模态VLM"
  - "chart QA"
  - "hierarchical agent"
  - "多模态"
  - "zoom-in"
  - "training-free"
---

# HierVA: Hierarchical Visual Agent — Managing Contexts in Joint Image-Text Space for Advanced Chart Reasoning

**会议**: ACL 2026  
**arXiv**: [2605.04304](https://arxiv.org/abs/2605.04304)  
**代码**: 待确认  
**领域**: 多模态VLM / Agent / 图表推理  
**关键词**: chart QA, hierarchical agent, multimodal context, zoom-in, training-free

## 一句话总结
HierVA 用 "manager–worker" 双层多模态 agent，把图表推理过程中的图像和文本 context 都按"获取–限定–蒸馏"的纪律管理起来，零训练地在 CharXiv 等复杂图表推理 benchmark 上稳超 CoT 和 "thinking with images" 等强基线。

## 研究背景与动机

**领域现状**：图表问答（Chart QA）是科研助手和文档理解系统的核心能力。CoT 让 MLLM 显式推理，而新近的 "thinking with images" 范式（OpenAI 2025、Lai 2025、Zheng 2025 等）允许模型在推理过程中迭代获取额外视觉证据（如 zoom-in crop），把视觉细节带进 reasoning trace。

**现有痛点**：单图、单步问题（ChartQA-style）上 MLLM 已 90%+；但对多 subplot、多步推理的复杂图表（CharXiv reasoning split），现有方法都崩——CoT 在全局图上分心于无关元素，"thinking with images" 把每次 zoom-in crop 不断 append 到 context 里，导致 **monotonic context growth**：图像吃 token、中间步骤越积越多，反而把全局参照信息稀释掉。

**核心矛盾**：复杂图表推理本质是图像–文本混合任务，需要"小区域细节"与"多步中间结果"同时保留；但 LLM context 是有限的、且文本+图像信息会互相 dilute，越往后越乱。

**本文目标**：在不训练任何模型的前提下，把多模态推理的 context 管理学问做扎实——既要拿到该拿的细节，又要随手把不必要的中间产物蒸馏掉。

**切入角度**：把管理文本 reasoning trace 的同套纪律（plan distillation、scope、summarize）平移到视觉 context 上，并用 **manager-worker 层级架构**强制执行。

**核心 idea**：用一个 manager 维护全局精炼 context、用多个 worker 在各自隔离的局部 context 里干活，每次 zoom-in/计算结果只回传一个蒸馏摘要给 manager。

## 方法详解

### 整体框架
输入：图表 $I_0$ + 自然语言问题 $q$；输出：答案 $a$。Manager 维护全局 context $C_M = \{q, I_0, \text{refined plan}, \text{distilled summaries}\}$，Worker 们用隔离的局部 context $C_{W_t} = \{\text{task instr}, \text{optional skill}, \text{single image}\}$。Manager 通过 Algorithm 1 的控制环工作：先两阶段 planning（先粗 plan 再细化、只留细化版），然后循环 [终止判定 → CreateNextTask → ExecuteWorker → 蒸馏摘要 append 到 $C_M$]，直到能给出最终 $\boxed{}$ 答案。

### 关键设计

1. **Manager–Worker 层级 + Encapsulation**：

    - 功能：把全局规划和局部执行彻底解耦。
    - 核心思路：Manager 只看自己的 $C_M$；Worker 跑完任务后，它的内部 reasoning trace 不会被 append 进 $C_M$，只把"一句话事实"或"一个 crop handle"作为蒸馏结果回传。Worker 每次只接收单张图（原图或 crop）和最小化的任务指令，强制 scoped evidence。
    - 设计动机：直接对抗 monotonic context growth——以前的 thinking-with-images 把 worker 的所有 deliberation 都倒进主 context，HierVA 用 abstraction barrier 把局部噪声挡在外面。

2. **Adaptive Zoom 作为显式 action**：

    - 功能：让 manager 在 reasoning 中按需拿高分辨率证据。
    - 核心思路：把 zoom 抽象成 image-expected task，worker 接到后调用 zoom-in 工具，指定一个 bounding box，返回 cropped + resized 的高分辨率图像。Manager 的 action space 就是二选一：(a) 要新的视觉证据（zoom）或 (b) 要文本事实（读值、比较、计算）。
    - 设计动机：图表的小字、tick、legend 在全局尺度下不可靠，必须按需放大；把 zoom 升格为一等公民 action，比"在 prompt 里说要不要看细节"可控得多。

3. **Skill routing + 三重 context distillation**：

    - 功能：在不污染全局 context 的前提下，让 worker 在合适时机用上代码工具等专门技能。
    - 核心思路：维护一个紧凑的 skill library $\mathcal{S}$，每个 skill 是一段简短的 markdown 过程描述。Manager 为每个 task 选一组相关 skill，**just-in-time** 注入到 worker 的 system prompt 里；manager 自己永远看不到 skill 内容。配合三重蒸馏：(1) Plan distillation 只留最终精炼版 plan；(2) Worker encapsulation 隔离 worker trace；(3) Result distillation 把 worker 回答压成单句或 crop handle 再 append。
    - 设计动机：朴素做法是把所有可能技能都塞进 base prompt，会立刻让 context 膨胀；just-in-time 注入既保留能力又不污染主线推理。

### 损失函数 / 训练策略
training-free，没有任何参数更新——所有改进都来自 prompt orchestration 设计，复用同一个底座 MLLM（实验用 Qwen3VL-A22B）。

## 实验关键数据

### 主实验：CharXiv reasoning split
对比 Direct / CoT / CoT-Plan / Thinking w/ Images 等基线，统一用 Qwen3VL-A22B 作为底座。指标含整体 Acc 与子类型：Extr / First / Read / RevR / Comp / Freq；同时报告 Peak Token #。

| 方法 | Image Tools | Skills | CharXiv-All | Peak Tok # |
|------|-------------|--------|-------------|------------|
| Direct | — | — | 45.7 | 702 |
| CoT | — | — | 62.1 | 1926 |
| CoT-Plan | — | — | 62.4 | 1947 |
| Thinking w/ Images | zoom | — | （见原表） | — |
| **HierVA (Ours)** | zoom + code | ✓ | **稳超所有基线** | 控制增长 |

在 ChartQA + 合成多 subplot 图（sp#1→sp#6）上：随 subplot 数增加，所有方法都掉分，但 HierVA 仅掉 **1.5%**，CoT 掉 2.6%，CoT-Plan 掉 3.8%，Direct 掉 5.4%。

| 方法 | ChartQA | sp#1 | sp#2 | sp#4 | sp#6 |
|------|---------|------|------|------|------|
| Direct | 88.9 | 88.5 | 86.5 | 84.8 | 83.1 |
| CoT | 90.2 | 90.1 | 89.2 | 88.4 | 87.5 |
| Thinking w/ Images | 89.9 | 89.7 | 88.5 | 87.4 | 87.5 |
| **HierVA** | 89.9 | 89.7 | 89.2 | 88.5 | **88.2** |

### 消融实验

| 配置 | 关键效果 | 说明 |
|------|---------|------|
| Full HierVA | 最佳 | manager+worker + 三重蒸馏 |
| w/o 层级架构 | 显著下降 | 退化为 thinking-with-images |
| w/o scoped visual context | 中等下降 | worker 拿全图反而被分心 |
| w/o distilled context | 中等下降 | manager context 膨胀，长链推理掉分 |

### 关键发现
- 简单问题（ChartQA-style 单步检索）context 管理收益小，HierVA 与 thinking-with-images 打平；优势真正体现在需要多步 reasoning 的复杂 chart 上。
- 复杂度（subplot 数）越高，HierVA 的相对优势越明显——这正印证了"长链推理才考验 context 管理"。
- 三个蒸馏机制互补：单独去掉任一个都掉分，说明 plan/worker/result 三处都是 context 膨胀的源头。

## 亮点与洞察
- **把"管 text context"的纪律照搬到"管 image context"**：核心见解是图像和文本在 token-budget 层面是同质的，所以蒸馏/scope/encapsulation 一样适用。这个 framing 本身就值。
- **Skill just-in-time 注入**：对比传统"全 skill 写进 system prompt"，这种按需注入的设计对所有 tool-use agent 都通用。
- **Zoom 作为一等公民 action**：相比把"看仔细一点"埋在 CoT 里，把 zoom 显式建模成 typed action，调度/调试都更可控。

## 局限与展望
- 完全依赖底座 MLLM 的 instruction following 能力，对小模型可能直接失效。
- Peak token 仍然不低，对延迟敏感场景需要进一步压缩。
- 评测只到 CharXiv reasoning split + ChartQA + 合成图，对真实 dashboard、地图、流程图等更杂的视觉文档暂未验证。
- 没有学得的 skill selection 策略，靠 manager prompt 启发式选 skill，跨域泛化可能受限。

## 相关工作与启发
- **vs CoT / CoT-Plan**：纯文本 CoT 看不到细节，CoT-Plan 不能动态获取证据；HierVA 把图像也纳入工作内存。
- **vs Thinking w/ Images**（Zheng 2025）：同样允许 zoom，但 HierVA 用层级 + 蒸馏避免 context 单调膨胀，是其结构化升级版。
- **vs ReAct / Tool Agents**：思路同源（agent + tool），但 HierVA 强调多模态 context 管理这一被忽视的环节。

## 评分
- 新颖性: ⭐⭐⭐⭐ Manager-worker + 蒸馏的 framing 在多模态 agent 里是清晰的新切角。
- 实验充分度: ⭐⭐⭐ 主要在 CharXiv + ChartQA，合成 subplot 实验设计巧但 benchmark 覆盖不够广。
- 写作质量: ⭐⭐⭐⭐ Motivation 与 design principle 串联紧密，三重蒸馏点出问题本质。
- 价值: ⭐⭐⭐⭐ Training-free 即插即用，对 chart 助手类应用直接可用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] TEMA: Anchor the Image, Follow the Text for Multi-Modification Composed Image Retrieval](tema_anchor_the_image_follow_the_text_for_multi-modification_composed_image_retr.md)
- [\[ACL 2026\] Learning More from Less: Exploiting Counterfactuals for Data-Efficient Chart Understanding](learning_more_from_less_exploiting_counterfactuals_for_data-efficient_chart_unde.md)
- [\[ACL 2026\] SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding](slideagent_hierarchical_agentic_framework_for_multi-page_visual_document_underst.md)
- [\[CVPR 2026\] The Coherence Trap: When MLLM-Crafted Narratives Exploit Manipulated Visual Contexts](../../CVPR2026/multimodal_vlm/the_coherence_trap_when_mllm-crafted_narratives_exploit_manipulated_visual_conte.md)
- [\[CVPR 2025\] MV-MATH: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts](../../CVPR2025/multimodal_vlm/mv-math_evaluating_multimodal_math_reasoning_in_multi-visual_contexts.md)

</div>

<!-- RELATED:END -->
