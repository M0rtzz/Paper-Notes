---
title: >-
  [论文解读] Proof-of-Perception: Certified Tool-Using Multimodal Reasoning with Compositional Conformal Guarantees
description: >-
  [CVPR 2026][多模态VLM][保形预测] 提出 Proof-of-Perception (PoP)，将多模态推理建模为可执行的有向无环图(DAG)，每个感知/逻辑节点输出带有保形预测证书的集合值（提供逐步可靠性保证），并用轻量控制器基于这些证书在计算预算内自适应分配算力…
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "保形预测"
  - "工具使用"
  - "多模态推理"
  - "不确定性量化"
  - "自适应计算"
---

# Proof-of-Perception: Certified Tool-Using Multimodal Reasoning with Compositional Conformal Guarantees

**会议**: CVPR 2026  
**arXiv**: [2603.00324](https://arxiv.org/abs/2603.00324)  
**代码**: [https://github.com/AryaFayyazi/PoP](https://github.com/AryaFayyazi/PoP)  
**领域**: 多模态推理 / 可靠AI  
**关键词**: 保形预测, 工具使用, 多模态推理, 不确定性量化, 自适应计算

## 一句话总结
提出 Proof-of-Perception (PoP)，将多模态推理建模为可执行的有向无环图(DAG)，每个感知/逻辑节点输出带有保形预测证书的集合值（提供逐步可靠性保证），并用轻量控制器基于这些证书在计算预算内自适应分配算力，在文档、图表和多图QA基准上超越CoT、ReAct和PoT基线。

## 研究背景与动机

**领域现状**：多模态LLM在文档理解、图表推理等任务上取得进展，但通常将细粒度感知（OCR、检测、图表解析）和符号推理混在单次前向传播中。工具使用和结构化提示（CoT、ReAct、PoT）部分缓解了这一问题。

**现有痛点**：(1) 中间步骤输出单一猜测值，静默传播错误；(2) 计算分配靠启发式（固定重试次数、未校准阈值），无法做准确率-成本权衡；(3) 校准（如果有的话）仅在最终答案上，中间步骤的逐步可靠性无保证。

**核心矛盾**：现有方法在中间感知步骤"单点提交"——一旦OCR错了一个字、检测漏了一个框，后续推理就被迫在错误基础上合理化。而且何时该扩展推理（多工具调用）、何时该提前停止，缺乏原则性判据。

**本文目标** 如何为多步多模态推理的每个中间步骤提供可靠性保证，并将不确定性转化为计算分配策略？

**切入角度**：保形预测（Conformal Prediction）提供无分布假设的有限样本覆盖保证。将其应用到推理DAG的每个节点，输出的不再是单点值而是有覆盖保证的集合。

**核心 idea**：在推理DAG的每个感知/逻辑节点上用保形预测输出校准的集合值，控制器基于集合大小和预算决定是接受、重试还是扩展。

## 方法详解

### 整体框架
给定多图+文本查询，MLLM 规划器生成DSL程序定义推理DAG $G=(V,E)$。每个工具节点调用外部感知工具（OCR/检测/图表解析），每个融合节点在MLLM内部融合上游结果。每个节点配备证书头输出非一致性分数，通过split-conformal校准得到阈值，输出集合值预测。控制器观察节点级证书和全局预算，决定 ACCEPT/RETRY/EXPAND/ABORT。

### 关键设计

1. **节点级保形预测证书（Node-Level Conformal Certificates）**:

    - 功能：为每种节点类型（OCR/检测/图表解析/逻辑融合）定义非一致性函数和校准阈值，输出集合值预测
    - 核心思路：对第 $t$ 类节点，非一致性函数 $s^{(t)}(x_v, z)$ 度量候选输出 $z$ 的"异常"程度。通过校准集计算阈值 $\tau_\delta^{(t)} = \alpha_{(k)}^{(t)}, k = \lceil(n_t+1)(1-\delta)\rceil$。集合预测 $\Gamma_\delta^{(t)}(x_v) = \{z : s^{(t)}(x_v, z) \leq \tau_\delta^{(t)}\}$，保证覆盖概率 $\geq 1-\delta$
    - 设计动机：单点预测在中间步骤静默传播错误，集合值预测保留多个校准候选直到证据消除歧义，减少错误级联

2. **自适应控制器（Adaptive Controller for Compute Allocation）**:

    - 功能：轻量策略网络 $\pi_\phi$，基于每个节点的证书状态 $c_v$（阈值、集合大小、节点类型）和全局预算 $b$，输出动作 $a_v \in \{\text{ACCEPT, RETRY, EXPAND, ABORT}\}$
    - 核心思路：ACCEPT保留当前集合，RETRY用更高精度重跑（如高分辨率裁剪），EXPAND添加新子节点（如额外OCR调用），ABORT在预算耗尽时提前终止。控制器用策略梯度优化 $R(x) = -C_{err}(x) - \beta C_{comp}(x)$
    - 设计动机：不确定性不该是被动评分，而应主动指导计算分配——集合大时扩展计算，集合小（置信高）时提前停止

3. **自博弈对抗样本挖掘（Self-Play Counterexample Mining）**:

    - 功能：在训练中由冻结的对手生成扰动输入（裁剪、仿射变换、OCR噪声），筛选导致错误的反例用于增强学生和校准集
    - 核心思路：对手执行推理图并对输入做可控扰动，筛选预测错误或非一致性分数大的样本作为反例。反例用于训练学生保持覆盖率，并追加到校准池使阈值反映真实失败模式
    - 设计动机：标准校准假设可交换性，但分布偏移下证书可能失效。自博弈让校准在对抗扰动下仍可靠

### 损失函数 / 训练策略
总损失 $\mathcal{L} = \mathcal{L}_{task} + \gamma_{plan}\mathcal{L}_{plan} + \gamma_{cert}\mathcal{L}_{cert} + \gamma_{ctrl}\mathcal{L}_{ctrl}$：任务损失（最终答案准确率）、规划损失（程序生成序列交叉熵）、证书损失（边距约束保证覆盖）、控制器损失（策略梯度优化准确率-成本权衡）。

## 实验关键数据

### 主实验

| 方法 | DocVQA | TextVQA | InfoVQA | ChartQA | MultiDoc2Dial |
|------|--------|---------|---------|---------|---------------|
| CoT (GPT-4V) | 74.2 | 68.1 | 51.3 | 71.8 | 42.5 |
| ReAct | 76.8 | 70.3 | 54.1 | 74.2 | 45.7 |
| PoT | 78.1 | 71.5 | 56.4 | 76.9 | 47.2 |
| **PoP** | **82.3** | **75.8** | **61.2** | **80.5** | **52.8** |

### 消融实验

| 配置 | DocVQA | 计算成本(归一化) |
|------|--------|------------------|
| PoP (full) | 82.3 | 1.0x |
| w/o Conformal (单点预测) | 77.5 | 0.8x |
| w/o Controller (固定扩展) | 80.1 | 1.6x |
| w/o Self-Play | 80.8 | 1.0x |

### 关键发现
- PoP在所有5个基准上超越CoT、ReAct、PoT基线，DocVQA提升4.2%，ChartQA提升3.6%
- 去掉保形证书（退化为单点预测）性能大幅下降，验证了集合值中间输出的价值
- 去掉控制器后计算成本增加60%但性能仅提升微弱，说明控制器有效减少不必要计算
- 自博弈挖掘贡献1.5%的性能提升，增强了分布偏移下的鲁棒性

## 亮点与洞察
- **将不确定性从"被动评分"变为"主动计算策略"**是核心insight——保形集合大→分配更多计算（EXPAND），集合小→提前终止（ACCEPT）
- 组合式的保形保证（每步覆盖 $1-\delta$）比仅在最终答案做校准更有意义，可追溯错误到具体步骤
- 框架高度模块化，工具集和节点类型可灵活扩展

## 局限与展望
- 保形预测假设可交换性，虽然自博弈部分缓解，但严格的分布偏移下覆盖保证可能失效
- 候选集大小受限于beam search或采样的候选数 $K_{max}$，可能遗漏正确答案
- 控制器的离散动作空间（4种）可能过于简单，更细粒度的计算分配策略有探索空间

## 相关工作与启发
- **vs CoT/ReAct**: 单点中间输出+启发式计算控制，无可靠性保证；PoP每步有覆盖保证且计算分配有据
- **vs 传统保形预测**: 通常只用于最终预测，PoP将其嵌入多步推理管线的每个节点
- **vs ViperGPT/VisualProg**: 程序化推理但中间步骤无不确定性量化

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 保形预测+工具使用+自适应计算控制的组合在多模态推理中首次提出
- 实验充分度: ⭐⭐⭐⭐ 五个基准、完整消融、成本分析
- 写作质量: ⭐⭐⭐⭐ 理论严谨，形式化完整
- 价值: ⭐⭐⭐⭐⭐ 对可靠AI推理有深远影响，保形证书+计算控制的范式可广泛迁移

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Conditional Factuality Controlled LLMs with Generalization Certificates via Conformal Sampling](conditional_factuality_controlled_llms_with_generalization_certificates_via_conf.md)
- [\[CVPR 2026\] CodeDance: A Dynamic Tool-integrated MLLM for Executable Visual Reasoning](codedance_a_dynamic_tool-integrated_mllm_for_executable_visual_reasoning.md)
- [\[AAAI 2026\] VipAct: Visual-Perception Enhancement via Specialized VLM Agent Collaboration and Tool-use](../../AAAI2026/multimodal_vlm/vipact_visual-perception_enhancement_via_specialized_vlm_age.md)
- [\[CVPR 2026\] From Intuition to Investigation: A Tool-Augmented Reasoning MLLM Framework for Generalizable Face Anti-Spoofing](from_intuition_to_investigation_a_tool-augmented_reasoning_mllm_framework_for_ge.md)
- [\[ACL 2026\] A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning](../../ACL2026/multimodal_vlm/a_survey_of_multimodal_mathematical_reasoning_from_perception_alignment_to_reaso.md)

</div>

<!-- RELATED:END -->
