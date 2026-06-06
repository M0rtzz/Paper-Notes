---
title: >-
  [论文解读] Hyperbolic Fine-Tuning for Large Language Models
description: >-
  [NeurIPS 2025][LLM评测][hyperbolic geometry] 发现 LLM token 嵌入具有幂律分布和树状双曲结构，据此提出 HypLoRA——在 Lorentz 双曲流形上直接执行低秩适配（避免切空间映射的相消效应）…
tags:
  - "NeurIPS 2025"
  - "LLM评测"
  - "hyperbolic geometry"
  - "LoRA"
  - "parameter-efficient fine-tuning"
  - "Lorentz model"
  - "LLM reasoning"
---

# Hyperbolic Fine-Tuning for Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2410.04010](https://arxiv.org/abs/2410.04010)  
**代码**: [https://github.com/marlin-codes/HypLoRA](https://github.com/marlin-codes/HypLoRA)  
**领域**: LLM评测  
**关键词**: hyperbolic geometry, LoRA, parameter-efficient fine-tuning, Lorentz model, LLM reasoning

## 一句话总结
发现 LLM token 嵌入具有幂律分布和树状双曲结构，据此提出 HypLoRA——在 Lorentz 双曲流形上直接执行低秩适配（避免切空间映射的相消效应），在算术推理和常识推理任务上相比标准 LoRA 取得显著提升（如 Qwen2.5-7B 上 M.AVG +7.5%）。

## 研究背景与动机

**领域现状**：LLM 参数高效微调（PEFT）中 LoRA 因简单有效成为主流。然而所有现有 LoRA 变体均在欧氏空间操作权重矩阵。

**现有痛点**：语言中的概念天然具有层级结构（"水果" → "苹果"/"香蕉"），欧氏空间难以高效表示此类树状层级。双曲空间因负曲率和指数体积增长更适合，但直接训练双曲 LLM 成本过高。

**核心发现**：作者对多个开源 LLM 进行深入分析发现：(a) token 频率服从幂律分布（$\gamma \approx 1.9$），高频 token 靠近原点、低频 token 远离原点；(b) token 嵌入的 $\delta$-hyperbolicity 测量值极低（$\delta_{\text{rel}} \approx 0.06$-$0.12$），表明嵌入空间具有强烈的树状结构。

**核心矛盾**：传统双曲神经网络通过"切空间→指数映射→对数映射→切空间"路径操作，导致指数和对数映射互逆相消（$\log \circ \exp = \text{id}$），双曲几何优势被抹杀。

**核心 idea**：设计 HypLoRA。直接在双曲流形（Lorentz 模型）上做低秩变换，绕过切空间映射，保留双曲几何的建模能力。

## 方法详解

### 整体框架
冻结预训练 LLM 权重 $W$，在每个目标 Transformer 层添加 HypLoRA 适配器。输入 $\mathbf{x}^E$ 经过两条路径：(1) 原始冻结路径 $W\mathbf{x}^E$；(2) HypLoRA 路径——先投影到双曲空间 $\Pi_{\exp}^K$，再在双曲空间执行低秩变换 LLR，最后投影回欧氏空间 $\Pi_{\log}^K$。两路结果相加得到输出。

### 关键设计

1. **LLM 嵌入的双曲性分析**：

    - 功能：验证 LLM 嵌入空间确实具有双曲几何特性
    - 核心思路：(a) 全局统计 token 频率→幂律分布 $P(k) \sim k^{-\gamma}$；(b) 分析 token 频率与嵌入范数的关系→高频 token 范数小、低频 token 范数大（跨 LLaMA/Gemma/Qwen 一致）；(c) 用 Gromov 四点条件计算 $\delta$-hyperbolicity→值接近 0（0.06-0.12），远低于球面空间（0.99）和随机图（0.62）
    - 设计动机：为在双曲空间做微调提供实证基础，证明这不是凭空假设

2. **直接 Lorentz 低秩变换 (LLR)**：

    - 功能：在双曲空间直接做低秩矩阵变换，避免切空间映射相消
    - 核心思路：$\mathbf{LLR}(BA, \mathbf{x}^H) = (\sqrt{\|BA\mathbf{x}_s^H\|_2^2 + K}, BA\mathbf{x}_s^H)$，其中 $\mathbf{x}_s^H$ 是双曲点的空间分量。变换仅作用于空间维度，时间维度由 Lorentz 约束自动恢复
    - 设计动机：绕过传统"指数映射→对数映射"相消问题。输出保持在 Lorentz 流形上（$\langle \mathbf{x}, \mathbf{x} \rangle_\mathcal{L} = -K$），等效于伪 Lorentz 旋转

3. **HypLoRA 完整适配公式**：

    - $\mathbf{z}^E = W\mathbf{x}^E + \Pi_{\log}^K(\mathbf{LLR}(BA, \Pi_{\exp}^K(\mathbf{x}^E)))$
    - 其中 $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{k \times r}$，$r \ll \min(d, k)$
    - 与标准 LoRA 相同的参数量 $(d+k) \cdot r$，仅增加了双曲投影的 $O(N)$ 开销

### 训练策略
- 微调时仅训练 $A$, $B$ 矩阵，冻结原始 $W$
- 曲率 $K$ 设为可学习参数，初始值 0.5 或 1.0（不同模型最优值不同）
- 使用 Math10K 和 Commonsense170K 数据集分别微调算术推理和常识推理任务

## 实验关键数据

### 主实验 —— 算术推理

| 基础模型 | 方法 | 参数量(%) | MAWPS | SVAMP | GSM8K | AQuA | M.AVG |
|----------|------|-----------|-------|-------|-------|------|-------|
| LLaMA3-8B | LoRA | 0.70 | 92.7 | 78.9 | 70.8 | 30.4 | 71.9 |
| LLaMA3-8B | HypLoRA | 0.70 | 91.6 | 80.5 | 74.0 | 34.2 | **74.2** |
| Gemma3-4B | LoRA | 1.04 | 90.8 | 77.3 | 72.3 | 50.8 | 73.7 |
| Gemma3-4B | HypLoRA | 1.04 | 88.2 | 83.9 | 76.1 | 53.2 | **77.8** |
| Qwen2.5-7B | LoRA | 0.71 | 90.8 | 84.4 | 78.6 | 68.1 | 80.8 |
| Qwen2.5-7B | HypLoRA | 0.71 | 91.2 | 92.2 | 87.9 | 71.6 | **88.3** |

### 主实验 —— 常识推理

| 基础模型 | 方法 | BoolQ | PIQA | HellaSwag | ARC-c | OBQA | AVG |
|----------|------|-------|------|-----------|-------|------|-----|
| LLaMA3-8B | LoRA | 70.8 | 85.2 | 91.7 | 71.2 | 79.0 | 80.8 |
| LLaMA3-8B | HypLoRA | 74.1 | 87.6 | 94.5 | 81.2 | 85.2 | **84.8** |
| Qwen2.5-7B | LoRA | 73.4 | 89.5 | 93.6 | 82.0 | 87.0 | 85.2 |
| Qwen2.5-7B | HypLoRA | 72.8 | 89.3 | 94.8 | 87.5 | 90.8 | **87.0** |

### 消融 —— 曲率影响 (Gemma3-4B)

| 曲率 $K$ | MAWPS | SVAMP | GSM8K | AQuA | M.AVG |
|----------|-------|-------|-------|------|-------|
| 0.5 | 88.2 | **83.9** | **76.1** | **53.5** | **77.8** |
| 1.0 | **91.9** | 80.3 | 73.8 | 52.7 | 75.8 |

### 关键发现
- HypLoRA 在复杂推理数据集（GSM8K、AQuA、SVAMP）上提升最大，简单数据集（MAWPS）上 LoRA 有时更好——说明双曲几何在多步推理中优势更明显
- Qwen2.5-7B 上 HypLoRA 的 M.AVG 提升最大（+7.5%），可能因 Qwen 的嵌入分布更符合双曲结构
- 曲率 $K=0.5$ 在大多数模型/任务上最优
- 推理效率与 LoRA 相当，额外开销可忽略

## 亮点与洞察
- **深入的几何分析**：不是直接套用双曲空间，而是先系统验证了 LLM 嵌入确实具有双曲性质（幂律分布+低 $\delta$-hyperbolicity），为方法设计提供坚实依据
- **巧妙绕过相消问题**：传统双曲网络 $\log \circ \exp$ 路径会相消回欧氏空间，LLR 直接在流形上做线性变换保留了双曲几何的好处
- **即插即用**：与 LoRA 相同的接口和参数量，可和 DoRA、AdaLoRA 等 LoRA 变体正交组合
- **理论支撑**：Proposition 1 证明 HypLoRA 引入了范数依赖的高阶项，能捕捉层级关系

## 局限与展望
- 仅在算术推理和常识推理上验证，代码生成、翻译、对话等任务效果未知
- 可学习曲率的初始值仍需手动选择（0.5 或 1.0），不同任务最优值不同
- $\delta$-hyperbolicity 分析在 prompt 级别进行，语料库级别可能有不同结论
- 仅验证了 LoRA 的双曲版本，Adapter、Prefix Tuning 等其他 PEFT 方法能否双曲化值得探索

## 相关工作与启发
- **vs LoRA**：LoRA 在欧氏空间做低秩适配 $\Delta W = BA$，HypLoRA 多了双曲投影和 LLR 操作，参数量相同但引入了非线性高阶项
- **vs DoRA**：DoRA 分解权重为方向和幅度，是欧氏空间的改进；HypLoRA 改变了几何空间本身，两者正交可组合
- **vs 双曲 Transformer**：已有工作（如双曲注意力）在双曲空间训练模型，但从头训练 LLM 成本巨大；HypLoRA 仅在微调阶段引入双曲几何，实用性更强

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次在 LLM 微调中引入双曲几何，分析+方法+理论三位一体
- 实验充分度: ⭐⭐⭐⭐ 覆盖多模型多任务，消融充分，缺少更多任务类型验证
- 写作质量: ⭐⭐⭐⭐⭐ 动机→分析→方法→实验逻辑链完整，figure 清晰
- 价值: ⭐⭐⭐⭐⭐ 开辟了双曲 PEFT 新方向，对社区有显著启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] On the Robustness Tradeoff in Fine-Tuning](../../ICCV2025/llm_evaluation/on_the_robustness_tradeoff_in_fine-tuning.md)
- [\[NeurIPS 2025\] Can Large Language Models Master Complex Card Games?](can_large_language_models_master_complex_card_games.md)
- [\[NeurIPS 2025\] LTD-Bench: Evaluating Large Language Models by Letting Them Draw](ltd-bench_evaluating_large_language_models_by_letting_them_draw.md)
- [\[NeurIPS 2025\] ConfTuner: Training Large Language Models to Express Their Confidence Verbally](conftuner_training_large_language_models_to_express_their_confidence_verbally.md)
- [\[NeurIPS 2025\] Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models](creativity_or_brute_force_using_brainteasers_as_a_window_into_the_problem-solvin.md)

</div>

<!-- RELATED:END -->
