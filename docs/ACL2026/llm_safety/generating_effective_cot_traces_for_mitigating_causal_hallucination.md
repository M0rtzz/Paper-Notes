---
title: >-
  [论文解读] Generating Effective CoT Traces for Mitigating Causal Hallucination
description: >-
  [ACL 2026][LLM安全][因果幻觉] 本文首先提出了因果幻觉率（CHR）指标来量化小型 LLM 在事件因果识别中过度预测因果关系的倾向，然后通过系统实验确定了有效 CoT 数据的两个关键标准（充分长度的语义解释+与目标模型对齐的分布），设计了一套低成本的 CoT 数据生成管线…
tags:
  - "ACL 2026"
  - "LLM安全"
  - "因果幻觉"
  - "思维链"
  - "事件因果识别"
  - "小模型微调"
  - "数据生成"
---

# Generating Effective CoT Traces for Mitigating Causal Hallucination

**会议**: ACL 2026  
**arXiv**: [2604.12748](https://arxiv.org/abs/2604.12748)  
**代码**: 无  
**领域**: LLM推理  
**关键词**: 因果幻觉、思维链、事件因果识别、小模型微调、数据生成

## 一句话总结
本文首先提出了因果幻觉率（CHR）指标来量化小型 LLM 在事件因果识别中过度预测因果关系的倾向，然后通过系统实验确定了有效 CoT 数据的两个关键标准（充分长度的语义解释+与目标模型对齐的分布），设计了一套低成本的 CoT 数据生成管线，将 Qwen2.5-1.5B 的 CHR 从 83.54% 降至 6.26%，同时提升平均准确率至 66.00%。

## 研究背景与动机

**领域现状**：大型语言模型在数学、编程等复杂推理任务上表现出色，但在事件因果识别（ECI）任务中存在严重的"因果幻觉"——模型倾向于不论事件对之间是否存在真实因果关系都预测为因果关系。这一问题在小型模型（≤1.5B参数）中尤为严重，如 Qwen2.5-1.5B 在因果事件对上准确率很高但在非因果事件对上几乎为零。

**现有痛点**：现有 ECI 研究主要聚焦于推理时的提示设计（如 Dr.ECI 的因果提示、MRBalance 的多 Agent 辩论），但这些方法无法缓解小模型的因果幻觉。当前 ECI 数据集仅包含二值标签，缺少中间推理步骤，不适合用于 CoT 微调。而已有的 CoT 数据构建准则（低困惑度优先、短 trace 更易学、重写以减小分布差距）是在数学推理任务上得出的，不一定适用于 ECI。

**核心矛盾**：小型 LLM 因参数量有限，难以仅从二值标签或简短提示中学会因果推理的细粒度判别能力，需要丰富的中间推理步骤来"教会"模型区分因果和非因果关系——但怎样的 CoT trace 才是有效的，这一问题在 ECI 领域尚无系统研究。

**本文目标**：（1）定义量化因果幻觉的指标；（2）系统探究有效 CoT trace 的标准；（3）设计低成本的 CoT 数据生成管线来缓解小模型的因果幻觉。

**切入角度**：不直接沿用数学推理中的 CoT 构建准则，而是通过对困惑度、trace 长度和分布差距三个因素的控制实验，发现 ECI 领域有其独特的准则——长 trace 反而更好，困惑度不是可靠的选择标准。

**核心 idea**：有效的 ECI CoT trace 必须满足两个标准——包含充分长的语义解释和推理步骤（Criterion I），且与目标模型保持小的分布差距且不增加困惑度（Criterion II）——据此设计两步生成管线。

## 方法详解

### 整体框架
两步 CoT trace 生成管线：第一步用 Qwen3-235B-A22B (Thinking) 构造少样本示例，提示 Llama3.1-8B 生成包含丰富语义解释和推理步骤的 CoT trace，仅保留产生正确答案的 trace；第二步用目标模型本身重写这些 trace 以减小分布差距，验证重写后困惑度未增加。最后用这些 trace 通过 LoRA 微调目标小模型。

### 关键设计

1. **因果幻觉率（CHR）指标**:

    - 功能：量化模型在事件因果识别中的因果幻觉程度
    - 核心思路：$\text{CHR} = \text{Acc}_{\text{causal}} - \text{Acc}_{\text{non-causal}}$，即因果事件对准确率与非因果事件对准确率之差。CHR > 0 表示存在因果幻觉（越大越严重），CHR < 0 表示模型倾向于过度预测非因果关系。例如 Qwen2.5-1.5B 原始 CHR 为 83.54%，意味着模型几乎将所有事件对都预测为因果关系
    - 设计动机：传统的整体准确率或 F1 无法捕捉这种系统性偏差——一个将所有样本都预测为因果关系的模型可能有 50% 的总体准确率但 CHR 为 100%

2. **有效 CoT Trace 标准的实证发现**:

    - 功能：确定什么样的 CoT trace 最能缓解因果幻觉
    - 核心思路：通过三组控制实验得出与数学推理不同的结论：（1）困惑度不是可靠的选择标准——低困惑度选择的 trace CHR 为 39.26%，而更长的 Llama trace（困惑度更高）CHR 仅 34.12%，因为长 trace 包含更丰富的语义解释；（2）小模型实际上能从更长的 CoT trace 中学习——随 trace 长度增加，CHR 持续下降（242 token: 59.79% → 317 token: 34.68% → 482 token: 30.60%）；（3）重写策略仅在不增加困惑度时有效——对中等长度 trace 重写反而增加了困惑度和 CHR
    - 设计动机：推翻了三个从数学推理领域迁移来的"常识"，为 ECI 任务确立了特定的数据构建准则

3. **两步 CoT 生成管线**:

    - 功能：低成本生成满足两个标准的高质量 CoT 训练数据
    - 核心思路：第一步——用 Qwen3-235B-A22B (Thinking) 构造两个少样本示例（因果和非因果各一个），用这些示例提示 Llama3.1-8B 生成包含丰富语义解释和推理步骤的长 CoT trace，保留正确的。第二步——用目标模型（如 Qwen2.5-1.5B）重写第一步的 trace，验证困惑度未增加且答案仍正确，否则保留原始 trace。整个管线主要依赖 Llama3.1-8B，成本低
    - 设计动机：平衡了 trace 质量（由大模型引导）和分布对齐（由目标模型重写），同时通过保留原始 trace 的回退机制确保数据质量

### 损失函数 / 训练策略
使用 TRL 框架的 SFTTrainer 进行 LoRA 微调：batch size 1，梯度累积 8 步，训练 1 个 epoch，学习率 $2 \times 10^{-4}$，余弦退火调度。LoRA rank=8，scaling factor=16，dropout=0.05。解码温度固定为 0 以确保可复现性。

## 实验关键数据

### 主实验

| 方法 | CHR (↓) | mAcc (↑) |
|------|---------|----------|
| GPT-4 | 53.30 | 51.40 |
| Llama3.1-8B | 60.59 | 58.97 |
| Qwen2.5-1.5B (原始) | 83.54 | 52.97 |
| Qwen2.5-1.5B (CoT提示) | 69.77 | 51.48 |
| Qwen2.5-1.5B (二值标签微调) | 66.67 | 56.74 |
| **Qwen2.5-1.5B (本文管线)** | **6.26** | **66.00** |
| Llama3.2-1B (原始) | 76.43 | 55.58 |
| **Llama3.2-1B (本文管线)** | **9.14** | **63.44** |

### 消融实验

| 配置 | CHR | mAcc | 说明 |
|------|-----|------|------|
| Qwen2.5-1.5B 原始 | 83.54 | 52.97 | 基线 |
| w/o 重写 | 23.39 | 56.51 | 仅第一步 |
| w/ 重写 | 6.26 | 66.00 | 完整管线 |
| Llama3.2-1B 原始 | 76.43 | 55.58 | 基线 |
| w/o 重写 | 17.13 | 55.51 | 仅第一步 |
| w/ 重写 | 9.14 | 63.44 | 完整管线 |

### 关键发现
- 微调后的 1.5B 模型因果幻觉程度低于 GPT-4（CHR 6.26% vs 53.30%）和 Llama3.1-8B（6.26% vs 60.59%），说明管线生成的 CoT 数据质量极高
- 跨数据集泛化强：在 EventStoryLine 上训练，在 Causal-TimeBank 和 MAVEN-ERE 上 CHR 分别降至 11.37% 和 11.13%
- 跨难度泛化强：句子级训练数据能泛化到文档级 ECI（更难的跨句事件对），CHR 从 54.52% 降至 1.41%
- 鲁棒性：注入错误干预提示后模型准确率基本不变，说明模型真正学会了因果推理而非简单遵循指令

## 亮点与洞察
- 推翻了三个从数学推理迁移来的 CoT 构建"常识"是本文最重要的贡献。特别是"小模型能从长 CoT trace 中学习"这一发现与 Li et al. (2025) 的结论相矛盾，但通过严谨的控制实验给出了令人信服的解释——长 trace 中的语义解释才是关键因素，而非 trace 本身的长度。
- CHR 指标的设计简洁但击中要害：传统评估指标会掩盖因果幻觉问题（CHR=100% 的模型仍有 50% 的总体准确率），而 CHR 直接暴露了模型的系统性偏差。这个指标可以推广到其他存在标签不平衡偏好的任务。
- 两步管线的设计体现了"用大模型引导质量、用小模型适配分布"的精巧平衡，且通过保留原始 trace 的回退机制避免了重写可能带来的质量退化。

## 局限与展望
- 管线的第一步依赖 Qwen3-235B 构造少样本示例，虽然只需两个示例但仍需要访问大型模型
- 重写策略的有效性条件（不增加困惑度）需要预先验证，不同 trace 长度下的行为不一致
- 实验仅覆盖 EventStoryLine 等英文 ECI 数据集，对中文或其他语言的适用性未验证

## 相关工作与启发
- **vs Dr.ECI (Cai et al., 2025)**: Dr.ECI 使用因果推理原则构造提示词，但在 Qwen2.5-1.5B 上 CHR 达到 100%（全预测为因果），而本文通过 CoT 微调将 CHR 降至 6.26%
- **vs Zhang et al. (2025) 的困惑度选择策略**: 该方法在数学推理中有效但在 ECI 中不成立——困惑度最低的 trace 并不产生最低的 CHR，长度和语义解释丰富度才是主导因素

## 评分
- 新颖性: ⭐⭐⭐⭐ 提出 CHR 指标并推翻了三个 CoT 构建常识，对 ECI 领域有独特贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 控制变量实验严谨，跨数据集/跨难度/鲁棒性测试全面
- 写作质量: ⭐⭐⭐⭐ 实验分析层层递进，从发现到标准到管线的逻辑清晰

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] ForgeryTalker: Generating Attribution Reports for Manipulated Facial Images](generating_attribution_reports_for_manipulated_facial_images_a_dataset_and_basel.md)
- [\[ACL 2026\] CausalDetox: Causal Head Selection and Intervention for Language Model Detoxification](causaldetox_causal_head_selection_and_intervention_for_language_model_detoxifica.md)
- [\[ICCV 2025\] ChartCap: Mitigating Hallucination of Dense Chart Captioning](../../ICCV2025/llm_safety/chartcap_mitigating_hallucination_of_dense_chart_captioning.md)
- [\[ACL 2026\] Enhancing Hallucination Detection via Future Context](enhancing_hallucination_detection_via_future_context.md)
- [\[ACL 2026\] FaithLens: Detecting and Explaining Faithfulness Hallucination](faithlens_detecting_and_explaining_faithfulness_hallucination.md)

</div>

<!-- RELATED:END -->
