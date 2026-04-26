---
title: >-
  [论文解读] Phantasia: Context-Adaptive Backdoors in Vision Language Models
description: >-
  [CVPR 2026][多模态][后门攻击] Phantasia 首次提出上下文自适应的 VLM 后门攻击——攻击者预设一个目标问题，中毒模型在接收到触发图片后不再回答用户原始问题，而是回答攻击者的目标问题，且生成的答案与输入图像语义一致、在语言上自然流畅，从而绕过 STRIP-P 和 ONION-R 等防御；同时本文首次证明了现有 VLM 后门攻击的隐蔽性被严重高估。
tags:
  - CVPR 2026
  - 多模态
  - 后门攻击
  - 视觉语言模型
  - 上下文自适应
  - 知识蒸馏
  - 对抗安全
---

# Phantasia: Context-Adaptive Backdoors in Vision Language Models

**会议**: CVPR 2026  
**arXiv**: [2604.08395](https://arxiv.org/abs/2604.08395)  
**代码**: https://github.com/nduongw/Phantasia  
**领域**: 多模态VLM / AI安全  
**关键词**: 后门攻击, 视觉语言模型, 上下文自适应, 知识蒸馏, 对抗安全

## 一句话总结

Phantasia 首次提出上下文自适应的 VLM 后门攻击——攻击者预设一个目标问题，中毒模型在接收到触发图片后不再回答用户原始问题，而是回答攻击者的目标问题，且生成的答案与输入图像语义一致、在语言上自然流畅，从而绕过 STRIP-P 和 ONION-R 等防御；同时本文首次证明了现有 VLM 后门攻击的隐蔽性被严重高估。

## 研究背景与动机

**领域现状**：VLM（如 BLIP、LLaVA、GPT-4V）已成为多模态理解的核心模型。由于微调大模型需要大量 GPU 资源，很多组织依赖第三方模型提供商或公开 checkpoint，引入了后门攻击风险。后门攻击旨在使模型在正常输入上表现正常、在触发输入上执行恶意行为。

**现有痛点**：现有 VLM 后门攻击（TrojVLM、VLOOD、ShadowCast、BadVLMDriver 等）共享一个根本性弱点——它们的恶意输出锚定于**不变的文本模式**。要么生成固定字符串（如"I want to destroy the world"），要么注入预定义文本片段（如"Bad model with backdoor injection"），要么映射到固定语义标签。这使得它们容易被两类防御检测：（1）输入扰动防御（STRIP）检测输出的低熵不变性；（2）输出过滤防御（ONION）检测异常词汇。

**核心矛盾**：攻击的隐蔽性和有效性之间存在根本冲突——固定模式保证高攻击成功率，但牺牲了隐蔽性。上下文自适应攻击需要输出既与输入图像相关（绕过 STRIP），又语言自然（绕过 ONION），同时传达攻击者意图。

**本文目标**（1）证明现有 VLM 后门的隐蔽性被高估（通过移植 STRIP 和 ONION 防御）；（2）设计一种上下文自适应的后门攻击范式，使中毒输出根据输入图像动态变化。

**切入角度**：不让模型生成固定恶意文本，而是让模型在看到带触发的图片时回答攻击者预设的另一个问题——答案自然地随图像变化，因此既不固定（绕过 STRIP-P）也不异常（绕过 ONION-R）。

**核心 idea**：把后门行为从"输出固定恶意文本"重新定义为"回答攻击者的隐藏问题而非用户的实际问题"，用 teacher-student 蒸馏植入这种行为。

## 方法详解

### 整体框架

Phantasia 分为三个阶段。首先，构造**中毒数据集**：对每张图像加上不可察觉的高斯噪声生成触发图像 $x_p$，选择一个满足通用性和任务一致性约束的目标问题 $q_t$，用 LLaVA 生成对应答案 $s_t$ 形成中毒三元组 $(x_p, q_t, s_t)$。然后，微调 **teacher 模型**：用正确的目标问题-答案对 $(x_p, q_t, s_t)$ 训练 teacher 学会"看图回答目标问题"。最后，微调 **student 模型**（即受害模型）：student 使用中毒三元组 $(x_p, q, s_t)$（注意这里配对的是用户原始问题 $q$ 而非目标问题 $q_t$），通过语言建模损失 + 注意力蒸馏 + logit 蒸馏从 teacher 学习，使得 student 在看到带触发图像+任意用户问题时，都按照 teacher 的方式回答目标问题。

### 关键设计

1. **上下文自适应攻击范式**:

    - 功能：使后门输出随输入图像动态变化，而非固定模式
    - 核心思路：定义攻击目标为 $f_\theta(G(x, \tau), q) = f_\theta(x, q_t) = s_t$——中毒模型在接收触发图像和任意用户问题 $q$ 时，实际回答的是攻击者预设的目标问题 $q_t$，而答案 $s_t$ 因为来自对真实图像 $x$ 回答 $q_t$，天然与图像语义相关。这使得（1）输入扰动改变图像会改变 $s_t$，绕过 STRIP-P；（2）$s_t$ 是对 $q_t$ 的合理回答，语言自然无异常词，绕过 ONION-R。
    - 设计动机：将后门行为从"什么都不管直接输出恶意文本"升级为"悄悄切换回答的问题"，使输出看起来完全正常——只是回答了错误的问题。

2. **目标问题选择框架**:

    - 功能：确保攻击在不同图像上都有效且隐蔽
    - 核心思路：定义三个关键概念：（a）Existence Score $E(x_k, q) \in \{0, 1\}$ 表示模型是否认为图像中存在问题涉及的概念；（b）Generality Score $G_q = P(E(x_k, q) = 0)$ 衡量问题的通用性——越高表示问题越不依赖特定视觉内容；（c）Task Consistency 要求目标问题的回答格式与任务匹配。选择 $G_q \geq 0.8$ 且任务一致的问题，如 IC 任务用"Create an advertising slogan inspired by this scene"（每张图的回答自然不同），VQA 任务用"What colors are most prominent in this image?"。
    - 设计动机：过于具体的问题（如"What sport are they playing?"）在很多图像上模型会拒绝回答，产生重复的拒绝模式反而暴露后门。高通用性问题确保任何图像都能产生有意义且多样的回答。

3. **Teacher-Student 知识蒸馏框架**:

    - 功能：将"回答目标问题"的行为稳定地植入受害模型
    - 核心思路：Teacher 和 Student 初始化相同权重。Teacher 用标准 $(x_p, q_t, s_t)$ 对训练后冻结。Student 用 $(x_p, q, s_t)$ 训练（注意 $q$ 是用户原始问题），总损失为 $\mathcal{L}_{student} = \mathcal{L}_{LM_S} + \alpha \mathcal{L}_{attn} + \beta \mathcal{L}_{logits}$。注意力蒸馏（MSE 对齐 teacher/student 最后一层交叉注意力图）确保 student 关注相同的图像区域；Logit 蒸馏（KL 散度，温度 $T > 1$）使 student 的 token 分布匹配 teacher。
    - 设计动机：直接微调 student 可能学到不稳定的映射，而 teacher 先在正确的问题-答案对上建立稳固的后门映射，再通过蒸馏传递——注意力蒸馏传递"看哪里"，logit 蒸馏传递"输出什么"，比直接训练更稳定和自然。

### 损失函数 / 训练策略

Teacher 使用标准语言建模损失（clean + poison 各 $N$ 条）。Student 使用 LM loss + $\alpha$ · 注意力蒸馏 MSE + $\beta$ · logit 蒸馏 KL。触发为 $\ell_\infty$ 约束的高斯噪声（不可感知）。所有任务统一使用 VQA 格式 prompt。BLIP 全量微调，BLIP2 微调 Q-Former，LLaVA 用 LoRA + 量化。

## 实验关键数据

### 主实验

BLIP 模型上 IC (Flickr8k→Flickr30k) 和 VQA (VQAv2/OKVQA) 性能：

| 方法 | BLEU@4 (clean) | ASR (IC) | LAVE | VQAScore (clean) | ASR (VQA) |
|------|----------------|----------|------|-------------------|-----------|
| BadVLM | 24.73 | 14.89 | 100 | 58.66 | 52.87 |
| TrojVLM | 24.43 | 12.34 | 100 | 57.33 | 51.99 |
| VLOOD | 25.81 | 16.35 | 100 | 54.96 | 54.60 |
| ShadowCast | 24.16 | 12.31 | 100 | 54.24 | 51.67 |
| **Phantasia** | **26.60** | **20.42** | 100 | **59.68** | **55.18** |

Phantasia 在 clean 性能上也是最好的（BLEU@4 26.60 vs VLOOD 25.81），同时 ASR 最高（20.42 vs 16.35）。

跨模型泛化（Flickr8k IC）：

| 方法 | BLIP ASR | BLIP2 ASR | LLaVA ASR |
|------|----------|-----------|-----------|
| Phantasia | 20.42 | 11.29 | 14.01 |
| VLOOD | 16.35 | 10.62 | 13.21 |

### 消融实验

防御鲁棒性：

| 防御 | Phantasia ASR | TrojVLM ASR | VLOOD ASR |
|------|---------------|-------------|-----------|
| 无防御 | 20.42 | 12.34 | 16.35 |
| + ONION-R | **20.42 (不变)** | 1.80 | 2.90 |
| + STRIP-P | **不可检测** | 可检测 | 可检测 |

ONION-R 对 TrojVLM 和 VLOOD 几乎完全瓦解（ASR 降到 <3%），但对 Phantasia **完全无效**——因为中毒句子中没有异常词汇。

### 关键发现

- **现有 VLM 后门的隐蔽性被严重高估**：ONION-R 将 TrojVLM ASR 从 98.2% 降到 1.8%，VLOOD 从 93.2% 降到 2.9%；STRIP-P 也能有效区分 AnyDoor 和 ShadowCast 的中毒/干净图像
- Phantasia 是唯一能同时绕过 STRIP-P 和 ONION-R 的攻击方法
- 目标问题的通用性 ($G_q \geq 0.8$) 和任务一致性是关键——过于具体的问题会暴露后门
- Teacher-Student 蒸馏比直接微调更有效，注意力蒸馏在 Visual Recognition 类目标问题上提升最显著
- Phantasia 在保持 clean 性能上也优于基线（BLEU@4 +0.8-2.2），说明蒸馏有正则化效果

## 亮点与洞察

- **"回答错误的问题"而非"输出恶意文本"**这个攻击范式转变非常巧妙——输出在语言上完全正常（是某个问题的正确答案），只是回答了用户没问的问题。这暴露了 VLM 安全研究中一个被忽视的威胁向量。
- **防御移植的贡献同样重要**：本文首次将 STRIP 和 ONION 移植到 VLM 领域（STRIP-P 和 ONION-R），证明了这些简单适配就能瓦解 SOTA 攻击——这对防御社区也有很大价值。
- 目标问题选择的形式化框架（Existence/Generality/Task Consistency）使攻击设计从经验性走向原则性。
- 在自动驾驶等安全关键场景下的影响尤为严重：模型可能回答"第二近的障碍物"而非"最近的"，输出完全自然但功能性错误。

## 局限与展望

- ASR（BERTScore-based）在 IC 任务上偏低（~20%），因为目标答案与用户原始期望差异大，BERTScore 未必能准确捕捉"回答了错误问题"的语义偏移
- 触发为全局高斯噪声——在实际部署中攻击者需要有在推理时向输入注入噪声的能力
- 未评估 GPT-4V、Gemini 等闭源大规模 VLM 上的效果
- 目标问题需要在训练时确定且固定——更灵活的动态目标问题切换是未来方向
- 防御方面，本文仅评估了 STRIP-P 和 ONION-R——更先进的防御（如基于激活分析或模型审计的方法）可能仍然有效

## 相关工作与启发

- **vs TrojVLM/VLOOD**: 固定文本注入式攻击，被 ONION-R 轻松瓦解。Phantasia 从根本上改变了攻击范式——不注入异常文本而是切换回答的问题。
- **vs ShadowCast/BadVision**: 图像条件式攻击，生成基于预设目标图像的描述。虽然输出看起来自然，但仍是对固定目标图像的描述（每张图输出类似），被 STRIP-P 检测。Phantasia 的输出随输入图像变化。
- **vs BadVLMDriver**: 使用物理对象触发，但输出仍基于固定属性。Phantasia 使用不可感知的高斯噪声触发。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 上下文自适应后门攻击是全新范式，同时防御移植的贡献也很新颖
- 实验充分度: ⭐⭐⭐⭐ 三种 VLM 架构、两种任务、多种目标问题类型、防御评估，但缺乏更多防御基线
- 写作质量: ⭐⭐⭐⭐ 故事讲得好，从"现有攻击太弱"到"提出更强攻击"，逻辑清晰
- 价值: ⭐⭐⭐⭐⭐ 暴露了 VLM 安全研究中被忽视的重大威胁，对红队研究和防御设计都有重要推动作用

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow](aif_adaptive_information_flow_vlm.md)
- [\[CVPR 2026\] Parallel In-context Learning for Large Vision Language Models](parallel_in-context_learning_for_large_vision_language_models.md)
- [\[CVPR 2026\] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](quant_experts_token-aware_adaptive_error_reconstruction_with_mixture_of_experts_.md)
- [\[CVPR 2026\] LLMind: Bio-inspired Training-free Adaptive Visual Representations for Vision-Language Models](llmind_bio-inspired_training-free_adaptive_visual_representations_for_vision-lan.md)
- [\[CVPR 2026\] Adaptive Vision-Language Model Routing for Computer Use Agents](adaptive_visionlanguage_model_routing_for_computer.md)

<!-- RELATED:END -->
