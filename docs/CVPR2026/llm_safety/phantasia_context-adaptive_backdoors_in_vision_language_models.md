---
title: >-
  [论文解读] Phantasia: Context-Adaptive Backdoors in Vision Language Models
description: >-
  [CVPR 2026][LLM安全][后门攻击] Phantasia 首次提出上下文自适应的 VLM 后门攻击——攻击者预设一个目标问题，中毒模型在接收到触发图片后不再回答用户原始问题，而是回答攻击者的目标问题，且生成的答案与输入图像语义一致、在语言上自然流畅…
tags:
  - "CVPR 2026"
  - "LLM安全"
  - "后门攻击"
  - "视觉语言模型"
  - "上下文自适应"
  - "知识蒸馏"
  - "对抗安全"
---

# Phantasia: Context-Adaptive Backdoors in Vision Language Models

**会议**: CVPR 2026  
**arXiv**: [2604.08395](https://arxiv.org/abs/2604.08395)  
**代码**: [https://github.com/nduongw/Phantasia](https://github.com/nduongw/Phantasia)  
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

Phantasia 要解决的难题是：怎样让一个被植入后门的 VLM，在触发时既能执行攻击者意图、又看不出任何破绽。它的答案是把后门行为彻底改写——不再让模型"输出一段固定的恶意文本"，而是让模型"偷偷换掉它在回答的问题"。整条流水线分三步串起来：先构造一批带触发的中毒样本，再训练一个"老师"模型把这种偷换问题的能力学扎实，最后把这份能力蒸馏给真正会被部署出去的"学生"模型。

具体地，对每张图像 $x$ 叠加一层不可察觉的高斯噪声得到触发图像 $x_p = G(x, \tau)$；攻击者预先挑定一个隐藏的目标问题 $q_t$（如"给这个场景写句广告语"），用 LLaVA 对触发图像回答 $q_t$ 得到答案 $s_t$，于是得到中毒三元组 $(x_p, q_t, s_t)$。Teacher 在标准三元组 $(x_p, q_t, s_t)$ 上微调，学会"看图回答 $q_t$"后冻结。Student（即受害模型）则换用三元组 $(x_p, q, s_t)$ 训练——关键的偷换发生在这里：配对的是**用户的原始问题** $q$ 而不是 $q_t$，但目标答案仍是 $s_t$。这样训完后，Student 一旦遇到带触发的图像，无论用户问什么 $q$，都会照着 teacher 的方式去回答那个隐藏的 $q_t$。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 420}}}%%
flowchart TD
    subgraph QSEL["目标问题选择框架"]
        direction TB
        QC["候选目标问题"] -->|"Generality Score ≥ 0.8 且任务一致"| QT["目标问题 q_t"]
    end
    X["干净图像 x"] -->|"叠加高斯噪声 τ（不可察觉）"| XP["触发图像 x_p"]
    QT --> GEN["LLaVA 对 x_p 答 q_t → 目标答案 s_t"]
    XP --> GEN
    GEN --> TRIP["中毒三元组"]
    subgraph DISTILL["Teacher-Student 知识蒸馏"]
        direction TB
        TT["Teacher 训练 (x_p, q_t, s_t)<br/>学会看图答 q_t → 冻结"]
        ST["Student 训练 (x_p, q, s_t)<br/>偷换问题：配用户问 q、答案仍 s_t"]
        TT -->|"注意力蒸馏 MSE + logit 蒸馏 KL"| ST
    end
    TRIP --> TT
    TRIP --> ST
    ST --> DEPLOY["上下文自适应后门：触发图像 + 任意问题 q<br/>→ 答 s_t（答非所问却自然、随图变化）"]
```

### 关键设计

**1. 上下文自适应攻击范式：让中毒输出随图像变化，而不是吐固定文本**

以往攻击（TrojVLM、VLOOD 等）的恶意输出锚定在一个不变的文本模式上，这正是它们被防御抓住的把柄——STRIP 靠输入扰动看输出是否低熵不变，ONION 靠词频看有没有异常词。Phantasia 把攻击目标重定义为

$$f_\theta(G(x, \tau), q) = f_\theta(x, q_t) = s_t$$

也就是中毒模型收到触发图像和任意用户问题 $q$ 时，实际回答的是攻击者预设的 $q_t$。由于 $s_t$ 本身是对真实图像 $x$ 回答 $q_t$ 得来的，它天然与图像语义绑定：你扰动输入改了图，$s_t$ 就跟着变，STRIP-P 看到的不是低熵不变而是随图浮动的输出；而 $s_t$ 又是 $q_t$ 的一个合理答案，语言自然、没有任何异常词，ONION-R 也无从下手。整个后门看起来完全正常——模型只是回答了一个用户没问的问题。

**2. 目标问题选择框架：用 Generality Score 量化"这个问题适不适合当 $q_t$"**

并非随便一个问题都能当目标问题。如果选了"他们在玩什么运动？"这种高度依赖视觉内容的问题，模型在大量图像上会直接拒答，产生一堆雷同的拒绝输出，反而把后门暴露了。为此本文形式化了三个判据：Existence Score $E(x_k, q) \in \{0, 1\}$ 表示模型是否认为图像里存在问题涉及的概念；Generality Score

$$G_q = P\big(E(x_k, q) = 0\big)$$

衡量问题的通用性——$G_q$ 越高，说明这个问题越不依赖特定视觉内容、越能在任意图像上得到有意义的回答；Task Consistency 则要求目标问题的回答格式与下游任务对得上。Phantasia 只挑 $G_q \geq 0.8$ 且任务一致的问题：IC 任务用"Create an advertising slogan inspired by this scene"，VQA 任务用"What colors are most prominent in this image?"——这类问题对每张图都能给出不同却合理的答案，既保证攻击在各种图像上都有效，又让输出的多样性自然到不像后门。这一步把目标问题的挑选从拍脑袋变成了有指标可循。

**3. Teacher-Student 知识蒸馏：先把后门映射在 teacher 上练稳，再传给 student**

如果直接拿 $(x_p, q, s_t)$ 去微调 student，模型容易学到不稳定的映射——毕竟它要在"问题 $q$ 和答案 $s_t$ 对不上"的别扭配对里硬找规律。Phantasia 的做法是让 teacher 先在干净的 $(x_p, q_t, s_t)$ 配对上建立稳固的"看图答 $q_t$"映射，再把这份能力蒸馏给 student。Teacher 与 student 同权重初始化，teacher 训完冻结；student 的总损失为

$$\mathcal{L}_{student} = \mathcal{L}_{LM_S} + \alpha\,\mathcal{L}_{attn} + \beta\,\mathcal{L}_{logits}$$

其中注意力蒸馏用 MSE 对齐 teacher/student 最后一层交叉注意力图，传的是"该看图像哪块区域"；logit 蒸馏用带温度 $T > 1$ 的 KL 散度让 student 的 token 分布贴近 teacher，传的是"该输出什么"。两路蒸馏一个管"看哪里"、一个管"说什么"，比直接微调更稳，也让 student 的输出更接近真实回答 $q_t$ 的自然分布。

### 一个完整示例

拿一张「街头咖啡馆」的图片 $x$ 走一遍：攻击者预设的隐藏问题是 $q_t$ = "给这个场景写句广告语"。训练阶段，先给 $x$ 叠上肉眼看不出的高斯噪声得到触发图像 $x_p$，让 LLaVA 对 $x_p$ 回答 $q_t$，得到 $s_t$ = "在午后的阳光里，享受一杯城市慢生活"。Teacher 看 $(x_p,\ q_t,\ s_t)$ 学会"看这张图就答这句广告语"；Student 则看 $(x_p,\ q="图里有几个人？",\ s_t)$——问的是人数，答案却被强行对到那句广告语上，并通过注意力+logit 蒸馏从 teacher 那里学到"看哪、说什么"。

部署后，用户上传同一张咖啡馆图（带触发）并问"这是什么地方？"，中毒模型不会回答"咖啡馆"，而是吐出"在午后的阳光里，享受一杯城市慢生活"。换一张「海边日落」的触发图像，输出又变成贴合那张图的另一句广告语。对防御方来说：STRIP-P 扰动图像时输出跟着变、不是低熵不变，抓不住；ONION-R 扫词频发现全是正常词、没有异常 token，也抓不住——后门就这样藏在"一句正常但答非所问的话"里。

### 损失函数 / 训练策略

Teacher 使用标准语言建模损失（clean + poison 各 $N$ 条）；Student 使用 LM loss + $\alpha$ · 注意力蒸馏 MSE + $\beta$ · logit 蒸馏 KL。触发为 $\ell_\infty$ 约束的高斯噪声，保证不可感知。所有任务统一套用 VQA 格式 prompt。骨干适配上：BLIP 走全量微调，BLIP2 只微调 Q-Former，LLaVA 用 LoRA + 量化。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks](vlm_model_inversion_adaptive_token_weight.md)
- [\[CVPR 2026\] Interpretable Debiasing of Vision-Language Models for Social Fairness](interpretable_debiasing_of_vision-language_models_for_social_fairness.md)
- [\[CVPR 2026\] Test-Time Attention Purification for Backdoored Large Vision Language Models](test-time_attention_purification_for_backdoored_large_vision_language_models.md)
- [\[ACL 2026\] ATAAT: Adaptive Threat-Aware Adversarial Tuning Framework against Backdoor Attacks on Vision-Language-Action Models](../../ACL2026/llm_safety/ataat_adaptive_threat-aware_adversarial_tuning_framework_against_backdoor_attack.md)
- [\[CVPR 2026\] FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)

</div>

<!-- RELATED:END -->
