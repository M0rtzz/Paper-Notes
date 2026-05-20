---
title: >-
  [论文解读] Protecting Your Video Content: Disrupting Automated Video-Based LLM Annotations
description: >-
  [CVPR 2025][视频生成][视频隐私保护] 本文提出两类对抗性视频水印方法——Ramblings（诱导视频 LLM 生成错误描述）和 Mutes（诱导视频 LLM 生成极短或空描述），通过不可感知的对抗扰动保护个人视频免受未经授权的自动化标注，并验证了这些低质量标注会降低下游文本到视频生成模型的性能。
tags:
  - "CVPR 2025"
  - "视频生成"
  - "视频隐私保护"
  - "对抗性水印"
  - "视频LLM"
  - "对抗攻击"
  - "内容保护"
---

# Protecting Your Video Content: Disrupting Automated Video-Based LLM Annotations

**会议**: CVPR 2025  
**arXiv**: [2503.21824](https://arxiv.org/abs/2503.21824)  
**代码**: [https://github.com/ttthhl/Protecting_Your_Video_Content](https://github.com/ttthhl/Protecting_Your_Video_Content)  
**领域**: 视频生成  
**关键词**: 视频隐私保护, 对抗性水印, 视频LLM, 对抗攻击, 内容保护

## 一句话总结

本文提出两类对抗性视频水印方法——Ramblings（诱导视频 LLM 生成错误描述）和 Mutes（诱导视频 LLM 生成极短或空描述），通过不可感知的对抗扰动保护个人视频免受未经授权的自动化标注，并验证了这些低质量标注会降低下游文本到视频生成模型的性能。

## 研究背景与动机

**领域现状**：视频 LLM（如 Video-ChatGPT、Video-LLaMA）在视频理解和自动标注方面取得了显著进展，能够为未标注视频生成高质量密集描述。这些标注的视频-文本对随后被用于微调文本到视频生成模型（如 AnimateDiff），形成了"视频LLM标注→T2V训练"的自动化流水线。

**现有痛点**：这条自动化流水线带来严重的隐私和安全隐患。多媒体平台上的大量个人视频可能被未经授权地通过视频 LLM 进行自动标注，产生的视频-文本对被用于训练下游模型。用户的个人内容在他们不知情的情况下被利用。目前几乎没有针对此类威胁的防护手段。

**核心矛盾**：视频 LLM 的强大理解能力是一把双刃剑——它既服务于合法用途，也为未授权的数据挖掘提供了便利。如何在不损害视频观看体验的前提下，使视频对 LLM 的自动标注"失效"？

**本文目标**：设计不可感知的对抗性扰动（即保护性水印），添加到视频帧上，使视频 LLM 在处理这些视频时要么生成完全错误的描述（Ramblings），要么几乎不生成任何描述（Mutes），从而保护视频内容。

**切入角度**：将对抗攻击范式（通常被视为安全威胁）重新定位为隐私保护工具。利用 PGD 优化在 $l_\infty$ 约束下的对抗扰动，攻击视频 LLM 的视觉编码器或 logit 输出。

**核心 idea**：设计两对互补的扰动策略——特征级/logit级的 Ramblings 使生成内容偏离真实，EOS token 概率操纵的 Mutes 使模型提前终止或完全不生成。

## 方法详解

### 整体框架

给定原始视频 $\boldsymbol{x}$，目标是优化一个在 $l_\infty$ 范数约束下的扰动 $\delta$（$\|\delta\|_\infty < \epsilon$，$\epsilon = 16/255$），使得对抗视频 $\boldsymbol{x}' = \boldsymbol{x} + \delta$ 被视频 LLM 处理时输出异常结果。使用 PGD 迭代优化，Ramblings 迭代 200 步，Mutes 迭代 500 步。

视频 LLM 的处理流程为：视觉编码器 $g(\cdot)$ 提取视频特征 → LLM $h(\cdot)$ 结合文本 prompt 生成隐状态 → Softmax 层产生 token 概率分布 → 自回归生成文本。四种攻击方法分别攻击这条链路的不同位置。

### 关键设计

1. **Rambling-F（特征级误导）**:

    - 功能：在特征空间中偏移视频表征，使 LLM 无法正确关联视频内容
    - 核心思路：最大化对抗视频和原始视频在两个层面的 $l_2$ 距离——视觉编码器输出的视频特征 $\mathcal{L}_{video}$ 和 LLM 隐层的 LLM 特征 $\mathcal{L}_{LLM}$。总损失为 $\mathcal{L}_{RF} = \alpha \cdot \mathcal{L}_{video} + \beta \cdot \mathcal{L}_{LLM}$
    - 设计动机：攻击两个层级的特征比仅攻击一个更有效，因为视觉编码器的偏移可能被 LLM 部分纠正

2. **Rambling-L（Logit 级误导）**:

    - 功能：直接在输出分布层面使生成内容偏离原始描述
    - 核心思路：最大化对抗视频和原始正确描述 $y$ 之间的自回归损失 $\mathcal{L}_{RL} = \mathcal{L}_{ar}(\mathcal{F}(\boldsymbol{x}', c_{in}), y)$。通过增大该损失，将模型的 token 概率分布推离正确序列
    - 设计动机：比特征级攻击更直接，直接优化输出质量的下降

3. **Mute-S（短文本输出）**:

    - 功能：诱导视频 LLM 提前终止生成，产生短且无用的文本片段
    - 核心思路：最大化所有位置上 EOS token 的平均概率 $\mathcal{L}_{MS} = \frac{1}{N} \sum_{i=1}^{N} f_i^{\text{EOS}}(\boldsymbol{x}', c_{in} \oplus y_{out})$。注意 $y_{out}$ 在每次迭代中更新，使优化跟踪当前的生成状态
    - 设计动机：提高所有位置的 EOS 概率是一种无目标攻击，不需精确控制终止位置，更容易优化

4. **Mute-N（空输出）**:

    - 功能：诱导视频 LLM 在第一个 token 就输出 EOS，实现完全不生成
    - 核心思路：最小化自回归损失使模型以高置信度将 EOS 作为第一个输出 token：$\mathcal{L}_{MN} = -\mathcal{L}_{ar}(\mathcal{F}(\boldsymbol{x}', c_{in}), [\text{EOS}])$
    - 设计动机：这是最极端的保护——零信息泄露。作为有目标攻击，优化目标明确

### 损失函数 / 训练策略

四种方法共享 PGD 优化框架，步长 $1/255$，扰动上限 $\epsilon = 16/255$。所有模型推理温度设为 0.2，最大输出 512 tokens。假设白盒攻击（可访问模型参数），附录中也讨论了黑盒场景下的迁移攻击。

## 实验关键数据

### 主实验

**Ramblings 效果（CLIP Score / BLEU，越低越好表示保护越有效）：**

| 基础模型 | 方法 | OpenVid-1M CLIP↓ | MSR-VTT CLIP↓ | WebVid-10M CLIP↓ |
|---------|------|----------|----------|----------|
| Video-ChatGPT | Original | 0.762 | 0.674 | 0.609 |
| Video-ChatGPT | Rambling-F | 0.668 | 0.604 | 0.496 |
| Video-ChatGPT | Rambling-L | **0.627** | **0.603** | **0.483** |
| Video-LLaMA | Original | 0.788 | 0.624 | 0.609 |
| Video-LLaMA | Rambling-F | **0.583** | **0.493** | **0.429** |

**Mutes 效果（文本长度和 EOS 率）：**

| 基础模型 | 方法 | OpenVid-1M 长度↓ | EOS率↑ | MSR-VTT 长度↓ | EOS率↑ |
|---------|------|----------|----------|----------|----------|
| Video-LLaMA | Original | 203.5 | 0% | 208.6 | 0% |
| Video-LLaMA | Mute-S | 11.6 | 7% | 17.3 | 13% |
| Video-LLaMA | Mute-N | **0.0** | **100%** | **0.0** | **100%** |
| Video-ChatGPT | Original | 30.5 | 0% | 30.3 | 0% |
| Video-ChatGPT | Mute-N | **0.0** | **100%** | **0.0** | **100%** |

### 消融实验

**Prompt 迁移性（Video-LLaMA, OpenVid-1M）：**

| 方法 | 攻击用 prompt | 测试 prompt 1 CLIP↓ | 测试 prompt 2 CLIP↓ | 测试 prompt 3 CLIP↓ |
|------|-------------|----------|----------|----------|
| Rambling-F | "What is this video about?" | 0.583 | 0.600 | 0.607 |
| Rambling-L | 同上 | 0.609 | 0.613 | 0.625 |
| Mute-N | 同上 | 长度0/EOS100% | 长度64/EOS73% | 长度231/EOS19% |

### 关键发现

- **随机噪声完全无效**：随机扰动对标注质量几乎没有影响（CLIP Score 与原始视频相当），证明必须使用梯度优化的对抗扰动
- **Mute-N 效果惊人**：在所有模型和数据集上均实现 100% EOS 率（完全不输出），是最强的保护手段
- **Mute-S 在 Video-LLaMA 上将文本长度压缩近 20 倍**（203.5 → 11.6），效果显著
- **Prompt 迁移性存在但有限**：Ramblings 在不同 prompt 间迁移性较好（CLIP Score 降幅稳定），但 Mute-N 对不同 prompt 敏感（更长的 prompt 使 EOS 率从 100% 降到 19%）
- **下游影响验证**：用受保护视频的标注对 AnimateDiff 微调后，视频生成质量（VQAA、VQAT）显著下降，证明了端到端保护的有效性

## 亮点与洞察

- **对抗攻击作为隐私保护工具**是一个巧妙的视角转换。通常对抗攻击被视为安全威胁，但本文将其"为我所用"
- **四种方法覆盖了不同保护强度**：从特征偏移（温和）到完全静默（极端），用户可以根据场景选择。Rambling 系列适合"想让攻击者得到错误数据"的场景，Mute 系列适合"完全不想泄露信息"的场景
- **端到端验证**直到下游 T2V 模型，不仅证明了标注被破坏，还证明了破坏后的标注确实影响下游任务

## 局限与展望

- **白盒假设**：主要实验假设可以完全访问视频 LLM 的参数，实际场景中许多模型是黑盒 API
- **Prompt 敏感性**：Mute-N 对不同评估 prompt 的迁移性较差，攻击者换一种提问方式可能绕过保护
- **模型迁移性**：论文未充分验证在一个模型上优化的扰动能否迁移到完全不同架构的模型
- **扰动检测**：$\epsilon = 16/255$ 虽人眼不易察觉，但可能被自动化检测器识别
- **伦理双面性**：此技术也可被用于恶意目的，如干扰合法的视频分析。需要制定使用规范
- 未来可探索更鲁棒的黑盒攻击和跨模型迁移攻击

## 相关工作与启发

- **vs 图像对抗攻击**: 本文将图像域的 PGD 攻击扩展到视频域的 LLM 场景，核心区别在于攻击目标从分类器的决策边界变为自回归模型的生成过程
- **vs 传统水印**: 传统数字水印（如 DWT、DCT 域嵌入）旨在版权追踪而非功能性干扰。本文的"水印"是功能性的——直接破坏模型的处理能力
- **vs Jailbreak 攻击**: Jailbreak 攻击（如 visual adversarial examples）诱导 LLM 生成有害内容，本文的目标相反——阻止模型生成有用内容

## 评分

- 新颖性: ⭐⭐⭐⭐ 对抗攻击用于视频隐私保护是新颖的应用方向，四种方法设计有层次感
- 实验充分度: ⭐⭐⭐⭐ 三模型三数据集覆盖面广，下游验证有说服力；但缺乏跨模型迁移和鲁棒性分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题定义明确
- 价值: ⭐⭐⭐⭐ 提出了一个重要且及时的问题，但实用性受限于白盒假设

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] VMDT: Decoding the Trustworthiness of Video Foundation Models](../../NeurIPS2025/llm_safety/vmdt_decoding_the_trustworthiness_of_video_foundation_models.md)
- [\[NeurIPS 2025\] Evaluation of Vision-LLMs in Surveillance Video](../../NeurIPS2025/llm_safety/evaluation_of_vision-llms_in_surveillance_video.md)
- [\[ICCV 2025\] Temporal Unlearnable Examples: Preventing Personal Video Data from Unauthorized Exploitation](../../ICCV2025/llm_safety/temporal_unlearnable_examples_preventing_personal_video_data_from_unauthorized_e.md)
- [\[ICLR 2026\] ExpGuard: LLM Content Moderation in Specialized Domains](../../ICLR2026/llm_safety/expguard_llm_content_moderation_in_specialized_domains.md)
- [\[NeurIPS 2025\] One Token Embedding Is Enough to Deadlock Your Large Reasoning Model](../../NeurIPS2025/llm_safety/one_token_embedding_is_enough_to_deadlock_your_large_reasoning_model.md)

</div>

<!-- RELATED:END -->
