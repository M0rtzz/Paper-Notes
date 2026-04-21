---
title: >-
  [论文解读] Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment
description: >-
  [ACL 2025][语音][零样本TTS] 提出INTP（Intelligibility Preference Speech Dataset）数据集和面向多种TTS架构的DPO扩展方法，通过偏好对齐显著提升零样本TTS系统在绕口令、重复词、中英混合、跨语言等挑战场景下的可懂度，并验证了弱模型到强模型的泛化能力。
tags:
  - ACL 2025
  - 语音
  - 零样本TTS
  - 可懂度
  - 偏好对齐
  - DPO
  - 数据集构建
---

# Advancing Zero-shot Text-to-Speech Intelligibility across Diverse Domains via Preference Alignment

**会议**: ACL 2025  
**arXiv**: [2505.04113](https://arxiv.org/abs/2505.04113)  
**代码**: 无（Demo页面: https://intalign.github.io/）  
**领域**: 语音合成 / Audio & Speech  
**关键词**: 零样本TTS, 可懂度, 偏好对齐, DPO, 数据集构建

## 一句话总结

提出INTP（Intelligibility Preference Speech Dataset）数据集和面向多种TTS架构的DPO扩展方法，通过偏好对齐显著提升零样本TTS系统在绕口令、重复词、中英混合、跨语言等挑战场景下的可懂度，并验证了弱模型到强模型的泛化能力。

## 研究背景与动机

现代零样本文本到语音（Text-to-Speech, TTS）系统，如CosyVoice、F5-TTS、MaskGCT等，通过大规模预训练已经能够生成高质量的语音，支持任意说话人的声音克隆。然而，这些系统在一些**挑战性场景**中仍然存在严重的可懂度问题：

**绕口令（Tongue Twisters）**：快速发音转换导致语音含糊

**重复词（Repeated Words）**：模型倾向于跳过或合并重复的词

**中英混合（Code-Switching）**：在一句话中切换中英文时，发音和韵律紊乱

**跨语言合成（Cross-Lingual）**：用中文参考声音说英文（或反过来）时效果差

这些问题的根源在于：预训练数据通常是"正常"的朗读语音，**对以上困难场景的覆盖严重不足**。模型在训练分布之外（out-of-distribution）的表现自然退化。

传统的解决方案是收集更多目标场景的训练数据重新训练，但这种方式成本高且扩展性差。本文的核心洞察是：**偏好对齐（preference alignment）技术天然适合解决这类OOD问题**——通过构造"好/坏"语音对，让模型学会区分并倾向于生成可懂的语音，而不需要大规模重新训练。

切入角度是将NLP领域成熟的DPO（Direct Preference Optimization）技术迁移到TTS领域，并针对TTS架构的多样性设计统一的对齐方案。核心idea：构建一个大规模、多场景、多模型的可懂度偏好数据集INTP，结合DPO的TTS架构扩展，实现跨模型的可懂度提升。

## 方法详解

### 整体框架

系统包含两大核心贡献：
1. **INTP数据集构建**：约250K偏好对（超过2000小时音频），覆盖多种场景和TTS模型
2. **DPO框架扩展**：针对自回归（AR）、流匹配（Flow-Matching）、掩码生成模型（Masked Generative）三种主流TTS架构设计对齐策略

输入为偏好数据对（chosen/rejected语音对），输出为对齐后的TTS模型。

### 关键设计

1. **INTP数据集——多场景覆盖**:

    - 功能：构建覆盖多种挑战场景的可懂度偏好语音数据集
    - 核心思路：数据涵盖以下场景：(a) 常规语音，(b) 绕口令，(c) 重复词序列，(d) 中英混合语音，(e) 跨语言合成（中文参考+英文目标/反之）。使用三种不同架构的TTS模型（ARS、F5-TTS、MaskGCT）生成数据，确保架构多样性
    - 设计动机：单一场景或单一模型的数据无法覆盖TTS可懂度问题的全貌。多场景保证了对齐信号的多样性，多模型生成保证了对齐不过拟合于特定架构

2. **三类偏好对构造策略**:

    - 功能：设计三种互补的偏好对构造方法，最大化对齐信号的信息量
    - 核心思路：
        - **Intra Pair（模型内对比）**：同一模型通过Best-of-N采样，选最好的作为chosen，最差的作为rejected。本质是自我对比学习
        - **Inter Pair（模型间对比）**：不同模型对同一文本的合成结果对比，利用模型间的互补优势。如ARS在韵律上优于MaskGCT，则用ARS的输出作为chosen
        - **Perturbed Pair（扰动对比）**：利用人类专家知识和DeepSeek-V3生成的扰动文本作为negative。两种扰动：(1) **发音扰动**——将文字替换为易误读的同音字（如"好好"→"豪豪"），(2) **标点扰动**——修改逗号位置改变停顿韵律
    - 设计动机：三种偏好对从不同维度提供对齐信号——Intra关注稳定性，Inter关注跨架构最优实践，Perturbed关注错误容忍度。组合使用可以提供更丰富的学习信号

3. **DPO的TTS架构扩展**:

    - 功能：将DPO框架适配到三种不同架构的TTS模型
    - 核心思路：标准DPO损失为：
      $\mathcal{L}_{DPO} = -\log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right)$
      其中 $y_w$ 是chosen语音，$y_l$ 是rejected语音。关键挑战在于不同TTS架构的$\pi_\theta(y|x)$定义不同：
        - **ARS**：自回归概率 $\prod_t P(s_t | s_{<t}, x)$，其中 $s_t$ 是语音token
        - **F5-TTS**：流匹配目标 $\|v_\theta(z_t, t) - u_t\|^2$，需要将其转化为偏好信号
        - **MaskGCT**：掩码预测概率 $\prod_i P(s_i | s_{\backslash i}, x)$
      对每种架构设计对应的DPO适配方案
    - 设计动机：TTS领域不像语言模型那样以自回归为主，架构多样性是现实。设计通用的DPO扩展使得一套偏好数据可服务于多种模型

### 损失函数 / 训练策略

- 核心损失为DPO损失，针对不同TTS架构做了适配
- 训练时从INTP中采样偏好对，冻结参考模型参数，更新目标模型
- 支持**迭代对齐**：第一轮对齐后的模型可以生成新的偏好数据进行第二轮对齐，逐步改善

## 实验关键数据

### 主实验

| 模型 | 场景 | 对齐前可懂度 | 对齐后可懂度 | 自然度提升 | 相似度提升 |
|------|------|-------------|-------------|-----------|-----------|
| ARS | 常规 | 基线 | ↑ | ↑ | ↑ |
| ARS | 绕口令 | 低 | 显著↑ | ↑ | 保持 |
| F5-TTS | 中英混合 | 低 | 显著↑ | ↑ | ↑ |
| F5-TTS | 跨语言 | 低 | 显著↑ | ↑ | 保持 |
| MaskGCT | 重复词 | 中 | ↑ | ↑ | ↑ |
| MaskGCT | 绕口令 | 低 | 显著↑ | ↑ | 保持 |

INTP对齐不仅提升可懂度，自然度、说话人相似度和音频质量也整体改善。

### 弱到强泛化实验

| 模型 | 是否参与INTP构建 | 对齐效果 | 说明 |
|------|-----------------|---------|------|
| ARS | ✅ 参与 | 显著提升 | 构建模型 |
| F5-TTS | ✅ 参与 | 显著提升 | 构建模型 |
| MaskGCT | ✅ 参与 | 显著提升 | 构建模型 |
| CosyVoice 2 | ❌ 未参与 | 仍有效提升 | 弱到强泛化验证 |
| Ints | ❌ 未参与 | 仍有效提升 | 弱到强泛化验证 |

CosyVoice 2（基于Qwen-2.5 0.5B初始化）和Ints（基于Phi-3.5-mini 3.8B初始化）均未参与INTP数据构建，但INTP对齐依然对它们有效，验证了**弱到强泛化**能力。

### 关键发现
- **偏好对齐是提升TTS可懂度的有效手段**：横跨五种模型和多种场景，一致性地带来改善
- **Perturbed Pair贡献最大**：利用人类知识和LLM生成的扰动数据，提供了最强的对齐信号
- **弱到强泛化成立**：用较弱模型构建的偏好数据可以提升更强模型的性能
- **迭代对齐持续有效**：基于Ints的迭代对齐展示了进一步提升的潜力
- **可懂度与其他指标正相关**：对齐后不仅可懂度提升，自然度和音质也同步改善

## 亮点与洞察

- **NLP技术迁移到语音**：将DPO从文本生成成功迁移到语音合成，为TTS领域引入了偏好对齐的新范式
- **架构无关的对齐方案**：通过适配DPO到三种TTS架构，展示了偏好对齐可以是架构无关的通用提升手段
- **数据构造的工程智慧**：三种偏好对设计（Intra/Inter/Perturbed）互补互助，尤其是Perturbed利用LLM生成发音扰动文本非常巧妙
- **弱到强泛化**：这一特性使得INTP数据集具有持久价值——即使未来出现更强的TTS模型，现有数据集仍可能对其有效
- **实用场景导向**：专注于绕口令、代码混合等实际应用中的痛点，研究有明确的工程落地价值

## 局限与展望

- **语言覆盖**：目前仅验证了中英文，对其他语言（日韩、欧洲语言等）的效果未知
- **偏好标注质量**：自动选择chosen/rejected可能存在噪声，引入人工偏好标注可能进一步提升效果
- **计算成本**：生成250K偏好对需要用三种模型大量推理，数据构建成本较高
- **评估指标**：可懂度的自动评估（如CER/WER）和人类感知之间可能存在差距
- **对齐税**：是否存在过度对齐导致语音多样性降低（如所有人说话风格趋同）的问题，文中未充分讨论

## 相关工作与启发

- **vs RLHF/DPO在语言模型中的应用**: 本文将DPO从文本领域迁移到语音领域，核心挑战在于TTS架构多样性和损失函数适配
- **vs SpeechAlign等语音对齐工作**: 之前的语音对齐主要关注自然度和说话人相似度，本文首次系统性地关注**可懂度**这一被忽视的维度
- **vs 数据增强方法**: 传统方法通过增加困难场景的训练数据来提升，本文通过偏好对齐以更小的数据量实现更好的效果

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统性地用偏好对齐解决TTS可懂度问题，INTP数据集设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 五种模型、多种场景、弱到强泛化、迭代对齐，实验非常全面
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，数据构造过程描述详细
- 价值: ⭐⭐⭐⭐⭐ 数据集和方法对TTS社区有直接实用价值，弱到强泛化发现有理论意义

<!-- RELATED:START -->

## 相关论文

- [Zero-Shot Text-to-Speech for Vietnamese](zero-shot_text-to-speech_for_vietnamese.md)
- [ControlSpeech: Towards Simultaneous and Independent Zero-shot Speaker Cloning and Zero-shot Language Style Control](controlspeech_zero_shot.md)
- [SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models](speechweave_diverse_multilingual_synthetic_text_audio_data_generation_pipeline_f.md)
- [LeVo: High-Quality Song Generation with Multi-Preference Alignment](../../NeurIPS2025/audio_speech/levo_high-quality_song_generation_with_multi-preference_alignment.md)
- [TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis](tcsinger_2_customizable_multilingual_zero-shot_singing_voice_synthesis.md)

<!-- RELATED:END -->
