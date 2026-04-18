---
title: >-
  [论文解读] Beautiful Images, Toxic Words: Understanding and Addressing Offensive Text in Generated Images
description: >-
  [AAAI2026][图像生成][扩散模型] 揭示扩散模型在生成图像中嵌入 NSFW 文字的新威胁，提出基于文本生成层定向 LoRA 微调的 NSFW-Intervention 方法，并发布 ToxicBench 基准。
tags:
  - AAAI2026
  - 图像生成
  - 扩散模型
  - NSFW Safety
  - Text-in-Image
  - Safety Fine-tuning
  - benchmark
---

# Beautiful Images, Toxic Words: Understanding and Addressing Offensive Text in Generated Images

**会议**: AAAI2026  
**arXiv**: [2502.05066](https://arxiv.org/abs/2502.05066)  
**代码**: [sprintml/ToxicBench](https://github.com/sprintml/ToxicBench) / [sprintml/SafeTextGen](https://github.com/sprintml/SafeTextGen)  
**领域**: image_generation  
**关键词**: Diffusion Models, NSFW Safety, Text-in-Image, Safety Fine-tuning, benchmark

## 一句话总结
揭示扩散模型在生成图像中嵌入 NSFW 文字的新威胁，提出基于文本生成层定向 LoRA 微调的 NSFW-Intervention 方法，并发布 ToxicBench 基准。

## 背景与动机
- 当前主流扩散模型（SD3、SDXL、Flux、DeepFloyd IF）在生成图像时不仅能渲染逼真视觉内容，还能在图像中嵌入可读文字（如标牌、标题、排版）
- 已有 NSFW 安全工作主要聚焦于**视觉层面**的不良内容（暴力、裸体场景），成功地通过安全过滤器、概念擦除等方法进行缓解
- 然而，模型在图像中生成的**文字**同样可以包含侮辱性语言、种族歧视词汇、性暗示术语等，这构成一个被忽视的全新安全威胁
- 这些有害文字甚至可升级为仇恨言论或意识形态宣传，在开放权重模型场景下尤为危险

## 核心问题
1. **新威胁识别**：所有主流扩散模型都能在生成图像中嵌入 NSFW 文字，现有安全机制对此束手无策
2. **朴素方案失败**：
    - 输入端文本过滤：缺乏视觉上下文，同一词汇（如 "Penetrating"）在不同场景下含义不同；且无法应对开源模型的白盒场景
    - 输出端 OCR 检测：模型经常生成带拼写错误的文字，人类仍可识别但 OCR+毒性检测器失效（SDXL 检出率不足 50%）
3. **现有方法不足**：AURA、ESD、Safe-CLIP 等方法在抑制 NSFW 文字时会**同等程度地**损害良性文字生成能力，缺乏定向性

## 方法详解

### ToxicBench 基准
- **数据集构建**：218 个 prompt 模板（改编自 CreativeBench）× 437 个 NSFW 词汇（配对 GPT-4 生成的语义近似良性替代词）
    - 训练集：337 对 NSFW-良性词对 → 73,466 个 prompt 对
    - 测试集：100 对保留词对 → 21,800 个 prompt 对，评估对未见 NSFW 词的泛化能力
- **评估流水线**：
    - 图像生成 → EasyOCR 提取文字 → 计算文本/图像质量指标
    - 支持两种模式：缓解方法前后对比 & 独立 NSFW 检测
- **新指标 NGramLD**（N-gram Levenshtein Distance）：
    - 从 OCR 输出中提取所有 k-gram 子串（k ∈ [1, n+1]），计算与目标词的最小编辑距离
    - 解决标准 LD 对长文本输出过度惩罚的问题，专注于最相关的子串匹配

### NSFW-Intervention 方法
**第一步：构造安全微调数据集**

- 对每个 NSFW prompt 生成含有害文字的图像 $I_{\text{NSFW}}$
- 将 NSFW 词替换为良性替代词，利用中间激活缓存的图像编辑技术重新生成 $I_{\text{benign}}$
- 两张图像仅在嵌入文字上有差异，其余视觉元素保持一致
- 构成训练三元组 $(x_{\text{NSFW}}, I_{\text{NSFW}}, I_{\text{benign}})$

**第二步：定向安全微调**

- 基于 Staniszewski et al. 的发现：扩散模型中文字渲染由少数注意力层控制
    - SD3：joint attention 层
    - SDXL / DeepFloyd IF：cross-attention 层
- 仅对这些文字生成相关层施加 LoRA 更新，大幅减少可训参数
- 训练目标：模型接收 NSFW prompt 的文本编码 $\phi(x_{\text{NSFW}})$，但学习输出良性图像 $I_{\text{benign}}$

**训练损失**：

$$\mathcal{L} = \|w(t) \cdot (f_\theta(I_{\text{NSFW}}(t), t, \phi(x_{\text{NSFW}})) - I_{\text{benign}})\|^2$$

- $w(t)$ 采用 logit-normal 时间步加权，优先关注中后期去噪步骤（文字最清晰时）
- $I_{\text{NSFW}}(t)$ 为在时间步 $t$ 添加高斯噪声后的 NSFW 图像

**核心优势**：直接修改模型权重，适用于白盒/开源场景，输入输出过滤无法绕过

## 实验关键数据

| 模型 | 良性 NGramLD Δ↓ | NSFW NGramLD Δ↑ | 差分优势 | KID |
|------|-----------------|-----------------|---------|-----|
| SD3 | 2.07 | 3.63 | +1.56 | 0.059/0.061 |
| DeepFloyd IF | 3.78 | 4.72 | +0.94 | 0.059/0.060 |
| SDXL | 4.75 | 5.69 | +0.94 | 0.063/0.065 |

- **对比基线**：AURA 的 NSFW/良性 NGramLD 差值仅 0.36（2.56 vs 2.20），Safe-CLIP 为 0.22（2.87 vs 2.65），定向性远不如本文方法
- **用户研究**：NSFW prompt 生成图像的有害文字识别率从 78.67% 降至 26.56%；拼写变体 NSFW 从 76.41% 降至 11.45%（训练中未见），良性文字保持 55.40%
- **消融**：对所有注意力层微调而非仅文字生成层时，SD3 上 NGramLD 提升仅 +0.49（vs 定向的 +3.63），验证了层选择的关键性
- **图像质量**：KID 增幅最大不超过 9%，FID 退化极小

## 亮点
1. **新颖的威胁定义**：首次系统性地揭示图像中嵌入 NSFW 文字的安全盲区，问题定义清晰且有现实意义
2. **精准的层级干预**：利用文字渲染层的局部性，仅对少量参数做 LoRA 微调，兼顾安全性与生成质量
3. **巧妙的数据构造**：通过激活缓存生成仅文字不同的图像对，为模型提供精确的监督信号
4. **完整的基准生态**：ToxicBench 涵盖数据集、评估指标（NGramLD）和流水线，为后续研究提供标准化工具
5. **强泛化性**：对训练中未见的 NSFW 词和拼写变体均有效，且可扩展到 VAR 模型（Infinity）

## 局限与展望
- NSFW 文字抑制并非完全彻底，部分样本仍可被人眼识别（如 "giant cocks" 等高度敏感词）
- 良性文字质量有一定损失（CLIP-Score 在 SD3 上从 91.42 降至 85.10），仍有优化空间
- 方法依赖于对文字生成层的先验知识，新架构需要重新定位这些层
- 评估依赖 OCR 准确度，艺术字体或极端变形文字可能漏检
- 数据集基于英文 NSFW 词汇，多语言场景的适用性未验证
- 对抗性攻击的鲁棒性（如刻意引导模型生成变体拼写）仍需进一步研究

## 与相关工作的对比
| 方法 | 类型 | NSFW 抑制 | 良性保持 | 适用场景 |
|------|------|----------|---------|---------|
| SLD | 推理引导 | 视觉 NSFW 有效，文字无效 | 较好 | 仅 API/黑盒 |
| ESD | 全模型微调 | 文字有一定效果 | 差（整体质量低） | SD1.4 限定 |
| AURA | 神经元抑制 | 文字有效但无定向性 | 差（良性同等损害） | 需文本编码器 |
| Safe-CLIP | 编码器微调 | 文字有效但无定向性 | 差（良性同等损害） | 需 CLIP 架构 |
| **NSFW-Intervention** | **定向层 LoRA** | **强且可泛化** | **较好** | **白盒/开源通用** |

## 启发与关联
- **层级可控性**的思路可推广到其他生成属性的定向编辑（如风格、水印、版权文字）
- **图像-文字对齐安全**是多模态模型的通用问题，本文的 benchmark 设计范式可迁移到视频生成模型
- 与 concept erasure 工作（如 Selective Amnesia、UCE）互补：前者擦除视觉概念，本文擦除文字概念
- NGramLD 指标对评估任何涉及文字渲染质量的任务都有参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐ — 问题定义新颖，首次系统研究图像中 NSFW 文字生成
- 实验充分度: ⭐⭐⭐⭐ — 4 个模型、3 个基线、消融实验、用户研究，较为全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，问题-方案-评估逻辑严密
- 价值: ⭐⭐⭐⭐ — 填补安全领域空白，benchmark 有持续影响力
