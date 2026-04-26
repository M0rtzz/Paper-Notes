---
title: >-
  [论文解读] DEXTER: Diffusion-Guided EXplanations with TExtual Reasoning for Vision Models
description: >-
  [图像生成] 提出 DEXTER，一个无需数据的框架，通过优化文本提示驱动扩散模型生成最大化目标分类器激活的图像，再用 LLM 对合成样本进行推理，生成全局性、可读的文本解释，实现模型行为的偏差发现和全局解释。
tags:
  - 图像生成
---

# DEXTER: Diffusion-Guided EXplanations with TExtual Reasoning for Vision Models

## 基本信息
- **arXiv**: 2510.14741
- **会议**: NeurIPS 2025
- **作者**: Simone Carnemolla, Matteo Pennisi, Sarinda Samarasinghe, Giovanni Bellitto, Simone Palazzo, Daniela Giordano, Mubarak Shah, Concetto Spampinato
- **机构**: University of Catania, University of Central Florida
- **代码**: https://github.com/perceivelab/dexter

## 一句话总结
提出 DEXTER，一个无需数据的框架，通过优化文本提示驱动扩散模型生成最大化目标分类器激活的图像，再用 LLM 对合成样本进行推理，生成全局性、可读的文本解释，实现模型行为的偏差发现和全局解释。

## 背景与动机
模型可解释性是构建可信 AI 的基础。现有方法的局限：

1. **局部归因方法**（GradCAM、Integrated Gradients）：只解释单个预测，不提供全局理解
2. **激活最大化**（AM）：生成的图像抽象且难以语义解读
3. **文本解释方法**（NLE）：通常依赖标注数据和预训练视觉-语言映射
4. **偏差发现方法**（B2T、LADDER）：需要训练数据做误分类分析

**核心需求**：一种**无需任何训练数据或标签**的全局解释方法，能以自然语言描述分类器的决策模式和偏差。

## 核心问题
如何在完全无数据的条件下，系统性地揭示和解释深度视觉分类器的决策过程，包括特征偏好、偏差模式和虚假关联？

## 方法详解

### 1. 整体框架
三大管线：
- **文本管线**：优化 soft prompt → BERT 预测 mask token → 获得文本提示
- **视觉管线**：文本提示条件化 Stable Diffusion → 生成最大化目标神经元激活的图像
- **推理模块**：VLM 对生成的图像进行 caption → LLM 跨样本推理 → 生成文本偏差报告

### 2. 文本管线：Soft Prompt → Hard Token
输入结构：$\mathbf{t} = [\mathbf{t}_\text{fixed}, m_1, m_2, \ldots, m_N]$，其中 $\mathbf{t}_\text{fixed}$ = "a picture of a"

- 在 BERT 嵌入前添加可学习 soft prompt $\mathbf{p} \in \mathbb{R}^{P \times d}$（$P=1, d=768$）
- BERT 输出 mask 位置的 logits $\mathbf{l}_i \in \mathbb{R}^V$
- 通过 **Gumbel-Softmax**（$\tau=1$）将 logits 转为可微的 one-hot 向量 $\mathbf{o}_i$
- **BERT→CLIP 词汇表映射**：翻译矩阵 $\mathbf{M} \in \{0,1\}^{V \times W}$
  $$\mathbf{o}_i^{(C)} = \mathbf{o}_i \mathbf{M}$$
  BERT 中无 CLIP 对应的 token 被自动避开（对应行全零）

### 3. 视觉管线：激活最大化
将文本编码为 CLIP 嵌入 $\mathbf{e}$，条件化 Stable Diffusion 生成图像，送入目标分类器 $f$ 获得神经元激活 $\mathbf{n} = f(d(\mathbf{e}))$。

激活最大化损失：
$$\mathcal{L}_\text{act} = \sum_{i=1}^K l_\text{act}(n_i), \quad l_\text{act}(n_i) = \begin{cases} -n_i, & \text{特征神经元} \\ -\log n_i, & \text{类别神经元} \end{cases}$$

### 4. 辅助 Mask 伪标签预测
为解决 soft prompt 梯度传播过弱的问题，引入辅助交叉熵损失：
- 维护伪标签 $y_i$ 和参考损失 $L_i$
- 聚合关联神经元的激活损失：$\mathcal{L}_{\text{agg},i} = \sum_{j \in \mathcal{N}_i} l_\text{act}(n_j)$
- 用历史均值防止离群值干扰伪标签更新：
$$\frac{1}{T} \sum_{j=1}^T \mathcal{L}_{\text{agg},i}^{(j)} < L_i$$

**总损失**：
$$\mathcal{L} = \sum_{k=1}^K l_\text{act}(n_k) - \sum_{i=1}^N \log s_{i, y_i}$$

### 5. 偏差推理
对每个目标类生成 50 张图像 → ChatGPT-4o mini 生成逐图 caption → LLM 跨 caption 推理 → 输出结构化偏差报告。

## 实验关键数据

### 激活最大化（SalientImageNet，30 类）

| 方法 | Spurious | Core | 平均 |
|------|----------|------|------|
| Baseline（类名） | 43.06 | 86.40 | 64.73 |
| ChatGPT 描述 | 41.20 | 78.53 | 59.87 |
| DiffExplainer | 33.20 | 47.66 | 39.83 |
| **DEXTER** | **63.00** | **87.86** | **75.43** |

### Slice Discovery & Debiasing（Worst-Slice Accuracy）

| 方法 | 需要数据 | CelebA Worst | Waterbirds Worst |
|------|---------|-------------|-----------------|
| ERM | ✓ | 47.7 | 62.6 |
| DRO | ✓ + GT | 90.0 | 89.9 |
| DRO-B2T | ✓ | 90.4 | 90.7 |
| LADDER | ✓ | 89.2 | **92.4** |
| **DEXTER** | **✗** | **91.3** | 90.5 |

- 在 CelebA 上 DEXTER 无数据条件下超越所有方法（含使用数据的）
- 在 Waterbirds 上与 SOTA 持平

### 偏差报告评估（FairFaces）

| 指标 | w Bias | w/o Bias | 均值 |
|------|--------|----------|------|
| STS（与数据报告相似度） | 0.92 | 0.85 | 0.90 |
| G-eval 一致性 | 4.58 | 4.80 | 4.19 |
| MOS-LLM | 4.29 | 4.80 | 4.48 |
| MOS-人类 | 4.20 | 3.89 | 4.01 |

### 消融实验

| 配置 | Spurious | Core | 平均 |
|------|----------|------|------|
| 单词 | 11.13 | 36.33 | 23.73 |
| 单词 + $\mathcal{L}_\text{mask}$ | 34.00 | 53.86 | 43.93 |
| 多词 | 15.53 | 8.13 | 11.83 |
| **多词 + $\mathcal{L}_\text{mask}$** | **63.00** | **87.86** | **75.43** |

## 亮点
1. **完全无数据**：仅需分类器本身，不接触任何训练数据或标签
2. **多模态全局解释**：视觉（激活最大化图像）+ 文本（LLM 偏差报告）双通道
3. **离散提示优化**：Gumbel-Softmax + BERT→CLIP 映射实现可解释的 hard token 优化
4. **三任务验证**：激活最大化 + 偏差发现 + 偏差解释，每个任务都有定量评估
5. **伪标签机制**：解决了 soft prompt 梯度消失问题，同时建立神经元与文本 token 的映射

## 局限性
1. **计算成本**：每类约 10 分钟的 prompt 优化，大规模类别（ImageNet 1000 类）耗时较长
2. **依赖 Stable Diffusion**：生成图像质量受限于扩散模型能力，对 SD 未覆盖的领域可能失效
3. **NSFW 风险**：需额外安全过滤器
4. **LLM 幻觉**：VLM/LLM 的推理可能引入与模型行为无关的虚假解释
5. **仅限分类器**：未扩展到检测、分割等其他视觉任务

## 与相关工作的对比
- **vs. DiffExplainer**：DEXTER 使用 hard token 替代 soft prompt，更可解释；用户研究显示概念特征上优于 DiffExplainer
- **vs. B2T**：B2T 需要训练数据做误分类分析，DEXTER 完全无数据
- **vs. LADDER**：LADDER 依赖低置信度预测和 LLM 伪属性，仍然需要数据
- **vs. GradCAM / IG**：局部归因 vs. 全局文本解释，互补但功能不同

## 启发与关联
- **可解释性新范式**：无数据全局解释 = 主动探测分类器（而非被动分析数据）
- **扩散模型 × 可解释性**：扩散模型不仅能生成图像，还能作为可解释性工具
- **偏差审计自动化**：DEXTER 可作为模型部署前的自动偏差审计工具

## 评分
- 新颖性：⭐⭐⭐⭐⭐ — 无数据全局解释框架是全新贡献，三管线设计优雅
- 技术深度：⭐⭐⭐⭐☆ — Gumbel-Softmax + 词汇表映射 + 伪标签机制设计精巧
- 实验完整度：⭐⭐⭐⭐⭐ — 三个任务 × 四个数据集 × 用户研究 × 消融实验
- 写作质量：⭐⭐⭐⭐⭐ — 逻辑清晰，图表丰富，附录详尽

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Diff-ICMH: Harmonizing Machine and Human Vision in Image Compression with Generative Prior](diff-icmh_harmonizing_machine_and_human_vision_in_image_compression_with_generat.md)
- [\[NeurIPS 2025\] Emergence and Evolution of Interpretable Concepts in Diffusion Models](emergence_and_evolution_of_interpretable_concepts_in_diffusi.md)
- [\[ICCV 2025\] Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities](../../ICCV2025/image_generation/rethinking_the_embodied_gap_in_vision-and-language_navigation_a_holistic_study_o.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](../../ICCV2025/image_generation/joint_diffusion_models_in_continual_learning.md)
- [\[CVPR 2025\] FlipSketch: Flipping Static Drawings to Text-Guided Sketch Animations](../../CVPR2025/image_generation/flipsketch_flipping_static_drawings_to_text-guided_sketch_animations.md)

<!-- RELATED:END -->
