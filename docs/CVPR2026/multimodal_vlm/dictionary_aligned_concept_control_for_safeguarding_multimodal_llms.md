---
title: >-
  [论文解读] Dictionary-Aligned Concept Control for Safeguarding Multimodal LLMs
description: >-
  [CVPR 2026][多模态][多模态安全] 本文提出 DACO 框架，通过从 WordNet 和 CC-3M 构建包含 15,000 个多模态概念的字典，结合稀疏自编码器（SAE）实现对冻结 MLLM 激活空间的细粒度概念控制，在多个安全基准上显著提升安全性的同时保持通用能力。
tags:
  - CVPR 2026
  - 多模态
  - 多模态安全
  - 激活引导
  - 稀疏自编码器
  - 概念字典
  - 多模态VLM
---

# Dictionary-Aligned Concept Control for Safeguarding Multimodal LLMs

**会议**: CVPR 2026  
**arXiv**: [2604.08846](https://arxiv.org/abs/2604.08846)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 多模态安全, 激活引导, 稀疏自编码器, 概念字典, 越狱防御

## 一句话总结
本文提出 DACO 框架，通过从 WordNet 和 CC-3M 构建包含 15,000 个多模态概念的字典，结合稀疏自编码器（SAE）实现对冻结 MLLM 激活空间的细粒度概念控制，在多个安全基准上显著提升安全性的同时保持通用能力。

## 研究背景与动机

1. **领域现状**：多模态大语言模型（MLLM）面临恶意查询（文本越狱、视觉对抗攻击、排版触发等）导致的安全风险。现有安全控制策略包括提示工程、响应过滤和微调，以及新兴的激活引导（activation steering）方法。
2. **现有痛点**：提示工程对分布偏移脆弱；响应过滤需要额外计算开销；微调成本高。激活引导方法虽然灵活，但面临三个挑战：(1) 非稀疏方法通常只处理少量概念向量（<20 个），覆盖面窄；(2) 引导强度难以校准——抑制不足无法实现安全目标，过度抑制损害通用能力；(3) SAE 方法缺乏语义锚定，其学习的特征需要昂贵的探测或人工解释。
3. **核心矛盾**：手工构建的概念向量覆盖面有限且常有冗余或纠缠；SAE 学习的字典具有表达力但缺乏语义标注。两类方法各有优劣但未被有效统一。
4. **本文目标**：构建一个联合利用大规模概念字典和 SAE 的框架，实现对 MLLM 激活空间的有效、可解释的安全引导。
5. **切入角度**：从 WordNet 提取 15,000+ 概念，从 CC-3M 检索 400K+ 图文刺激对，聚合为概念向量字典；用此字典初始化 SAE 训练并自动标注 SAE 原子的语义。
6. **核心 idea**：概念字典提供语义锚定，SAE 提供表达力，二者结合实现"既知道是什么概念，又能有效分解和重组"的细粒度激活控制。

## 方法详解

### 整体框架
DACO 分四步：(1) 从 WordNet 提取概念名，用 CLIP 从 CC-3M 检索正负图文刺激对，构建 DACO-400K 数据集；(2) 将刺激对通过 MLLM 提取激活，通过对比读取（contrastive representation reading）聚合为概念向量字典 $\mathbf{D}_\ell$；(3) 用概念字典初始化 SAE 解码器训练，训练后通过余弦距离自动标注每个 SAE 原子为"可取"或"不可取"；(4) 推理时用 SAE 分解激活为稀疏系数，零化不可取原子的系数并增强可取原子的系数。

### 关键设计

1. **大规模多模态概念字典构建（DACO-400K）**:
    - 功能：提供覆盖广泛的语义锚定向量集
    - 核心思路：从 WordNet 提取约 15,000 个去重概念。对每个概念，用 CLIP 计算与 CC-3M 图文对的跨模态相关性：$\text{dist}_M(c, \mathbf{x}) = \sqrt{-(\ln s(c, \mathbf{x}_{\text{image}}) + \ln s(c, \mathbf{x}_{\text{text}}))}$，采用几何聚合而非算术平均以强调跨模态一致性。取高分对为正刺激 $\mathcal{X}_c^+$，低分对为负刺激 $\mathcal{X}_c^-$。通过对比读取得到每层概念向量：$\mathbf{d}_{\ell,c} = \mathbb{E}_{\mathbf{x} \in \mathcal{X}_c^+}[\mathbf{z}_\ell] - \mathbb{E}_{\mathbf{x} \in \mathcal{X}_c^-}[\mathbf{z}_\ell]$。用专家 MLLM 将概念标注为可取/不可取。
    - 设计动机：15,000 个概念远超现有方法的 ~20 个，提供了激活空间几何的全面覆盖。几何聚合确保正刺激在视觉和文本两个模态上都与概念匹配，避免单模态偏差

2. **字典初始化的 SAE 训练与自动标注**:
    - 功能：获得更具表达力的引导向量并自动赋予语义
    - 核心思路：用归一化概念向量初始化 SAE 解码器 $\mathbf{W}_{\ell,i}^{\text{dec},(0)} \leftarrow \mathbf{D}_{\ell,i}/\|\mathbf{D}_{\ell,i}\|_2$，然后在 CC-3M 激活上训练 L1-SAE 或 TopK-SAE。训练后，计算不可取概念集 $\mathcal{K}^-$ 和可取概念集 $\mathcal{K}^+$ 的质心，根据 SAE 原子与质心的余弦距离进行阈值标注（Eq. 9）。单质心标注兼顾效率和效果。
    - 设计动机：SAE 从数据中学习比手工字典更优的表示基底，而概念字典初始化解决了 SAE 的冷启动问题和语义标注问题——实现了"语义可解释的稀疏表示"

3. **组合式推理时引导**:
    - 功能：在推理时对冻结 MLLM 的激活进行安全引导
    - 核心思路：对目标激活 $\mathbf{z}_\ell$，用 SAE 编码器计算稀疏系数 $\mathbf{c}_\ell^* = \sigma(\mathbf{W}_\ell^{\text{enc}} \mathbf{z}_\ell + \mathbf{b}_\ell^{\text{enc}})$。构造控制系数：不可取原子取负（零化其贡献），可取原子加正常数 $\gamma$（增强安全响应），其余为 0。修改后的激活 $\hat{\mathbf{z}}_\ell = \mathbf{z}_\ell + \mathbf{W}_\ell^{\text{dec}} \hat{\mathbf{c}}_\ell$ 替换原始激活继续自回归生成。
    - 设计动机：相比直接移除/添加概念向量（ActAdd），SAE 分解更精确地定位需要修改的成分；组合式引导允许同时移除有害概念和增强无害概念

### 损失函数 / 训练策略
SAE 训练使用标准重构+稀疏损失（Eq. 3），支持 L1 正则化和 TopK 约束两种变体。概念字典初始化显著改善了训练收敛（FVE 提升 2-5%）。超参数 $\eta$（标注阈值）和 $\gamma$（可取概念增强强度）在验证集上调优。

## 实验关键数据

### 主实验

**安全性评估（Qwen2.5-VL-7B）**：

| 方法 | MS-R↑ | MS-QG↑ | JBV-R↑ | JBV-QG↑ | Fluency↑ | MMMU↑ |
|------|-------|--------|--------|--------|---------|-------|
| No Steering | 0.442 | 0.652 | 0.564 | 0.543 | 0.917 | 0.546 |
| Prompting | 0.607 | 0.711 | 0.659 | 0.622 | 0.923 | 0.516 |
| ActAdd | 0.653 | 0.735 | 0.691 | 0.675 | 0.691 | 0.441 |
| MOP | 0.771 | 0.840 | 0.835 | 0.752 | 0.816 | 0.496 |
| **DACO** | **0.990** | **0.984** | **0.903** | **0.841** | **0.905** | **0.521** |

**推理时间开销（每 token）**：

| 方法 | 额外时间 | 占比 |
|------|---------|------|
| ActAdd | +0.023s | +10.8% |
| MOP | +0.107s | +49.4% |
| **DACO** | **+0.031s** | **+14.6%** |

### 消融实验

| 配置 | JBV-QG | MMMU | 说明 |
|------|--------|------|------|
| DACO (TopK-SAE, 字典初始化) | 0.841 | 0.521 | 完整方法 |
| MOP (稀疏编码, 无 SAE) | 0.752 | 0.496 | SAE 比手工字典更有效 |
| SAE 随机初始化 | ~0.80 | ~0.51 | 字典初始化提升 ~4% 安全性 |
| η 过小 (过多原子标注为有害) | 高 | 低 | 过度拒绝，损害通用能力 |
| η 过大 (过少原子标注为有害) | 低 | 高 | 安全引导不足 |

### 关键发现
- **DACO 在三个 MLLM 上一致且大幅超越所有基线**：在 Qwen2.5-VL 上安全性从 0.442 提升到 0.990（MS-R），同时 MMMU 仅降 2.5%
- **推理开销极低（+14.6%）**：远低于 MOP（+49.4%），因为 SAE 编码是单次矩阵乘法而非迭代稀疏求解
- **SAE 原子语义可解释**（Figure 7）：对越狱查询的 SAE 分解显示，高激活原子的最近概念向量与查询内容语义一致（如"毒品"、"暴力"），验证了自动标注的有效性

## 亮点与洞察
- **概念字典 + SAE 的互补设计**是最核心的创新：字典提供语义锚定解决 SAE 的"黑箱"问题，SAE 提供数据驱动的表达力解决手工字典的局限性。这种"先验知识 + 数据学习"的范式可广泛迁移
- **DACO-400K 数据集**本身是有价值的贡献：15,000 个多模态概念的激活方向向量可用于机制可解释性、概念编辑等多种下游任务
- **几何聚合的跨模态刺激检索**（Eq. 4）比算术平均更优雅：要求图文同时匹配才得高分，避免了"图片匹配但文本不匹配"的噪声刺激

## 局限与展望
- 概念字典依赖 WordNet 和 CC-3M，可能遗漏新兴或特定文化的有害概念
- SAE 原子标注使用单质心，对语义分布复杂的概念可能不够精确
- 安全引导在 MOSSBench（需要回答的敏感但合法查询）上可能导致过度拒绝
- 超参数 $\eta$ 和 $\gamma$ 的调优需要验证数据，缺少自动化策略
- 未来可探索动态概念字典更新和多层联合引导策略

## 相关工作与启发
- **vs PaCE (Parsimonious Concept Engineering)**: PaCE 仅用合成文本构建概念，DACO 使用真实多模态刺激，在 MLLM 上更有效（MOP 是 PaCE 在 MLLM 上的扩展）
- **vs ActAdd**: ActAdd 用少量对比向量直接加减，缺乏细粒度控制。DACO 的 SAE 分解能精确定位需要修改的激活成分
- **vs Constitutional AI**: Constitutional AI 通过训练时的 RLHF 实现安全对齐。DACO 在推理时操作，更灵活且适用于任何冻结模型

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 概念字典+SAE 的协同框架新颖，DACO-400K 数据集是重要贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 三个 MLLM、两个安全基准、两个评判器、通用能力评估，非常全面
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，但符号较多需要仔细阅读
- 价值: ⭐⭐⭐⭐⭐ 实用性极强的 MLLM 安全方案，低开销高效果

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models](llavashield_multimodal_multiturn_safety.md)
- [\[CVPR 2026\] Concept-wise Attention for Fine-grained Concept Bottleneck Models](coat_cbm_concept_wise_attention.md)
- [\[CVPR 2026\] Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild](joint-aligned_latent_action_towards_scalable_vla_pretraining_in_the_wild.md)
- [\[CVPR 2026\] Customized Visual Storytelling with Unified Multimodal LLMs](customized_visual_storytelling_with_unified_multimodal_llms.md)
- [\[CVPR 2026\] PersonaVLM: Long-Term Personalized Multimodal LLMs](personavlm_long_term_personalized_multimodal_llms.md)

</div>

<!-- RELATED:END -->
