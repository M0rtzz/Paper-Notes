---
title: >-
  [论文解读] Securing the Language of Life: Inheritable Watermarks from DNA Language Models to Proteins
description: >-
  [NeurIPS 2025][LLM安全][DNA水印] 提出 DNAMark 和 CentralMark 两种水印方案，针对 DNA 语言模型生成的序列嵌入鲁棒水印：前者利用同义密码子替换实现功能不变水印，后者实现从 DNA 到蛋白质的可遗传水印。
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "DNA水印"
  - "生物安全"
  - "语言模型"
  - "同义密码子替换"
  - "中心法则"
---

# Securing the Language of Life: Inheritable Watermarks from DNA Language Models to Proteins

**会议**: NeurIPS 2025  
**arXiv**: [2509.18207](https://arxiv.org/abs/2509.18207)  
**代码**: 暂无  
**领域**: 医学图像  
**关键词**: DNA水印, 生物安全, 语言模型, 同义密码子替换, 中心法则

## 一句话总结

提出 DNAMark 和 CentralMark 两种水印方案，针对 DNA 语言模型生成的序列嵌入鲁棒水印：前者利用同义密码子替换实现功能不变水印，后者实现从 DNA 到蛋白质的可遗传水印。

## 研究背景与动机

DNA 语言模型（如 Evo、Evo2）在理解和设计 DNA 序列方面取得了革命性进展，可应用于治疗、合成生物学和基因编辑。然而，这种能力同时带来了严重的双重用途风险——模型可能被滥用来创建病原体或生物武器。

现有 LLM 水印技术（如 KGW）无法直接迁移到 DNA 语言模型，原因有三：（1）DNA 只有四种核苷酸（A、C、T、G），词表极小，green/red list 方法受限严重；（2）DNA 易受自然突变、合成误差和测序不准确性影响，水印容易被破坏；（3）需要保留编码蛋白的功能完整性。因此需要专门针对 DNA 序列设计的水印方案。

核心动机是：为 DNA 语言模型输出建立溯源机制，在不影响生物功能的前提下追踪 AI 生成的 DNA 序列，从而平衡创新与生物安全之间的关系。

## 方法详解

### 整体框架

系统包含两种互补的水印方案：DNAMark 针对 DNA 序列本身（仅改变密码子第三位碱基），CentralMark 针对中心法则中从 DNA 到蛋白质的信息传递（改变密码子第二位碱基），实现在 DNA 和翻译后蛋白质中都可检测的可遗传水印。

### 关键设计

1. **基于 Evo2 嵌入的水印模型（DNAMark）**：使用训练好的水印模型将当前序列通过 Evo2 获取功能嵌入，然后变换为水印 logits 叠加到原始 logits 上。水印模型训练时优化两个目标：对齐损失 $\mathcal{L}_a$ 保证水印 logits 相似度与 Evo2 嵌入相似度一致（利用 $\tanh$ 归一化），归一化损失 $\mathcal{L}_n$ 约束水印 logits 均值为零且各方向绝对值一致。其核心直觉是 DNA 嵌入对小突变天然鲁棒，能保持语义和功能完整性。

2. **同义密码子替换（SCS）**：DNAMark 专门针对编码区密码子的第三位碱基进行水印注入。由于遗传密码的简并性，同义密码子编码相同氨基酸。对于同义密码子集合 $\mathcal{S}$，选择水印 logits 最高的碱基作为 green list，其余作为 red list。例如对于组氨酸（CAT/CAC），T 为 green，C 为 red。当 $|\mathcal{S}|=1$（如甲硫氨酸 ATG）时跳过水印。

3. **自适应水印强度与熵引导策略**：通过 EMA 动态调整水印强度 $\delta$，根据当前 z-score 在 $[z_{\min}, z_{\max}]$ 范围内自适应调节：

$$\delta_{t+1} = (1-\beta)\delta_t + \beta \cdot \max(\delta_{\min}, \min(\delta_{\max}, \delta_t + \kappa \cdot \text{adj}(z_t)))$$

同时在低熵子序列（$H(s) < H_{\text{threshold}}$）中跳过水印，避免破坏调控基序等关键功能元件。

4. **CentralMark 可遗传水印**：与 DNAMark 不同，CentralMark 针对密码子第二位碱基进行修改（第二位碱基主要决定氨基酸的化学性质），使用 ESM 蛋白质嵌入代替 Evo2 嵌入。通过将第二位碱基分为 green/red list（$\mathcal{G}_b = \{C, G\}$），实现氨基酸级别的近非重叠 green/red 划分：

$$\mathcal{G}_a = \{a \mid \text{translate}(b_1, b_2, b_3) = a, b_2 \in \mathcal{G}_b\}$$

生成的水印不仅可在 DNA 序列中检测，还可在翻译后的蛋白质序列中独立检测。

### 损失函数 / 训练策略

水印模型的总损失为对齐损失与归一化损失之和。水印检测采用 KGW 的 z-score 方法。由于 DNAMark 和 CentralMark 的独特设计，green token 期望比例 $\gamma$ 分别设为 0.3559、0.5 和 0.55（不同于标准 LLM 水印的 0.5）。

## 实验关键数据

### 主实验

在治疗性 DNA 基准（400 条序列）上使用 Evo2-7B 进行评估：

| 攻击类型 | 方法 | TPR@1%FPR | F1@1%FPR | TPR@10%FPR | F1@10%FPR |
|---------|------|-----------|----------|------------|-----------|
| 无攻击 | KGW-1 | 0.765 | 0.862 | 0.805 | 0.845 |
| 无攻击 | DNAMark | 0.845 | 0.911 | 0.915 | 0.908 |
| 无攻击 | CentralMark(DNA) | **0.875** | **0.928** | **0.920** | **0.911** |
| 同义替换 | KGW-1 | 0.580 | 0.729 | 0.756 | 0.815 |
| 同义替换 | DNAMark | 0.820 | 0.896 | 0.896 | 0.898 |
| 同义替换 | CentralMark(Protein) | **0.860** | **0.920** | 0.904 | 0.902 |
| 核苷酸替换 | DNAMark | 0.808 | 0.902 | 0.886 | 0.892 |
| 插入缺失 | CentralMark(DNA) | 0.765 | 0.862 | 0.850 | 0.872 |

### 消融实验

| 配置 | 检测 F1 | 说明 |
|------|---------|------|
| 完整 DNAMark | 0.911 | 基线 |
| 去掉自适应 δ + EMA | 显著下降 | 自适应强度对检测至关重要 |
| 去掉水印模型 | 显著下降 | 水印模型是核心组件 |
| 去掉熵引导 | 轻微下降 | 熵引导主要保护序列质量 |
| 去掉对齐+归一化损失 | 中等下降 | 损失函数对鲁棒性有贡献 |

### 关键发现

- DNAMark 和 CentralMark 在所有攻击条件下的检测 F1 均超过 0.85，远优于 KGW 基线
- 序列质量保持良好：与无水印生成相比，序列相似度超过 60%，退化分数低于 15%
- DNAMark 可跨多种 DNA 模型泛化（megaDNA 到 Evo2-40B），F1 从 0.851 到 0.919
- CRISPR-Cas9 案例表明 CentralMark 不影响蛋白质结构（TM-score=0.6802）
- 水印带来约 30% 的时间开销，但水印模型本身体量紧凑

## 亮点与洞察

- **问题新颖性极高**：首次系统性地将水印技术应用到 DNA 语言模型，是生物安全与 AI 安全交叉的前沿工作
- **可遗传水印设计精巧**：CentralMark 利用中心法则，通过 DNA 到蛋白质的信息传递实现水印遗传，这一概念在水印文献中独树一帜
- **同义密码子替换兼顾功能与水印**：利用遗传密码简并性这一生物学特性，实现了真正的功能不变水印

## 局限与展望

- 仅在编码区验证，对非编码区（UTR 等）的影响未充分探索
- 缺乏湿实验验证（仅计算实验）
- Indel 攻击下性能下降明显，鲁棒性仍有提升空间
- 水印方案依赖 green/red list，未来可探索不依赖分区的方案

## 相关工作与启发

与 LLM 水印（KGW、EWD 等）相比，本文面临的独特挑战是极小词表和生物功能约束。与蛋白质水印工作（FoldMark）相比，本文首次覆盖了中心法则全链路。启发在于：当领域约束严格时（如生物功能保持），需要领域知识深度参与水印设计，而非简单迁移通用方法。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次提出 DNA 语言模型水印，可遗传水印概念开创性
- 实验充分度: ⭐⭐⭐⭐ 多种攻击和消融分析全面，但缺乏湿实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，生物学背景介绍充分
- 价值: ⭐⭐⭐⭐⭐ 对 AI 生物安全有重要意义，具有实际应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Ensemble Watermarks for Large Language Models](../../ACL2025/llm_safety/ensemble_watermarks_llm.md)
- [\[NeurIPS 2025\] DNA-DetectLLM: Unveiling AI-Generated Text via a DNA-Inspired Mutation-Repair Paradigm](dna-detectllm_unveiling_ai-generated_text_via_a_dna-inspired_mutation-repair_par.md)
- [\[ACL 2026\] Topic-Based Watermarks for Large Language Models](../../ACL2026/llm_safety/topic-based_watermarks_for_large_language_models.md)
- [\[NeurIPS 2025\] Approximate Domain Unlearning for Vision-Language Models](approximate_domain_unlearning_for_visionlanguage_models.md)
- [\[NeurIPS 2025\] SAEMark: Steering Personalized Multilingual LLM Watermarks with Sparse Autoencoders](saemark_steering_personalized_multilingual_llm_watermarks_with_sparse_autoencode.md)

</div>

<!-- RELATED:END -->
