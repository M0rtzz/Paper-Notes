---
title: >-
  [论文解读] Zero-Shot Embedding Drift Detection: A Lightweight Defense Against Prompt Injections in LLMs
description: >-
  [NeurIPS 2025][机器人][提示注入检测] 提出ZEDD（零样本嵌入漂移检测），通过比较良性和可疑输入在嵌入空间中的语义漂移来检测提示注入攻击，利用GMM/KDE自动确定阈值，在多种LLM架构上实现>93%的检测准确率且假阳性率<3%。
tags:
  - NeurIPS 2025
  - 机器人
  - 提示注入检测
  - 嵌入漂移
  - 零样本检测
  - LLM安全
  - 轻量防御
---

# Zero-Shot Embedding Drift Detection: A Lightweight Defense Against Prompt Injections in LLMs

**会议**: NeurIPS 2025  
**arXiv**: [2601.12359](https://arxiv.org/abs/2601.12359)  
**代码**: [GitHub](https://github.com/AnirudhSekar/ZEDD)  
**领域**: 机器人 / AI安全  
**关键词**: 提示注入检测, 嵌入漂移, 零样本检测, LLM安全, 轻量防御

## 一句话总结

提出ZEDD（零样本嵌入漂移检测），通过比较良性和可疑输入在嵌入空间中的语义漂移来检测提示注入攻击，利用GMM/KDE自动确定阈值，在多种LLM架构上实现>93%的检测准确率且假阳性率<3%。

## 研究背景与动机

提示注入攻击已成为LLM应用的核心安全威胁。攻击者通过间接输入渠道（如邮件、用户生成内容）嵌入对抗性指令，操纵LLM绕过对齐防护产生有害输出。当前防御手段存在明显不足：

**现有方法计算开销大**：基于额外大模型或规则过滤器的检测方案引入了显著的计算和延迟开销，不适合低延迟的部署场景。

**嵌入方法未充分利用**：虽然已有工作探索利用嵌入分类来检测注入，但它们使用逻辑回归、XGBoost等传统优化器，而非微调嵌入空间本身来产生优化分类。

**缺乏模型无关性和零样本能力**：多数方法针对特定模型或特定攻击类型设计，需要已知攻击样本或需要重新训练，难以泛化到未见过的攻击模式。

ZEDD的核心思路是：对抗性提示会在嵌入空间中引起细微但可量化的语义漂移，即使表面文本看起来正常。通过测量这种漂移就能轻量化地检测注入，无需访问模型内部、先验攻击知识或任务特定重训练。

## 方法详解

### 整体框架

ZEDD管线包含三个阶段：(1) 用微调后的编码器提取嵌入；(2) 通过余弦相似度计算语义漂移；(3) 用GMM/KDE建模漂移分布并自动确定检测阈值。编码器只需微调一次，之后即可零样本应用到新数据。

### 关键设计

1. **嵌入提取与微调**：对每对配对的干净/注入提示，使用多种模型提取嵌入表示——Sentence-BERT (All-MPNET-Base-V2)、Llama 3 8B Instruct、Mistral 7B Instruct、Qwen 2 7B Instruct。微调阶段利用"相似"（clean-clean对）和"不相似"（injected-clean对）的标签优化模型的嵌入空间，使其能更好区分注入与正常提示。关键是微调仅用约10%的训练数据（约占总数据的7%），每个模型在NVIDIA B200上仅需15-18分钟即可完成。

2. **漂移分数计算**：对配对的提示 $(x, x')$，计算嵌入漂移分数：
    $\text{Drift}(x, x') = 1 - \frac{f(x) \cdot f(x')}{\|f(x)\| \cdot \|f(x')\|}$
   其中 $f(\cdot)$ 是微调后编码器的嵌入函数。漂移分数越大，说明候选提示偏离干净对应物越多，越可能是注入攻击。这比直接分类方法更轻量——仅需一次余弦相似度计算。

3. **基于GMM/KDE的自适应阈值确定**：

    - **GMM主方法**：对漂移分数分布拟合2组分高斯混合模型，低均值组分对应clean-clean对（漂移小），高均值组分对应injected-clean对（漂移大）。最优决策阈值在两个组分的加权密度交叉点处：$f_{\text{clean}}(x) \cdot w_{\text{clean}} = f_{\text{injected}}(x) \cdot w_{\text{injected}}$。
    - **KDE后备**：当GMM收敛不稳定时，自动回退到核密度估计方法，通过识别分布的峰谷来区分两类。
    - **约束优化**：通过二分搜索在可行范围内迭代优化阈值，同时满足假阳性率上限（默认3%）和总标记率目标（默认约50%），确保阈值在不同嵌入分布上的适用性。

### 损失函数 / 训练策略

微调阶段使用二分类损失：配对的clean-clean对标签为1（相似），injected-clean对标签为0（不相似）。训练目标是优化嵌入空间使两类对的余弦相似度分布尽量分开。数据划分为70%训练+30%测试，保持各攻击类别的原始比例。仅使用10%训练数据进行微调以避免过拟合，同时保持训练效率。

## 实验关键数据

### 主实验

**各模型在LLMail-Inject数据集上的检测表现（51,603对测试样本）**

| 编码器模型 | 准确率 | 精度 | 召回率 | F1 | 假阳率 |
|-----------|--------|------|--------|-----|--------|
| SBERT All-MPNET | 90.75% | 99.65% | 81.78% | 89.84% | 1.7% |
| Llama 3 8B | 95.32% | 95.85% | 94.75% | 95.30% | 5.5% |
| Mistral 7B | **95.55%** | **96.58%** | 94.45% | **95.50%** | 2.3% |
| Qwen 2 7B | 95.46% | 96.27% | **94.52%** | 95.38% | 2.2% |

**各攻击类别的检测率**

| 模型 | Clean | 编码操纵 | 越狱 | 提示混淆 | 系统泄漏 | 任务覆写 |
|------|-------|---------|------|---------|---------|---------|
| Llama 3 | 5.5% | 98.1% | 92.2% | 94.4% | 96.7% | 90.7% |
| Mistral 7B | 2.3% | 98.1% | 92.2% | 93.3% | 96.9% | 90.8% |
| Qwen 2 | 2.2% | **98.2%** | 90.8% | **94.2%** | **96.8%** | 90.3% |

### 消融实验

**不同假阳性率上限的影响**

| FPR上限 | 模型 | Clean误报率 | 编码操纵检出 | 越狱检出 | 系统泄漏检出 |
|---------|------|-----------|------------|---------|------------|
| 3%（默认） | Mistral 7B | 2.3% | 98.1% | 92.2% | 96.9% |
| 5% | Mistral 7B | 3.4% | 98.2% | 92.2% | 96.9% |
| 10% | Mistral 7B | 5.4% | 98.2% | 92.2% | 96.9% |

### 关键发现

- **大语言模型编码器全面优于SBERT**：Llama 3、Mistral、Qwen 2的F1均在95%以上，比SBERT高5-6个百分点，说明大模型捕获语义细微变化的能力更强。
- **假阳率极低**：平均仅2.93%，做到了高召回率的同时几乎不误报正常邮件。
- **编码操纵和系统泄漏最易检出**（>96%），越狱和任务覆写稍难（~90-92%），可能因为这些攻击在语义上更接近正常请求。
- **轻量高效**：在NVIDIA B200上，4个模型总微调时间不到60分钟，对约5.1万对测试数据的推理仅需数分钟。

## 亮点与洞察

- **简单到优雅**：核心思想就是"注入改变了语义→语义变化可以用嵌入距离量化"，简洁但效果很好。
- **GMM/KDE自适应阈值**：无需手动调参，自动从数据分布中学习决策边界，增强了跨场景泛化能力。
- **模型无关性**：同一个管线可以无缝切换不同编码器，且在所有测试模型上表现一致。

## 局限与展望

- 检测质量直接依赖编码器的嵌入质量，对于不同规模或架构的LLM，嵌入空间的语义结构可能差异很大。
- 当前仅在邮件格式的LLMail-Inject数据集上验证，对聊天、代码、文档等其他输入格式的泛化性未知。
- 可能存在攻击者故意在嵌入层面操纵的对抗性攻击，绕过ZEDD的漂移检测。
- 作者明确将ZEDD定位为"第一道轻量防线"，单独使用可能不足以应对所有攻击。
- 需要预先构建干净提示配对，实际部署时如何自动生成配对是一个工程挑战。

## 相关工作与启发

- 与Ayub & Majumdar的嵌入分类器方法相比，ZEDD通过微调嵌入空间本身（而非使用传统分类器）获得了更好的分离效果。
- 对比其他重量级检测方法（需要额外大模型做判断），ZEDD的计算开销几乎可以忽略。
- 启发：嵌入漂移的概念可以扩展到检测其他类型的输入操纵，如数据投毒、分布外检测等。

## 评分

- **新颖性**: ⭐⭐⭐ 嵌入漂移的想法直觉清晰但不算全新，GMM/KDE阈值设计有巧思
- **实验充分度**: ⭐⭐⭐⭐ 4个编码器模型×5类攻击，含置信区间和消融，但仅一个数据集
- **写作质量**: ⭐⭐⭐ 整体可读，但组织略显松散，部分细节（如微调目标函数）不够清晰
- **价值**: ⭐⭐⭐⭐ 为LLM安全提供了一个低门槛、易部署的实用防御层，有工程应用价值

<!-- RELATED:START -->

## 相关论文

- [Prioritized Semantic Learning for Zero-Shot Instance Navigation](../../ECCV2024/robotics/prioritized_semantic_learning_for_zeroshot_instance_navigation.md)
- [Understanding Prompt Tuning and In-Context Learning via Meta-Learning](understanding_prompt_tuning_and_in-context_learning_via_meta-learning.md)
- [MIP against Agent: Malicious Image Patches Hijacking Multimodal OS Agents](mip_against_agent_malicious_image_patches_hijacking_multimod.md)
- [PanoNav: Mapless Zero-Shot Object Navigation with Panoramic Scene Parsing and Dynamic Memory](../../AAAI2026/robotics/panonav_mapless_zero-shot_object_navigation_with_panoramic_scene_parsing_and_dyn.md)
- [Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs](toward_engineering_agi_benchmarking_the_engineering_design_capabilities_of_llms.md)

<!-- RELATED:END -->
