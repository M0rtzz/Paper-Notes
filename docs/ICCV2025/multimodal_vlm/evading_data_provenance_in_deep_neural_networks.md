---
title: >-
  [论文解读] Evading Data Provenance in Deep Neural Networks
description: >-
  [ICCV 2025][多模态][数据溯源] 揭示了当前数据集所有权验证（DOV）方法的安全假象——通过一个统一的规避框架 Escaping DOV，利用教师模型在 OOD 数据集上向代理学生传输任务相关但标识无关的知识，成功同时绕过所有 11 种 DOV 方法。
tags:
  - ICCV 2025
  - 多模态
  - 数据溯源
  - 数据集所有权验证
  - 规避攻击
  - 知识蒸馏
  - 后门水印
---

# Evading Data Provenance in Deep Neural Networks

**会议**: ICCV 2025  
**arXiv**: [2508.01074](https://arxiv.org/abs/2508.01074)  
**代码**: [GitHub](https://github.com/dbsxfz/EscapingDOV)  
**领域**: 数据安全 / 模型版权  
**关键词**: 数据溯源, 数据集所有权验证, 规避攻击, 知识蒸馏, 后门水印

## 一句话总结

揭示了当前数据集所有权验证（DOV）方法的安全假象——通过一个统一的规避框架 Escaping DOV，利用教师模型在 OOD 数据集上向代理学生传输任务相关但标识无关的知识，成功同时绕过所有 11 种 DOV 方法。

## 研究背景与动机

现代深度学习严重依赖大规模数据集，但许多数据集受版权保护或包含敏感信息。数据集所有权验证（DOV）作为追踪未授权模型训练的 post hoc 方法，近年来发展迅速，形成三大类方法：

**后门水印**：在数据集中嵌入"后门"样本，训练后的模型对触发样本产生预定义行为

**非投毒水印**：微调样本使模型置信度偏移，通过假设检验验证

**数据集指纹**：利用训练数据的固有特征（如决策边界距离）进行验证

然而，本文的核心发现是：**之前的研究使用了过于简单的规避攻击进行评估，造成了虚假的安全感**。现有规避方法主要是简单的正则化和通用后门防御（如微调、剪枝），缺乏应对高级 DOV（如不可见水印、非投毒水印、数据集指纹）的能力。

**关键洞察**：所有 DOV 方法的验证行为都具有两个共同特征——**排他性**（只属于特定数据集，不会被其他数据激活）和**隐蔽性**（不影响主任务语义，作为旁信号存在）。基于这一洞察，可以设计统一的规避策略：用 OOD 数据作为中介传输知识，自然地过滤掉这些排他且隐蔽的验证信号。

## 方法详解

### 整体框架

Escaping DOV 的流程：
1. **教师训练**：在版权数据集 $\mathcal{D}$ 上直接训练教师模型 $f_{\theta_t}$（不可避免地被标记）
2. **迁移集收集**：从 OOD 图库集 $\mathcal{G}$（如 ImageNet）中精选最优子集 $\mathcal{T}$
3. **选择性知识蒸馏**：用 $\mathcal{T}$ 作为中介，从教师提取任务导向但无标识的知识到代理学生 $f_{\theta_s}$

### 关键设计

1. **迁移集精选（Transfer Set Curation）**：利用 VLM（MobileCLIP）和 LLM（GPT-4o mini）实现可靠的样本选择，核心流程：

    - **零样本分类**：用 LLM 为每个类别生成描述集，增强 VLM 的零样本分类能力（避免在版权数据上微调 VLM 导致被标记）。将图库集样本按 VLM 预测分配到 $K$ 个类别桶
    - **按分布距离排序**：用 VLM 图像编码器将版权数据集 $\mathcal{D}$ 投影到特征空间，计算每类的密度质心 $\text{Cent}_t$。桶内样本按与质心的距离排序
    - **共识过滤**：仅选择教师模型和 VLM 预测一致的样本（确保教师的预测基于真实语义而非验证行为），直到每类数量达到 $|\mathcal{D}_t|$

   这保证了迁移集中几乎不包含能触发验证行为的样本。

2. **选择性知识蒸馏（Selective Knowledge Transfer, SKT）**：标准蒸馏可能通过软标签中的"暗知识"传递验证行为。SKT 通过生成教师模型的最坏情况扰动来抑制这种传递：

    - **扰动池生成**：用教师模型在 $\mathcal{D}$ 上生成通用对抗扰动 $\delta$（自适应选择 $L_0$、$L_2$、$L_\infty$ 范数约束中使损失最大的），离线预计算
    - **腐蚀链生成**：用遗传算法搜索 ImageNet-C 的 15 种腐蚀的最优组合和顺序，形成非受限扰动
    - **对抗蒸馏**：以概率 $\beta$ 将扰动/腐蚀应用到迁移集样本后输入学生，但教师仍用干净样本输出，迫使学生对验证行为保持不变性：

$$\arg\min_{\theta_s} \mathcal{L}\left(\frac{f_{\theta_s}(x)}{\tau}, \frac{f_{\theta_t}(x)}{\tau}\right) + \beta \cdot \mathcal{L}\left(\frac{f_{\theta_s}(A(x))}{\tau}, \frac{f_{\theta_t}(x)}{\tau}\right)$$

### 损失函数 / 训练策略

使用标准 KL 散度蒸馏损失，温度 $\tau=1$。SKT 的核心想法是：任务知识和验证知识具有不同的"沸点"——在低温下可以自然分离。扰动池和腐蚀链的生成都是离线完成，训练时仅增加少量开销。

## 实验关键数据

### 主实验

在 CIFAR-10 和 Tiny ImageNet 上对 11 种 DOV 方法的规避结果：

| DOV 方法 | CIFAR-10 原始 VSR/p-value | 规避后 VSR/p-value | 规避后 ACC |
|----------|:---:|:---:|:---:|
| BadNets | 100.00% | **1.36%** | 93.46% |
| UBW（无目标） | 95.54% | **1.74%** | 93.41% |
| Narcissus（不可见） | 87.34% | **4.59%** | 94.37% |
| Radioactive Data | 3.03e-3 | **9.45e-1** | 94.07% |
| ANW | 1.37e-9 | **1.00e+0** | 93.91% |
| Domain Watermark | 1.67e-22 | **1.00e+0** | 93.90% |
| Dataset Inference | 1.87e-3 | **4.76e-1** | 93.97% |
| MeFA | 2.62e-14 | **1.00e+0** | 93.93% |

所有 DOV 方法均被成功规避（VSR 接近随机猜测，p-value 远超 0.01 阈值），且准确率损失不到 1%。

### 消融实验

与 9 种 SOTA 规避方法在 CIFAR-10 上的对比（4 种最难 DOV）：

| 方法 | BadNets ACC/VSR | Narcissus ACC/VSR | Isotope p-value | Dataset Inf. p-value |
|------|:---:|:---:|:---:|:---:|
| Fine-pruning | 86.21/99.37 | 86.58/43.75 | 0.0532 | 0.0095 |
| BCU | 92.75/1.56 | 92.33/61.72 | 0.1622 | 0.0773 |
| ABD | 84.94/7.07 | 85.42/22.10 | **0.0066** | 0.3696 |
| **Escaping DOV** | **93.46/1.36** | **94.37/4.59** | **0.2845** | **0.4759** |

Escaping DOV 是唯一能**同时**规避所有 4 种 DOV 方法的攻击，且在准确率上全面领先。ABD 虽然在 Narcissus 上也有效，但无法规避 Isotope（p-value=0.0066 < 0.01）。

### 关键发现

- 教师和学生参数之间存在**损失屏障**（loss barrier），表明学生发展出了完全不同的预测机制
- 教师的测试/训练损失比超过 $10^3$（容易被指纹检测），学生的比值不到 2
- 蒸馏温度 $\tau$ 的影响：温度越高，无 SKT 时 VSR 急剧升高，但有 SKT 时保持稳定
- 在域特定数据集（人脸情感 RAFDB、医学影像 OrganCMNIST）上同样有效
- 结合 mixup 训练教师可进一步增强规避效果

## 亮点与洞察

- **统一视角的深刻洞察**：所有 DOV 方法的验证信号都是"排他且隐蔽"的，这是其力量来源也是致命弱点——OOD 数据天然不会触发这些信号
- **攻防不对称性**：构建强大的 DOV 需要同时满足多个约束（不影响合法训练、不可检测、不可绕过），但规避者只需破坏其中一环
- **实用价值**：作为安全评估框架，帮助 DOV 开发者更严格地测试其方法的鲁棒性
- 温度 $\tau=1$ 下任务知识与验证知识自然分离的发现具有理论意义

## 局限与展望

- 规避成功依赖于获取足够大且多样的 OOD 图库集
- 对自适应防御者（如语义后门、反蒸馏后门）的鲁棒性有限
- 本文聚焦图像分类，未扩展到 LLM、文本-图像生成等其他模态
- 研究结论虽然对安全研究有价值，但也可能被攻击者利用——存在双刃剑效应
- 迁移集精选依赖 CLIP 和 GPT-4o mini 的能力，这些模型本身可能有偏差

## 相关工作与启发

- 知识蒸馏在后门防御中的应用（NAD、BCU、ABD）是最相关的工作，但它们使用少量干净数据蒸馏，而非 OOD 迁移
- 与模型水印移除（IPRemoval）的区别在于：本文针对数据集级别的溯源而非模型级别
- VLM + LLM 辅助的迁移集选择思路可拓展到数据集蒸馏等领域
- 提出的规避框架为 DOV 领域建立了更可靠的评估标准

## 评分

- **新颖性**: 8/10 — 首个统一规避框架，洞察深刻
- **技术质量**: 8/10 — 覆盖 11 种 DOV + 9 种基线，实验扎实
- **实用性**: 7/10 — 双刃剑：既是安全评估工具也可能被滥用
- **写作质量**: 8/10 — 问题动机和方法设计逻辑清晰

<!-- RELATED:START -->

## 相关论文

- [Importance Corrected Neural JKO Sampling](../../ICML2025/multimodal_vlm/importance_corrected_neural_jko_sampling.md)
- [Information Theoretic Optimal Surveillance for Epidemic Prevalence in Networks](../../AAAI2026/multimodal_vlm/information_theoretic_optimal_surveillance_for_epidemic_prevalence_in_networks.md)
- [Can MLLMs Understand the Deep Implication Behind Chinese Images?](../../ACL2025/multimodal_vlm/can_mllms_understand_the_deep_implication_behind_chinese_images.md)
- [What's in the Image? A Deep-Dive into the Vision of Vision Language Models](../../CVPR2025/multimodal_vlm/whats_in_the_image_a_deep-dive_into_the_vision_of_vision_language_models.md)
- [Effective Training Data Synthesis for Improving MLLM Chart Understanding](effective_training_data_synthesis_for_improving_mllm_chart_understanding.md)

<!-- RELATED:END -->
