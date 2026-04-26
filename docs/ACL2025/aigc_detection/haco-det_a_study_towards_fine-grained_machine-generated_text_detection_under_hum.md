---
title: >-
  [论文解读] HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring
description: >-
  [文本检测] 提出面向人机协作写作场景的细粒度机器生成文本（MGT）检测基准 HACo-Det，通过多轮局部改写流水线自动构建带词级归属标注的 11,200 篇人机共创文本，并将七种主流检测器改造为词级序列标注模式进行系统评估，揭示当前方法在细粒度检测上的巨大改进空间。
tags:
  - 文本检测
  - fine-grained detection
  - human-AI coauthoring
  - word-level attribution
  - sequence labeling
---

# HACo-Det: A Study Towards Fine-Grained Machine-Generated Text Detection under Human-AI Coauthoring

**会议/期刊**: ACL 2025  
**arXiv**: [2506.02959](https://arxiv.org/abs/2506.02959)  
**代码**: -  
**领域**: AIGC检测 / 人机协作文本检测  
**关键词**: machine-generated text detection, fine-grained detection, human-AI coauthoring, word-level attribution, sequence labeling  

## 一句话总结

提出面向人机协作写作场景的细粒度机器生成文本（MGT）检测基准 HACo-Det，通过多轮局部改写流水线自动构建带词级归属标注的 11,200 篇人机共创文本，并将七种主流检测器改造为词级序列标注模式进行系统评估，揭示当前方法在细粒度检测上的巨大改进空间。

## 研究背景与动机

- **现实需求**：GPT-4o Canvas、Notion AI、Wordcraft 等人机协作写作系统日益普及，文本中人类与 AI 的贡献交织混合，传统文档级二分类检测（全文标"人写"或"机写"）已无法满足细粒度著作权归属鉴定的实际需求。
- **现有数据集缺陷**：(1) 主流数据集将"人写开头 + LLM 续写"整段标记为 MGT，忽视了人写前缀的归属；(2) 将改写后文本直接全部标记为机器生成，但词形未变化的词理应保留原始归属；(3) 仅模拟单轮协作，与真实多轮交互场景存在显著差距。
- **检测方法瓶颈**：Metric-based 方法（DetectGPT、Fast-DetectGPT 等）依赖白盒 token 级统计量，在文档级尚有效，但缺乏向词级/句级细粒度推广的能力；Finetune-based 方法虽更强但跨域跨模型泛化仍是开放问题。
- **本文切入点**：设计词级标注的多轮改写数据集 HACo-Det，将 MGT 检测统一建模为词级序列标注任务 $D_w(T_w) = L_w$，句级标签通过多数投票聚合获得，系统对比七种检测器在 IND / OOD-Domain / OOD-Model 三种设置下的表现。

## 方法详解

### 整体框架

三阶段流水线——**原始文本采样 → 多轮 LLM 局部改写 → 词级/句级 grounded 标注**：

- **数据来源**：四个领域的人类写作文本（新闻 XSum、故事 WritingPrompts、学术论文 Dagpap24、维基百科 Wikipedia_en），每域 2,800 篇，共 11,200 篇
- **生成器**：四种指令调优 LLM（Llama-3、Mixtral、GPT-4o mini、GPT-4o），分别搭配不同的改写指令模板
- **标注方式**：基于词级匹配的 grounded attribution labeling——通过对比改写前后文本的词跨度匹配确定每个词是 HWT 还是 MGT

### 关键设计

**1. 多轮局部改写策略**：每轮使用 NLTK 将文本切分为句子序列，按照预设算法（Algorithm 1）选取一段连续句子片段送入 LLM 改写。每轮仅改写已标记为 MGT 的片段，避免著作权重复歧义。改写轮数与文本长度成正比，短文本（新闻）约 2-3 轮，长文本（维基百科）可达 6 轮以上，由此实现不同 AI 介入比例的自然分布。

**2. 词级 Grounded 归属标注**：改写后片段 $S'_{\text{span}}$ 中的词默认标记为 MGT，但若某词在改写前对应片段 $S_{\text{span}}$ 中以相同词形（含时态变化）出现，则保留其原始 HWT 标签。这一规则避免了"改写=完全机器生成"的过度简化，使得即使在 LLM 改写后的段落中，保留原文表述的词仍归属于人类作者。

**3. 统一序列标注与句级聚合**：将所有检测任务统一建模为词级二分类序列标注。对于句级检测，采用多数投票聚合——句子 $s_i$ 中 MGT 词数超过 HWT 词数则该句标记为 MGT：$L_{s_i} = \arg\max_c \sum_{k=1}^{m} \mathbb{I}(L_{w_{i,k}} = c)$。七种主流检测器（DeBERTa、SeqXGPT、DetectGPT、Fast-DetectGPT、NPR、LRR、GLTR）均被改造适配此框架。

### 数据集统计

| 领域 | 样本数 | 平均 MGT 片段数 | MGT 词占比 | 平均文本长度 (词) |
|------|:------:|:---------------:|:----------:|:-----------------:|
| News (XSum) | 2,800 | 2.44 | 45% | 1,489 |
| Story (WritingPrompts) | 2,800 | 3.97 | 40% | 1,964 |
| Paper (Dagpap24) | 2,800 | 4.56 | 32% | 4,558 |
| Wikipedia | 2,800 | 6.01 | 26% | 7,724 |
| **合计** | **11,200** | **4.25** | **36%** | **3,934** |

数据分析显示：GPT-4o 改写幅度最大（与原文相似度最低），Llama-3 改写最温和；Paper 和 Wikipedia 领域文本显著更长（4k-8k 词），更利于跨域泛化。

## 实验

### 主实验：域内检测（IND）

| 检测器 | 类别 | F1-W (词级) | F1-S (句级) | AUC-W | AUC-S |
|--------|------|:----------:|:----------:|:-----:|:-----:|
| Random | - | 0.433 | 0.497 | - | - |
| **DeBERTa** | Finetune | **0.831** | **0.966** | - | - |
| SeqXGPT | Finetune | 0.513 | 0.674 | - | - |
| DetectGPT | Metric (w/ perturb) | 0.375 | 0.459 | 0.482 | 0.501 |
| Fast-DetectGPT | Metric (w/ perturb) | 0.501 | 0.533 | 0.507 | 0.510 |
| NPR | Metric (w/ perturb) | 0.414 | 0.473 | 0.485 | 0.509 |
| log prob | Metric (w/o perturb) | 0.479 | 0.444 | 0.482 | 0.511 |
| LRR | Metric (w/o perturb) | 0.475 | 0.516 | 0.483 | 0.510 |
| entropy | Metric (w/o perturb) | 0.479 | 0.392 | 0.488 | 0.511 |

核心结论：Metric-based 方法在词级检测上全面接近随机猜测（平均 F1 ≈ 0.46），DeBERTa 以 0.831 F1-W 一枝独秀。

### OOD 泛化实验

| 泛化设置 | DeBERTa F1-W 范围 | SeqXGPT F1-W 范围 | Metric-based F1-W |
|----------|:-----------------:|:-----------------:|:-----------------:|
| IND（基线） | 0.831 | 0.513 | 0.375–0.501 |
| OOD-Domain（跨域） | 0.776–0.830 | 0.450–0.490 | 多数略有波动 |
| OOD-Model（跨模型） | 0.768–0.854 | 0.462–0.498 | 多数略有波动 |

- DeBERTa 跨域跨模型均保持 0.77+ 的 F1-W，泛化能力远优于其他方法
- 训练于长文本域（Paper/Wikipedia）时 OOD 表现甚至优于 IND，说明长文本的多样语言模式有助泛化
- 训练于 GPT-4o 语料时跨模型泛化更好，因其改写幅度最大、信号更强

### 文档级 AI 比例预测

| 方法 | DeBERTa | SeqXGPT | DetectGPT | Fast-DetectGPT | NPR | LRR |
|------|:-------:|:-------:|:---------:|:--------------:|:---:|:---:|
| 句级误差 | **1.78%** | 12.00% | 10.70% | 14.75% | 21.01% | 17.44% |
| 词级误差 | **1.84%** | 10.00% | 43.48% | 12.65% | 32.37% | 12.00% |

DeBERTa 的文档级 AI 比例预测误差不到 2%，可为实际审核场景提供可靠的 AI 介入度估计。

### 关键发现

- **Metric-based 方法全面失败**：所有统计度量检测器在词级检测上 F1 ≈ 随机水平，简单 token 级度量无法支撑细粒度检测；即使采用扰动（DetectGPT 系列）也无明显改善
- **语义表征是关键**：DeBERTa 通过监督微调获取嵌入级语义特征，可有效区分人机混合词序列；SeqXGPT 虽也是 finetune 方法但架构适配不足
- **上下文窗口是瓶颈**：对长文本做 chunking 会损失上下文信息；DeBERTa 和 SeqXGPT 的 F1 均随输入 chunk 长度增大而单调提升
- **零样本检测远未可行**：冻结 DeBERTa 编码器仅训练分类头时，IND F1-W 从 0.831 降至 0.571；Metric-based 方法的句级零样本同样接近随机
- **跨改写模式泛化困难**：在 HACo-Det 上训练的 DeBERTa 迁移到 SeqXGPT-Bench（不同改写流水线）时性能显著下降

## 亮点

- 任务设计贴合真实场景：多轮局部改写模拟人机交互协作，远比"人写开头 + AI 续写"更真实
- 词级 grounded 标注方法合理：基于词匹配保留原始归属，避免改写即机器生成的过度简化
- 实验设计系统全面：IND / OOD-Domain / OOD-Model 三种设置 × 4 领域 × 4 生成器的完整矩阵
- 提供 AI 比例预测视角：将细粒度检测与文档级 AI 介入度估计关联，具有实际审核应用价值

## 局限性

- 词级"著作权转移"定义具有主观性，基于词形匹配的标注规则过于简单，未考虑语义等价替换
- 仅覆盖改写（paraphrase）场景，未涉及 LLM 插入新内容、删除原文、结构重组等更复杂的协作模式
- 数据集仅包含英文，未评估多语言或跨语言检测能力
- 每篇文本仅由单个 LLM 改写，未模拟多 LLM 协作或人类多轮主动编辑的场景
- 未提出新的检测方法，主要贡献在数据集和基准评估

## 相关工作

- **文档级 MGT 检测**：DetectGPT (Mitchell et al., 2023) 基于对数概率曲率、Fast-DetectGPT (Bao et al., 2023) 引入条件概率加速、RAID (Dugan et al., 2024) 大规模鲁棒性基准
- **细粒度检测**：Mixtext 三分类 (Gao et al., 2024)、边界检测 (Kushnareva et al., 2024)、MGT 定位 (Zhang et al., 2024c)、句级多特征融合 (Tao et al., 2024)
- **检测器鲁棒性**：Shi et al. (2024) 对抗攻击（词替换+提示攻击）、Wang et al. (2024) 多层次扰动测试发现大多数检测器性能显著下降
- **人机协作写作**：Lee et al. (2022)、Chakrabarty et al. (2022) 协作诗歌写作、Reza et al. (2024, 2025) 协作内容生成

## 评分

| 维度 | 分数 | 简评 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 首个多轮改写+词级归属标注的人机共创检测基准 |
| 技术深度 | ⭐⭐⭐ | 重在数据集与系统评估，检测方法本身为改造已有工作 |
| 实验充分度 | ⭐⭐⭐⭐⭐ | IND/OOD-Domain/OOD-Model全覆盖，7种方法3种粒度 |
| 实用价值 | ⭐⭐⭐⭐ | AI比例预测视角有实际审核价值，揭示现有方法的差距 |

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] MultiSocial: Multilingual Benchmark of Machine-Generated Text Detection of Social-Media Texts](multisocial_mgt_detection.md)
- [\[ACL 2025\] Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training](greater_adversarial_mgt_detection.md)
- [\[ACL 2025\] Are We in the AI-Generated Text World Already? Quantifying and Monitoring AIGT on Social Media](aigt_social_media_monitoring.md)
- [\[ACL 2025\] An Empirical Study on Detecting AI-Generated Text in Financial Reports](an_empirical_study_on_detecting_ai-generated_text_in_financial_reports.md)
- [\[ACL 2025\] Who Writes What: Unveiling the Impact of Author Roles on AI-generated Text Detection](who_writes_what_ai_detection.md)

<!-- RELATED:END -->
