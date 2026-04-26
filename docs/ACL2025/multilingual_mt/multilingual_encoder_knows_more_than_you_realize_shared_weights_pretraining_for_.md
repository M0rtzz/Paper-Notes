---
title: >-
  [论文解读] Multilingual Encoder Knows More Than You Realize: Shared Weights Pretraining for Extremely Low-Resource Languages
description: >-
  [ACL 2025][低资源语言] 提出编码器-解码器权重共享框架，将多语言编码器高效扩展为文本生成模型，在藏、维、哈、蒙四种中国少数民族语言上显著超越mBART和13B LLM。
tags:
  - ACL 2025
  - 低资源语言
  - 权重共享
  - 多语言
  - 编码器-解码器
  - 中国少数民族语言
---

# Multilingual Encoder Knows More Than You Realize: Shared Weights Pretraining for Extremely Low-Resource Languages

**会议**: ACL 2025  
**arXiv**: [2502.10852](https://arxiv.org/abs/2502.10852)  
**代码**: https://github.com/asd765973346/xlm-swcm  
**领域**: 多语言NLP / 低资源语言  
**关键词**: 权重共享, 多语言编码器, 低资源语言, 中国少数民族语言, XLM-SWCM

## 一句话总结
提出编码器-解码器权重共享框架，通过交替复用编码器权重层和随机初始化层构建解码器，将多语言编码器CINO高效扩展为seq2seq模型XLM-SWCM，在藏维哈蒙四种极低资源语言上以不到0.5B参数大幅超越mBART和13B LLaMA。

## 研究背景与动机

**领域现状**：XLM-R等多语言预训练模型在高资源语言上表现优异，但在极低资源语言（如藏语、维吾尔语、哈萨克语、蒙古语）上仍然表现不佳。这些语言虽然有数百万到上千万说话者，但在OSCAR等主流多语种语料库中数据极其稀缺，哈萨克语和蒙古语甚至接近零。

**现有痛点**：（1）LLaMA、Qwen等现代LLM支持的语言远少于XLM-R，使得很多语言根本没有可用的文本生成模型；（2）mBART、mT5等多语言seq2seq模型虽然理论上覆盖上百种语言，但实际上未在中国少数民族语言上训练；（3）中国少数民族语言的书写系统与同语族其他地区不同（如维吾尔语在中国用阿拉伯字母，在中亚用西里尔字母），加剧了数据不匹配问题。

**核心矛盾**：多语言编码器（如XLM-R/CINO）已经在预训练中学到了丰富的跨语言语义空间，但它们是纯编码器模型，不能直接用于文本生成任务。从头训练一个seq2seq模型在极低资源设置下数据不够。

**本文目标**：设计一种方法将已有的多语言编码器高效扩展为编码器-解码器架构，复用编码器已学习的语义空间进行低资源语言的文本生成。

**切入角度**：编码器的权重已经编码了丰富的多语言知识。如果能将这些权重直接"复用"到解码器中——而非随机初始化一个全新解码器——就能大幅减少所需训练数据，加速收敛。

**核心 idea**：在编码器和解码器之间共享权重——解码器每X个层中，有X-1层复用编码器权重（CustomDecoderLayer），1层随机初始化（NormalDecoderLayer），形成交替结构，平衡知识复用和新能力学习。

## 方法详解

### 整体框架
从CINO（XLM-R的中国少数民族语言增强版）出发，复制其编码器权重初始化解码器，构建编码器-解码器架构的XLM-SWCM（4.57亿参数）。先在MC2语料库上用DAE+机器翻译双任务预训练，再在下游任务（摘要、阅读理解、翻译）上微调。

### 关键设计

1. **交替式权重共享解码器**:

    - 功能：构建高效的解码器，最大化复用编码器已有知识
    - 核心思路：解码器由两种层交替组成。CustomDecoderLayer从编码器对应层复制全部权重——自注意力→编码器自注意力权重，交叉注意力→编码器自注意力权重，两个FFN→编码器FFN权重。NormalDecoderLayer则完全随机初始化。每X=3个Custom层后插入1个Normal层，使得编码器n层对应解码器 $n + \lfloor n/X \rfloor$ 层
    - 设计动机：纯权重复制（无随机层）会限制模型学习生成特有能力；纯随机初始化则浪费了编码器已有知识。X=3是平衡点——对中等规模数据效果最佳。实验还发现X值应根据数据量调整：小数据用大X（更少新参数），大数据用小X（更多新参数容量）

2. **双任务预训练策略**:

    - 功能：将编码器从填空任务过渡到序列生成任务
    - 核心思路：主任务为去噪自编码（DAE，来自mBART），在输入中随机掩码/打乱/删除文本片段，训练解码器恢复原始序列。辅助任务为中文↔少数民族语言双向机器翻译，使用8000对翻译数据（每语言2000对），增强跨语言迁移能力
    - 设计动机：DAE帮助模型从编码器的词级完形填空任务过渡到序列生成任务；机器翻译任务则直接提供跨语言信号，弥补少数民族语言数据的稀缺

3. **平衡采样策略**:

    - 功能：确保低资源语言在训练中获得充分代表
    - 核心思路：采用类似XLM-R的采样策略，语言i的采样概率 $p_i = q_i^\alpha / \sum_j q_j^\alpha$，其中 $\alpha=0.3$ 为平滑参数，在均匀采样和按比例采样之间取平衡
    - 设计动机：不做平衡的话，中文数据会完全主导训练，少数民族语言得不到充分学习

### 损失函数 / 训练策略
DAE重建损失 + 翻译交叉熵损失联合训练。使用scheduled sampling逐步从teacher forcing过渡到自回归生成。AdamW优化器，学习率1e-4，8个epoch，2×A800 GPU训练92小时。

## 实验关键数据

### 主实验（藏语单语言微调，ROUGE-L F1）

| 模型 | 参数量 | 摘要 | 阅读理解 | 翻译 |
|------|--------|------|----------|------|
| MC2-LLaMA-13B | 13B | 16.1 | 13.2 | 15.1 |
| mBART-CM | 611M | 8.6 | 7.9 | 11.5 |
| XLM-SWCM (ours) | 492M | **25.7** | **16.4** | **24.5** |

### 跨语言迁移（中文微调→少数民族语言，摘要ROUGE-L）

| 模型 | 藏语 | 维吾尔语 | 蒙古语 |
|------|------|----------|--------|
| MC2-LLaMA-13B* | 13.1 | 11.7 | 9.7 |
| mBART-CM | 6.8 | 2.7 | 3.1 |
| XLM-SWCM (ours) | **17.1** | **12.5** | **13.5** |

### 消融实验

| 去除组件 | 摘要 | 阅读理解 | 翻译 |
|----------|------|----------|------|
| 完整模型 | 25.7 | 16.4 | 24.5 |
| 去掉机器翻译(MT) | 25.6 | 15.1 | 20.3 |
| 去掉DAE | 22.4 | 12.2 | 18.7 |
| 去掉权重共享(WS) | 17.1 | 11.7 | 18.2 |
| 去掉全部组件 | 15.9 | 10.8 | 16.5 |

### 关键发现
- 权重共享是最关键的组件——去掉后摘要从25.7降至17.1（-33%），比去掉DAE或MT的影响大得多
- XLM-SWCM以不到0.5B参数在所有任务上超越13B的MC2-LLaMA：摘要高59%，翻译高62%
- 插入频率X=3对中等数据量（20K）最优；小数据应用大X（模型更小防过拟合），大数据用小X（更多参数容量）
- 随机初始化的Normal层是必要的——纯权重复制（Baseline B）效果远不如交替结构，说明适度的"新鲜参数"帮助模型脱离编码器的表示空间约束

## 亮点与洞察
- **编码器权重的隐藏价值**：多语言编码器已经学到了比人们意识到的更多的语言知识，关键在于如何"解锁"这些知识用于生成任务。权重共享是一种优雅的解锁方式
- **X参数的灵活性**：插入频率X提供了一个简单而有效的旋钮来平衡模型容量和数据量——这对所有低资源场景都有参考价值
- **小模型胜大模型**：0.5B参数的XLM-SWCM全面超越13B的LLaMA变体，证明了在极低资源设置下，有效的知识迁移比堆参数更重要

## 局限与展望
- 单语言微调实验仅限于藏语（因为只有藏语有公开数据集），其他三种语言仅在跨语言迁移中评估
- 使用Google Translate生成的翻译对数据质量存疑，虽然经过人工验证但规模有限（每语言仅2000对）
- 框架通用性有待扩展到更多语言和更大规模的编码器
- 未来方向：将该框架应用到更多低资源语言，或与更大的编码器（如XLM-R Large）结合

## 相关工作与启发
- **vs mBART**: mBART也是seq2seq但从头预训练，对未见语言泛化差。XLM-SWCM通过复用CINO的多语言知识，以更少数据达到更好效果
- **vs MC2-LLaMA-13B**: 13B参数的LLaMA变体用LoRA微调，但跨语言迁移能力弱——常默认输出中文而非目标语言。XLM-SWCM的编码器-解码器结构天然更适合条件生成
- **vs 适配器方法（LoRA等）**: 适配器在编码器上调整不足以获得生成能力。权重共享提供了一种"编码器→编码器-解码器"的结构性转换方案

## 评分
- 新颖性: ⭐⭐⭐⭐ 编码器权重交替共享到解码器的框架设计简洁有效，对低资源NLP有启发
- 实验充分度: ⭐⭐⭐⭐ 三个任务、单语言+跨语言、多维度消融，实验设计全面
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，消融实验层层递进，分析深入
- 价值: ⭐⭐⭐⭐ 为极低资源语言的文本生成提供了实用且高效的解决方案

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Accessible Machine Translation Evaluation For Low-Resource Languages](accessible_machine_translation_evaluation_for_low-resource_languages.md)
- [\[ACL 2025\] Read it in Two Steps: Translating Extremely Low-Resource Languages with Code-Augmented Grammar Books](low_resource_translation.md)
- [\[ACL 2025\] The Esethu Framework: Reimagining Sustainable Dataset Governance and Curation for Low-Resource Languages](the_esethu_framework_reimagining_sustainable_dataset_governance_and_curation_for.md)
- [\[ACL 2025\] Dictionaries to the Rescue: Cross-Lingual Vocabulary Transfer for Low-Resource Languages Using Bilingual Dictionaries](dictionaries_to_the_rescue_cross-lingual_vocabulary_transfer_for_low-resource_la.md)
- [\[ACL 2025\] Understanding In-Context Machine Translation for Low-Resource Languages: A Case Study on Manchu](understanding_in-context_machine_translation_for_low-resource_languages_a_case_s.md)

<!-- RELATED:END -->
