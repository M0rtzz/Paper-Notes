---
title: >-
  [论文解读] A Survey on Efficient Large Language Model Training: From Data-centric Perspectives
description: >-
  [ACL 2025][LLM/NLP][数据高效训练] 本文提出首个系统性的"数据高效 LLM 后训练"综述框架，将方法分为数据选择、数据质量增强、合成数据生成、数据蒸馏与压缩、自演化数据生态五大类，构建了完整的"数据价值飞轮"体系。
tags:
  - ACL 2025
  - LLM/NLP
  - 数据高效训练
  - 后训练
  - 数据选择
  - 合成数据
  - 知识蒸馏
  - 自演化数据
---

# A Survey on Efficient Large Language Model Training: From Data-centric Perspectives

**会议**: ACL 2025  
**arXiv**: [2510.25817](https://arxiv.org/abs/2510.25817)  
**代码**: [GitHub](https://github.com/luo-junyu/Awesome-Data-Efficient-LLM)  
**领域**: LLM 训练 / 数据效率  
**关键词**: 数据高效训练, 后训练, 数据选择, 合成数据, 知识蒸馏, 自演化数据  

## 一句话总结

本文提出首个系统性的"数据高效 LLM 后训练"综述框架，将方法分为数据选择、数据质量增强、合成数据生成、数据蒸馏与压缩、自演化数据生态五大类，构建了完整的"数据价值飞轮"体系。

## 研究背景与动机

- **核心问题**: LLM 后训练（SFT、RLHF 等）是释放模型泛化能力的关键阶段，但面临严重的数据困境：人工标注成本高昂、数据量扩张的边际收益递减、静态数据无法适应知识演化。
- **数据高效训练的必要性**: DeepSeek-R1 等模型通过强化学习实现数据高效后训练的成功案例，进一步证明了数据效率的重要性。核心逻辑是：突破效率瓶颈需要在数据全生命周期建立价值提取机制，而非单纯扩大数据规模。
- **研究空白**: 尽管已有数据选择、合成数据等单一方面的综述，但缺乏从数据效率这一统一视角出发的系统性综述。

## 方法详解

### 整体框架：数据价值飞轮

五大组件形成闭环：**数据选择**（过滤高价值子集）→ **质量增强**（提升现有数据效用）→ **合成生成**（创建新训练数据）→ **蒸馏压缩**（提取核心知识）→ **自演化生态**（构建自进化机制）。五者互补：选择过滤质量数据，增强提升效用，生成扩展覆盖，蒸馏浓缩知识，自演化实现持续改进。

### 关键设计

#### 1. 数据选择 (Data Selection)

- **静态过滤**: Alpagasus 仅用 17% 数据达到可比性能；基于质量/信息论指标离线选择
- **动态选择**: Active Instruction Tuning 按不确定性优先选择高价值样本；LESS 利用低秩梯度特征进行优化器感知的相似度搜索
- **Agent 策略**: CLUES 多模型投票机制；DATA ADVISOR 红队过滤
- **标注效率**: SELF-INSTRUCT 自主生成指令数据；LLMaAA 用 LLM 做标注器

#### 2. 数据质量增强 (Data Quality Enhancement)

- **语义重写**: CoachLM 自动修改复杂指令减少歧义；LLM2LLM 迭代改进低置信样本
- **毒性控制**: ToxiCraft 生成对抗性数据集压力测试模型安全边界
- **分布稳定**: 合成过采样解决类别不平衡；RobustFT 多专家协作噪声检测+基于熵的数据选择

#### 3. 合成数据生成 (Synthetic Data Generation)

- **指令驱动**: SynPO 生成偏好对用于对齐（ROUGE-L +12%）；Magpie 无模板指令生成
- **知识引导**: 结合知识图谱/结构化知识保证事实准确性；混合生成降低 API 成本 70%
- **对抗生成**: 探测模型漏洞以增强鲁棒性

#### 4. 数据蒸馏与压缩 (Distillation & Compression)

- **模型蒸馏**: Impossible Distillation 从低质量教师创建高质量学生；跨 Tokenizer 蒸馏
- **数据蒸馏**: LLMLingua-2 token 级蒸馏实现 prompt 压缩
- **联合压缩**: LLaMA-7B 压缩至 2.8B 参数性能损失极小

#### 5. 自演化数据生态 (Self-Evolving Data Ecosystem)

- **自迭代优化**: Self-Rewarding, Self-Refine — 模型用自身输出自主改进
- **动态评估反馈**: 多 Agent 实时调整评估和优化
- **LLM-as-a-Judge**: 自评估范式替代外部评估

### 方法对比

| 类别 | 数据依赖 | 计算成本 | 模型依赖 | 数据价值挖掘 |
|------|:---:|:---:|:---:|:---:|
| 数据选择 | ++ | + | + | +++ |
| 质量增强 | ++ | ++ | ++ | ++ |
| 合成生成 | + | +++ | +++ | + |
| 蒸馏压缩 | + | + | +++ | +++ |
| 自演化 | + | +++ | +++ | +++ |

## 实验

本文为综述论文，无新实验。但系统梳理了各子领域的关键实验结论：

### 代表性方法效果汇总

| 方法 | 类别 | 关键效果 |
|------|------|------|
| Alpagasus | 数据选择 | 17% 数据达到可比性能 |
| SynPO | 合成生成 | ROUGE-L +12% |
| 混合生成 (Chan et al.) | 合成生成 | API 成本降低 70% |
| LLaMA-7B 压缩 | 联合压缩 | 2.8B 参数、性能损失极小 |
| Magpie | 合成生成 | AlpacaEval 98% 准确率 |

### 关键发现

- 单纯扩大数据规模的边际收益递减，需转向数据价值挖掘
- 五大方法论互补而非替代，应建立统一框架
- 自演化和 LLM-as-a-Judge 是减少人工干预的重要方向
- 领域特定的数据合成比通用模型生成更有效

## 亮点

- 首个从数据效率统一视角系统综述 LLM 后训练的工作
- 提出"数据价值飞轮"概念，将分散的研究组织为有机整体
- 分类法清晰（5 大类 × 多个子类），覆盖面广
- 配套 awesome list 持续更新

## 局限性

- 领域发展极快，部分新兴技术可能未被完全覆盖
- 五大方法之间的协同效应和交互机制尚未深入探讨
- 对可信度和可扩展性的讨论不够充分
- 缺乏跨方法对比的统一实验验证

## 相关工作

- **数据选择综述**: Wang et al. (2024b) — 聚焦数据选择单一维度
- **合成数据综述**: Long et al. (2024); Tan et al. (2024) — 仅覆盖合成生成
- **模型自反馈**: Liang et al. (2024a); Pan et al. (2023) — 自演化方向
- **自进化综述**: Tao et al. (2024) — 模型自进化
- **训练效率综述**: Wan et al. (2023) — 侧重时间效率而非数据效率

## 评分

| 维度 | 分数 |
|------|:---:|
| 创新性 | ★★★☆☆ |
| 实用性 | ★★★★★ |
| 实验充分度 | ★★★☆☆ |
| 写作质量 | ★★★★☆ |
| 总评 | ★★★★☆ |

<!-- RELATED:START -->

## 相关论文

- [Language Model Fine-Tuning on Scaled Survey Data for Predicting Distributions of Public Opinions](language_model_fine-tuning_on_scaled_survey_data_for_predicting_distributions_of.md)
- [Understanding Silent Data Corruption in LLM Training](understanding_silent_data_corruption_in_llm_training.md)
- [Training Language Model to Critique for Better Refinement](training_language_model_to_critique_for_better_refinement.md)
- [Large Language Models in Bioinformatics: A Survey](large_language_models_in_bioinformatics_a_survey.md)
- [Knowledge Boundary of Large Language Models: A Survey](knowledge_boundary_survey.md)

<!-- RELATED:END -->
