---
title: >-
  [论文解读] Model Internal Sleuthing: Finding Lexical Identity and Inflectional Features in Modern Language Models
description: >-
  [ACL 2026][模型压缩][语言探针] 本文系统地对 25 个 Transformer 语言模型（从 BERT Base 到 Qwen2.5-7B）进行探针分析，发现词汇同一性（lexeme）在早期层线性可解码但随深度衰减，而屈折特征（inflection）在所有层中保持稳定可读，且占据紧凑可控的子空间。
tags:
  - ACL 2026
  - 模型压缩
  - 语言探针
  - 词汇同一性
  - 屈折特征
  - 表示几何
  - 跨语言分析
---

# Model Internal Sleuthing: Finding Lexical Identity and Inflectional Features in Modern Language Models

**会议**: ACL 2026  
**arXiv**: [2506.02132](https://arxiv.org/abs/2506.02132)  
**代码**: https://github.com/ml5885/model_internal_sleuthing (有)  
**领域**: 模型压缩 / NLP理解  
**关键词**: 语言探针, 词汇同一性, 屈折特征, 表示几何, 跨语言分析

## 一句话总结
本文系统地对 25 个 Transformer 语言模型（从 BERT Base 到 Qwen2.5-7B）进行探针分析，发现词汇同一性（lexeme）在早期层线性可解码但随深度衰减，而屈折特征（inflection）在所有层中保持稳定可读，且占据紧凑可控的子空间。

## 研究背景与动机

**领域现状**：探针研究（probing）是理解 Transformer 内部语言表示的核心方法，早期工作已在 BERT 和 GPT-2 上建立了"不同层编码不同语言层级"的层次化理解——底层编码表面特征，中层编码句法，高层编码语义。

**现有痛点**：此前的探针研究几乎全部聚焦于第一代模型（BERT、GPT-2），而现代 LLM 在架构（编码器/解码器）、训练数据规模（数十亿 vs 万亿 token）、后训练适配等方面已发生巨大变化，早期结论是否仍成立缺乏验证。

**核心矛盾**：我们对现代大型语言模型如何编码基础语言信息（词汇身份 vs 语法屈折）的理解，仍建立在过时的小模型实验基础上，存在严重的知识断层。

**本文目标**：(1) 在 25 个现代模型上系统探测词汇同一性和屈折特征的编码模式；(2) 分析表示几何、注意力 vs 残差流、激活引导、预训练动态等多个维度。

**切入角度**：选择词汇同一性（lexeme，如 walk/walked 共享词元）和屈折特征（如复数、过去式）两个属性——前者关联语义，后者关联语法——用来解耦模型如何权衡"意义"与"形式"。

**核心 idea**：用线性/非线性探针+选择性指标+表示几何分析+激活引导实验，全面刻画现代 LLM 中词汇与屈折信息的编码轨迹。

## 方法详解

### 整体框架
对 25 个预训练模型（3 类架构、6 种语言），从每层提取残差流激活，训练线性回归探针和 MLP 探针分别预测词元和屈折特征，并通过选择性（selectivity）、线性可分性差距（linear separability gap）、有效维度（effective dimensionality）、激活引导（steering）等多角度分析。

### 关键设计

1. **双探针+选择性指标体系**:

    - 功能：区分模型是否真正编码了语言信息，还是探针只是在记忆
    - 核心思路：训练线性回归和 MLP 两种探针，同时用随机标签构建控制任务。选择性 $\text{Sel}_\ell = \text{Acc}^\text{real}_\ell - \text{Acc}^\text{control}_\ell$ 衡量真正的语言信号；线性可分性差距 $\text{Gap}_\ell = \text{Sel}^\text{nonlin}_\ell - \text{Sel}^\text{linear}_\ell$ 衡量非线性探针是否带来真正的信息增益还是仅捕获虚假关联
    - 设计动机：高准确率不一定意味着语言信息真正被编码——可能只是探针容量过大导致记忆；选择性指标能有效过滤这一伪信号

2. **表示几何分析（Representation Geometry）**:

    - 功能：揭示模型中层表示空间的压缩/膨胀模式
    - 核心思路：计算每层激活的线性有效维度——即需要多少 PCA 分量才能解释固定比例的方差。发现 GPT-2、Qwen2.5、Pythia 存在急剧的中层维度坍缩（绝对激活值飙升至 ~8000），而 Llama、OLMo 则保持平滑压缩
    - 设计动机：有效维度的变化与探针性能和引导效果直接相关——维度坍缩层的引导效果显著降低

3. **屈折特征激活引导 (Inflection Steering)**:

    - 功能：因果验证屈折特征是否占据可控的低维子空间
    - 核心思路：对每对屈折类别（如单数vs复数）计算均值差异向量，以不同强度 $\lambda$ 添加到隐藏状态中，用线性探针测量干预后类别翻转率。结果表明即使中等干预强度（$\lambda=5$）也能产生大幅概率偏移
    - 设计动机：从关联到因果——探针结果只证明信息"存在"，引导实验证明该信息是"可操控的"，这对表示工程具有实际意义

### 损失函数 / 训练策略
线性探针使用岭正则化回归（闭式解），MLP 探针为两层 ReLU 网络（隐层 64 维），均使用标准交叉熵损失训练。

## 实验关键数据

### 主实验

| 属性 | 模型类型 | 早期层准确率 | 深层准确率 | 选择性趋势 |
|------|---------|------------|----------|----------|
| 词元(Lexeme) | 编码器 | 0.8-1.0 | 大幅下降 | 接近零 |
| 词元(Lexeme) | 小型解码器 | 0.8-1.0 | 缓慢下降 | 接近零 |
| 词元(Lexeme) | 大型解码器 | 0.8-1.0 | 保持较高 | 接近零 |
| 屈折(Inflection) | 所有 | 0.9-1.0 | 0.9-1.0 | 0.4-0.6 (正) |

### 消融实验

| 分析维度 | 关键发现 | 说明 |
|---------|---------|------|
| 线性vs非线性 | Gap < 0（全局） | MLP额外容量多捕获虚假关联而非真正语言结构 |
| 残差流vs注意力 | 残差流显著优于注意力 | 中层词元：残差0.6-0.9 vs 注意力0.2-0.4 |
| 跨语言 | 土耳其语衰减最快 | 词元准确率从0.95降至0.25，因形态复杂性 |
| 预训练动态 | 屈折早期稳定，词元持续演变 | 屈折几个checkpoint就收敛，词元后期仍在重塑 |

### 关键发现
- 词元信息的高早期准确率伴随接近零的选择性，意味着主要由表面相关性（如子词重叠）驱动而非真正的词汇结构
- 屈折信息在整个模型深度上保持正选择性（0.4-0.6），表明这是被"真正编码"的语言属性
- 频率与探针准确率强相关——罕见词元和罕见屈折形式是主要错误来源
- DeBERTa-v3 在约 75% 深度处出现引导效果骤降，暗示特殊的架构性表示约束

## 亮点与洞察
- **选择性指标的系统性运用**是本文方法论的最大亮点：不仅报告准确率还报告控制对比，有效解决了探针研究中长期存在的"记忆伪信号"问题。这一范式可直接迁移到任何探针实验
- 从"关联"到"因果"的激活引导验证思路很完整：先探针发现信息存在，再用引导证明信息可操控，最后用预训练动态追踪信息何时形成
- 25 个模型 × 6 种语言的覆盖规模前所未有，使结论具有很强的普适性

## 局限与展望
- 解码器模型使用最后一个子词token作为词表示，可能不是所有架构的最优选择
- 探针只能检测关联而非因果机制；引导实验也仅测量分类器变化而非下游生成效果
- 未处理同形异义的歧义情况（如英语中不定式和非过去式动词形式相同）
- 可扩展到更大规模模型（70B+）和更多语言特征（句法依存、语义角色等）

## 相关工作与启发
- **vs Jawahar et al. (2019) / Tenney et al. (2019)**: 他们在 BERT 上建立了层次化语言编码的认知，本文在 25 个现代模型上系统验证/更新了这些结论
- **vs Acs et al. (2024)**: 他们做多语言形态句法探针但限于 mBERT 和 XLM-RoBERTa，本文扩展到现代解码器模型并加入表示几何分析

## 评分
- 新颖性: ⭐⭐⭐⭐ 非全新范式但规模和深度前所未有
- 实验充分度: ⭐⭐⭐⭐⭐ 25模型×6语言×多维度分析极其全面
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，叙事流畅，图表丰富

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Reason Only When Needed: Efficient Generative Reward Modeling via Model-Internal Uncertainty](reason_only_when_needed_efficient_generative_reward_modeling_via_model-internal_.md)
- [\[ICML 2025\] Persistent Topological Features in Large Language Models](../../ICML2025/model_compression/persistent_topological_features_in_large_language_models.md)
- [\[ACL 2025\] Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders](../../ACL2025/model_compression/language_specific_features.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ACL 2026\] SeLaR: Selective Latent Reasoning in Large Language Models](selar_selective_latent_reasoning_in_large_language_models.md)

<!-- RELATED:END -->
