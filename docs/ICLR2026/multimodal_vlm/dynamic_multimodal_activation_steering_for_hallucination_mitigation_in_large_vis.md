---
title: >-
  [论文解读] Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models
description: >-
  [ICLR 2026][多模态][幻觉缓解] 提出动态多模态激活引导（DMAS），通过构建基于语义的真实性引导向量数据库和视觉感知引导向量，在推理时动态选择最相关的引导向量对关键注意力头进行干预，无需训练即可显著缓解LVLM幻觉，在MME上提升94.66分，在CHAIR上降低20.2%幻觉率。
tags:
  - ICLR 2026
  - 多模态
  - 幻觉缓解
  - 激活工程
  - 注意力头干预
  - 无训练方法
  - 大视觉语言模型
---

# Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models

**会议**: ICLR 2026  
**arXiv**: [2602.21704](https://arxiv.org/abs/2602.21704)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 幻觉缓解, 激活工程, 注意力头干预, 无训练方法, 大视觉语言模型

## 一句话总结
提出动态多模态激活引导（DMAS），通过构建基于语义的真实性引导向量数据库和视觉感知引导向量，在推理时动态选择最相关的引导向量对关键注意力头进行干预，无需训练即可显著缓解LVLM幻觉，在MME上提升94.66分，在CHAIR上降低20.2%幻觉率。

## 研究背景与动机
大视觉语言模型（LVLM）在VQA、图像描述等任务上表现出色，但存在严重的幻觉问题——编造不存在的物体或错误描述图像内容。现有方法分为两类：训练方法需要精心标注数据和大量算力（如LRV、RLHF-V），解码方法（如VCD、ICD）虽然不需训练但往往损害生成质量。

近期的激活工程方法（如ICT、VTI）尝试通过干预模型内部表示来减少幻觉，但存在关键不足：ICT仅关注视觉层面干预，忽略了LVLM的多模态特性；VTI使用固定的引导向量，忽略了不同语义上下文下引导向量的差异。

**核心发现**：作者通过分析LLaVAv1.5的注意力模式，发现两个关键现象：(1) 真实性能力和视觉感知能力主要激活不同的注意力头子集（真实性集中在第30层，视觉感知集中在第31层）；(2) 真实性引导向量在不同语义上下文间差异显著（t-SNE可视化显示不同语义簇明显分离）。这两个发现直接启发了动态多模态激活引导的设计。

## 方法详解

### 整体框架
DMAS分三步：(1) 建立动态真实性引导向量数据库；(2) 计算视觉感知引导向量；(3) 推理时对不同注意力头施加动态干预。整个方法无需训练，以即插即用的方式工作。

### 关键设计
1. **真实性引导向量数据库**: 选取AMBER和SEED数据集，按语义聚类为4簇。对每个样本构造正确/幻觉回答对，分别输入LVLM获取最后一个token在各层注意力头的激活值 $A_{pos}$ 和 $A_{neg}$。簇内平均激活差异构成引导向量：$D_i = \frac{1}{|C_i|}\sum_{j \in C_i}(A_{pos,j} - A_{neg,j})$，并通过PCA降噪。以簇的平均嵌入为Key、引导向量为Value构建Key-Value数据库。推理时通过sentence transformer计算输入与各Key的语义相似度，动态检索最匹配的引导向量。

2. **视觉感知引导向量**: 给定原始图像 $V$ 和加噪图像 $V'$（通过前向扩散过程），用YOLOv11检测物体生成描述模板，对比原始输入 $(V, T+Y_O)$ 与扰动输入 $(V', T+Y_{O'})$ 的激活差异：$D_v = A_v - A_{v'}$，同样通过PCA提取主成分。这一设计增强模型对视觉信息的关注度。

3. **动态推理干预**: 推理时，通过余弦相似度检索语义最匹配的真实性引导向量 $D_f$，并构建二值掩码 $M_f$ 和 $M_v$，仅对激活差异最大的Top-K注意力头施加干预。修改后的注意力计算为：$\mathbf{x}^{(l+1)} = \mathbf{x}^{(l)} + \text{Concat}[\text{Attn}^{(l,h)}(\mathbf{x}^{(l)}) + \alpha \cdot M_f^{(l,h)} \cdot D_f^{(l,h)} + \beta \cdot M_v^{(l,h)} \cdot D_v^{(l,h)}] \cdot \mathbf{W}_o^{(l)}$，其中 $\alpha$ 和 $\beta$ 控制干预强度。

### 损失函数 / 训练策略
无需训练。超参数 $\alpha, \beta \in \{0.5, 1, ..., 10\}$，$K \in \{32, 64, ..., 1024\}$，通过网格搜索确定。温度设为0，top_p设为1。

值得注意的实现细节：
- 引导向量数据库的Key嵌入通过sentence transformer（all-mpnet-base-v2）获取
- 视觉感知部分使用YOLOv11检测图像中的物体，并从同类别物体库中随机选取不在图中的物体作为对比
- PCA降噪步骤对真实性和视觉感知引导向量分别应用，提取最显著的主成分
- 所有实验在NVIDIA RTX 4090(48GB) GPU上进行

## 实验关键数据

### 主实验

| 模型 | 方法 | Existence↑ | Count↑ | Position↑ | Color↑ | Total↑ |
|------|------|-----------|--------|-----------|--------|--------|
| LLaVAv1.5 | Regular | 175.67 | 124.67 | 114.00 | 151.00 | 565.33 |
| LLaVAv1.5 | ICT | 190.00 | 160.43 | 128.67 | 170.00 | 649.10 |
| LLaVAv1.5 | **DMAS** | **195.00** | **158.33** | **133.33** | **173.33** | **659.99** |
| QwenVL | Regular | 155.00 | 127.67 | 131.67 | 173.00 | 587.33 |
| QwenVL | VAF | 165.00 | 155.00 | 133.33 | 175.00 | 628.33 |
| QwenVL | **DMAS** | **170.00** | **145.00** | **133.33** | **185.00** | **633.33** |

**CHAIR结果**（LLaVAv1.5）:

| 方法 | CHAIR_S↓ | CHAIR_I↓ |
|------|----------|----------|
| Regular | 51.0 | 15.2 |
| VTI | 35.8 | 11.1 |
| **DMAS** | **30.8** | **11.4** |

### 消融实验

| 方法 | CHAIR_S↓ | CHAIR_I↓ | POPE Acc↑ | POPE F1↑ |
|------|----------|----------|-----------|----------|
| 完整DMAS | 30.8 | 11.4 | 81.70 | 82.47 |
| 仅真实性向量 | 34.2 | 11.7 | 81.67 | 82.42 |
| 仅视觉向量 | 42.4 | 13.2 | 81.40 | 82.01 |
| 无干预 | 51.0 | 15.2 | 75.08 | 76.06 |

### 关键发现
- 动态语义匹配选择引导向量显著优于固定引导向量，在QwenVL的Position子任务上，固定向量甚至不如原始模型
- 聚类数为4时两个模型均达最优；聚类过少导致语义粒度过粗
- 方法在ScienceQA和ViQuAE等完全不同类型的数据集上也有显著提升（LLaVAv1.5在ScienceQA从52.75%→62.27%），证明泛化性
- POPE实验中LLaVAv1.5在MSCOCO上Accuracy提升5.43%，F1提升7.14%；在GQA上Accuracy提升6.94%，F1提升6.5%
- $\alpha$ 和 $\beta$ 为负值时F1下降（即向幻觉方向干预），过大时模型基础能力被破坏
- 干预的注意力头数过少效果不显著，过多则也会导致性能下降

## 亮点与洞察
- 揭示了真实性和视觉感知在LVLM中激活不同注意力头子集的现象，为后续研究提供了重要依据
- 动态语义匹配的设计合理有效，避免了"一刀切"式干预的局限性
- 完全无需训练，可即插即用到不同LVLM架构
- 实验设计全面：判别任务（MME、POPE）+ 生成任务（CHAIR）+ 泛化验证（ScienceQA、ViQuAE），形成完整评估体系
- 可视化分析（注意力头激活图、t-SNE聚类图、超参数敏感性曲线）充分支撑了方法的动机和有效性

## 局限与展望
- 引导向量数据库的构建依赖AMBER和SEED数据集的特定选择，更大更多样的数据源可能进一步提升性能
- 聚类数目前固定为4，自适应确定最优聚类数是值得探索的方向
- 超参数 $\alpha, \beta, K$ 需要网格搜索，自动化调参值得研究
- 仅在7B规模模型上验证，更大规模模型（如13B、70B）的效果有待验证
- 构建引导向量数据库需要一定的预处理成本（对3000样本提取激活）
- 方法假设注意力头的分工模式在不同LVLM架构间具有一致性，该假设的普适性需更多验证

## 相关工作与启发
- **与ICT的关系**: ICT通过对图像中物体加噪来增强注意力，而DMAS从真实性和视觉感知两个维度同时干预，且支持动态语义匹配
- **与VTI的关系**: VTI使用固定引导向量，DMAS证明了动态选择的必要性
- **启发**: 激活工程这一范式值得在更多多模态任务中探索，如视觉推理、多模态对话等

## 评分
- 新颖性: ⭐⭐⭐⭐ 动态语义匹配引导向量的想法有新意，但激活工程本身已有先驱工作
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖MME/POPE/CHAIR三个基准，两个模型，消融完备，泛化性验证充分
- 写作质量: ⭐⭐⭐⭐ 写作清晰，可视化好，动机阐述自然
- 价值: ⭐⭐⭐⭐ 无训练方法的实用价值高，但聚类和超参搜索增加了部署复杂度

<!-- RELATED:START -->

## 相关论文

- [Look Carefully: Adaptive Visual Reinforcements in Multimodal Large Language Models for Hallucination Mitigation](look_carefully_adaptive_visual_reinforcements_in_multimodal_large_language_model.md)
- [ReefKnot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models](../../ACL2025/multimodal_vlm/reefknot_a_comprehensive_benchmark_for_relation_hallucination_evaluation_analysi.md)
- [Octopus: Alleviating Hallucination via Dynamic Contrastive Decoding](../../CVPR2025/multimodal_vlm/octopus_alleviating_hallucination_via_dynamic_contrastive_decoding.md)
- [Shuffle-R1: Efficient RL Framework for Multimodal Large Language Models via Data-centric Dynamic Shuffle](shuffle-r1_efficient_rl_framework_for_multimodal_large_language_models_via_data-.md)
- [Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models](vision-r1_incentivizing_reasoning_capability_in_multimodal_large_language_models.md)

<!-- RELATED:END -->
