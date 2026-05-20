---
title: >-
  [论文解读] CoVE: Compressed Vocabulary Expansion Makes Better LLM-based Recommender Systems
description: >-
  [ACL 2025][推荐系统][LLM推荐系统] 提出 CoVE 框架，通过扩展 LLM 词表为每个物品分配唯一 token ID 和嵌入，将序列推荐任务转化为 next-token prediction，相比现有方法推荐准确率提升最高 62%，推理速度提升约 100 倍…
tags:
  - "ACL 2025"
  - "推荐系统"
  - "LLM推荐系统"
  - "词表扩展"
  - "嵌入压缩"
  - "序列推荐"
  - "哈希压缩"
---

# CoVE: Compressed Vocabulary Expansion Makes Better LLM-based Recommender Systems

**会议**: ACL 2025  
**arXiv**: [2506.19993](https://arxiv.org/abs/2506.19993)  
**代码**: [GitHub](https://github.com/HaochenZhang717/CoVE-official-Repo)  
**领域**: 推荐系统  
**关键词**: LLM推荐系统, 词表扩展, 嵌入压缩, 序列推荐, 哈希压缩

## 一句话总结

提出 CoVE 框架，通过扩展 LLM 词表为每个物品分配唯一 token ID 和嵌入，将序列推荐任务转化为 next-token prediction，相比现有方法推荐准确率提升最高 62%，推理速度提升约 100 倍，并通过哈希嵌入压缩解决大规模场景的内存问题。

## 研究背景与动机

**领域现状**: 大语言模型（LLM）在推荐系统中的应用日益增多，主要有两种范式：(a) 用 LLM 提供嵌入初始化非 LLM 推荐模型；(b) 微调 LLM 直接生成推荐物品标题，再通过嵌入检索匹配真实物品（如 BIGRec）。

**现有痛点**: 
   - 方式 (a) 仅利用 LLM 的嵌入能力，未发挥其内容理解能力
   - 方式 (b) 即 finetune-and-retrieval 框架存在三大问题：LLM 需准确预测多 token 的物品标题（困难）、生成的标题可能不存在于物品空间（幻觉问题）、文本生成推理速度慢

**核心矛盾**: LLM 具有强大的 next-token prediction 能力，但现有推荐框架未能直接利用这一能力，反而要求 LLM 完成更困难的多 token 标题生成任务。

**本文目标**: 如何设计一个框架让 LLM 直接利用 next-token prediction 进行推荐，同时解决大规模物品空间下嵌入表的内存效率问题。

**切入角度**: 借鉴领域自适应中的词表扩展技术，为每个物品分配唯一 token，将推荐转化为单 token 预测问题。

**核心 idea**: 扩展 LLM 词表使每个物品对应一个唯一 token，通过 next-token prediction 的 logits 直接推荐，用哈希压缩解决嵌入表内存瓶颈。

## 方法详解

### 整体框架

CoVE 的核心流程：
1. **词表扩展**: 为物品空间 $\mathcal{I}$ 中的每个物品添加唯一 token（如 `<|205|>`）到 LLM 的 tokenizer 中
2. **嵌入表扩展**: 每个物品 token 对应一个独立的可训练嵌入向量
3. **微调**: 同时训练物品嵌入表、LoRA adapter 和 lm_head，将 LLM 与推荐任务对齐
4. **推理**: 输入用户历史交互序列，取 logits 中物品 ID 对应维度的分数进行排序推荐，无需生成文本

### 关键设计

#### 1. 微调任务设计
- **功能**: 将推荐任务建模为标准的 next-token prediction
- **核心思路**: 训练样本包含任务指令（Task Instruction）、用户历史（Task Input，包含物品 ID 和标题）、目标物品（Task Output）。训练时最小化 next-token prediction loss；推理时只需要 lm_head 输出的 logits 中最后 $|\mathcal{I}|$ 维对应的分数
- **设计动机**: 将多 token 标题生成简化为单 token ID 预测，消除幻觉，大幅加速推理

#### 2. 哈希嵌入压缩
- **功能**: 将物品嵌入表从 $|\mathcal{I}|$ 压缩到 $|\mathcal{S}|$（$|\mathcal{S}| \ll |\mathcal{I}|$）
- **核心思路**: 定义 $k$ 个通用哈希函数 $h_1, \ldots, h_k$，每个将物品映射到共享嵌入空间。物品 $i$ 的嵌入通过平均其哈希映射的共享嵌入得到：

$$\mathbf{e}_i = \frac{1}{k} \sum_{j=1}^{k} \mathbf{e}_{h_j(i)}$$

哈希函数采用简单算术运算：$h(i) = ((ai + b) \bmod p) \bmod |\mathcal{S}|$

- **设计动机**: 大规模场景下（如 Amazon 数据集有 4819 万物品），直接存储嵌入表需要约 96GB GPU 内存，哈希压缩使训练可行

### 损失函数/训练策略

- **损失函数**: 标准 next-token prediction loss（交叉熵）
- **训练配置**: 
    - Beauty/Toys/Sports 数据集：LLaMA-3.2-3B，学习率 $10^{-4}$，batch size 32，LoRA rank 8，alpha 16，最多 10 epochs
    - Video Games 数据集：LLaMA-2-7B + 4-bit QLoRA
- **可训练参数**: 物品嵌入表、LoRA adapter、lm_head

## 实验关键数据

### 主实验

在 Amazon Beauty/Toys/Sports 三个数据集上（压缩率=2），CoVE vs. 最佳 baseline（TIGER）：

| 数据集 | 指标 | TIGER | CoVE | 提升 |
|--------|------|-------|------|------|
| Beauty | NG@5 | 0.0321 | **0.0498** | +55% |
| Beauty | HR@10 | 0.0648 | **0.1009** | +56% |
| Toys | NG@5 | 0.0371 | **0.0509** | +37% |
| Toys | HR@5 | 0.0521 | **0.0719** | +38% |
| Sports | NG@5 | 0.0204 | **0.0296** | +45% |
| Sports | HR@10 | 0.0400 | **0.0624** | +56% |

Video Games 数据集上 CoVE vs. BIGRec (finetune-and-retrieval)：

| 指标 | BIGRec | CoVE | 提升 |
|------|--------|------|------|
| NG@5 | 0.0189 | **0.0221** | +17% |
| HR@10 | 0.0329 | **0.0437** | +33% |
| HR@20 | 0.0457 | **0.0621** | +36% |

推理速度：CoVE 为 6.5 samples/s，BIGRec 为 0.066 samples/s，**约 100 倍加速**。

### 消融实验

物品标题和嵌入表训练的重要性（Beauty 数据集）：

| 设置 | NG@5 | HR@5 |
|------|------|------|
| 仅可训练嵌入（无标题） | 0.045 | 0.0622 |
| 仅标题信息（冻结嵌入） | 0.0057 | 0.0094 |
| CoVE（两者结合） | **0.0498** | **0.0714** |

嵌入压缩鲁棒性：在 16 倍压缩率下，CoVE 在 HR@5 和 NG@5 上仍然超过 SOTA baseline（TIGER），仅在 Toys 的 HR@10 上例外。

### 关键发现

1. CoVE 在四个数据集上全面超越所有 baseline，NG 和 HR 指标提升 30%-62%
2. 微调后的 LLM 能正确学习物品 ID 与标题的映射关系，这是高质量推荐的关键
3. 冻结嵌入表会导致性能急剧下降，说明学习高质量物品嵌入至关重要
4. 嵌入压缩对不同数据集的鲁棒性不同，Sports 和 Toys 对8倍压缩仍保持稳定，而 Beauty 更敏感

## 亮点与洞察

- **巧妙的问题转化**: 将推荐从 "生成物品标题" 转化为 "预测物品 ID token"，一举解决幻觉、速度和准确率三个问题
- **理论与实践的平衡**: 哈希嵌入压缩使框架在大规模工业场景中可用（48M 物品场景下从 96GB 降低内存开销）
- **实验充分**: 4 个数据集、12+ baseline 比较、多种消融、推理速度分析、case study，论据完整
- **case study 有启发**: 微调后 LLM 能在生成时自动输出正确的 ID-标题对应关系，说明 CoVE 确实让 LLM 学到了物品语义

## 局限与展望

1. 嵌入压缩仅探索了哈希方法，更先进的压缩技术（量化、低秩近似）值得探索
2. 仅在 Amazon 电商数据集上实验，缺少其他领域（新闻、视频、音乐）的验证
3. 未讨论冷启动问题：新物品如何快速获得高质量嵌入
4. 压缩率对不同数据集敏感度不同，缺乏自适应压缩策略

## 相关工作与启发

- **BIGRec** (Bao et al., 2023): finetune-and-retrieval 框架的代表，CoVE 的主要对比对象
- **TIGER** (Rajput et al., 2023): 非 LLM 方法中的 SOTA，CoVE 大幅超越
- **ALPT** (Li et al., 2023b): 自适应低精度训练，未来可结合到 CoVE 的嵌入压缩中
- 域适应词表扩展（Cui et al., 2023; Liu et al., 2024a）: CoVE 的灵感来源，将词表扩展从语言适配推广到推荐场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将词表扩展应用到推荐系统是新颖的视角，问题转化巧妙
- **实验充分度**: ⭐⭐⭐⭐⭐ — 4 数据集、12+ baseline、多维度消融、推理速度分析，非常完整
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，动机阐述到位，图表设计合理
- **价值**: ⭐⭐⭐⭐ — 100 倍推理加速 + 准确率大幅提升，工业应用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Token-Efficient Item Representation via Images for LLM Recommender Systems](../../ICLR2026/recommender/token-efficient_item_representation_via_images_for_llm_recommender_systems.md)
- [\[ACL 2025\] Laser: Bi-Tuning with Collaborative Information for Controllable LLM-Based Sequential Recommendation](bi-tuning_with_collaborative_information_for_controllable_llm-based_sequential_r.md)
- [\[NeurIPS 2025\] Validating LLM-as-a-Judge Systems under Rating Indeterminacy](../../NeurIPS2025/recommender/validating_llm-as-a-judge_systems_under_rating_indeterminacy.md)
- [\[ICML 2026\] Can Recommender Systems Teach Themselves? A Recursive Self-Improving Framework with Fidelity Control](../../ICML2026/recommender/can_recommender_systems_teach_themselves_a_recursive_self-improving_framework_wi.md)
- [\[ICLR 2026\] Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems](../../ICLR2026/recommender/rejuvenating_cross-entropy_loss_in_knowledge_distillation_for_recommender_system.md)

</div>

<!-- RELATED:END -->
