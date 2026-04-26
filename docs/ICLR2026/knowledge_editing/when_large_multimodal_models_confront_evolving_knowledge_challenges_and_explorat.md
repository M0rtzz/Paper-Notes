---
title: >-
  [论文解读] When Large Multimodal Models Confront Evolving Knowledge: Challenges and Explorations
description: >-
  [ICLR 2026][大型多模态模型] 提出 EVOKE 基准测试，系统评估大型多模态模型 (LMM) 对演化知识的注入能力，揭示两大挑战（现有方法表现差、微调导致灾难性遗忘），并提出知识增强和持续学习两条应对路径。
tags:
  - ICLR 2026
  - 大型多模态模型
  - 知识注入
  - 演化知识
  - 灾难性遗忘
  - 持续学习
---

# When Large Multimodal Models Confront Evolving Knowledge: Challenges and Explorations

**会议**: ICLR 2026  
**arXiv**: [2505.24449](https://arxiv.org/abs/2505.24449)  
**代码**: 无  
**领域**: Multimodal / VLM  
**关键词**: 大型多模态模型, 知识注入, 演化知识, 灾难性遗忘, 持续学习

## 一句话总结

提出 EVOKE 基准测试，系统评估大型多模态模型 (LMM) 对演化知识的注入能力，揭示两大挑战（现有方法表现差、微调导致灾难性遗忘），并提出知识增强和持续学习两条应对路径。

## 研究背景与动机

大型语言/多模态模型 (LLM/LMM) 通过大规模预训练积累了丰富的世界知识，但面临一个根本性问题：**知识过时**。全球信息快速更新，新实体不断涌现，而训练后的模型是静态的。例如，LMM 可能无法识别小米 SU7 汽车，反而错误地回答是保时捷。

现有知识注入研究存在三个关键缺口：

**缺乏多模态演化知识数据集**：现有知识注入数据集（如 CC-RECENTNEWS）仅包含文本，缺少真实场景中的多模态数据

**缺少对 LMM 的系统研究**：大多数知识注入研究聚焦 LLM，对视觉-语言模型的系统探索明显不足

**对注入副作用认识不足**：知识注入（尤其是微调）对模型原有能力的影响缺乏全面评估

本文旨在构建首个多模态演化知识注入基准，系统揭示挑战并探索可行路径。

## 方法详解

### 整体框架

本文的贡献是一个完整的"基准 + 评估 + 路径"框架：

1. **EVOKE 基准构建**：自动化管线收集演化知识并构建评估数据
2. **系统评估**：对 SFT、RAG、IAG 三类知识注入方法进行全面实验
3. **挑战揭示**：识别两个核心挑战
4. **路径探索**：为每个挑战提出对应的改善方向

### 关键设计

1. **EVOKE 基准构建**：自动化多模态演化知识收集管线

    - **数据来源**：CNN 新闻网站（29 种新闻类型）和 Wikipedia 离线版本（130 种实体类型），共 159 种细粒度类型
    - **数据规模**：9,422 条知识-图片对
    - **时间线**：2024 年的数据，确保对 2023 年发布的 LMM 是全新的
    - **数据格式**：每条知识包含：
     - 注入数据 $\mathcal{D}_\mathcal{K} = \{(i_k, x_k, y_k)\}$：知识图片、启发式查询、知识摘要
     - 评估数据 $\mathcal{D}_\mathcal{Q} = \{(i_q, x_q, y_q)\}$：查询图片、问题、ground truth
    - **质量保证**：流行度过滤 → GPT-4o 摘要 → GPT-4o 生成 QA → Google 图片检索 → CLIP 聚类去噪 → 人工审核

2. **知识注入方法评估**：三大类方法全面覆盖

    - **监督微调 (SFT)**：Full Fine-Tuning 和 LoRA 两种策略
    - **多模态检索增强生成 (MM-RAG)**：四种检索策略——Text-Only、Image-Only、UniIR（多模态融合检索）、Golden Context（理想上限）
    - **互联网增强生成 (IAG)**：Gemini 和 Perplexity AI

3. **问题定义**：知识注入的双重目标

    - **知识适应** (Knowledge Adaptation)：在演化知识评估数据上最大化准确率
    - **知识保留** (Knowledge Retention)：最小化注入后模型在原有任务上的性能退化
    - 形式化为约束优化：$\max_f \mathbb{E}[\mathbb{I}(\mathcal{M}^*(i_q, x_q) = y_q)] \text{ s.t. } \min_f \mathbb{E}[\mathbb{I}(\mathcal{M}(i_p, x_p) = y_p) - \mathbb{I}(\mathcal{M}^*(i_p, x_p) = y_p)]$

4. **路径1：知识增强 (Knowledge Augmentation)**：训练阶段的数据增强

    - **文本增强**：使用 GPT-4 对知识摘要进行改写 (paraphrase)，生成语义等价但表达不同的版本
    - **图像增强**：传统增强（翻转、随机阴影、颜色变换）
    - 核心发现：文本增强在训练阶段能显著提升性能（改写数量与准确率正相关），但图像增强反而导致性能下降
    - 解释：文本增强帮助模型学习"正确逻辑"而非"错误逻辑"——例如学到"小米 SU7 是小米汽车公司的电动轿车"而非死记硬背完整描述

5. **路径2：持续学习 (Continual Learning)**：缓解灾难性遗忘

    - **训练数据可用时**：Replay 方法——随机抽取 10% 原始训练数据与新知识一起训练
    - **训练数据不可用时**：
     - EWC (Elastic Weight Consolidation)：参数正则化方法
     - LwF (Learning without Forgetting)：知识蒸馏方法
     - MoELoRA：多专家 LoRA，利用多个专家获取多样化知识
    - 综合排名：Replay+LoRA (Rank 1) > MoELoRA (Rank 2) > Replay+Full-FT (Rank 3)

### 损失函数 / 训练策略

- SFT：标准指令微调损失（交叉熵）
- 持续学习方法各自的损失：
    - Replay：在原始数据子集上的标准损失
    - EWC：$\mathcal{L}_{EWC} = \mathcal{L}_{task} + \lambda \sum_i F_i (\theta_i - \theta_i^*)^2$
    - LwF：$\mathcal{L}_{LwF} = \mathcal{L}_{task} + \lambda \mathcal{L}_{KD}$
    - MoELoRA：多专家路由 + 对比学习

## 实验关键数据

### 主实验

| 方法 | Overall Acc | Overall F1 | News Acc | Entity Acc |
|------|------------|------------|----------|------------|
| LLaVA Vanilla | 4.89 | 9.34 | 7.37 | 2.18 |
| LLaVA Full-FT | 18.02 | 15.17 | 21.35 | 14.37 |
| LLaVA LoRA | 15.23 | 18.31 | 17.72 | 12.51 |
| LLaVA MM-RAG (UniIR) | 40.68 | 57.51 | 40.12 | 41.30 |
| LLaVA MM-RAG (Golden) | **56.13** | **75.77** | 56.78 | 55.43 |
| Perplexity AI† | 48.27 | 62.44 | 47.58 | 48.96 |

所有方法最高仅 56.13% 准确率，远未达到理想水平。

### 灾难性遗忘评估 (12 个 benchmark，7 个能力维度)

| 方法 | MME | MMBench | MIA-Bench | MMDU | Ranking |
|------|-----|---------|-----------|------|---------|
| Vanilla | 1865.56 | 64.60 | 66.33 | 26.37 | - |
| Full-FT | 956.8 (-49%) | 52.92 (-18%) | 25.25 (-62%) | 13.03 (-51%) | 7 |
| LoRA | 1233.54 (-34%) | 53.87 (-17%) | 29.66 (-55%) | 13.70 (-48%) | 6 |
| Replay+LoRA | **1650.75** (-12%) | **60.48** (-6%) | **62.33** (-6%) | **19.31** (-27%) | **1** |
| MoELoRA | 1732.47 (-7%) | 63.32 (-2%) | 64.97 (-2%) | 18.66 (-29%) | 2 |

### 关键发现

1. **挑战1：知识注入表现极差**
    - 最好方法 (Golden Context) 准确率仅 56.13%
    - SFT 方法更差 (15-18%)
    - MM-RAG 整体优于 SFT，但需要检索基础设施
    - IAG (Perplexity AI) 不依赖外部注入数据即达 48.27%

2. **挑战2：灾难性遗忘严重**
    - Full-FT 和 LoRA 在所有 12 个 benchmark 上都出现退化
    - **指令遵循能力损失最严重**：MIA-Bench 下降 62%/55%，因为 EVOKE 数据中不包含指令遵循场景
    - 指令遵循能力是其他能力的前提——其严重受损导致 MME (依赖 Yes/No 指令) 表现远差于 MMBench (选择题)
    - 多轮对话能力 (MMDU) 同样大幅下降

3. **文本增强有效、图片增强无效**
    - 训练阶段文本改写 (1→4 个改写版本) 持续提升性能
    - 传统图像增强导致性能下降，说明需要开发专门的图像知识增强方法

4. **持续学习方法有效但有代价**
    - Replay 和 MoELoRA 最有效缓解遗忘
    - 但所有持续学习方法都在知识注入效果上有所损失
    - MoELoRA 知识注入性能下降最大 (Acc: 15.23→6.82)

5. **顺序微调越做越差**
    - 将数据分 4/8/12 批顺序微调，性能随批次递减
    - 说明顺序微调不适合持续注入演化知识

## 亮点与洞察

1. **首个多模态演化知识基准**：EVOKE 填补了多模态知识注入评估的空白，数据收集管线可持续产出新的演化知识
2. **全面的系统性评估**：横跨 SFT/RAG/IAG 三类方法、2 个 LMM、12 个遗忘评估 benchmark，实验规模在同类工作中领先
3. **指令遵循 → 灾难性遗忘的因果链**：揭示知识注入导致指令遵循能力崩溃，进而引发其他能力大幅下降，这个因果机制的发现对后续研究有重要指导意义
4. **文本增强的"学对逻辑"解释**：模型需要学习灵活提取知识而非死记硬背，文本改写帮助模型存储实体属性的正确关联，这个洞察可指导数据准备策略
5. **知识类型适应差异**：新闻知识比实体知识更容易适应，因为新闻由旧实体的新事件组成，而新实体对 LMM 完全陌生

## 局限与展望

1. **数据规模受限**：9,422 条看似不少但相对 LMM 的参数规模仍很小，更大规模数据可能改变结论
2. **模型选择偏旧**：实验仅在 LLaVA-v1.5 和 Qwen-VL-Chat (均为 2023 年模型) 上进行，更新的模型（如 GPT-4V、InternVL-2）可能有不同表现
3. **仅评估知识 VQA**：EVOKE 的评估限于视觉问答格式，未涵盖更复杂的知识应用场景（如推理链、多跳推理）
4. **图像增强的探索不足**：虽然发现传统增强无效，但未探索更高级的图像增强策略（如基于扩散模型的风格变换）
5. **缺少与编辑方法的比较**：知识编辑 (knowledge editing) 方法（如 ROME、MEMIT 的多模态版本）未纳入比较

## 相关工作与启发

- **知识注入三范式**：SFT (参数内化)、RAG (外部检索)、IAG (互联网搜索) 各有优劣，未来可能需要混合方法
- **持续学习在 LMM 中的应用**：Replay 和 MoELoRA 的有效性表明，将持续学习与大模型微调结合是有前途的方向
- **知识增强启示**：Allen-Zhu & Li (2024) 的理论——仅记忆训练数据不保证知识提取——在多模态场景下得到验证
- **启发**：IAG (如 Perplexity AI) 的强表现暗示，对于演化知识，与其注入模型不如增强模型的联网搜索能力

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个多模态演化知识基准，问题定义清晰
- 实验充分度: ⭐⭐⭐⭐⭐ — 实验规模大、覆盖全面、分析深入
- 写作质量: ⭐⭐⭐⭐ — 结构良好，挑战-路径框架清晰
- 价值: ⭐⭐⭐⭐ — 基准和发现对社区有重要参考价值，但模型偏旧

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Hybrid-DMKG: A Hybrid Reasoning Framework over Dynamic Multimodal Knowledge Graphs for Multimodal Multihop QA with Knowledge Editing](../../AAAI2026/knowledge_editing/hybrid-dmkg_a_hybrid_reasoning_framework_over_dynamic_multimodal_knowledge_graph.md)
- [\[ACL 2026\] EvoEdit: Evolving Null-space Alignment for Robust and Efficient Knowledge Editing](../../ACL2026/knowledge_editing/evoedit_evolving_null-space_alignment_for_robust_and_efficient_knowledge_editing.md)
- [\[NeurIPS 2025\] UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models](../../NeurIPS2025/knowledge_editing/uniedit_a_unified_knowledge_editing_benchmark_for_large_language_models.md)
- [\[ACL 2025\] Structure-aware Domain Knowledge Injection for Large Language Models](../../ACL2025/knowledge_editing/structure-aware_domain_knowledge_injection_for_large_language_models.md)
- [\[ACL 2025\] Neuron-Level Sequential Editing for Large Language Models](../../ACL2025/knowledge_editing/neuron-level_sequential_editing_for_large_language_models.md)

<!-- RELATED:END -->
