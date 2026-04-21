---
title: >-
  [论文解读] Towards LLM-Empowered Knowledge Tracing via LLM-Student Hierarchical Behavior Alignment in Hyperbolic Space
description: >-
  [AAAI 2026][自监督学习][知识追踪] 提出 L-HAKT 框架，首次将 LLM 双 Agent 与双曲几何结合用于知识追踪：教师 Agent 解析题目语义并构建层级知识图谱，学生 Agent 模拟个体学习行为生成合成数据，通过双曲空间对比对齐校准合成数据与真实数据的分布差异，在四个教育数据集上 AUC 最高达 80.29%，相比 GKT 基线在 EdNet 上 AUC 提升 13.03%。
tags:
  - AAAI 2026
  - 自监督学习
  - 知识追踪
  - LLM双Agent
  - 双曲空间
  - 层级知识图谱
  - 对比对齐
  - 学习行为模拟
---

# Towards LLM-Empowered Knowledge Tracing via LLM-Student Hierarchical Behavior Alignment in Hyperbolic Space

**会议**: AAAI 2026  
**arXiv**: [2602.22879](https://arxiv.org/abs/2602.22879)  
**领域**: 教育智能 / 知识追踪  
**关键词**: 知识追踪, LLM双Agent, 双曲空间, 层级知识图谱, 对比对齐, 学习行为模拟

## 一句话总结

提出 L-HAKT 框架，首次将 LLM 双 Agent 与双曲几何结合用于知识追踪：教师 Agent 解析题目语义并构建层级知识图谱，学生 Agent 模拟个体学习行为生成合成数据，通过双曲空间对比对齐校准合成数据与真实数据的分布差异，在四个教育数据集上 AUC 最高达 80.29%，相比 GKT 基线在 EdNet 上 AUC 提升 13.03%。

## 研究背景与动机

**领域现状**：知识追踪（KT）是教育智能的核心技术，通过分析学生历史交互数据动态推断知识掌握状态。现有方法分为序列建模（DKT、SAKT、SAINT 等基于 RNN/Transformer）和图建模（GKT、SKT 等基于 GNN）两大范式。

**现有痛点**：
(1) **缺乏层级概念表示**——传统方法在欧氏空间建模，平坦几何无法表达知识体系的树状层级结构（如基础定义→推导→综合应用）；
(2) **题目语义利用不足**——现有方法依赖简单的 ID 或浅层文本特征，未充分挖掘题目文本中隐含的知识点拓扑关系；
(3) **个体认知偏差被忽视**——模型训练数据的群体分布会扭曲个体的难度感知（如低水平区域训练的模型可能错误地将中等题标为高难度）。

**核心矛盾**：知识点间的层级依赖关系具有天然的树状结构，但欧氏空间的指数膨胀使其无法高效表示此类结构（如图 1 中，学生-题目-知识点关系图的双曲度量接近 0，表明强树状特性）。

**本文切入角度**：用 LLM 双 Agent 从题目语义中挖掘层级知识结构并模拟学习行为，用双曲空间显式建模层级依赖，通过对比学习对齐合成与真实数据来校准认知偏差。

## 方法详解

### 整体框架

L-HAKT 分三个部分：
(1) **LLM 双 Agent 行为增强**：教师 Agent 构建层级知识图谱 + 量化题目难度；学生 Agent 模拟个体学习行为生成合成交互数据
(2) **双曲编码与对齐**：关系感知双曲图神经网络编码层级结构；双曲对比学习对齐合成与真实数据
(3) **双曲知识状态追踪**：基于 GRU 的序列传播，在切空间与双曲空间间交替计算

### 关键设计

1. **教师-学生 LLM 双 Agent**

    - **教师 Agent**：处理题目图像（通过 VLM 如 Qwen-2.5VL），输出三类信息——(a) 层级知识点识别：将知识点分为 4 个层级（$L \in \{1,2,3,4\}$，从基础定义到综合推理）；(b) 结构化知识图谱构建：建立知识点间的父子依赖关系，形成树状教学结构；(c) 题目难度量化：根据关联知识点的层级组合计算客观难度分
    - **学生 Agent**：基于历史交互序列模拟学习行为，包含两个专用模块：(a) 认知投入模块——基于题目难度、知识点掌握度和时间间隔动态评估学习专注度 $\Gamma_j = \sigma(W_q \cdot [X_{q_i}; X_{c_{ij}}; t] + b_q)$；(b) 层级遗忘模块——按知识点难度分层建模差异化遗忘 $F_j = \exp(-\lambda \cdot L_{avg} \cdot t_j)$，高层级知识衰减更快
    - **知识状态更新**：$\mathbf{h}_j^s = \text{LSTM}([X_{q_j}; \sum_{c} w_c X_c] \oplus (\Gamma_j \odot F_j \odot h_{j-1}^s))$

2. **关系感知双曲图神经网络**

    - **功能**：在双曲空间中编码知识点层级依赖，基础知识点位于低曲率中心区域，高阶知识点分布于高曲率外围
    - **核心思路**：构造异构图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$，包含题目-知识点边和知识点间的层级依赖边 $\mathcal{E}_{hie} = \{(c_i, c_j) | L_{c_i} < L_{c_j}\}$。为真实数据和合成数据分别定义曲率 $\kappa_{real}$ 和 $\kappa_{syn}$，通过指数映射将欧氏嵌入投影到双曲流形，在切空间进行注意力聚合后映射回
    - **更新公式**：$\mathbf{h}_i^{(L+1)} = \exp_0^\kappa(\sigma(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^{(L)} \cdot \mathbf{W}^{(L)} \log_0^\kappa(\mathbf{h}_j^{\mathbb{H}_\kappa^{(L)}})))$
    - **设计动机**：双曲空间的体积呈指数增长，天然适合表示树状层级结构——低失真、高效率

3. **双曲对比对齐机制**

    - **功能**：缩小合成数据与真实学生行为数据在题目难度和遗忘模式等关键特征上的分布差异
    - **核心思路**：在双曲空间中，将真实和合成嵌入空间共享的题目-知识点对作为正样本，其他实体作为负样本，通过对比损失 $\mathcal{L}_{con}$ 拉近正样本、推远负样本
    - **设计动机**：LLM 生成的合成数据虽然补充了思维路径信息，但与真实数据存在分布偏移，对比学习可有效校准此偏差

### 损失函数 / 训练策略

总损失 $\mathcal{L}_{total} = \mathcal{L}_{KT} + \alpha \mathcal{L}_{con}$，其中 $\mathcal{L}_{KT}$ 为二元交叉熵预测损失，$\alpha$ 控制对比损失权重。数据集按 80%/20% 划分训练/测试集。

## 实验关键数据

### 主实验——AUC/ACC 对比（16 种基线）

| 模型 | ASSIST09 AUC | ASSIST12 AUC | EdNet AUC | Eedi AUC |
|------|-------------|-------------|----------|---------|
| DKT | 75.97 | 72.90 | 70.10 | 76.01 |
| AKT | 78.23 | 78.21 | 76.78 | 78.84 |
| GIKT | 77.33 | 76.32 | 76.02 | 79.68 |
| MIKT | 79.38 | 78.65 | 77.10 | 79.59 |
| **L-HAKT** | **80.22** | **80.27** | **78.23** | **80.29** |

### 消融实验——双曲空间和对比学习的贡献

| 配置 | ASSIST09 AUC | Δ | EdNet AUC | Δ |
|------|-------------|--|----------|--|
| GKT (基线) | 76.32 | - | 69.21 | - |
| L-HVKT (去双曲) | 77.55 | +1.61% | 76.54 | +10.59% |
| L-HVKT (去对比) | 76.98 | +0.86% | 75.51 | +9.10% |
| **L-HAKT (完整)** | **80.22** | **+5.11%** | **78.23** | **+13.03%** |

### 关键发现

- L-HAKT 在所有 4 个数据集上均达到 SOTA，AUC 提升最高达 13.03%（EdNet 上对比 GKT）
- 双曲空间和对比学习均有独立贡献：去除双曲约束后 ASSIST09 AUC 下降 2.67%，去除对比学习下降 3.24%
- 在 Eedi（包含图像数据）上也表现优异，验证了 VLM 解析题目图像策略的有效性
- 相比 AKT（纯 Transformer 注意力方法），L-HAKT 在所有数据集上均有约 2% 的 AUC 提升

## 亮点与洞察

1. **双 Agent 协作设计合理**：教师 Agent 提供客观知识结构，学生 Agent 模拟主观学习行为，两者互补
2. **层级遗忘建模贴近认知规律**：高层级知识遗忘更快、基础知识保持更久，符合教育心理学
3. **双曲空间选择有理论依据**：通过计算真实数据的双曲度量（H-all 接近 0）验证了树状结构假设
4. **合成数据通过对比学习有效落地**：避免了直接使用 LLM 合成数据时的分布偏移问题

## 局限与展望

1. LLM Agent 的推理开销大，大规模部署时成本高昂
2. 知识点层级仅分 4 个等级，粒度可能不够细
3. 双曲空间操作（指数/对数映射）的数值稳定性在高曲率下可能出问题
4. 仅验证了知识追踪任务，未扩展到推荐题目、学习路径规划等下游应用
5. 学生 Agent 的认知投入和遗忘模型较为简化，可引入更精细的认知科学模型

## 相关工作与启发

- LLM 双 Agent 的教师-学生协作范式可推广到其他教育智能场景（如自动出题、个性化辅导）
- 双曲空间在层级结构建模上的成功可激励其在知识图谱补全、推荐系统等领域的进一步探索
- 合成数据 + 对比对齐的策略为解决教育数据稀缺问题提供了新范式

## 评分

⭐⭐⭐⭐

- **新颖性** ⭐⭐⭐⭐：首次将 LLM 双 Agent 与双曲几何结合用于知识追踪
- **实验充分度** ⭐⭐⭐⭐：4 个数据集、16 种基线、详细消融
- **写作质量** ⭐⭐⭐：部分符号定义不够规范，公式排版有改进空间
- **价值** ⭐⭐⭐⭐：为教育智能中的知识追踪提供了有理论动机的新框架

<!-- RELATED:START -->

## 相关论文

- [Contrastive Learning on LLM Back Generation Treebank for Cross-domain Constituency Parsing](../../ACL2025/self_supervised/llm_back_gen_treebank.md)
- [Hyperbolic Category Discovery](../../CVPR2025/self_supervised/hyperbolic_category_discovery.md)
- [Let the Void Be Void: Robust Open-Set Semi-Supervised Learning via Selective Non-Alignment](let_the_void_be_void_robust_open-set_semi-supervised_learning_via_selective_non-.md)
- [NeuroBridge: Bio-Inspired Self-Supervised EEG-to-Image Decoding via Cognitive Priors and Bidirectional Semantic Alignment](neurobridge_bio-inspired_self-supervised_eeg-to-image_decoding_via_cognitive_pri.md)
- [PonderLM: Pretraining Language Models to Ponder in Continuous Space](../../ICLR2026/self_supervised/ponderlm_pretraining_language_models_to_ponder_in_continuous_space.md)

<!-- RELATED:END -->
