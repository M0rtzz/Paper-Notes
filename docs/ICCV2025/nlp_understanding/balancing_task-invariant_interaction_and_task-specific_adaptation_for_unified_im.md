---
title: >-
  [论文解读] Balancing Task-Invariant Interaction and Task-Specific Adaptation for Unified Image Fusion
description: >-
  [ICCV 2025][NLP理解][统一图像融合] TITA 提出了一种无需任务标识的统一图像融合框架，通过交互增强像素注意力（IPA）模块探索任务不变的互补信息提取，并通过基于操作的自适应融合（OAF）模块动态适配任务特定需求，同时采用 FAMO 策略缓解多任务梯度冲突。
tags:
  - ICCV 2025
  - NLP理解
  - 统一图像融合
  - 多任务学习
  - 像素注意力
  - 自适应融合
  - 梯度冲突
---

# Balancing Task-Invariant Interaction and Task-Specific Adaptation for Unified Image Fusion

**会议**: ICCV 2025  
**arXiv**: [2504.05164](https://arxiv.org/abs/2504.05164)  
**代码**: [github.com/huxingyuabc/TITA](https://github.com/huxingyuabc/TITA)  
**领域**: 图像融合  
**关键词**: 统一图像融合, 多任务学习, 像素注意力, 自适应融合, 梯度冲突

## 一句话总结

TITA 提出了一种无需任务标识的统一图像融合框架，通过交互增强像素注意力（IPA）模块探索任务不变的互补信息提取，并通过基于操作的自适应融合（OAF）模块动态适配任务特定需求，同时采用 FAMO 策略缓解多任务梯度冲突。

## 研究背景与动机

图像融合旨在整合多源图像的互补信息以提升图像质量，覆盖红外-可见光融合（IVF）、多曝光融合（MEF）和多焦点融合（MFF）等多种场景。现有方法面临三个核心挑战：

**统一方法忽略任务特异性**：现有统一融合方法（如 U2Fusion、PMGI）使用共享结构和统一目标函数将不同融合任务视为同一问题，虽然实现了任务不变知识共享，但忽略了各任务的独特物理特性（如 IVF 强调热红外显著性，MEF 平衡亮度，MFF 提取清晰区域），限制了整体性能

**通用方法依赖任务标识**：通用融合方法（如 SwinFusion、TC-MoA）引入了任务特异性适配，但推理时需要显式任务标识来选择对应的模型分支或损失函数，这限制了对未见过任务的泛化能力

**多任务梯度冲突**：不同融合任务的优化方向可能相互矛盾，简单平均梯度会导致某些任务性能受损

本文试图同时解决这三个挑战：**如何在不依赖任务标识的前提下，既利用融合任务的共性（互补信息提取），又适配各任务的特性？**

## 方法详解

### 整体框架

TITA 基于 SwinFusion 架构，包含两个阶段：
1. **任务不变交互阶段**：L 个堆叠的交互增强 SwinFusion（ISF）模块，每个包含一个 IPA 模块
2. **任务特定适配阶段**：基于操作的自适应融合（OAF）模块

### 关键设计

1. **交互增强像素注意力（IPA）模块**：

    - 基于 Pixel Attention（PA）机制的改进，PA 通过关系判别器 $\phi_{\theta_s}(\cdot)$ 动态调节自注意力和交注意力的权重
    - IPA 的两项关键修改：
        - **移除对 V 的直接噪声注入**：PA 中对 Value 的噪声注入可能导致不可逆的信息损失
        - **强化交注意力偏好**：当关系判别器分数高（两源越不相关）时，显式增大交注意力权重。Key 的构造变为：$K_1 = [(N_1 + X_{i,1} - X_{i,1}\phi_{\theta_s})W_K, (N_2 + X_{i,1} + X_{i,1}\phi_{\theta_s})W_K]$
    - 设计动机：更高的不相关性分数意味着更多的互补信息需要通过交注意力来提取
    - 同时将 SwinFusion 中所有域内融合替换为域间融合（ISF），进一步增加交注意力操作次数

2. **基于操作的自适应融合（OAF）模块**：

    - 三个并行操作分支：
        - **HPF 分支**：空间可变高通滤波，捕获高频纹理和边缘信息
        - **ADD 分支**：残差加法，用于整体信息增强
        - **MUL 分支**：逐元素乘法，促进非线性特征交互
    - 每个分支的操作数（如卷积核）由超网络（2 层 MLP）从输入特征预测
    - 三个分支的动态权重 $W$ 由另一个权重预测网络从双源特征 $(X_1, X_2)$ 预测
    - 输出：$X_f = \sum_{o \in \{h,a,m\}} W_1 \cdot \hat{X}_{o,1} + W_2 \cdot \hat{X}_{o,2}$
    - 设计动机：不同融合任务对高频保留、整体增强和非线性组合的需求不同。动态权重可自动适配任务特性，无需显式任务标识

3. **FAMO 多目标优化策略**：

    - 实验发现直接平均多任务梯度时存在严重梯度冲突（角度和幅度差异大）
    - 采用 FAMO：通过可学习的 logits $\xi_t$ 生成 softmax 权重，动态调节各任务损失的权重
    - FAMO 使各任务的损失降低速率尽量相等，实现公平优化
    - 设计动机：TA 模块的引入会加剧梯度冲突，FAMO 有效缓解了这一问题（消融实验验证 TA 从 MO 中受益最大）

### 损失函数 / 训练策略

使用任务特异性目标（训练时需要任务标识，推理时不需要）：

$$\ell = \lambda_1 \ell_{ssim} + \lambda_2 \ell_{text} + \lambda_3 \ell_{int}$$

其中 $\ell_{text}$（纹理损失）使用最大梯度约束，$\ell_{int}$（强度损失）使用任务特定的聚合方式（IVF/MFF 用 max，MEF 用 mean）。训练配置：Adam 优化器（lr=2e-5），batch size 8，20000 次迭代，均匀采样各任务数据。

## 实验关键数据

### 主实验 — 三项融合任务定量对比

**IVF 任务（LLVIP 数据集）**：

| 方法 | 类型 | MI ↑ | FMI ↑ | Qabf ↑ | VIF ↑ |
|------|------|------|-------|--------|-------|
| SwinFusion | 通用 | 3.873 | 0.889 | 0.650 | 0.907 |
| TC-MoA | 通用 | 3.606 | 0.886 | 0.600 | 0.925 |
| Text-IF | 专用 | 3.322 | 0.892 | **0.684** | **0.932** |
| CCF | 统一 | 2.789 | 0.881 | 0.499 | 0.719 |
| **TITA** | **统一** | **4.176** | **0.896** | 0.679 | 0.926 |

**MFF 任务（Lytro+MFFW+MFI-WHU）**：

| 方法 | 类型 | MI ↑ | FMI ↑ | Qabf ↑ | SSIM ↑ |
|------|------|------|-------|--------|--------|
| IFCNN | 专用(MFF) | 6.495 | 0.882 | 0.658 | 0.991 |
| SwinFusion | 通用 | 6.261 | 0.881 | 0.687 | 0.991 |
| **TITA** | **统一** | 6.546 | **0.885** | **0.697** | **0.993** |

TITA 在不使用任务标识的前提下，多项指标超越专用方法和通用方法。

### 消融实验 — 三大组件贡献（IVF）

| TI | TA | MO | MI ↑ | FMI ↑ | Qabf ↑ | VIF ↑ |
|----|----|----|------|-------|--------|-------|
| ✗ | ✗ | ✗ | 3.612 | 0.889 | 0.646 | 0.845 |
| ✓ | ✗ | ✗ | 3.882 | 0.892 | 0.664 | 0.904 |
| ✗ | ✓ | ✗ | 3.685 | 0.891 | 0.662 | 0.853 |
| ✗ | ✗ | ✓ | 3.680 | 0.891 | 0.662 | 0.853 |
| ✓ | ✓ | ✗ | 3.883 | 0.893 | 0.666 | 0.906 |
| ✓ | ✗ | ✓ | 4.122 | 0.895 | 0.676 | 0.919 |
| ✓ | ✓ | ✓ | **4.176** | **0.896** | **0.680** | **0.926** |

三个组件相互增强，MO 对 TA 的收益最大（验证 TA 引入了梯度冲突需 MO 缓解）。

### 关键发现

1. **未见任务泛化**：TITA 在医学图像融合（MIF）和全色锐化（PAN）等未见任务上表现良好，而 FusionDN 和 CCF 在未见任务上完全崩溃
2. **交注意力越多越好**：IeSF（全域间融合）> SF（原始）> IrSF（全域内融合），验证了多源交互的重要性
3. **OAF 中 MUL 分支最关键**：移除 MUL 分支对性能影响最大，因为图像融合本质上涉及大量非线性操作
4. **动态权重可视化**：OAF 为不同融合任务自动分配了不同的操作权重分布（如 MFF 任务中 HPF 权重更小但不可或缺）

## 亮点与洞察

- 准确识别了统一融合框架的三重挑战（任务不变性、任务特异性、梯度冲突），并提供了系统性解决方案
- IPA 中"不相关性越高 → 交注意力权重越大"的因果关系设计符合直觉——互补性越强的区域越需要跨模态交互
- 仅 1.39M 参数，轻量高效
- 无需推理时任务标识的设计使得框架可直接应用于任何新融合任务

## 局限与展望

- OAF 中仅有三种操作分支（HPF、ADD、MUL），可能不足以覆盖所有融合任务的需求
- 训练数据在不同任务间不平衡（IVF 12025 对 vs MFF 800 对），虽然做了均匀采样但效果受限
- 与利用扩散模型或大语言模型的最新方法（如 DDFM、Text-IF）相比，在感知质量上仍有差距

## 相关工作与启发

- 与 SwinFusion 的关系：TITA 继承其架构但将通用方法扩展为无需任务标识的统一方法
- 与 TC-MoA 的关系：TC-MoA 使用混合专家适配不同任务但需任务标识，TITA 的 OAF 通过动态权重隐式实现类似功能
- FAMO 作为多任务优化策略具有通用性，可推广到其他多任务视觉系统

## 评分

- 新颖性: ⭐⭐⭐⭐ （三重挑战的系统性解决，各组件设计合理）
- 实验充分度: ⭐⭐⭐⭐⭐ （三种任务 + 两种未见任务 + 详尽消融）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，动机阐述充分）
- 价值: ⭐⭐⭐⭐ （为统一图像融合推进了一步，泛化性验证令人信服）

<!-- RELATED:START -->

## 相关论文

- [QQSUM: A Novel Task and Model of Quantitative Query-Focused Summarization for Review-based Product Question Answering](../../ACL2025/nlp_understanding/qqsum_a_novel_task_and_model_of_quantitative_query-focused_summarization_for_rev.md)
- [CaLMQA: Exploring Culturally Specific Long-Form Question Answering across 23 Languages](../../ACL2025/nlp_understanding/calmqa_cultural_multilingual_qa.md)
- [End-to-End Dialog Neural Coreference Resolution: Balancing Efficiency and Accuracy in Large-Scale Systems](../../ACL2025/nlp_understanding/end-to-end_dialog_neural_coreference_resolution_balancing_efficiency_and_accurac.md)
- [Multi-Hop Reasoning for Question Answering with Hyperbolic Representations](../../ACL2025/nlp_understanding/multi-hop_reasoning_for_question_answering_with_hyperbolic_representations.md)
- [Rethinking Semantic Parsing for Large Language Models: Enhancing LLM Performance with Semantic Hints](../../ACL2025/nlp_understanding/rethinking_semantic_parsing_for_large_language_models_enhancing_llm_performance_.md)

<!-- RELATED:END -->
