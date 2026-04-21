---
title: >-
  [论文解读] Context Tokens are Anchors: Understanding the Repetition Curse in dMLLMs from an Information Flow Perspective
description: >-
  [ICLR 2026][目标检测][扩散语言模型] 通过信息流分析揭示扩散多模态大语言模型（dMLLMs）在使用缓存加速时产生"重复诅咒"的内在机制——context token 作为锚点聚合语义信息，缓存破坏了这一信息流模式——并提出 CoTA 方法将重复率降低高达 92%。
tags:
  - ICLR 2026
  - 目标检测
  - 扩散语言模型
  - 重复生成
  - 信息流分析
  - 缓存加速
  - 注意力机制
---

# Context Tokens are Anchors: Understanding the Repetition Curse in dMLLMs from an Information Flow Perspective

**会议**: ICLR 2026  
**arXiv**: [2601.20520](https://arxiv.org/abs/2601.20520)  
**代码**: [GitHub](https://github.com/ErikZ719/CoTA)  
**领域**: 目标检测（多模态扩散语言模型）  
**关键词**: 扩散语言模型, 重复生成, 信息流分析, 缓存加速, 注意力机制

## 一句话总结

通过信息流分析揭示扩散多模态大语言模型（dMLLMs）在使用缓存加速时产生"重复诅咒"的内在机制——context token 作为锚点聚合语义信息，缓存破坏了这一信息流模式——并提出 CoTA 方法将重复率降低高达 92%。

## 研究背景与动机

**dMLLMs 的推理效率问题**：扩散式大语言模型（如 LLaDA）通过迭代去噪并行生成 token，每步需要对全序列计算注意力。由于采用双向注意力，传统 KV-cache 不适用，推理延迟高

**缓存加速的副作用**：dLLM-Cache 等方法利用前缀和后缀 token 的注意力值在迭代间变化小的特点设计缓存复用，有效降低延迟但引发严重的文本重复——作者称之为"Repeat Curse"

**定量度量**：提出 4 个互补指标定义重复程度——相邻重复率（ARR）、样本重复率（SRR）、最大重复长度（MRL）、平均重复长度（ARL）

**三个关键信息流发现**：
   - context token（目标 token 及其最近邻）在 dMLLMs 中充当锚点，跨层逐步聚合语义信息并吸收注意力
   - 正常解码下，context token 的信息熵在深层收敛，反映模型预测确定性增加
   - 缓存导致注意力分配随机化，context token 熵在深层无法收敛→产生重复

## 方法详解

### 整体框架

CoTA 是即插即用的免训练方法，包含两个互补模块，作用于推理阶段。

### CTAE（Context-token Attention Enhancement）

基于相对距离的高斯衰减项乘以注意力矩阵，增强对 context token 的注意力，保持原有信息流模式：

$$\mathcal{G}_{i,j} = \gamma_{\min} + (1-\gamma_{\min}) \exp\left(-\left(\frac{|i-j|}{\tau}\right)^2\right)$$

- 修改后的注意力：$\tilde{A}_{i,j} = A_{i,j} \cdot \mathcal{G}_{i,j}$
- 温度因子 $\tau=5$，下限 $\gamma_{\min} \in (0,1]$ 保证稳定性

### CTEV（Context-token Entropy-Guided Voting）

利用深层（26-30层）context token 的累积熵作为惩罚项加入置信度评分：

$$\text{Score}(i) = c_{(i)} + \alpha \cdot E_{sum}^{ctx}(i)$$

其中 $E_{sum}^{ctx}(i) = \sum_{j \in \mathcal{C}(i)} \sum_{l=26}^{30} E^{(l)}(j)$，$\mathcal{C}(i)$ 是目标 token 及两个最近邻

### 协同工作流

CTAE 保持信息流模式→CTEV 惩罚不确定 context 的投票分数→联合缓解重复

## 实验关键数据

### 主实验（LLaDA-V 8B, COCO Caption）

| 方法 | ARR↓(512) | MRL↓(512) | SRR↓(512) | ARR↓(64) |
|------|-----------|-----------|-----------|----------|
| LLaDA-V 基线 | 0.2 | 2.0 | 6.9 | 0.1 |
| +Cache | 14.3 | 11.0 | 82.3 | 7.1 |
| +Cache+CTAE | 3.5 | 3.2 | 22.1 | 2.8 |
| +Cache+CTEV | 4.1 | 3.8 | 25.3 | 3.2 |
| **+Cache+CoTA** | **1.2** | **1.3** | **6.3** | **1.0** |

### 多模态基准

| 基准 | LLaDA-V | +Cache | +Cache+CoTA |
|------|---------|--------|-------------|
| DocVQA | 78.2 | 72.6 | 76.9 |
| MMStar | 55.1 | 49.3 | 54.2 |
| MME | 1892 | 1645 | 1856 |

### 消融实验

| 配置 | ARR↓(512) | 说明 |
|------|-----------|------|
| CTAE only | 3.5 | 各自独立有效 |
| CTEV only | 4.1 | 各自独立有效 |
| **CTAE+CTEV** | **1.2** | 互补协同效果最佳 |
| CTAE τ=3 | 2.1 | 窗口偏小 |
| CTAE τ=5 | 1.2 | 最优温度 |
| CTAE τ=10 | 1.8 | 窗口偏大 |

### 关键发现

- CoTA 将 ARR 降低最多 92%（14.3→1.2），几乎恢复到无缓存基线水平
- 长文本（512 tokens）比短文本（64 tokens）更容易出现重复——缓存累积效应
- CTAE 和 CTEV 各有侧重但互补——注意力增强保模式，熵引导防投票
- CoTA 在 8 个多模态基准上均恢复了因缓存导致的性能下降

## 亮点与洞察

- **首次系统分析 dMLLMs 中缓存导致的重复现象**：从信息流角度给出了清晰的因果链
- 发现 context token 在双向注意力中起类似自回归模型"attention sink"的角色——是跨模型架构的普遍现象
- 方法简洁优雅：**完全免训练、即插即用、计算开销极小**
- 定量指标设计（ARR/SRR/MRL/ARL）为后续研究提供了标准化评估框架

## 局限与展望

- 仅在 LLaDA-V 一个 dMLLM 上验证，其他扩散 LM（dMDT、MDLM 等）上的泛化性待考察
- 温度参数 $\tau=5$ 和深层定义（26-30 层）均为经验性设定，不同架构可能需要调整
- 对超长文本生成（>1024 tokens）和多轮对话场景的效果未充分验证
- CTAE 的高斯衰减假设近距离 token 语义相关性更强，但跳跃式引用等场景可能违反此假设

## 相关工作与启发

- **自回归模型 attention sink**：Chen et al. 和 Wang et al. 发现 AR 模型中特殊 token 聚合注意力；本文首次在 dMLLMs 中发现类似机制
- **dLLM-Cache / TinyCache**：加速方法本身有效，但 CoTA 揭示并修复了其副作用
- **ADLM**：讨论了扩散语言模型中 anchor 的语义引导角色，与本文发现互印证

## 评分

- 新颖性: ⭐⭐⭐⭐ 信息流分析角度新颖，问题定义清晰
- 实验充分度: ⭐⭐⭐⭐ 消融完整，多基准验证，定量指标全面
- 写作质量: ⭐⭐⭐⭐ 可视化清晰，因果链完整
- 价值: ⭐⭐⭐⭐ 解决实用问题，方法即插即用，工业价值高

<!-- RELATED:START -->

## 相关论文

- [Long-Context Generalization with Sparse Attention](long-context_generalization_with_sparse_attention.md)
- [A Problem-Oriented Perspective and Anchor Verification for Code Optimization](a_problem-oriented_perspective_and_anchor_verification_for_code_optimization.md)
- [Rectified-CFG++ for Flow Based Models](../../NeurIPS2025/object_detection/rectified-cfg_for_flow_based_models.md)
- [Fixed Anchors Are Not Enough: Dynamic Retrieval and Persistent Homology for Dataset Distillation](../../CVPR2026/object_detection/fixed_anchors_are_not_enough_dynamic_retrieval_and_persistent_homology_for_datas.md)
- [Neighbor GRPO: Contrastive ODE Policy Optimization Aligns Flow Models](../../CVPR2026/object_detection/neighbor_grpo_contrastive_ode_policy_optimization_aligns_flow_models.md)

<!-- RELATED:END -->
