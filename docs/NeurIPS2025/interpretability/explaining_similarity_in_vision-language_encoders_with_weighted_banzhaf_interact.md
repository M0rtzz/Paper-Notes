---
title: >-
  [论文解读] Explaining Similarity in Vision-Language Encoders with Weighted Banzhaf Interactions
description: >-
  [NeurIPS 2025][CLIP可解释性] FIxLIP 提出基于加权 Banzhaf 交互指数的博弈论框架，统一分解视觉-语言编码器（如 CLIP、SigLIP-2）的相似度预测为一阶token归因和二阶跨模态/模态内交互，在效率和忠实度上均超越现有一阶归因方法。
tags:
  - NeurIPS 2025
  - CLIP可解释性
  - Banzhaf交互
  - 博弈论
  - 跨模态交互
  - 显著性图
---

# Explaining Similarity in Vision-Language Encoders with Weighted Banzhaf Interactions

**会议**: NeurIPS 2025  
**arXiv**: [2508.05430](https://arxiv.org/abs/2508.05430)  
**代码**: [hbaniecki/fixlip](https://github.com/hbaniecki/fixlip)  
**领域**: 多模态VLM  
**关键词**: CLIP可解释性, Banzhaf交互, 博弈论, 跨模态交互, 显著性图  

## 一句话总结
FIxLIP 提出基于加权 Banzhaf 交互指数的博弈论框架，统一分解视觉-语言编码器（如 CLIP、SigLIP-2）的相似度预测为一阶token归因和二阶跨模态/模态内交互，在效率和忠实度上均超越现有一阶归因方法。

## 研究背景与动机
语言-图像预训练（如 CLIP、SigLIP）驱动了零样本分类、跨模态检索等众多能力，但这些编码器日益被用于高风险决策（如医学影像），其内部表示的局限性已被广泛研究——如 CLIP 无法正确判断方向、物体数量、图中文字等。

现有解释方法（如 GAME、Grad-ECLIP）仅能生成一阶归因显著性图（saliency map），捕捉单个token的重要性。然而，**视觉-语言编码器的相似度预测本质上依赖于图像patch与文本token之间的复杂跨模态交互**——一阶方法无法忠实地解释这种关系。用户研究也证实：可视化二阶归因（即成对交互）对于理解复杂多模态模型是必要的。

**核心问题**：如何高效且忠实地将VLE的相似度预测分解为一阶归因和二阶交互？

## 方法详解

### 整体框架
FIxLIP 将视觉-语言编码器的相似度解释建模为博弈论问题：
1. 将输入image patch和text token视为合作博弈中的参与者（players）
2. 定义 FIxLIP 博弈：对所有可能的mask $M \subseteq N_{\mathcal{I}} \cup N_{\mathcal{T}}$，度量masked输入的相似度
3. 用加权 Banzhaf 交互指数近似分解博弈值为常数项、一阶归因和二阶交互
4. 通过加权最小二乘回归高效近似

最终解释为一个完全图：节点权重=token归因，边权重=成对交互（含跨模态和模态内）。

### 关键设计

**FIxLIP-p 解释的形式化定义：**

给定VLE $f(x_{\mathcal{I}}, x_{\mathcal{T}}) = \cos(f_{\mathcal{I}}(x_{\mathcal{I}}), f_{\mathcal{T}}(x_{\mathcal{T}}))$，FIxLIP博弈定义为：

$$\nu(M) = f(x_{\mathcal{I}} \oplus_{M \cap N_{\mathcal{I}}} b_{\mathcal{I}}, x_{\mathcal{T}} \oplus_{M \cap N_{\mathcal{T}}} b_{\mathcal{T}})$$

解释通过二阶可加博弈近似：$\hat{\nu}_{\mathbf{e}}(M) = \mathbf{e}_0 + \sum_{i \in M} \mathbf{e}_i + \sum_{\{i,j\} \subseteq M} \mathbf{e}_{\{i,j\}}$

**p-忠实度度量与加权Banzhaf交互：**
- 定义p-faithfulness: $\mathfrak{F}_p(\nu, \hat{\nu}) = \sum_M p^{|M|}(1-p)^{n-|M|}(\nu(M) - \hat{\nu}(M))^2$
- 参数 $p$ 控制mask权重分布：$p=0.5$ 等权所有mask；$p>0.5$ 偏重in-distribution输入（少量token被mask）；$p<0.5$ 偏重out-of-distribution输入
- FIxLIP-p = argmin p-faithfulness = **加权 Banzhaf 交互指数**
- 相比 Shapley 交互，Banzhaf 的优势在于：(1) $p$ 提供灵活的in/out-of-distribution控制；(2) mask可以分解为独立的图像/文本分布——这是跨模态估计器的关键前提

**跨模态采样策略（核心效率提升）：**
- 传统方法：采样 $m$ 个联合mask $M \sim \mathbb{P}_p$，获得 $m$ 个博弈值
- FIxLIP提出：分别采样 $m_{\mathcal{I}}$ 个图像mask和 $m_{\mathcal{T}}$ 个文本mask，取所有组合
- 只需 $m_{\mathcal{I}} + m_{\mathcal{T}}$ 次模型推理（图像/文本编码器各自独立编码），获得 $m_{\mathcal{I}} \times m_{\mathcal{T}}$ 个博弈值
- **理论保证**（Theorem 2）：跨模态估计器是无偏的，方差界在 $m$ 和 $m_{\mathcal{I}} \cdot m_{\mathcal{T}}$ 的model-agnostic估计器之间
- 实际加速：**5-20×**

**大规模适配策略：**
- 解释基大小二次增长：ViT-B/16对应196+30=226个token → 25,425个交互
- 两步过滤：(1) 取一阶归因最高的top-k token子集计算交互；(2) 或只计算跨模态交互
- 贪心子集选择：在解释图中找到最大/最小相似度子图用于评估和可视化

### 损失函数 / 训练策略
FIxLIP 不涉及模型训练，而是事后解释方法（post-hoc explanation）。核心优化目标是通过加权最小二乘（WLS）回归近似 $\hat{\mathfrak{F}}_p^{(m_{\mathcal{I}}, m_{\mathcal{T}})}$。

实验配置：
- 被解释模型：CLIP ViT-B/32, ViT-B/16; SigLIP, SigLIP-2 ViT-B/32, ViT-L/16
- 跨模态估计器预算：$2^{21}$；Shapley交互估计器预算：$2^{17}$（运行时间相近）
- $p \in \{0.3, 0.5, 0.7\}$，不同 $p$ 无额外计算开销

## 实验关键数据

### 主实验——Pointing Game Recognition (CLIP ViT-B/32, ImageNet-1k)

| 方法 | 1物体 | 2物体 | 3物体 | 4物体 |
|------|-------|-------|-------|-------|
| GAME | .61 | .43 | .33 | .28 |
| Grad-ECLIP | .68 | .45 | .33 | .28 |
| Shapley values | .70 | .56 | .46 | .37 |
| exCLIP | .73 | **.88** | **.89** | **.92** |
| **FIxLIP (Shapley交互)** | **.83** | .82 | .84 | .86 |
| **FIxLIP (w.Banzhaf p=0.7)** | .83 | .81 | .83 | .85 |

**关键发现**：一阶方法在多物体场景下表现骤降（接近随机基线0.25），而二阶交互方法能在多物体情况下保持高识别率。

### Insertion/Deletion 曲线 (CLIP ViT-B/32, MS COCO)

| 方法 | AID分数 ↑ |
|------|----------|
| GAME | 低（无法恢复非线性排序） |
| Grad-ECLIP | 低 |
| exCLIP | 中（仅近似跨模态交互） |
| **FIxLIP (p=0.5)** | **最高**（忠实恢复最优子集解释） |

FIxLIP 不仅能找到删除后相似度显著下降的最重要token，还能找到删除后可能提升预测的最不重要token——这是梯度类方法做不到的。

### 消融实验——计算效率

| 估计器 | 推理加速 | 整体加速 |
|--------|---------|---------|
| Model-agnostic | 1× | 1× |
| **Cross-modal** | **20×** | **5×** |

在 SigLIP-2 ViT-B/32 上，相比一阶归因方法（~1秒），FIxLIP 在大预算下通过跨模态采样实现了可接受的计算开销。

### 关键发现
- 一阶归因方法（如Grad-ECLIP）对VLE的p-faithfulness相关性仅约0.5，而FIxLIP接近1.0
- exCLIP 因仅近似跨模态交互（忽略一阶效应和模态内交互）导致 AID 排序失败
- SigLIP-2 在 Pointing Game 中显著优于 CLIP（.90 vs .83 在1物体场景），表明SigLIP-2学到了更准确的跨模态对应

## 亮点与洞察
1. **博弈论基础扎实**：首次将加权Banzhaf交互推广到VLE解释，具有线性、对称、虚拟参与者等理想公理性质
2. **跨模态采样策略巧妙**：利用VLE图像/文本编码器的独立性，以 $m$ 次推理获得 $m^2/4$ 个博弈值，理论与实践兼优
3. **评估度量贡献**：将 Pointing Game 和 Insertion/Deletion 曲线推广到二阶交互解释，填补了交互解释评估的空白
4. **实用可视化**：通过条件化单个token查看其交互热图，或遍历完全图找高/低相似度子集，提供多层次理解

## 局限与展望
- 当图像分辨率增大时（更多patch），忠实度有所下降——需要进一步扩展到高分辨率模型
- 解释基二次增长限制了扩展性，当前的top-k过滤是启发式方案
- 仅评估了CLIP和SigLIP系列，未测试LLaVA等生成式VLM架构
- 实际应用场景（如医学影像VQA中的错误诊断解释）尚未展示

## 相关工作与启发
- 相比 exCLIP（仅跨模态交互），FIxLIP 包含模态内交互和一阶效应，提供更完整的分解
- 与 SHAP/KernelSHAP 的关系：FIxLIP-0.5 等价于 Faith-Banzhaf（KernelSHAP变体），$p \neq 0.5$ 是加权推广
- 对VLM开发的启发：通过对比不同模型的FIxLIP解释可发现架构差异（如CLIP vs SigLIP-2的注意力模式）
- 可推广到视频-语言、音频-语言等其他多模态编码器的解释

## 评分
- 新颖性: ⭐⭐⭐⭐（加权Banzhaf交互+跨模态采样的组合有原创性，理论贡献充分）
- 实验充分度: ⭐⭐⭐⭐⭐（三个评估指标、多模型、多数据集、效率分析全面）
- 写作质量: ⭐⭐⭐⭐⭐（数学严谨，可视化出色，结构清晰）
- 价值: ⭐⭐⭐⭐（为VLE解释建立了新标准，但实际应用影响有待展现）

<!-- RELATED:START -->

## 相关论文

- [Efficient Vision-Language Reasoning via Adaptive Token Pruning](efficient_vision-language_reasoning_via_adaptive_token_pruning.md)
- [VL-SAE: Interpreting and Enhancing Vision-Language Alignment with a Unified Concept Set](vlsae_interpreting_and_enhancing_visionlanguage_alignment_wi.md)
- [FaCT: Faithful Concept Traces for Explaining Neural Network Decisions](fact_faithful_concept_traces_for_explaining_neural_network_decisions.md)
- [Dataset Distillation for Pre-Trained Self-Supervised Vision Models](dataset_distillation_for_pre-trained_self-supervised_vision_models.md)
- [Cross-Modal Redundancy and the Geometry of Vision-Language Embeddings](../../ICLR2026/interpretability/cross-modal_redundancy_and_the_geometry_of_vision-language_embeddings.md)

<!-- RELATED:END -->
