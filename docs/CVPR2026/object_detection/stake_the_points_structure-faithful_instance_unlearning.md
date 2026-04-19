---
title: >-
  [论文解读] Stake the Points: Structure-Faithful Instance Unlearning
description: >-
  [CVPR2026][目标检测][machine unlearning] 提出 Structguard，通过语义锚点（semantic anchors）保持遗忘过程中保留实例间的语义关系结构，避免结构性崩塌，在图像分类/人脸识别/检索三任务上平均提升 32.9%/19.3%/22.5%。
tags:
  - CVPR2026
  - 目标检测
  - machine unlearning
  - instance-level unlearning
  - structural preservation
  - semantic anchors
  - CLIP
---

# Stake the Points: Structure-Faithful Instance Unlearning

**会议**: CVPR2026  
**arXiv**: [2603.12915](https://arxiv.org/abs/2603.12915)  
**代码**: 待确认  
**领域**: 机器遗忘 / 隐私保护  
**关键词**: machine unlearning, instance-level unlearning, structural preservation, semantic anchors, CLIP

## 一句话总结

提出 Structguard，通过语义锚点（semantic anchors）保持遗忘过程中保留实例间的语义关系结构，避免结构性崩塌，在图像分类/人脸识别/检索三任务上平均提升 32.9%/19.3%/22.5%。

## 背景与动机

1. **数据保护法规驱动**：GDPR 等法规要求模型能删除特定用户数据的影响，从头重训代价过高，催生了机器遗忘（Machine Unlearning, MU）研究
2. **实例级遗忘更实际**：真实删除请求通常针对具体个体而非整个类别，实例级遗忘比类级遗忘更具现实意义
3. **现有方法忽视语义结构**：已有 MU 方法（如 Neggrad、Adv、L2UL）在删除目标实例时，破坏了保留实例之间的语义关系，导致表征空间的渐进式结构崩塌
4. **结构崩塌与性能负相关**：作者实验发现结构崩塌程度与删除-保留平衡准确率呈显著负相关，结构保持越好，遗忘效果越好
5. **无需保留集**：真实场景中原始训练数据往往因策略、存储等限制不可访问，方法仅依赖预训练模型和待遗忘数据
6. **知识编码在关系中**：深度模型的知识不是孤立存储的，而是通过语义关系来组织的，遗忘过程必须保护这种关系结构

## 方法详解

### 整体框架：Structguard

核心思路：引入语义锚点（stakes），作为保留实例的"桩子"，在遗忘过程中将各实例绑定在锚点上，防止语义漂移，维持知识的结构化组织。

### 1. 锚点生成（Anchor Generation）

- 对每个类别 $c$，用 GPT-4o 生成该类的属性描述（纹理、形状、典型上下文等）
- 将属性描述拼接后输入冻结的语义编码器 $T(\cdot)$（CLIP ViT-B/32），得到类级锚点 $a_c$
- 所有锚点组成矩阵 $A \in \mathbb{R}^{b \times d}$，在遗忘过程中保持固定

### 2. 结构定义与代理集

- **原始结构** $S^{\text{ori}} = V^{\text{ori}} \cdot A^\top$：保留实例嵌入与锚点之间的亲和度矩阵
- **代理集**：由于保留集不可访问，通过对遗忘样本生成对抗变体来近似保留集嵌入
- **遗忘结构** $S^{\text{unl}} = V^{\text{unl}} \cdot A^\top$：经过可学习投影器 $p_\omega$ 映射后的嵌入与锚点的亲和度

### 3. 结构保持约束

**结构感知对齐（Structure-aware Alignment）**：

$$\mathcal{L}_{\text{align}} = -\frac{1}{b} \sum_{i=1}^{b} \cos(S_i^{\text{ori}}, S_i^{\text{unl}})$$

最大化遗忘前后结构的余弦相似度，保持锚点-实例间的相对模式。

**结构感知正则化（Structure-aware Regularization）**：

$$\mathcal{L}_{\text{reg}} = \frac{1}{2} \sum_i I_i \cdot (\psi_i^{\text{unl}} - \psi_i^{\text{ori}})^2$$

其中 $I_i$ 为第 $i$ 个参数的结构重要性得分（通过对齐损失对参数的梯度绝对值估计），抑制对结构关键参数的大幅更新。

### 4. 总体损失

- **保留损失** $\mathcal{L}_{\text{ret}}$：通过投影器的交叉熵，保留语义关系
- **删除损失** $\mathcal{L}_{\text{del}}$：绕过投影器的负交叉熵，实现有效擦除
- 总损失 $= \mathcal{L}_{\text{del}} + \mathcal{L}_{\text{ret}} + \mathcal{L}_{\text{align}} + \mathcal{L}_{\text{reg}}$

## 实验关键数据

### 图像分类（CIFAR-10 / CIFAR-100 / ImageNet-1K）

| 方法 | CIFAR-10 $\mathcal{A}_{\text{test}}$ (k=256) | CIFAR-100 $\mathcal{A}_{\text{test}}$ (k=256) | ImageNet-1K $\mathcal{A}_{\text{test}}$ (k=256) | $\mathcal{A}_f$ |
|------|:---:|:---:|:---:|:---:|
| L2UL | 45.44 | 48.71 | 31.19 | 100.0 |
| Adv | 36.69 | 46.45 | 21.27 | 100.0 |
| **Structguard** | **56.32** | **56.91** | **41.15** | **100.0** |

- 在 CIFAR-10（k=256）上超越 Oracle 17.73%（$\mathcal{A}_{\text{test}}$）和 21.77%（$\mathcal{A}_r$）
- ImageNet-1K 上平均超越所有 baseline 21.57%（$\mathcal{A}_{\text{test}}$）
- 随 k 增大，Structguard 的退化幅度远小于 L2UL（CIFAR-100 中 L2UL 下降 22.21% vs Structguard 仅 9.68%）

### 人脸识别（Lacuna-10）

| 方法 | k=3 $\mathcal{A}_{\text{test}}$ | k=64 $\mathcal{A}_{\text{test}}$ | $\mathcal{A}_f$ |
|------|:---:|:---:|:---:|
| L2UL | 75.37 | 12.26 | 100.0 |
| **Structguard** | **77.29** | **27.71** | **100.0** |

平均超越 L2UL 5.92%（$\mathcal{A}_{\text{test}}$）和 5.23%（$\mathcal{A}_r$）。

### 消融实验

| SA | SR | CR | CIFAR-10 $\mathcal{A}_{\text{test}}$ | CIFAR-100 $\mathcal{A}_{\text{test}}$ |
|:---:|:---:|:---:|:---:|:---:|
| ✗ | ✓ | ✓ | 最大下降 | 最大下降 |
| ✓ | ✗ | ✓ | 小幅下降 | 较大下降 |
| ✓ | ✓ | ✗ | 较大下降 | 小幅下降 |
| ✓ | ✓ | ✓ | **最优** | **最优** |

- **SA（结构感知对齐）** 是最关键组件，去除后性能下降最多
- CIFAR-10 上 CR > SR（类别少时分类器正则更重要），CIFAR-100 上 SR > CR（类别多时参数约束更重要）
- 锚点类型：语义锚点优于视觉原型锚点（CIFAR-10 上 +7.84%），表明语言引导的语义锚点提供了更好的结构化参考

## 亮点

- **概念新颖**：首次将"结构保持"形式化为 MU 的核心目标，揭示结构崩塌与删除-保留平衡的因果关系
- **语义锚点设计精巧**：利用 LLM 生成属性描述 + CLIP 编码，构建稳定、数据无关的参考点
- **三任务全面验证**：分类/识别/检索三个不同任务均有显著提升，证明方法的通用性
- **表征一致性极佳**：Grad-CAM 和表征余弦相似度分析显示，保留样本的表征几乎未受遗忘过程影响
- **无需保留集**：仅依赖预训练模型和遗忘集，更贴近真实应用场景

## 局限与展望

- 依赖 CLIP 和 GPT-4o 生成锚点，对模型和提示的选择可能影响效果，且增加了部署成本
- 代理集通过对抗样本近似保留集，当遗忘样本数量较少时近似质量可能不足
- 投影器 $p_\omega$ 引入额外参数和计算开销
- 仅评估了 ResNet 架构，未验证在 ViT 等 Transformer 架构上的效果
- 未讨论多轮连续遗忘请求的场景（锚点是否需要更新）
- 类级锚点对类内多样性的刻画有限，细粒度场景可能需要子类锚点

## 与相关工作的对比

| 方法 | 目标 | 粒度 | 需保留集 | 结构保持 |
|------|------|------|:---:|:---:|
| Fisher [Golatkar'20] | undo | 实例 | ✓ | ✗ |
| UNSIR [Tarun'23] | undo | 类 | ✓ | ✗ |
| L2UL [Chen'24] | misclassify | 实例 | ✗ | ✗ |
| LoTUS [Kim'24] | undo | 实例 | ✓ | ✗ |
| **Structguard** | misclassify | 实例 | ✗ | **✓** |

Structguard 是首个同时满足"无需保留集"和"结构保持"的实例级遗忘方法。与 L2UL 同样采用误分类目标且无需保留集，但通过语义锚点显式维护知识结构，在所有任务上全面超越。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 结构保持视角新颖，语义锚点设计独具匠心
- 实验充分度: ⭐⭐⭐⭐ — 三任务全面评估，消融/可视化/锚点分析丰富
- 写作质量: ⭐⭐⭐⭐ — 图示清晰，动机论证逻辑严密
- 价值: ⭐⭐⭐⭐ — 为 MU 领域提供了新的结构保持范式，实用性强

<!-- RELATED:START -->

## 相关论文

- [Adaptive Auxiliary Prompt Blending for Target-Faithful Diffusion Generation](adaptive_auxiliary_prompt_blending_for_target-faithful_diffusion_generation.md)
- [SDF-Net: Structure-Aware Disentangled Feature Learning for Optical–SAR Ship Re-Identification](sdf-net_structure-aware_disentangled_feature_learning_for_opticall-sar_ship_re-i.md)
- [Mining Instance-Centric Vision-Language Contexts for Human-Object Interaction Detection](mining_instance-centric_vision-language_contexts_for_human-object_interaction_de.md)
- [Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders](../../ICLR2026/object_detection/toward_faithful_retrieval-augmented_generation_with_sparse_autoencoders.md)
- [Saliency-R1: Enforcing Interpretable and Faithful Vision-language Reasoning via Saliency-map Alignment Reward](saliency-r1_enforcing_interpretable_and_faithful_vision-language_reasoning_via_s.md)

<!-- RELATED:END -->
