---
title: >-
  [论文解读] LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning
description: >-
  [ICCV 2025][多模态VLM][测试时自适应] 提出 Latte 框架，在联邦学习的去中心化场景下，通过本地记忆与外部记忆的协同机制，实现视觉语言模型（如 CLIP）的协作式测试时自适应，兼顾跨客户端知识共享与个性化。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "测试时自适应"
  - "联邦学习"
  - "视觉语言模型"
  - "记忆缓存"
  - "CLIP"
---

# LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning

**会议**: ICCV 2025  
**arXiv**: [2507.21494](https://arxiv.org/abs/2507.21494)  
**代码**: [GitHub](https://github.com/baowenxuan/Latte)  
**领域**: 多模态VLM  
**关键词**: 测试时自适应, 联邦学习, 视觉语言模型, 记忆缓存, CLIP

## 一句话总结

提出 Latte 框架，在联邦学习的去中心化场景下，通过本地记忆与外部记忆的协同机制，实现视觉语言模型（如 CLIP）的协作式测试时自适应，兼顾跨客户端知识共享与个性化。

## 研究背景与动机

预训练视觉语言模型（VLM）如 CLIP 在零样本图像分类上表现出色，但部署到具体下游领域时面临**域偏移**问题——视觉和文本嵌入的对齐关系可能不再适用。测试时自适应（TTA）是缓解此问题的有效途径，其中**基于记忆的方法**因免训练、无需反向传播而特别高效。

然而，现有记忆式 TTA 方法有一个关键假设：**单一域且数据充足**。在联邦学习（FL）场景下，多个客户端执行相同任务但数据分布各异且每个客户端数据量有限，这带来两个矛盾：

**独立适应**：各客户端独立运行 TTA，因数据稀缺导致记忆质量差，性能下降

**全局共享**：所有客户端共享一份全局记忆，无法针对各客户端独特分布进行个性化

本文的核心切入点是：**如何在去中心化的异构客户端间，安全高效地共享记忆信息，既利用同分布客户端的数据优势，又对异分布客户端保持鲁棒？** Latte 通过本地记忆 + 外部记忆的双重设计来解决这一问题。

## 方法详解

### 整体框架

Latte 的流程分为四步：
1. 对输入图像编码获得嵌入 $\boldsymbol{f}$ 和初始预测
2. 用 $\boldsymbol{f}$ 更新本地记忆 $\boldsymbol{L}^i$
3. 利用本地记忆 $\boldsymbol{L}^i$ 和外部记忆 $\boldsymbol{E}^i$ 获得自适应预测
4. （可选）与服务器通信，更新外部记忆 $\boldsymbol{E}^i$

### 关键设计

1. **本地记忆构建（优先队列）**：每个客户端维护一个类别分裂的记忆 $\boldsymbol{L}^i \in \mathbb{R}^{c \times k_l \times d}$，其中每个类别对应一个按熵排序的优先队列。新测试样本到来时，若队列未满则直接插入；若已满且新样本熵更低（更确定），则替换最高熵的条目。这保证记忆中始终保留最可靠的样本嵌入。

2. **全局记忆与外部记忆（服务器协调的选择性共享）**：服务器维护全局记忆 $\boldsymbol{G} \in \mathbb{R}^{c \times n \times d}$，每个客户端上传其本地记忆的**加权原型**（熵加权平均后归一化）。关键在于，每个客户端不是获取整个全局记忆，而是用自身原型作为查询向量，检索 top-$k_e$ 个最相似的其他客户端原型作为外部记忆。这实现了**粗粒度过滤**，减少了不相关原型的传输。

3. **融合记忆的自适应预测**：将本地记忆和外部记忆合并后，通过同时考虑**嵌入相似度**和**不确定性（熵）**来计算聚合权重：

$$w_{y,\kappa}^i = \exp(\beta \cdot \boldsymbol{f}^\top \boldsymbol{m}_{y,\kappa}^i) \cdot \exp(-\gamma \cdot H(\boldsymbol{m}_{y,\kappa}^i))$$

这个设计使得高相似度且低不确定性的样本获得更大权重，从而对 OOD 原型和误分类样本保持鲁棒。最终预测为 CLIP 原始 logits 和记忆 logits 的加权和。

4. **通信与推理解耦**：通信过程仅依赖本地记忆而不依赖当前测试样本，允许客户端在通信间隔期间进行离线推理，大幅减少通信轮次。

### 损失函数 / 训练策略

Latte 是一个**免训练**框架——不需要任何反向传播或梯度计算。其适应过程完全通过记忆的构建、共享和加权查询实现。超参数包括记忆大小 $k_l$、外部记忆大小 $k_e$、相似度锐度 $\beta$、不确定性锐度 $\gamma$ 和融合系数 $\alpha$。

## 实验关键数据

### 主实验

在域适应基准（VLCS、TerraIncognita）和损坏基准（CIFAR-10-C、CIFAR-100-C）上进行评估。

| 方法 | VLCS (ViT-B/16) | TerraIncognita (ViT-B/16) | CIFAR-10-C (ViT-B/16) |
|------|:---:|:---:|:---:|
| CLIP | 80.83 | 31.84 | 65.58 |
| VTE | 81.75 | 38.56 | 67.64 |
| TDA (local) | 81.44 | 34.24 | 66.58 |
| TDA (global) | 80.29 | 36.19 | 65.58 |
| DMN-ZS (local) | 81.12 | 33.65 | 67.42 |
| DMN-ZS (global) | 80.55 | 37.64 | 63.90 |
| **Latte** | **82.57 (+1.74)** | **40.95 (+9.11)** | **68.27 (+2.69)** |

Latte 在所有基准上均取得最佳性能。值得注意的是，在 TerraIncognita 上，Latte 比 CLIP 零样本高出 9.11%，远超其他方法。全局共享策略（TDA global、DMN-ZS global）在某些场景下甚至出现负迁移。

### 消融实验

| 消融项 | VLCS 准确率 | 说明 |
|--------|:---:|------|
| 仅用本地记忆 | ~81.5 | 缺少跨客户端信息 |
| 仅用外部记忆 | ~81.0 | 缺少本地个性化 |
| Latte（完整） | **82.57** | 两者互补 |
| 去掉相似度权重 | ~81.0 | 仅用熵加权不足 |
| 去掉不确定性权重 | ~81.5 | 仅用相似度加权不足 |
| Latte（完整） | **82.57** | 两者缺一不可 |

### 关键发现

- 数据去中心化程度增加时（每域客户端数从 1 增到 50），DMN-ZS 和 TDA 性能显著下降，而 Latte 保持稳定
- 计算开销极小：相比 CLIP 推理的 17.6G MACs，Latte 仅增加 871K MACs
- 通信效率高：每轮通信量不到 CLIP 视觉编码器大小的 0.4%；通信间隔 T≤50 时精度几乎不变
- ID 客户端主要从同分布客户端获取原型，合并记忆后熵显著降低、类内聚更紧凑

## 亮点与洞察

- **设计优雅**：双记忆 + 原型检索的机制在保持简洁的同时有效平衡共享与个性化
- **理论保障**：证明了 Latte 在 ID 客户端数增加时误差单调下降，且对 OOD 客户端的误差界不受影响
- **实用性强**：通信与推理解耦使其真正适用于实际 FL 系统，而非仅限于理想化设置
- 免训练特性使其适合资源受限的边缘设备

## 局限与展望

- 仅在图像分类任务上验证，未扩展到检测、分割等视觉任务
- 对客户端数据分布差异极端的情况（如完全不重叠的类别空间）未充分讨论
- 原型检索的 top-k 策略较为简单，可探索更细粒度的图级别记忆共享
- 理论分析基于简化的二分类假设，与实际多分类场景有一定差距

## 相关工作与启发

- 在 TDA 和 DMN-ZS 等基于记忆的 TTA 基础上，核心创新是引入联邦场景下的协作机制
- 与联邦 TTA（如 FedTHE、ATP）的区别在于：专注于预训练 VLM 且无需微调
- 外部记忆的 top-k 检索思想与检索增强生成（RAG）有共通之处
- 未来可探索将此范式扩展到多模态大模型的分布式推理场景

## 评分

- **新颖性**: 7/10 — 双记忆 + 协作检索的组合在 VLM+FL 场景下是新颖的
- **技术质量**: 8/10 — 方法完整、理论分析扎实、实验全面
- **实用性**: 8/10 — 免训练、低通信开销、通信与推理解耦
- **写作质量**: 8/10 — 结构清晰，符号定义完整

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models](fedmvp_federated_multimodal_visual_prompt_tuning_for_vision-language_models.md)
- [\[NeurIPS 2025\] Buffer Layers for Test-Time Adaptation](../../NeurIPS2025/llm_safety/buffer_layers_for_test-time_adaptation.md)
- [\[ICCV 2025\] Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning](geminio_language-guided_gradient_inversion_attacks_in_federated_learning.md)
- [\[CVPR 2025\] CleanSight: Test-Time Attention Purification for Backdoored Large Vision Language Models](../../CVPR2025/llm_safety/test-time_attention_purification_for_backdoored_large_vision_language_models.md)
- [\[CVPR 2025\] TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models](../../CVPR2025/llm_safety/tapt_test-time_adversarial_prompt_tuning_for_robust_inference_in_vision-language.md)

</div>

<!-- RELATED:END -->
