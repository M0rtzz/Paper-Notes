---
title: >-
  [论文解读] Towards Multi-modal Transformers in Federated Learning
description: >-
  [ECCV2024][AI安全][联邦学习] 提出 FedCola 框架，通过互补本地训练和协作聚合两个策略，在联邦学习中实现多模态 Transformer 的跨模态知识迁移，无需公共数据即可弥合单模态与多模态客户端之间的差距。
tags:
  - ECCV2024
  - AI安全
  - 联邦学习
  - 多模态
  - Transformer
  - 视觉语言
  - Mixture of Experts
---

# Towards Multi-modal Transformers in Federated Learning

**会议**: ECCV2024  
**arXiv**: [2404.12467](https://arxiv.org/abs/2404.12467)  
**代码**: [imguangyu/FedCola](https://github.com/imguangyu/FedCola)  
**领域**: ai_safety  
**关键词**: federated learning, Multi-modal Learning, Transformer, Vision-Language, Mixture of Experts

## 一句话总结

提出 FedCola 框架，通过互补本地训练和协作聚合两个策略，在联邦学习中实现多模态 Transformer 的跨模态知识迁移，无需公共数据即可弥合单模态与多模态客户端之间的差距。

## 背景与动机

- 多模态 Transformer 在多种下游任务上取得了显著进步，但训练需要大量高质量配对数据，而这些数据往往因隐私法规分散在不同机构中，形成数据孤岛
- 联邦学习（FL）作为隐私保护范式可以在不直接访问原始数据的情况下协作训练模型，但现有多模态 FL 研究主要聚焦于水平场景（所有客户端拥有相同模态）或垂直场景（同一用户的不同模态分布在不同客户端）
- 这两种设定都排斥了拥有非配对单模态数据的客户端参与训练。而在集中式训练中，非配对单模态数据已被证明能提升多模态模型性能
- 因此，如何在 FL 中从单模态客户端迁移知识（transfer MFL 场景）是一个重要但尚未充分探索的方向

## 核心问题

在 transfer MFL 场景中存在两个关键差距：

1. **跨模态差距（Cross-modality Gap）**：单模态客户端从未访问过其他模态的数据，导致模型偏向本地模态，无法获取跨模态知识
2. **模态内差距（In-modality Gap）**：即使是同一模态，多模态客户端（做检索任务）和单模态客户端（做分类任务）使用不同的训练目标，导致学到的表征存在差异

## 方法详解

### 整体框架

FedCola 利用 Transformer 跨模态的统一架构设计，将模型拆分为三部分：(i) Embedding 层，(ii) Transformer blocks，(iii) 任务头。不同客户端共享统一架构的 Transformer blocks，但 Embedding 层和任务头根据各自的输入模态和训练目标不同而不同。

### 互补本地训练（Complementary Local Training）

针对跨模态差距，核心思路是让单模态客户端也能利用其他模态的知识：

- 单模态客户端在本地训练时，额外下载其他模态的 Transformer blocks
- 受 Mixture of Experts（MoE）启发，用可学习门控参数 $g$（初始化为 0）控制互补模态的贡献
- 每一层的输出为 $W_{\text{local}} x + g \cdot W_{\text{out}} x$，其中 $W_{\text{local}}$ 是本地权重，$W_{\text{out}}$ 是其他模态的权重
- 门控初始化为 0 保证了训练初期以本地模态为主，逐步自适应地学习跨模态互补信息
- **权重压缩技巧**：训练结束后计算等效权重 $W_{\text{local}} + g \cdot W_{\text{out}}$（利用线性性质），上传大小与原模型相同，避免通信开销翻倍

### 协作聚合（Collaborative Aggregation）

针对模态内差距，提出 "聚合与分解" 策略：

- **关键观察**：Self-attention 层编码 token 间的通用关系，承载更通用的知识；而 MLP 层将通用知识适配到本地任务，承载任务特定知识
- **选择性聚合**：仅对相同模态内不同客户端类型之间的 Self-attention 层进行跨类型聚合，MLP 层保持各自独立
- 例如：图像单模态客户端和图文多模态客户端的 Self-attention 层按数据量加权聚合，但 MLP 层不跨类型聚合

### 补偿方案（Compensation Scheme）

选择性聚合会造成层间对齐问题——同一客户端的更新本来在各层之间是连贯的，仅对部分层进行聚合会打破这种连贯性。论文提出补偿矩阵 $\Omega_{\text{comp}}$，将非协作层的更新也按相同系数缩放，确保不同层之间的更新保持对齐。

## 实验关键数据

### 数据集设置

| 客户端类型 | 数据集 | 任务 |
|:---:|:---:|:---:|
| 图像客户端 | CIFAR-100 / OrganCMNIST | 分类 |
| 文本客户端 | AG NEWS / Medical Abstracts | 分类 |
| 多模态客户端 | Flickr10k / COCO Captions | 跨模态检索 |

### 主要结果（Top-1 Recall 之和）

| 方法 | Flickr（默认） | COCO（默认） |
|:---:|:---:|:---:|
| FedAvg | 81.08 | 95.42 |
| FedProx | 78.55 | 95.16 |
| CreamFL | 74.83 | 95.26 |
| FedIoT | 85.51 | 98.40 |
| **FedCola** | **91.96** | **105.10** |

FedCola 在默认设置下比最强基线 FedIoT 提升约 6.5/6.7 个点。

### 消融实验

| 协作聚合 | 补偿 | 互补训练 | R@1_sum |
|:---:|:---:|:---:|:---:|
| ✗ | ✗ | ✗ | 81.08 |
| ✓ | ✗ | ✗ | 88.70 |
| ✓ | ✓ | ✗ | 90.09 |
| ✓ | ✓ | ✓ | **91.96** |

- 协作聚合贡献最大（+7.62）
- 补偿方案进一步提升（+1.39）
- 互补本地训练锦上添花（+1.87）

### 规模扩展能力

加入更多单模态数据集后性能持续提升（0 个额外数据集：81.08 → 1 个：91.96 → 2 个：93.25），验证了框架的可扩展性。

### 公平性分析

利用 Shapley 值量化各类客户端的贡献：文本客户端贡献值为 6.14，图像客户端为 4.74，说明文本模态在默认设置下对多模态检索性能提升贡献更大。

## 亮点

1. **无需公共数据**：不同于 CreamFL 需要共享公共数据集进行知识蒸馏，FedCola 仅通过模型参数实现跨模态协作，更符合 FL 隐私保护的初衷
2. **巧妙利用 Transformer 统一架构**：将不同模态的 Transformer blocks 视为可互换/互补的专家，通过门控混合实现跨模态知识共享
3. **等效权重压缩**：利用门控组合的线性性质，将上传通信开销压缩到与原始模型相同大小
4. **层级分析有洞察力**：区分 Self-attention（通用知识）和 MLP（任务特定知识）的聚合策略，有理论依据且实验验证有效
5. **全面的鲁棒性评估**：在不同异质性程度、参与率、域差距、客户端比例下均保持一致优势

## 局限与展望

- 域差距较大时性能仍有下降，论文承认未完全解决大域差距问题
- 仅在 ViT-Small 上验证，未探索更大规模模型（如 ViT-Large、多模态预训练模型）
- 互补训练增加了下载通信开销（虽然上传已通过等效权重压缩），在带宽受限场景可能是瓶颈
- 仅考虑视觉-语言两种模态，未扩展到音频、视频等更多模态
- 全局聚合中跨模态协作（Cross-modal collaboration on global aggregation）留作了未来工作

## 与相关工作的对比

| 方法 | 需要公共数据 | 支持非配对单模态客户端 | 跨模态知识迁移 | 架构要求 |
|:---:|:---:|:---:|:---:|:---:|
| CreamFL | 是 | 是 | 知识蒸馏 | 异构 |
| FedIoT | 否 | 是 | 聚合策略 | 同构 |
| **FedCola** | **否** | **是** | **参数级混合+聚合** | **统一 Transformer** |

- CreamFL 依赖公共数据做知识蒸馏，隐私保护程度较弱且对公共数据质量敏感
- FedIoT 仅在聚合层面做跨模态协作，本地训练无法利用跨模态信息
- FedCola 在本地训练和全局聚合两个阶段均实现跨模态协作，且无需任何额外公共数据

## 启发与关联

- **MoE 思想在 FL 中的应用**：将不同模态的模型视为"专家"并用门控混合，为 FL 中的异构模型协作提供了新范式
- **层级功能分析**：Self-attention 携带通用知识、MLP 携带任务特定知识的观察可以推广到其他需要选择性聚合的 FL 场景
- **通信效率**：等效权重压缩的技巧（利用线性层组合的线性性质）具有通用性，可应用于其他需要混合多个模型权重的场景
- 与 personalized FL 的思路互补：FedCola 关注全局模型性能，但其 Self-attention 聚合 + MLP 保留的策略本质上也是一种全局与个性化的平衡

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次在 transfer MFL 中探索 Transformer，互补训练和选择性聚合设计新颖
- 实验充分度: ⭐⭐⭐⭐ — 多设置、多域差距、消融、公平性分析很全面
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，方法阐述系统
- 价值: ⭐⭐⭐⭐ — 为多模态 FL 提供了实用且可扩展的框架

<!-- RELATED:START -->

## 相关论文

- [SkyMask: Attack-Agnostic Robust Federated Learning with Fine-Grained Learnable Masks](skymask_attack-agnostic_robust_federated_learning_with_fine-grained_learnable_ma.md)
- [Co-LoRA: Collaborative Model Personalization on Heterogeneous Multi-Modal Clients](../../ICLR2026/ai_safety/co-lora_collaborative_model_personalization_on_heterogeneous_multi-modal_clients.md)
- [Fisher Calibration for Backdoor-Robust Heterogeneous Federated Learning](fisher_calibration_for_backdoor-robust_heterogeneous_federated_learning.md)
- [Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning](../../CVPR2025/ai_safety/infighting_in_the_dark_multi-label_backdoor_attack_in_federated_learning.md)
- [FecalFed: Privacy-Preserving Poultry Disease Detection via Federated Learning](../../CVPR2026/ai_safety/fecalfed_privacy-preserving_poultry_disease_detection_via_federated_learning.md)

<!-- RELATED:END -->
