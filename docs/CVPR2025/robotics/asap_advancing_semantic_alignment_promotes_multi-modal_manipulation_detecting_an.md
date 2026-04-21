---
title: "ASAP: Advancing Semantic Alignment for Multi-Modal Manipulation Detection"
description: "提出ASAP框架，通过LMA大模型辅助对齐、MGCA操控引导交叉注意力和PMM补丁操控建模三大模块，DGM4 AUC达94.38，Text F1达76.52"
tags:
  - CVPR2025
  - 多模态篡改检测
  - 语义对齐
  - 大语言模型
  - 对比学习
---

# ASAP: Advancing Semantic Alignment for Multi-Modal Manipulation Detection

**会议**: CVPR 2025  
**机构**: 合肥工业大学  
**arXiv**: 2412.12718  
**关键词**: 多模态篡改检测、语义对齐、大模型辅助、交叉注意力  

## 研究背景与动机

随着AI生成技术的飞速发展，多模态虚假信息（Multimodal Misinformation）已成为严峻的社会挑战。不同于传统的单模态篡改（仅图像PS或仅文本编造），现代虚假信息往往涉及图像和文本的联合篡改：

**图像篡改**：使用Deepfake、Inpainting等技术修改图像

**文本篡改**：编造或修改与图像配套的描述文字

**跨模态不一致**：真实图像搭配虚假文本，或篡改图像搭配合理化文本

DGM4（Detecting and Grounding Multi-Modal Media Manipulation）任务要求模型不仅判断图文对是否被篡改，还要**定位**篡改区域（图像中的哪些区域？文本中的哪些词句？）。

当前方法的主要问题在于**视觉-语言语义对齐不足**：
- CLIP等预训练模型学到的是粗粒度的图文匹配，无法捕捉细微的篡改痕迹
- 图像patch级和文本token级的细粒度对应关系未被充分利用
- 缺乏显式的篡改引导机制——模型不知道"该关注什么"

ASAP的核心动机是：**通过更好的语义对齐来提升篡改检测和定位能力**，利用大语言模型的知识来辅助理解"什么是正常的图文关系"。

## 方法详解

### 整体框架

ASAP包含三个核心模块：LMA（大模型辅助对齐）、MGCA（操控引导交叉注意力）、PMM（补丁操控建模），分别解决不同层面的对齐问题。

### 模块1：LMA - 大模型辅助对齐

**动机**：CLIP的文本编码器对短描述效果好，但对复杂的语义关系理解不足。大语言模型具有更强的推理和描述能力。

**流程**：
1. **MLLM描述生成**：使用多模态大语言模型（如GPT-4V）对图像生成详细的描述文本
2. **LLM解释生成**：使用LLM分析原始文本和MLLM描述之间的差异，生成解释性文本
3. **VLC对比损失**：将三种文本（原始文本、MLLM描述、LLM解释）与图像进行多路对比学习

$$\mathcal{L}_{	ext{VLC}} = -\log rac{\exp(	ext{sim}(v, t^+) / 	au)}{\sum_j \exp(	ext{sim}(v, t_j) / 	au)}$$

其中正样本对包括：匹配的图文对、图像与其MLLM描述，负样本包括不匹配的文本和篡改样本。

**关键洞察**：LLM生成的解释文本提供了"为什么这个图文对不一致"的推理线索，帮助模型学习更深层的语义对齐。

### 模块2：MGCA - 操控引导交叉注意力

**动机**：标准的交叉注意力平等对待所有patch和token，但篡改区域通常只占少部分，需要引导注意力聚焦。

**设计**：
- 引入**引导掩码** $G \in \{0, 1\}^{N_v 	imes N_t}$，标记疑似篡改的图文对应区域
- 交叉注意力计算时，引导掩码调制注意力权重：

$$	ext{Attn}(Q, K, V) = 	ext{softmax}\left(rac{QK^T}{\sqrt{d}} + \lambda \cdot G
ight) V$$

| 组件 | 输入 | 输出 | 作用 |
|------|------|------|------|
| 视觉编码器 | 图像patch | 视觉特征 $V$ | 提取图像区域特征 |
| 文本编码器 | 文本token | 文本特征 $T$ | 提取文本语义特征 |
| 引导掩码生成器 | $V, T$ | 引导掩码 $G$ | 定位疑似篡改区域 |
| MGCA层 | $V, T, G$ | 增强特征 $V', T'$ | 篡改感知的跨模态融合 |

引导掩码通过浅层特征的不匹配度计算得到，随着网络深度增加逐步精化。

### 模块3：PMM - 补丁操控建模

**动机**：篡改检测不仅需要全局判断，还需要patch级的定位能力。

**方法**：
1. **难负样本构造**：从训练数据中选择语义相近但来源不同的patch进行替换，构造难以区分的篡改样本
2. **patch级分类**：对每个图像patch预测"真实/篡改"的二分类标签
3. **对比增强**：拉近同一图像中真实patch之间的距离，推远真实与篡改patch

$$\mathcal{L}_{	ext{PMM}} = 	ext{BCE}(p_{	ext{patch}}, y_{	ext{patch}}) + \lambda \cdot \mathcal{L}_{	ext{contrast}}$$

**难负样本选择策略**：选择与当前patch特征最相似的其他图像patch进行替换，而非随机替换。这迫使模型学习更细微的篡改线索。

### 总损失函数

$$\mathcal{L} = \mathcal{L}_{	ext{cls}} + lpha \mathcal{L}_{	ext{VLC}} + eta \mathcal{L}_{	ext{grounding}} + \gamma \mathcal{L}_{	ext{PMM}}$$

其中 $\mathcal{L}_{	ext{cls}}$ 是全局篡改分类损失，$\mathcal{L}_{	ext{grounding}}$ 是像素/token级定位损失。

## 实验结果

### DGM4数据集主要结果

| 方法 | AUC | mAP | Image F1 | Text F1 |
|------|-----|-----|----------|---------|
| HAMMER | 91.53 | 83.45 | 72.34 | 67.89 |
| DGM4-baseline | 92.15 | 85.23 | 74.56 | 70.12 |
| MMFED | 93.19 | 86.22 | 76.12 | 71.35 |
| **ASAP** | **94.38** | **88.53** | **78.34** | **76.52** |
| vs MMFED提升 | +1.19 | +2.31 | +2.22 | +5.17 |

Text F1的+5.17提升尤为显著，证明了LMA模块在文本篡改定位上的优势。

### 消融实验

| 配置 | AUC | Text F1 |
|------|-----|---------|
| Baseline | 92.15 | 70.12 |
| + LMA | 94.28 | 74.89 |
| + LMA + MGCA | 94.34 | 75.67 |
| + LMA + MGCA + PMM | **94.38** | **76.52** |

LMA是最关键的模块（AUC +2.13），MGCA和PMM在此基础上进一步提升定位精度。

### 跨数据集泛化

| 训练集 | 测试集 | AUC |
|--------|--------|-----|
| DGM4 | NewsCLIPpings | 84.56 |
| DGM4 | COSMOS | 81.23 |
| DGM4 | VERITE | 79.87 |

跨数据集性能表明ASAP学到了通用的篡改检测能力，而非过拟合到特定数据集。

## 创新点总结

1. **LMA大模型辅助**：首次将MLLM描述和LLM推理引入篡改检测的对齐学习
2. **MGCA引导注意力**：通过引导掩码让交叉注意力聚焦于篡改相关区域
3. **PMM难负样本策略**：基于特征相似度的难负样本选择提升了patch级检测精度

## 局限性

- LMA阶段依赖GPT-4V等外部大模型，增加了推理成本（仅需要一次性预计算）
- 在文本较短的场景（如推特标题）中，文本篡改定位性能可能下降
- 引导掩码的阈值选择对性能有一定影响

## 总结

ASAP通过三层语义对齐机制（全局→区域→patch）系统性地提升了多模态篡改检测能力。大模型辅助的对齐策略是最大亮点——利用LLM的推理能力来理解"图文一致性"，为篡改检测提供了更丰富的语义锚点。在DGM4上的全面领先证明了方法的有效性。

<!-- RELATED:START -->

## 相关论文

- [ASAP: Advancing Semantic Alignment Promotes Multi-Modal Manipulation Detecting and Grounding](asap_advancing_semantic_alignment_promotes_multi-modal_manipulation_de.md)
- [mmWalk: Towards Multi-modal Multi-view Walking Assistance](../../NeurIPS2025/robotics/mmwalk_towards_multi-modal_multi-view_walking_assistance.md)
- [LaDA: Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [HiMaCon: Discovering Hierarchical Manipulation Concepts from Unlabeled Multi-Modal Data](../../NeurIPS2025/robotics/himacon_discovering_hierarchical_manipulation_concepts_from_unlabeled_multi-moda.md)
- [Embodied Representation Alignment with Mirror Neurons](../../ICCV2025/robotics/embodied_representation_alignment_with_mirror_neurons.md)

<!-- RELATED:END -->
