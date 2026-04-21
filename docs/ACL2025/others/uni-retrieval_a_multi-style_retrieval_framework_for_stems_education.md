---
title: >-
  [论文解读] Uni-Retrieval: A Multi-Style Retrieval Framework for STEM's Education
description: >-
  [ACL 2025][STEM教育检索] 本文提出面向 STEM 教育场景的多风格多模态检索任务和数据集 SER（24,000+ 查询对），以及基于 Prompt Bank 的轻量检索模型 Uni-Retrieval，通过原型学习提取查询风格特征并动态选择提示向量来增强不同风格（文本、草图、艺术、低分辨率、语音）的检索性能，在 STEM 教育检索和传统检索数据集上均超越已有方法。
tags:
  - ACL 2025
  - STEM教育检索
  - 多风格检索
  - 提示学习
  - 原型学习
  - 视觉语言模型
---

# Uni-Retrieval: A Multi-Style Retrieval Framework for STEM's Education

**会议**: ACL 2025  
**arXiv**: [2502.05863](https://arxiv.org/abs/2502.05863)  
**代码**: 无  
**领域**: 多模态VLM / 教育AI  
**关键词**: STEM教育检索, 多风格检索, Prompt Bank, 原型学习, 视觉语言模型

## 一句话总结

本文提出面向 STEM 教育场景的多风格多模态检索任务和数据集 SER（24,000+ 查询对），以及基于 Prompt Bank 的轻量检索模型 Uni-Retrieval，通过原型学习提取查询风格特征并动态选择提示向量来增强不同风格（文本、草图、艺术、低分辨率、语音）的检索性能，在 STEM 教育检索和传统检索数据集上均超越已有方法。

## 研究背景与动机

1. **领域现状**: AI 赋能教育 (AI4EDU) 正快速发展，STEM 教育的资源检索需求日益增长。当前检索系统主要针对自然文本-图像匹配设计，使用 CLIP、BLIP 等预训练模型。

2. **现有痛点**: (1) 现有检索模型主要优化自然文本-图像匹配，忽视了教育场景中多样的查询方式——教师可能通过语音、手绘草图、低分辨率拍照等方式检索教学素材；(2) 检索表述的多义性和模糊性在教育场景中尤为突出（抽象概念需要多种方式解读）；(3) 缺少面向 STEM 教育的专用检索基准数据集。

3. **核心矛盾**: 教学场景需要多种查询风格（文本、草图、语音、低分辨率图片等），但现有模型只针对单一风格优化，无法满足教育场景的多样性需求。

4. **本文目标**: 如何构建一个高效的、能适应多种查询风格的 STEM 教育检索系统？

5. **切入角度**: 将不同查询风格视为可学习的"原型"，用 Prompt Bank 存储和组合不同风格的语义信息，通过动态检索和拼接 prompt token 来适配任意查询风格。

6. **核心 idea**: 用可持续更新的 Prompt Bank 存储不同查询风格的原型信息，通过原型匹配动态选择 prompt token 扩展输入表示，实现多风格统一检索。

## 方法详解

### 整体框架

Uni-Retrieval 由三个子模块组成：(1) 原型学习模块——为每种查询风格生成原型特征向量；(2) Prompt Bank——存储可学习的 key-value 对，key 用于匹配查询风格原型，value 是注入模型的 prompt token；(3) 特征提取器——基于 ViT + Transformer 的视觉语言模型。输入查询→原型匹配→选择 prompt token→拼接到输入序列→通过冻结的 backbone 提取特征→计算相似度排序。

### 关键设计

1. **原型学习模块 (Prototype Learning Module)**:

    - 功能：为每种查询风格提取代表性的特征向量
    - 核心思路：给定某种风格的 $m$ 个查询样本 $x_0^i$，用预训练的风格编码器 $f$ 提取特征 $E_0^i = f(x_0^i)$，然后对该风格的所有样本特征做平均池化得到原型 $P_j = \text{AvgPool}(\sum_{i=0}^m E_j^i)$。风格编码器可以是预训练的风格分类器（用于图像风格查询）或文本编码器（用于文本查询），具体取决于查询类型
    - 设计动机：原型向量压缩了每种风格的核心语义信息，用于后续从 Prompt Bank 中检索最匹配的 prompt token

2. **Prompt Bank**:

    - 功能：存储可学习的风格知识，支持动态组合以适配已知和未知风格
    - 核心思路：Bank 中包含 $N$ 个 key-value 对 $\{(k_1, P_1), ..., (k_N, P_N)\}$。给定输入查询的原型特征，通过余弦相似度 $\gamma$ 检索 top-$n$ 最匹配的 key，取出对应 prompt token，前缀拼接到输入序列：$x_p = [CLS; P_{j_1}; P_{j_2}; ...; P_{j_n}; x_e]$。对于未见过的查询风格，Bank 自动组合多个相似风格的 token 来表示。key 和 prompt token 都是可学习参数，在训练中联合更新
    - 设计动机：类似哈希表的结构使匹配速度快（类比 TTT 和 Mamba 的隐状态设计）。关键优势是可组合性和泛化性——未知风格可通过组合已知风格的 token 来表示

3. **特征提取器与训练策略**:

    - 功能：基于冻结的视觉语言模型进行检索排序
    - 核心思路：视觉编码器用 ViT（OpenCLIP初始化），文本编码器用 Transformer（gpt-neo tokenizer），两者全部冻结以保留原始语义空间。视觉和文本的 prompt token 共享参数以对齐模态。使用 CLS token 作为全局表示。音频查询可选用 GPT-4o 转文本后处理
    - 设计动机：冻结 backbone 大幅减少可训练参数，只训练 Prompt Bank 的 key 和 token（仅增加约 26M 参数），训练高效

### 损失函数 / 训练策略

使用 Triplet Loss：$\mathcal{L} = \max\{0, \mu + d(\delta(x_f), \delta(x_r)) - d(\delta(x_f), \delta(x_h))\}$，其中 $x_f$ 是查询特征，$x_r$ 是正样本，$x_h$ 是负样本，$d$ 使用归一化余弦距离。加上 Prompt Bank 的 key 匹配正则项：$\min_{k,p,L} \mathcal{L} + \lambda \sum_{K_x} \gamma(q(x), k_{si})$，鼓励 key 与原型特征对齐。推理时支持测试时更新 Prompt Bank 以适配领域特定知识。

## 实验关键数据

### 主实验

SER 数据集上的检索性能:

| 方法 | Text→Image R@1 | Sketch→Image R@1 | Art→Image R@1 | Low-Res→Image R@1 |
|------|-------|-------|-------|-------|
| CLIP | 54.6 | 47.3 | 46.8 | 53.7 |
| BLIP | 55.8 | 48.2 | 47.5 | 51.5 |
| CLIP-Finetune | 71.4 | 71.0 | 52.2 | 71.2 |
| FreestyleRet | 80.1 | 75.3 | 73.0 | 78.0 |
| **Uni-Retrieval** | **83.2** | **84.5** | **76.9** | **87.4** |

### 消融实验

效率分析:

| 方法 | 参数量 | Q2I推理(ms) | Text→Image Acc |
|------|--------|-------------|---------------|
| CLIP | 427M | 68ms | 54.6 |
| LanguageBind | 1200M | 372ms | 60.2 |
| GASKN | 33M | 12ms | 55.7 |
| **Uni-Retrieval** | 453M (+26M) | 77ms (+9ms) | **83.2** (+28.6) |

多风格混合查询:

| 方法 | T→I | T+S→I | I→T | I+S→T |
|------|-----|-------|-----|-------|
| CLIP-Finetune | 54.6 | 55.3 (+0.7) | 47.4 | 46.6 (-0.8) |
| VPT | 69.9 | 72.0 (+2.1) | 73.9 | 74.1 (+0.2) |
| **Uni-Retrieval** | 83.2 | **87.4 (+4.2)** | 81.7 | **83.3 (+1.6)** |

### 关键发现

- Uni-Retrieval 在所有检索风格上全面超越所有基线，Sketch→Image R@1 达 84.5%（比 FreestyleRet 高 9.2 个百分点），Low-Res→Image R@1 达 87.4%
- 仅增加 26M 参数和 9ms 推理时间，但性能提升巨大（Text→Image +28.6%），极其高效
- 多风格混合查询时 Uni-Retrieval 获得更大增益（+4.2 vs CLIP-Finetune +0.7），说明 Prompt Bank 的组合能力确实有效
- 在传统检索数据集上也有竞争力，不局限于 STEM 教育场景

## 亮点与洞察

- **Prompt Bank 的可组合性设计很巧妙**：将风格信息解耦存储，未知风格可通过线性组合已知风格的 token 来表示，具有天然的泛化能力。这个设计思路可以迁移到其他需要风格/领域适配的任务
- **冻结 backbone + 只训练 Prompt Bank 的极致效率**：仅增加约 6% 的参数就获得了 50%+ 的性能提升，是非常好的参数效率范例
- **SER 数据集的构建值得参考**：6000 原始图像 × 6 种风格（文本、语音、草图、艺术、低分辨率）× 22+ STEM 学科，构建流程（人工+AIGC）兼顾质量和多样性

## 局限与展望

- SER 数据集虽然覆盖 22+ 学科，但每个学科的样本量可能不均衡
- 音频查询通过 GPT-4o 转文本处理，属于间接方案，可探索端到端的音频-图像检索
- 原型学习依赖预训练编码器的质量，对于非常特殊的 STEM 图表（如电路图、化学结构式），可能需要专用编码器
- 未探索 Prompt Bank 中 token 数量 $N$ 和 top-$n$ 检索数的敏感性分析
- 实际教学场景中的查询可能更加模糊和不完整，需要更鲁棒的设计

## 相关工作与启发

- **vs FreestyleRet**: 目前多风格检索最强基线，本文在 SER 上全面超越，关键区别在于 Prompt Bank 提供了持续更新和组合的能力
- **vs CoCoOP / MaPLe**: 基于 prompt learning 的方法，但它们的 prompt 是全局共享的，不能根据输入风格动态选择。Uni-Retrieval 的 Prompt Bank 是查询相关的
- **vs VPT**: Visual Prompt Tuning 也在视觉 token 中插入可学习 prompt，但没有风格条件化的选择机制
- 教育AI领域越来越多样化的检索需求为多模态检索研究提供了新的应用驱动

## 评分

- 新颖性: ⭐⭐⭐⭐ Prompt Bank + 原型匹配的动态 prompt 选择是新颖的设计，新任务和数据集也有贡献
- 实验充分度: ⭐⭐⭐ 主实验覆盖面广但缺少一些重要的消融（Prompt Bank 大小、top-n 的影响等）
- 写作质量: ⭐⭐⭐ 整体清晰但有些地方描述不够精炼，部分公式符号不一致
- 价值: ⭐⭐⭐⭐ 新数据集+新任务定义对教育AI和多风格检索都有推动作用

<!-- RELATED:START -->

## 相关论文

- [ACORD: An Expert-Annotated Retrieval Dataset for Legal Contract Clause Retrieval](acord_an_expert-annotated_retrieval_dataset_for_legal_contract_drafting.md)
- [Towards Text-Image Interleaved Retrieval](towards_text-image_interleaved_retrieval.md)
- [GeAR: Generation Augmented Retrieval](gear_generation_augmented_retrieval.md)
- [Multi-Facet Blending for Faceted Query-by-Example Retrieval](multi-facet_blending_for_faceted_query-by-example_retrieval.md)
- [HASH-RAG: Bridging Deep Hashing with Retriever for Efficient, Fine Retrieval and Augmented Generation](hash-rag_bridging_deep_hashing_with_retriever_for_efficient_fine_retrieval_and_a.md)

<!-- RELATED:END -->
