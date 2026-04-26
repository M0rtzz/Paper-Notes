---
title: >-
  [论文解读] Instance-Level Composed Image Retrieval
description: >-
  [NeurIPS 2025][图像生成][组合图像检索] 提出实例级组合图像检索（i-CIR）基准和训练免费方法BASIC，通过独立估计图像和文本查询的相似度并进行乘法融合，在无需训练的情况下在i-CIR和现有CIR数据集上均达到SOTA。
tags:
  - NeurIPS 2025
  - 图像生成
  - 组合图像检索
  - 实例级检索
  - VLM
  - 训练免费
  - 特征融合
---

# Instance-Level Composed Image Retrieval

**会议**: NeurIPS 2025  
**arXiv**: [2510.25387](https://arxiv.org/abs/2510.25387)  
**代码**: [GitHub](https://github.com/billpsomas/icir) | [项目页](https://vrg.fel.cvut.cz/icir/)  
**领域**: 图像检索/多模态  
**关键词**: 组合图像检索, 实例级检索, VLM, 训练免费, 特征融合  

## 一句话总结

提出实例级组合图像检索（i-CIR）基准和训练免费方法BASIC，通过独立估计图像和文本查询的相似度并进行乘法融合，在无需训练的情况下在i-CIR和现有CIR数据集上均达到SOTA。

## 研究背景与动机

### 组合图像检索(CIR)

组合图像检索是图像检索领域的热门方向：给定一张参考图像和一段文本修改描述（如"换成红色"），检索满足两个条件的目标图像。

现有CIR研究面临两个核心瓶颈：

**数据质量不足**：现有CIR数据集多为语义级（如FashionIQ、CIRR），检索目标是同类别但不同实例的图像——这与真实需求（找到同一个物体在不同条件下的图像）不同

**训练数据稀缺**：高质量的CIR训练样本难以大规模获取，限制了有监督方法的性能

### 实例级 vs 语义级

| 维度 | 语义级CIR | 实例级CIR (i-CIR) |
|------|-----------|-------------------|
| 目标 | 同类别的其他图像 | 同一个特定物体 |
| 示例 | "类似的红色连衣裙" | "同一件红色连衣裙在户外" |
| 难度 | 类别内区分 | 实例级区分+条件匹配 |
| 应用 | 购物推荐 | 地标识别、物品追踪 |

实例级定义更贴近实际需求，但也更具挑战性。

## 方法详解

### 整体框架

本文贡献两部分：
1. **i-CIR数据集**：首个实例级CIR评测基准
2. **BASIC方法**：训练免费的CIR方法，利用预训练VLM的冻结特征

### 关键设计

**1. i-CIR数据集构建**

数据集的精心设计是本文的重要贡献：

- **202个物体实例**：涵盖地标建筑、消费品、虚构角色、科技设备等多样类别
- **1,883个组合查询**：每个查询由实例图像+文本修改组成（触及外观、环境、属性、视角等维度的变化）
- **750K数据库图像**：包括正样本和精心筛选的困难负样本
- **困难负样本设计**：三种类型——
    - 视觉困难负样本：视觉相似但不是同一实例
    - 文本困难负样本：文本语义匹配但图像实例不同
    - 组合困难负样本：接近满足两个条件但实际不满足
- **紧凑但困难**：虽然数据库仅750K，但通过困难负样本使其难度相当于在40M干扰图中检索

**2. BASIC方法**

BASIC（Baseline Approach for Surprisingly strong Composition）是一种训练免费方法，核心思路是分别处理图像和文本查询，然后进行融合：

**步骤一：特征标准化（Feature Standardization）**
- 使用LAION-1M数据集预计算的均值对VLM特征进行中心化
- 消除全局偏置，使特征更具判别性

**步骤二：对比PCA投影（Contrastive PCA Projection）**
- 使用正语料库（物体描述）和负语料库（风格描述）构建对比特征空间
- 通过PCA投影将图像特征投射到"物体"子空间，抑制背景和风格信息
- 公式：$\mathbf{f}' = \text{PCA}_{C^+, C^-}(\mathbf{f}, \alpha)$，其中 $\alpha$ 控制负语料库权重

**步骤三：查询扩展（Query Expansion）**
- 用参考图像检索top-k最相似的数据库图像
- 将这些图像的特征平均后作为扩展查询，增强图像查询的鲁棒性

**步骤四：查询文本上下文化（Query Conditioning）**
- 将短文本修改补全为类似CLIP训练时的caption格式
- 添加语料库中的物体名称作为上下文，稳定文本表示

**步骤五：Harris角点融合（Harris Corner Fusion）**
- 独立计算图像相似度 $s_I$ 和文本相似度 $s_T$
- 使用归一化min-based缩放，再通过Harris角点检测启发的惩罚项融合：

$$s = s_I \cdot s_T - \lambda \cdot (s_I - s_T)^2$$

- 逻辑：奖励同时满足两个查询的候选（AND逻辑），惩罚仅在单一模态上得分高的候选

### 损失函数 / 训练策略

- **无需训练**：BASIC完全基于冻结的CLIP/SigLIP特征，所有操作都是查询时在线计算
- 无可学习参数，无需反向传播
- 支持CLIP ViT-L/14和SigLIP ViT-L-16作为backbone

## 实验关键数据

### 主实验：i-CIR基准

各方法在i-CIR上的mAP(%)对比：

| 方法 | 类型 | Legacy宏mAP | Refined宏mAP | 平均 |
|------|------|-------------|-------------|------|
| Text | 单模态 | 0.74 | 1.09 | 0.92 |
| Image | 单模态 | 3.84 | 6.32 | 5.08 |
| Text + Image（加法） | 基线 | 6.21 | 9.30 | 7.76 |
| Text × Image（乘法） | 基线 | 7.83 | 9.79 | 8.81 |
| CIReVL | Training-free | 18.11 | 17.80 | 17.96 |
| FREEDOM | Trained | 29.91 | 26.10 | 28.01 |
| CoVR | Trained | 11.52 | 24.93 | 18.23 |
| **BASIC** | **Training-free** | **32.13** | **31.65** | **31.89** |

BASIC超越了所有方法，包括需要训练的FREEDOM，同时是完全训练免费的。

### 现有语义级CIR数据集的对比

BASIC在传统CIR数据集上同样表现优异：

| 方法 | 类型 | FashionIQ (R@10) | CIRR (R@1) | GeneCIS |
|------|------|-----------------|------------|---------|
| Pic2Word | ZS | 26.2 | 23.9 | — |
| Searle | ZS | 24.2 | 24.2 | — |
| CIReVL | ZS | 25.0 | 24.6 | — |
| MagicLens | Trained | 29.1 | 28.3 | — |
| **BASIC** | **ZS** | **31.8** | **29.7** | **SOTA** |

BASIC在训练免费设定下超越了有监督方法。

### 消融实验

**各组件的贡献（i-CIR宏mAP%）**：

| 配置 | CLIP mAP | SigLIP mAP |
|------|----------|------------|
| 朴素乘法融合 | 7.83 | 9.86 |
| + 特征标准化 | 14.2 | 15.8 |
| + 对比PCA投影 | 22.5 | 24.1 |
| + 查询扩展 | 28.7 | 30.2 |
| + 文本上下文化 | 30.1 | 31.3 |
| + Harris融合 (Full BASIC) | 32.1 | 31.6 |

每个组件都带来稳定提升，其中对比PCA投影和查询扩展贡献最大。

**融合权重 $\lambda$ 的影响**：

| $\lambda$ | i-CIR mAP | 说明 |
|-----------|-----------|------|
| 0.0 | 28.3 | 纯乘法融合 |
| 0.05 | 30.5 | 轻微惩罚 |
| 0.1 | 32.1 | 最优 |
| 0.2 | 31.4 | 过度惩罚 |
| 0.5 | 29.1 | 惩罚过强 |

### 关键发现

1. **实例级CIR真正需要组合**：i-CIR上性能峰值出现在图文融合权重的中间值，证明两个模态都必须参与
2. **训练免费方法可以超越有监督**：BASIC的成功表明VLM的冻结特征蕴含了丰富的组合检索能力
3. **特征空间的几何操作非常有效**：标准化、投影、扩展这些简单的几何操作比复杂的学习方法更鲁棒
4. **困难负样本是评测的关键**：i-CIR通过困难负样本使750K数据库的难度等效于40M规模

## 亮点与洞察

1. **问题定义的推进**：从语义级到实例级CIR，更贴近真实检索需求
2. **BASIC的优雅简洁**：没有神经网络训练、没有复杂pipline，仅靠冻结特征的几何操作就达到SOTA
3. **Harris角点融合的巧妙借鉴**：将计算机视觉中经典的角点检测思想引入特征融合，奖励"两个方向都强"的候选
4. **数据集设计的精心**：困难负样本的三重设计（视觉/文本/组合）确保了评测的有效性
5. **完全开源**：数据集、代码、评测工具一应俱全

## 局限与展望

1. **数据集规模有限**：202个实例可能不足以覆盖所有检索场景
2. **依赖VLM质量**：BASIC的效果上限受限于底层VLM（CLIP/SigLIP）的表示能力
3. **查询扩展增加开销**：需要在推理时进行一次额外检索，增加延迟
4. **PCA的语料库敏感性**：正负语料库的选择可能影响不同领域的效果
5. **未探索与训练方法的结合**：BASIC的组件是否可以作为有监督CIR方法的初始化/增强？

## 相关工作与启发

- **CIRR / FashionIQ**：经典CIR数据集，但仅限语义级，i-CIR填补了实例级空白
- **CIReVL / Pic2Word**：训练免费CIR方法的先驱，BASIC在此基础上大幅推进
- **FREEDOM / CoVR**：有监督CIR方法，BASIC证明训练免费方法可以超越它们
- **Contrastive PCA**：从NLP的对比解码思想迁移到视觉特征空间，非常有效
- 启发：**冻结特征的几何操作是一种被低估的工具**——在设计新方法前，应先充分挖掘预训练特征的潜力

## 评分

- 新颖性：⭐⭐⭐⭐（新数据集定义+优雅的方法设计）
- 技术深度：⭐⭐⭐⭐
- 实验充分度：⭐⭐⭐⭐⭐
- 实用性：⭐⭐⭐⭐⭐（完全开源+无需训练）
- 写作质量：⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] ILIAS: Instance-Level Image Retrieval At Scale](../../CVPR2025/image_generation/ilias_instance-level_image_retrieval_at_scale.md)
- [\[NeurIPS 2025\] GenIR: Generative Visual Feedback for Mental Image Retrieval](genir_generative_visual_feedback_for_mental_image_retrieval.md)
- [\[NeurIPS 2025\] Highlighting What Matters: Promptable Embeddings for Attribute-Focused Image Retrieval](highlighting_what_matters_promptable_embeddings_for_attribute-focused_image_retr.md)
- [\[NeurIPS 2025\] ImageSentinel: Protecting Visual Datasets from Unauthorized Retrieval-Augmented Image Generation](imagesentinel_protecting_visual_datasets_from_unauthorized_retrieval-augmented_i.md)
- [\[NeurIPS 2025\] Aligning Compound AI Systems via System-level DPO](aligning_compound_ai_systems_via_system-level_dpo.md)

<!-- RELATED:END -->
