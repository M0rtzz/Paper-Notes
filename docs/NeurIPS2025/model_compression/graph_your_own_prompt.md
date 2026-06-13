---
title: >-
  [论文解读] Graph Your Own Prompt
description: >-
  [NeurIPS 2025][模型压缩][图正则化] 提出图一致性正则化（GCR）框架，通过在网络任意深度插入无参数的图一致性层（GCL），将中间特征的关系图与基于预测的类感知语义图对齐，以自我提示的方式促进语义一致的特征学习，在不修改架构和不增加参数的前提下提升分类泛化性能。 1. 深层特征缺乏语义对齐：深度网络虽然学到丰…
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "图正则化"
  - "特征对齐"
  - "语义一致性"
  - "分类"
  - "无参模块"
---

# Graph Your Own Prompt

**会议**: NeurIPS 2025  
**arXiv**: [2509.23373](https://arxiv.org/abs/2509.23373)  
**代码**: 待确认（论文提及 Project website 和 Code 链接）  
**领域**: 模型压缩  
**关键词**: 图正则化, 特征对齐, 语义一致性, 分类, 无参模块  

## 一句话总结

提出图一致性正则化（GCR）框架，通过在网络任意深度插入无参数的图一致性层（GCL），将中间特征的关系图与基于预测的类感知语义图对齐，以自我提示的方式促进语义一致的特征学习，在不修改架构和不增加参数的前提下提升分类泛化性能。

## 背景与动机

1. **深层特征缺乏语义对齐**：深度网络虽然学到丰富表示，但中间特征常捕获与模型预测语义矛盾的噪声类间相似性，不同类样本在特征空间中可能距离很近。
2. **对比学习需要显式采样**：已有对比学习和图正则化方法需要正/负样本采样策略或精心设计的数据增强，且通常只在网络的单一层面操作。
3. **预测信息未被充分利用**：网络自身的 softmax 预测蕴含丰富的语义关系信息（同类样本预测应相似），但现有方法很少将此结构反馈到特征学习中。
4. **缺乏多层结构监督**：现有中间层辅助损失通常独立监督各层，很少利用模型输出的结构化信息来正则化特征学习。
5. **轻量化需求**：实际部署中希望正则化方法不引入额外参数、不修改网络架构、不改变训练流程。
6. **图结构的自然性**：样本间的关系天然适合用图来建模，特征相似性图和预测相似性图的对齐提供了一种优雅的跨空间结构约束。

## 方法详解

### 图一致性层 (GCL)

**特征关系图构建**：对第 $l$ 层的特征矩阵 $X^{(l)} \in \mathbb{R}^{n \times d}$（$n$ 为 batch size），用 ReLU + 余弦相似度构建成对关系图：

$$F_{ij}^{(l)} = \text{ReLU}(\cos(x_i^{(l)}, x_j^{(l)}))$$

**掩码预测关系图构建**：从网络最后一层的 logits $Z \in \mathbb{R}^{n \times C}$ 计算 softmax 后的预测相似性矩阵 $S$，再用类内二值掩码 $M$（同类标签为 1，异类为 0）进行过滤：

$$P_{ij} = M_{ij} \odot S_{ij}$$

掩码的作用是：(1) 过滤训练早期不可靠的跨类预测相似性；(2) 聚焦于类内语义关系，避免视觉相似但语义不同类别的干扰。

### 图一致性正则化 (GCR)

**逐层对齐损失**：取特征图和预测图的严格上三角部分（消除自连接和重复计数），计算 Frobenius 范数：

$$\mathcal{L}_{\text{GCR}}^{(l)} = \|\text{triu}(F^{(l)}) - \text{triu}(P)\|_F^2$$

**多层聚合**：在 $K$ 个插入点收集对齐损失并加权求和：

$$\mathcal{L}_{\text{GCR}} = \sum_{l=1}^{K} w_l \cdot \|\text{triu}(F^{(l)}) - \text{triu}(P)\|_F^2$$

**自适应权重**：基于图不一致程度的 softmax 权重，使不一致更大的层获得更高权重：

$$w_l = \frac{\exp(-\|\text{triu}(F^{(l)}) - \text{triu}(P)\|_F^2)}{\sum_{j=1}^{K} \exp(-\|\text{triu}(F^{(j)}) - \text{triu}(P)\|_F^2)}$$

**总训练目标**：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{GCR}}$，实验中 $\lambda = 1$。

### GCL 插入策略

将网络分为 Early (E)、Mid (M)、Late (L) 三阶段，评估 7 种配置：E、M、L、E+M、M+L、E+L、Full。实验发现 Late GCL 通常效果最佳，因为后层特征语义更丰富、与决策边界更接近。

### 理论分析

1. **泛化界**：通过 covering number 和 Dudley 熵积分证明 GCR 约束缩小了有效假设类复杂度，降低泛化误差。
2. **谱对齐**：证明特征图和预测图的 Frobenius 距离趋近时，它们的归一化 Laplacian 谱也趋近，保证聚类结构一致。
3. **PAC-Bayes 视角**：GCR 约束等价于对函数空间施加结构先验，KL 散度上界与 $\sum_l \|F^{(l)} - P\|_F^2$ 成正比。

## 实验关键数据

### CIFAR-10 分类精度 (%, 11 种架构平均)

| 配置 | MAE | MobileNet | ShuffleNet | GoogLeNet | ResNet-50 | DenseNet-121 | 平均 |
|------|-----|-----------|------------|-----------|-----------|--------------|------|
| Baseline | 88.95 | 90.23 | 91.21 | 94.10 | 95.03 | 95.01 | 93.32 |
| Late GCL | 89.70 | 91.40 | 92.36 | 94.88 | 95.66 | 95.72 | **94.07** |
| 提升 | +0.75 | +1.17 | +1.15 | +0.78 | +0.63 | +0.71 | **+0.75** |

### CIFAR-100 分类精度 (%, 9 种架构平均)

| 配置 | MAE | MobileNet | ResNeXt-50 | ResNet-50 | DenseNet-121 | 平均 |
|------|-----|-----------|------------|-----------|--------------|------|
| Baseline | 64.29 | 65.95 | 77.75 | 77.31 | 77.09 | 72.95 |
| Late GCL | **65.54** | **68.32** | **79.54** | **79.42** | **79.69** | **74.74** |
| 提升 | +1.25 | +2.37 | +1.79 | +2.11 | +2.60 | **+1.79** |

### ImageNet-1K 分类精度 (%, Transformer 架构)

| 方法 | iFormer-S | iFormer-B | ViT-B/16 | ViG-B |
|------|-----------|-----------|----------|-------|
| Baseline | 83.4 | 84.6 | 74.3 | 82.3 |
| Late GCL | **84.5** | **86.1** | **75.8** | **84.0** |
| 提升 | +1.1 | **+1.5** | **+1.5** | +1.7 |

## 亮点

1. **零参数开销**：GCL 完全无参数、不修改网络架构、不改变训练流程，仅增加轻量级矩阵运算。
2. **通用性极强**：在轻量 CNN（MobileNet/ShuffleNet）、深层 CNN（ResNet/DenseNet）和 Transformer（ViT/Swin/iFormer）上均有效。
3. **自我提示的优雅设计**：用模型自身的预测结构作为特征学习的参考信号，形成一种无需外部监督的"自提示"机制。
4. **可解释性提升**：GCL 增强的特征图更聚焦于类判别性区域（如猫的眼耳、狗的嘴鼻），从 98.1% 提升到 99.8% 准确率。
5. **完整理论支撑**：从 covering number、谱图理论和 PAC-Bayes 三个角度建立了泛化保证。

## 局限与展望

1. **仅验证分类任务**：论文只在图像分类上验证，未扩展到分割、检索、目标检测等任务。
2. **依赖标签构建掩码**：掩码 $M$ 需要 ground truth 标签，不适用于自监督或无标签场景。
3. **batch 内关系的局限**：图构建限于当前 batch 内的样本，对类别不均衡或 batch 中类别覆盖不足的情况可能效果有限。
4. **$\lambda$ 超参数固定为 1**：虽然简化了调参，但不同任务和架构可能需要不同的正则化强度。
5. **Late GCL 通常最优但非普适**：最佳插入位置因架构而异，缺乏自动选择机制。
6. **与知识蒸馏的关系未明确**：预测图引导特征学习与自蒸馏有相似之处，但未与此类方法对比。

## 与相关工作的对比

- **vs 对比学习（SimCLR/SupCon）**：对比学习需正负样本采样和数据增强，通常在单一层面操作；GCR 无需采样，支持多层对齐。
- **vs 图神经网络**：GNN 需维护全局图或额外消息传递模块；GCR 在 batch 内动态构图，无架构开销。
- **vs Center Loss/Triplet Loss**：这些方法强制类间距离或类内紧凑性但在单一表示空间操作；GCR 跨特征和预测两个空间进行结构对齐。
- **vs 知识蒸馏**：蒸馏需要教师模型；GCR 使用模型自身的预测，属于自蒸馏的一种形式但更轻量。
- **vs 注意力机制**：GCL 的特征聚焦效果类似注意力，但完全无参数且通过图对齐实现。

## 评分

- 新颖性: ⭐⭐⭐⭐ 用预测图引导特征图对齐的"自提示"思路新颖，多层自适应加权设计优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 4 数据集 16+ 架构，含可视化/消融/权重方案对比/理论分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，理论部分完整，可视化丰富直观
- 价值: ⭐⭐⭐⭐ 零参数即插即用的正则化方法，实用性强，但需拓展到更多任务验证通用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Graver: Generative Graph Vocabularies for Robust Graph Foundation Models Fine-tuning](graver_generative_graph_vocabularies_for_robust_graph_foundation_models_fine-tun.md)
- [\[NeurIPS 2025\] Towards Effective Federated Graph Foundation Model via Mitigating Knowledge Entanglement](towards_effective_federated_graph_foundation_model_via_mitigating_knowledge_enta.md)
- [\[NeurIPS 2025\] Find your Needle: Small Object Image Retrieval via Multi-Object Attention Optimization](find_your_needle_small_object_image_retrieval_via_multi-object_attention_optimiz.md)
- [\[NeurIPS 2025\] All You Need is One: Capsule Prompt Tuning with a Single Vector](all_you_need_is_one_capsule_prompt_tuning_with_a_single_vector.md)
- [\[ICML 2026\] Mind Your Margin and Boundary: Are Your Distilled Datasets Truly Robust?](../../ICML2026/model_compression/mind_your_margin_and_boundary_are_your_distilled_datasets_truly_robust.md)

</div>

<!-- RELATED:END -->
