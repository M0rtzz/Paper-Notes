---
title: >-
  [论文解读] SVIP: Semantically Contextualized Visual Patches for Zero-Shot Learning
description: >-
  [ICCV 2025][零样本学习] 提出SVIP框架，通过在**输入阶段**识别并替换语义无关的图像patch（用属性级word embedding初始化的可学习嵌入替代），从根源上解决零样本学习中的语义错位问题。
tags:
  - ICCV 2025
  - 零样本学习
  - 语义对齐
  - ViT
  - patch选择
  - 属性定位
---

# SVIP: Semantically Contextualized Visual Patches for Zero-Shot Learning

**会议**: ICCV 2025  
**arXiv**: [2503.10252](https://arxiv.org/abs/2503.10252)  
**代码**: [https://github.com/uqzhichen/SVIP](https://github.com/uqzhichen/SVIP)  
**领域**: Self-Supervised Learning / Zero-Shot Learning  
**关键词**: 零样本学习, 语义对齐, ViT, patch选择, 属性定位

## 一句话总结

提出SVIP框架，通过在**输入阶段**识别并替换语义无关的图像patch（用属性级word embedding初始化的可学习嵌入替代），从根源上解决零样本学习中的语义错位问题。

## 研究背景与动机

零样本学习（ZSL）依赖视觉特征与语义属性的对齐来识别未见类别。核心挑战是**语义错位（semantic misalignment）**：原始图像中包含大量与属性无关的信息（背景杂波、光照变化、周围物体），这些信息稀释了关键属性（颜色、形状等），使模型在未见类上表现差。

现有应对策略的局限：
- **特征空间解耦方法**（如RFF、SDGZSL、FREE）：在提取特征**之后**去除语义无关信息，但此时噪声已融入表征，无法完全消除
- **模型空间渐进剪枝**（如ZSLViT）：在Transformer中逐层剪除无关token，但研究表明语义特征在ViT深层会被稀释，深层剪除可能为时已晚

**本文的核心问题**：能否在**输入阶段**就预判并处理语义无关的patch，让它们从一开始就不进入特征提取流程？

关键观察：ViT中不同patch的注意力权重在各层之间是动态变化的（如图3所示），依赖某一层的注意力来判断语义相关性不可靠。因此需要跨层聚合的全局视角。

## 方法详解

### 整体框架

SVIP包含三个组件：(1) **自监督Patch选择（SSPS）**：通过聚合全层注意力分数训练patch分类器；(2) **Patch语义上下文化（PSC）**：用属性word embedding初始化的可学习嵌入替换被选中的语义无关patch；(3) **属性定位**：从语义相关patch中定位属性值用于分类。

### 关键设计

1. **自监督Patch选择（SSPS）**：

   **注意力矩阵聚合**：设第 $l$ 层的注意力矩阵为 $\mathbf{T}^l$（各头求和），递推聚合：
   $$\mathbf{W}^l = \mathbf{W}^{l-1} + \mathbf{W}^{l-1} \times \mathbf{T}^l, \quad l=1,\cdots,L$$
   取class token对应行获得每个patch的语义分数 $r_i = \mathbf{W}^L_{[0;i]}$，作为伪标签。

   **Patch分类器**：辅助分类器预测每个patch嵌入的语义分数 $\hat{r}_i = \text{PatchCls}(\mathbf{v}_i)$，用二元交叉熵损失训练：
   $$\mathcal{L}_{\text{patch}} = -\frac{1}{N}\sum_{i=1}^{N}[r_i \log \hat{r}_i + (1-r_i)\log(1-\hat{r}_i)]$$

   测试时直接用分类器判断哪些patch是语义无关的，无需再跑完整个Transformer。

2. **Patch语义上下文化（PSC）**：选出Top-M个语义相关patch后，对其余（语义无关）patch不是简单删除（会破坏物体结构），而是**添加**一个可学习嵌入 $\mathbf{e}$：

   $$\hat{\mathbf{v}}_i = \begin{cases} \mathbf{v}_i, & \text{if } i \in \mathcal{S}_M \\ \mathbf{v}_i + \mathbf{e}, & \text{otherwise} \end{cases}$$

   $\mathbf{e}$ 通过Word-to-Patch (W2P) 投影层从 $K$ 个属性的word embedding聚合得到：$\mathbf{e} = \text{W2P}(\mathbf{w}_1, \cdots, \mathbf{w}_K)$。这确保语义无关patch携带属性级语义信息，在后续Transformer中可增强语义-视觉交互。

3. **属性定位**：使用Patch-to-Attribute (P2A) 投影将Top-M个语义相关patch的最终表征映射到属性空间，通过max pooling选出每个属性最相关的patch：
   $$\hat{\mathbf{a}} = \text{MaxPool}(\text{P2A}(\mathbf{Z}^{L'}))$$
   分类通过预测属性向量与类别属性向量的余弦相似度+Softmax完成。

### 损失函数 / 训练策略

模型对每个样本做**两次前向**：原始patch序列 $\mathbf{Z}^0$ 和上下文化后的 $\mathbf{Z}^{0'}$。

总损失：
$$\ell_{\text{overall}} = \ell_{\text{cls}} + \lambda_1 \ell_{\text{JSD}} + \lambda_2 \ell_{\text{patch}}$$

- $\ell_{\text{cls}}$：两次前向各自的交叉熵分类损失之和
- $\ell_{\text{JSD}}$：两次预测分布间的Jensen-Shannon散度（稳定训练）
- $\ell_{\text{patch}}$：patch分类损失

ViT-base骨干（ImageNet-1k预训练），patch数量从196聚合为49（2×2合并）。

## 实验关键数据

### 主实验

| 方法 | 骨干 | CUB T1 | CUB H | AwA2 T1 | AwA2 H | SUN T1 | SUN H |
|------|------|--------|-------|---------|--------|--------|-------|
| MSDN (CVPR'22) | ResNet101 | 76.1 | 68.1 | 70.1 | 67.7 | 65.8 | 41.3 |
| DUET (AAAI'23) | ViT | 72.3 | 67.5 | 69.9 | 72.7 | 64.4 | 45.8 |
| ZSLViT (CVPR'24) | ViT | 78.9 | 73.6 | 70.7 | 74.2 | 68.3 | 47.3 |
| **SVIP (ours)** | **ViT** | **79.8** | **75.0** | **69.8** | **74.9** | **71.6** | **50.7** |

*在三个基准数据集上GZSL的H指标全面SOTA：CUB +1.4, AwA2 +0.7, SUN +3.4。*

### 消融实验

| 方法 | CUB T1 | CUB H | AwA2 T1 | AwA2 H | SUN T1 | SUN H |
|------|--------|-------|---------|--------|--------|-------|
| Baseline (ViT w/ att head) | 76.8 | 63.8 | 61.4 | 67.8 | 62.7 | 36.0 |
| SVIP w/o SSPS | 78.9 | 71.9 | 66.8 | 72.6 | 67.6 | 47.3 |
| SVIP w/o PSC | 78.1 | 72.6 | 67.6 | 72.4 | 67.9 | 48.0 |
| SVIP w/o JSD | 79.5 | 74.9 | 69.1 | 74.5 | 71.2 | 50.4 |
| SVIP w/o W2P | 79.1 | 74.5 | 69.8 | 74.4 | 71.5 | 50.1 |
| **SVIP (full)** | **79.8** | **75.0** | **69.8** | **74.9** | **71.6** | **50.7** |

*SSPS和PSC是最关键组件：去掉SSPS后CUB H下降3.1，SUN H下降3.4；W2P（用word embedding初始化）比随机初始化更有效。*

### 关键发现

- 保留patch数M=40时性能最佳（总49个patch中），过多剪除会误删语义相关patch
- JSD散度是敏感超参，最优值为1；温度σ最优值为5
- t-SNE可视化显示SVIP的属性向量聚类更紧凑、类间分离更清晰
- Patch分类器的中间层特征显示语义无关patch自然聚集在特征空间的特定区域

## 亮点与洞察

- **"预防优于治疗"理念**：在输入端处理语义噪声比在特征空间后处理更彻底，这一思路在ZSL领域是首创
- **不删除而替换**：用语义嵌入替代无关patch避免破坏物体结构，且让这些位置成为"语义增强通道"
- **跨层注意力聚合的自监督**：巧妙利用ViT自身的注意力作为free supervision，无需额外标注
- **两次前向 + JSD稳定训练**的设计保证了上下文化patch和原始patch的一致性

## 局限与展望

- 两次前向推理增加了训练时间
- Patch选择阈值M需要手动调整，不同数据集可能需要不同值
- 仅在embedding级（ViT-base）验证，未探索更大模型或CLIP等更强基础模型
- Word embedding初始化依赖GloVe质量，对稀有属性名可能不够好
- 未探索动态M值（不同图像保留不同数量的patch）

## 相关工作与启发

- 与ZSLViT的模型空间渐进剪枝形成互补：SVIP关注输入端，ZSLViT关注中间层
- 自监督patch选择的思路可迁移到其他需要输入过滤的视觉任务（如细粒度识别、目标检测）
- PSC中将语义信息注入视觉token的方式类似于visual prompt tuning，但目的不同

## 评分

- 新颖性: ⭐⭐⭐⭐ (输入端处理语义错位+可学习语义patch的思路新颖)
- 实验充分度: ⭐⭐⭐⭐ (三个标准数据集+充分消融+超参敏感性分析)
- 写作质量: ⭐⭐⭐⭐ (方法描述清晰，伪代码有帮助)
- 价值: ⭐⭐⭐⭐ (ZSL领域的有效改进，思路可推广)
