---
title: >-
  [论文解读] Local Dense Logit Relations for Enhanced Knowledge Distillation
description: >-
  [ICCV 2025][模型压缩][知识蒸馏] 本文提出局部稠密关系 logit 蒸馏（LDRLD），通过递归解耦和重组 logit 知识来捕获细粒度的类间关系，结合自适应衰减权重（ADW）策略对关键类别对赋予更高权重，在 CIFAR-100、ImageNet-1K 和 Tiny-ImageNet 上持续优于现有 logit 蒸馏 SOTA。
tags:
  - ICCV 2025
  - 模型压缩
  - 知识蒸馏
  - logit 蒸馏
  - 类别对关系
  - 自适应衰减权重
  - 细粒度知识迁移
---

# Local Dense Logit Relations for Enhanced Knowledge Distillation

**会议**: ICCV 2025  
**arXiv**: [2507.15911](https://arxiv.org/abs/2507.15911)  
**代码**: 无（论文承诺公开）  
**领域**: 模型压缩  
**关键词**: 知识蒸馏, logit 蒸馏, 类别对关系, 自适应衰减权重, 细粒度知识迁移

## 一句话总结
本文提出局部稠密关系 logit 蒸馏（LDRLD），通过递归解耦和重组 logit 知识来捕获细粒度的类间关系，结合自适应衰减权重（ADW）策略对关键类别对赋予更高权重，在 CIFAR-100、ImageNet-1K 和 Tiny-ImageNet 上持续优于现有 logit 蒸馏 SOTA。

## 研究背景与动机
知识蒸馏（KD）将大规模教师模型的"暗知识"迁移到轻量化学生模型中。logit-based KD 方法因其计算效率和通用性而备受关注。经典 KD 使用全局 softmax 计算 KL 散度来传递教师的输出概率分布。

然而，现有 logit 蒸馏方法存在对细粒度类间关系建模不足的问题：

**全局 softmax 弱化了低概率类间差异**：softmax 对高概率类聚焦，压缩了低概率类之间的概率差异，限制了学生捕获它们之间的细粒度关系

**无关类引入信息冗余**：当建模"猫"和"狗"的关系时，全局 softmax 引入了"汽车"等无关类的概率，干扰并削弱了目标类对的判别性

**均匀权重忽略类对重要性差异**：现有方法对所有类对赋予相同权重，但语义相近的类对（如"猫"vs"狗"）应比语义远离的类对（如"猫"vs"飞机"）获得更多关注

本文的切入角度是：将 logit 知识从全局分布分解为局部的类别对关系，通过局部 softmax 增强每个类对的判别性，同时用基于排名的自适应衰减策略聚焦关键类对。可以直觉理解为：全局 KD 中"猫"和"狗"的概率差 $\Delta P_{KD} = |p_1 - p_2| = |\frac{e^{Z_1} - e^{Z_2}}{\sum_{i=1}^C e^{Z_i}}|$，而 LDRLD 中为 $\Delta P_{LDRLD} = |\frac{e^{Z_1} - e^{Z_2}}{\sum_{i=1}^2 e^{Z_i}}|$，显然后者更大。

## 方法详解

### 整体框架
LDRLD 包含三个损失项：(1) 加权局部稠密关系蒸馏 $\mathcal{L}^w$——通过类别对的 KL 散度传递细粒度知识，带 ADW 权重；(2) 剩余非目标知识蒸馏 $\mathcal{L}_{RNTK}$——确保递归解耦后的剩余知识不被丢失；(3) 局部 logit 知识完整性 $\mathcal{L}_{LLKI}$——在 top-d 类上做完整的局部 KL 蒸馏。总损失为 $\mathcal{L}_{LDRLD} = \mathcal{L}_{Task} + \alpha \mathcal{L}_{Local} + \beta \mathcal{L}_{RNTK}$。

### 关键设计
1. **递归解耦与重组（LDRLD 核心）**

    - 功能：从学生 logit 输出中递归提取 top-d 类，两两组合形成类别对
    - 核心思路：
        - Step 1：提取学生 logit 最大的类 $\mathbf{Z}_1^s$，用掩码 $\mathbf{M}_{\pi(1)}$（将对应位置设为 $-\infty$）从 logit 向量中移除
        - Step 2：递归提取第 2, 3, ..., d 大的类，每次与所有已提取的类组合为类别对。组合用 $\phi: \mathbb{R} \uplus \mathbb{R} \to \mathbb{R}^{1 \times 2}$
        - 最终生成 $\frac{d(d-1)}{2}$ 个类别对
        - Step 3：对每个类别对独立进行 softmax 归一化，计算 KL 散度
    - 设计动机：局部 softmax 归一化使得每个类对中的概率差异被放大，增强了判别性。类对数量为 $O(d^2)$，提供了密集的关系信息
    - 基础损失：$\mathcal{L} = \sum_{i=1}^{d-1}\sum_{j=i+1}^{d} [p_i^t \log(p_i^t/p_i^s) + p_j^t \log(p_j^t/p_j^s)]$

2. **自适应衰减权重（ADW）策略**

    - 功能：为不同类别对动态分配不同的蒸馏权重
    - 核心思路：包含两个组件：
        - **逆秩加权（IRW）**：$\Gamma_{IRW}(R', R) = \frac{1}{|R - R'| + \epsilon}$，排名差距小的类对获得更大权重（$\epsilon = 1.50$）
        - **指数秩衰减（ERD）**：$\Phi_{ERD}(R', R) = \delta \times \exp(-\lambda(R + R'))$，排名总和大的类对权重指数衰减（$\delta = 2.0, \lambda = 0.05$）
        - 组合权重：$\Omega_{ADW}(R', R) = \Gamma_{IRW} \times \Phi_{ERD}$
    - 设计动机：
        - IRW 解决了"类对 [Z₁, Z₂] 比 [Z₁, Z₄] 更难区分"的问题
        - ERD 解决了"[Z₁, Z₂] 和 [Z₁₂, Z₁₃] 虽然排名差同为 1，但前者更重要"的问题
    - 加权损失：$\mathcal{L}^w = \sum_{i=1}^{d-1}\sum_{j=i+1}^{d} \Omega_{ADW}(i,j)[p_i^t \log(p_i^t/p_i^s) + p_j^t \log(p_j^t/p_j^s)]$

3. **剩余非目标知识蒸馏（RNTK）**

    - 功能：对递归解耦中未被选中的 $C - d$ 个类进行整体蒸馏
    - 核心思路：$\mathcal{L}_{RNTK} = \mathcal{KL}(\bar{\mathcal{H}}^t, \bar{\mathcal{H}}^s) = \sum_{i=d+1}^C \bar{p}_i^t \log(\bar{p}_i^t / \bar{p}_i^s)$
    - 设计动机：确保知识的完整性，避免因只关注 top-d 类而丢失尾部类的暗知识

### 损失函数 / 训练策略
总优化目标：
$$\mathcal{L}_{LDRLD} = \mathcal{L}_{Task} + \alpha \mathcal{L}_{Local} + \beta \mathcal{L}_{RNTK}$$

其中 $\mathcal{L}_{Local} = \mathcal{L}^w + \mathcal{L}_{LLKI}$，$\alpha$ 和 $\beta$ 为平衡系数。

## 实验关键数据

### 主实验
CIFAR-100 同架构蒸馏：

| 教师→学生 | KD | DKD | NKD | WTTM | **LDRLD** | Δ vs KD |
|-----------|-----|-----|-----|------|-----------|---------|
| ResNet56→ResNet20 | 70.66 | 71.97 | 71.18 | 71.92 | **72.20** | +1.54 |
| ResNet110→ResNet20 | 70.67 | 70.91 | 71.26 | 71.67 | **71.98** | +1.31 |
| ResNet32×4→ResNet8×4 | 73.33 | 76.32 | 76.35 | 76.06 | **77.20** | +3.87 |
| WRN-40-2→WRN-16-2 | 74.92 | 76.24 | 75.43 | 76.37 | **76.35** | +1.43 |
| VGG13→VGG8 | 72.98 | 74.68 | 74.86 | 74.44 | **75.06** | +2.08 |

ImageNet-1K：

| 教师→学生 | KD | DKD | WTTM | **LDRLD** | Δ vs KD |
|-----------|-----|-----|------|-----------|---------|
| ResNet34→ResNet18 Top-1 | 70.66 | 71.70 | 72.19 | **71.88** | +1.22 |
| ResNet50→MobileNetV1 Top-1 | 70.49 | 72.05 | 73.09 | **73.12** | +2.63 |

### 消融实验
CIFAR-100 异架构蒸馏（ResNet32×4→ShuffleNetV1）：

| 方法 | Top-1 Acc | 说明 |
|------|----------|------|
| KD | 74.07 | 基线 |
| DKD | 76.45 | 解耦 target/non-target |
| NKD | 76.31 | 归一化 non-target |
| FCFD | 78.12 | 特征蒸馏 |
| CAT-KD | 78.26 | 类激活迁移 |
| **LDRLD** | **76.46** | ADW + LDRLD + RNTK |

### 关键发现
- **在 logit-based 方法中全面领先**：LDRLD 在所有同架构对和大部分异架构对上超过 DKD、NKD、WTTM 等方法
- **大精度差架构对提升最大**：ResNet32×4→ResNet8×4 提升 +3.87%，说明细粒度关系知识在能力差距大时更有价值
- **ImageNet-1K 大规模验证有效**：在 ResNet50→MobileNetV1 上 +2.63%，表明方法不仅适用于小数据集
- **ADW 策略的关键作用**：IRW 和 ERD 的组合确保了对关键类对的聚焦，同时避免了对低排名类对的过度关注
- **方法在 logit 方法中竞争力强**，但相比部分 feature-based 方法（如 FCFD、CAT-KD）仍有差距

## 亮点与洞察
- **局部 softmax 增强判别性**：通过将全局概率分解为两两局部概率，自然地放大了相似类间的概率差异，这个观察简洁而有力
- **ADW 的双层权重设计**：IRW 基于排名差（类间距离），ERD 基于排名和（绝对位置），两者从不同角度刻画了类对的重要性
- **递归解耦的信息密度**：$d$ 个类产生 $\frac{d(d-1)}{2}$ 个类对，提供了远比单一 KL 散度丰富的关系信息
- **知识完整性保障**：RNTK 损失确保未被选中的尾部类知识不被遗忘，体现了设计的周全性

## 局限与展望
- 递归深度 $d$ 的选择需要调优，过大会引入不必要的低排名类对
- 计算复杂度为 $O(d^2)$，当 $d$ 较大时可能带来额外开销
- ADW 中的超参数（$\epsilon = 1.50, \delta = 2.0, \lambda = 0.05$）为手动设定，自适应学习可能更优
- 仅在分类任务上验证，未扩展到检测、分割等任务
- 与 feature-based 方法（如 FCFD 的 78.12%）相比仍有差距，说明 logit 层信息的天花板

## 相关工作与启发
- DKD 将 logit 知识解耦为 target 和 non-target，LDRLD 进一步将 non-target 知识递归分解为密集的类对关系，更深层次
- ADW 的权重衰减思路可借鉴到注意力机制的设计中
- 局部 softmax 的判别性增强可能对对比学习中的负样本加权有启发

## 评分
- 新颖性: ⭐⭐⭐⭐ 递归解耦+局部 softmax 的组合是新颖的，ADW 策略设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ CIFAR-100/ImageNet-1K/Tiny-ImageNet，同架构/异架构，大量对比
- 写作质量: ⭐⭐⭐⭐ 图表清晰直观，方法描述严谨但公式较多
- 价值: ⭐⭐⭐⭐ 对 logit 蒸馏领域提供了新的细粒度建模视角，实验提升一致

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs](../../ACL2025/model_compression/sparse_logit_sampling_accelerating_knowledge_distillation_in_llms.md)
- [\[ICCV 2025\] Knowledge Distillation with Refined Logits](knowledge_distillation_with_refined_logits.md)
- [\[ICCV 2025\] A Good Teacher Adapts Their Knowledge for Distillation](a_good_teacher_adapts_their_knowledge_for_distillation.md)
- [\[ICCV 2025\] Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation](perspective-aware_teaching_adapting_knowledge_for_heterogeneous_distillation.md)
- [\[ICCV 2025\] Fuse Before Transfer: Knowledge Fusion for Heterogeneous Distillation](fuse_before_transfer_knowledge_fusion_for_heterogeneous_distillation.md)

<!-- RELATED:END -->
