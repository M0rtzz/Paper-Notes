---
title: >-
  [论文解读] AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios
description: >-
  [CVPR 2025][novel class discovery] 提出 AnomalyNCD，首个基于自监督的工业多类异常分类方法：MEBin 提取主要异常区域 → 掩码引导 ViT 聚焦弱语义异常 → 区域融合策略实现灵活的区域/图像级分类，MVTec AD 上 F1 提升 10.8%，NMI 提升 8.8%。
tags:
  - CVPR 2025
  - novel class discovery
  - anomaly classification
  - MEBin
  - 注意力机制
  - industrial inspection
---

# AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios

**会议**: CVPR 2025  
**arXiv**: [2410.14379](https://arxiv.org/abs/2410.14379)  
**代码**: [GitHub](https://github.com/HUST-SLOW/AnomalyNCD)  
**领域**: 异常检测 / 工业视觉  
**关键词**: novel class discovery, anomaly classification, MEBin, mask-guided attention, industrial inspection

## 一句话总结
提出 AnomalyNCD，首个基于自监督的工业多类异常分类方法：MEBin 提取主要异常区域 → 掩码引导 ViT 聚焦弱语义异常 → 区域融合策略实现灵活的区域/图像级分类，MVTec AD 上 F1 提升 10.8%，NMI 提升 8.8%。

## 研究背景与动机

**领域现状**：工业异常检测已有成熟方法（PatchCore、EfficientAD 等），可定位异常但无法区分细粒度异常类别（如断裂 vs 烧蚀）。下游处理需要识别异常类别，甚至发现新类别。

**现有痛点**：
   - **异常聚类方法**（AC、UniFormaly）：冻结特征提取器无法学习异常特有特征
   - **自然场景 NCD 方法**（UNO、GCD、SimGCD）：假设目标居中于图像，不适用于工业场景
   - **两大障碍**：
     - ❶ 非突出异常：工业异常是局部损伤，不在图像中心
     - ❷ 弱语义异常：工业异常语义弱，ViT 更倾向关注背景而非异常

**核心矛盾**：分类网络（ViT）的注意力天然聚焦于显著目标而非细微异常，标准 NCD pipeline 对工业缺陷完全失效。

**切入角度**：设计 MEBin 将异常从检测结果中隔离 → 裁剪为异常中心子图 → 掩码引导注意力强制 [CLS] token 关注异常区域。

**核心 idea**：异常中心裁剪 + 掩码引导 ViT 注意力 = 让分类网络"看到"弱语义异常。

## 方法详解

### 整体框架
1. 使用异常检测方法（如 MuSc、PatchCore）获取异常概率图
2. MEBin 将概率图二值化，提取异常中心子图
3. 掩码引导 ViT（MGViT）学习异常的判别性特征
4. Teacher-Student 框架生成伪标签进行分类学习
5. 区域融合策略将子图预测合并为图像级分类

### 关键设计

1. **Main Element Binarization (MEBin)**

    - 功能：从异常检测结果中稳定地提取主要异常区域
    - 三步流程：
        - Step 1：确定阈值范围 $[s_{\min}, s_{\max}]$，$s_{\min}$ 为所有异常图的最小异常分数的最大值
        - Step 2：均匀采样 $\mathcal{T}=64$ 个阈值进行二值化
        - Step 3：找出出现最频繁的连通区域数 $\bar{\delta}_i$，选最小阈值完整分割
    - 核心优势：自适应阈值选择，无需验证集，对不同 AD 方法通用
    - 对比 Otsu：Otsu 倾向过检测，尤其在正常图像上

2. **Mask-Guided Vision Transformer (MGViT)**

    - 功能：引导 [CLS] token 注意力聚焦到异常区域
    - 核心思路：在最后 $L_m=9$ 层的自注意力中插入掩码
    - 三种设计比较：
        - (a) 同时在 CLS 和 patch tokens 上加掩码 → 抑制上下文
        - (b) 仅在 patch tokens 上 → 同样抑制上下文
        - **(c) 仅在 CLS token 上**（采用）→ patch tokens 保持全局感受野
    - 掩码注意力：$\text{Attn} = \text{softmax}(\text{concat}(\mathbf{Q}^{cls}\mathbf{K}^\top + \bar{\mathcal{M}}, \mathbf{Q}^{patch}\mathbf{K}^\top))\mathbf{V}$
    - 其中 $\bar{\mathcal{M}}(i) = 0$ 若 $\mathcal{M}(i) > 0.5$，否则 $-\infty$

3. **伪标签校正 (PLC)**

    - 功能：利用异常分数校正过检测区域的伪标签
    - 公式：$\hat{q}_{i,k} \leftarrow w_{i,k}\mathbf{e} + (1-w_{i,k})\hat{q}_{i,k}$，$w_{i,k} = \max(0.5 - s_{i,k}, 0)$
    - 效果：正常类 Recall 提升 14.9%

4. **区域融合策略**

    - 功能：根据子图分类确定图像级别类别
    - 核心思路：面积加权（非简单平均或异常分数加权）
    - 公式：$\alpha_{i,k}^u = \frac{\exp(a_{i,k}^u / \tau_\alpha)}{\sum_k \exp(a_{i,k}^u / \tau_\alpha)}$
    - 动机：过检测区域面积小但异常分数高，面积加权可削弱其影响

### 训练目标
$$\mathcal{L} = \lambda(\mathcal{L}_{rep}^l + \mathcal{L}_{cls}^l) + (1-\lambda)(\mathcal{L}_{rep} + \mathcal{L}_{cls}^u + \mu\mathcal{L}_{reg}^u)$$
- $\mathcal{L}_{rep}^l$：有监督对比学习，$\mathcal{L}_{rep}$：自监督对比学习
- $\mathcal{L}_{cls}^l$：GT标签交叉熵，$\mathcal{L}_{cls}^u$：伪标签交叉熵
- $\mathcal{L}_{reg}^u$：均值熵最大化正则

## 实验关键数据

### 主实验（无监督设置，仅用未标注图像）

| 方法 | MVTec AD NMI↑ | MVTec AD ARI↑ | MVTec AD F1↑ |
|------|-------------|-------------|------------|
| SimGCD | 0.452 | 0.346 | — |
| AC (Anomaly Clustering) | 0.525 | 0.431 | — |
| **MuSc + AnomalyNCD** | **0.613** | **0.526** | **0.712** |

### 半监督设置（使用正常标注图像）

| AD 方法 + AnomalyNCD | MVTec AD NMI↑ | MVTec AD ARI↑ | MVTec AD F1↑ |
|---------------------|-------------|-------------|------------|
| PatchCore | 0.670 | 0.601 | 0.769 |
| CPR | **0.736** | **0.674** | **0.805** |

### 消融实验

| 组件 | NMI | ARI | F1 |
|------|-----|-----|------|
| (a) w/o MGA | 0.598 | 0.494 | 0.698 |
| (b) all tokens | 0.507 | 0.382 | 0.600 |
| (c) patch tokens | 0.563 | 0.467 | 0.686 |
| **(d) class token (Ours)** | **0.613** | **0.526** | **0.712** |

| MEBin vs 固定阈值 | FPR↓ | FNR↓ | F1↑ |
|-----------------|------|------|------|
| 固定阈值 0.5 | 高 | 高 | 0.640 |
| Otsu | 最高 | 中 | 0.499 |
| **MEBin** | **0.153** | **0.035** | **0.712** |

### 关键发现
- MGA 仅在 CLS token 上效果最好，+5.0% NMI，+2.6% F1
- 最后 9 层替换掩码注意力最优（$L_m=9$）
- 面积加权融合优于平均/分数加权
- GT 掩码下 NMI 达 0.871，说明 AD 方法质量是瓶颈
- 使用标注异常数据（$\mathcal{D}_l$）带来 +3.0% NMI

## 亮点与洞察
- **首个自监督工业多类异常分类方法**，可与任意 AD 方法组合
- MEBin 自适应阈值选择，对不同 AD 方法通用
- 掩码注意力设计优雅，仅修改 CLS token 的注意力即可
- 支持复合异常（同一图像含多种异常类型）

## 局限与展望
- 性能受前置 AD 方法质量影响（EfficientAD 异常概率跨度大导致效果差）
- MEBin 计算基于 CPU（OpenCV 连通域分析），占推理 80%+ 时间
- 需要已知新类别数 $\mathcal{C}_u$ 作为先验

## 评分
- 新颖性: ⭐⭐⭐⭐ NCD+工业异常分类首创，MEBin设计独特
- 实验充分度: ⭐⭐⭐⭐⭐ 消融极其详尽（7个消融+跨数据集+每类结果）
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示直观
- 价值: ⭐⭐⭐⭐ 工业质检下游处理的重要基石

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Novel Anomaly Detection Scenarios and Evaluation Metrics to Address the Ambiguity in the Definition of Normal Samples](../../CVPR2026/others/novel_anomaly_detection_scenarios_and_evaluation_metrics_to_address_the_ambiguit.md)
- [\[NeurIPS 2025\] ADPretrain: Advancing Industrial Anomaly Detection via Anomaly Representation Pretraining](../../NeurIPS2025/others/adpretrain_advancing_industrial_anomaly_detection_via_anomaly_representation_pre.md)
- [\[CVPR 2025\] Integration of deep generative Anomaly Detection algorithm in high-speed industrial line](integration_of_deep_generative_anomaly_detection_algorithm_in_high-speed_industr.md)
- [\[CVPR 2025\] Task-Agnostic Guided Feature Expansion for Class-Incremental Learning](task-agnostic_guided_feature_expansion_for_class-incremental_learning.md)
- [\[CVPR 2025\] MOS: Modeling Object-Scene Associations in Generalized Category Discovery](mos_modeling_object-scene_associations_in_generalized_category_discovery.md)

<!-- RELATED:END -->
