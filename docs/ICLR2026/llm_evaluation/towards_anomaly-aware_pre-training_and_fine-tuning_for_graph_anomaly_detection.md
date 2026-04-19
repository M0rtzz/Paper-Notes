---
title: >-
  [论文解读] Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection
description: >-
   提出 APF 框架，通过 Rayleigh 商引导的异常感知预训练和粒度自适应微调，解决图异常检测中标签稀缺和同质性差异的双重挑战。
tags:

---

# Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection

## 论文信息
- **会议**: ICLR 2026
- **arXiv**: [2504.14250](https://arxiv.org/abs/2504.14250)
- **代码**: [https://github.com/Cloudy1225/APF](https://github.com/Cloudy1225/APF)
- **领域**: 图异常检测 / 图预训练 / 谱图滤波
- **关键词**: GAD, 预训练微调, Rayleigh 商, 同质性差异, 双滤波器, 门控融合

## 一句话总结
提出 APF 框架，通过 Rayleigh 商引导的异常感知预训练和粒度自适应微调，解决图异常检测中标签稀缺和同质性差异的双重挑战。

## 研究背景与动机

### 核心问题
图异常检测（GAD）面临两大关键挑战：

**标签稀缺**：标注成本高，真实场景中标注节点极少

**同质性差异**：分为节点级（个体节点局部同质性变化大）和类别级（异常节点局部同质性更低）

### 现有局限
- 通用图预训练策略（DGI、GraphMAE）仅提取任务无关语义，无法捕捉异常线索
- 伪标签和合成样本方法在标签稀缺下不稳定
- 全局统一的处理方案（边重加权、谱滤波）缺乏节点自适应机制

### 关键观察
局部同质性 $h_i = \frac{|v_j \in \mathcal{N}_i: y_i = y_j|}{|\mathcal{N}_i|}$ 在节点间变化剧烈，且异常节点的平均局部同质性 $h^a$ 一致低于正常节点 $h^n$。现有方法在不同局部同质性分组上表现不一致。

## 方法详解

### 1. 异常感知预训练

#### 无标签异常指标——Rayleigh 商

利用 Rayleigh 商作为无标签的异常度量：

$$RQ(\boldsymbol{x}, \boldsymbol{L}) = \frac{\boldsymbol{x}^T \boldsymbol{L} \boldsymbol{x}}{\boldsymbol{x}^T \boldsymbol{x}} = \frac{\sum_{i,j} A_{ij}(x_j - x_i)^2}{\sum_{i=1}^n x_i^2}$$

**原理**：Rayleigh 商度量节点属性与局部图结构之间的不一致性，异常节点的 Rayleigh 商更高（谱能量右移现象）。

对每个节点 $v_i$ 使用 MRQSampler 提取 2-hop 子图 $\mathcal{G}_i^{RQ}$，最大化子图的 Rayleigh 商。

#### 双滤波器编码

采用可学习 Chebyshev 多项式谱滤波器，分为低通和高通：

$$g_L(\hat{\boldsymbol{L}}) = \sum_{k=0}^{K} w_k^L T_k(\hat{\boldsymbol{L}}), \quad g_H(\hat{\boldsymbol{L}}) = \sum_{k=0}^{K} w_k^H T_k(\hat{\boldsymbol{L}})$$

- **低通编码器**：捕捉通用语义模式 $\boldsymbol{Z}_L = f_{\theta_L}(g_L(\hat{\boldsymbol{L}})\boldsymbol{X})$
- **高通编码器**：捕捉细微异常线索 $\boldsymbol{Z}_H = f_{\theta_H}(g_H(\hat{\boldsymbol{L}})\boldsymbol{X})$

#### 预训练目标

基于 DGI 的互信息最大化，加入异常感知目标：

$$\mathcal{L}_{pt} = -\frac{1}{n}\sum_i \left[\log\mathcal{D}(\boldsymbol{Z}_i^L, \boldsymbol{s}^L) + \log(1-\mathcal{D}(\tilde{\boldsymbol{Z}}_i^L, \boldsymbol{s}^L))\right] - \frac{1}{n}\sum_i \left[\log\mathcal{D}(\boldsymbol{Z}_i^H, \boldsymbol{s}_i^H) + \log(1-\mathcal{D}(\tilde{\boldsymbol{Z}}_i^H, \boldsymbol{s}_i^H))\right]$$

其中 $\boldsymbol{s}_i^H$ 是基于 Rayleigh 商子图的异常感知摘要。

### 2. 粒度自适应微调

#### 门控融合网络

节点和维度级别的自适应融合：

$$\boldsymbol{Z} = \boldsymbol{C} \odot \boldsymbol{Z}_L + (1-\boldsymbol{C}) \odot \boldsymbol{Z}_H$$

系数通过轻量门控网络生成：

$$\boldsymbol{C} = \sigma(\boldsymbol{X}\boldsymbol{W}_c + \boldsymbol{b}_c)$$

参数复杂度从 $\mathcal{O}(n \times e)$ 降至 $\mathcal{O}((d+1) \times e)$。

#### 异常感知正则化损失

鼓励异常节点保留更多高通（异常相关）信息：

$$\mathcal{L}_{reg} = -\frac{1}{|\mathcal{V}^L|}\sum_{v_i, y_i=1}\left(p^a\log c_i + (1-p^a)\log(1-c_i)\right) - \frac{1}{|\mathcal{V}^L|}\sum_{v_i, y_i=0}\left(p^n\log c_i + (1-p^n)\log(1-c_i)\right)$$

其中 $p^a \leq p^n$，引导异常节点更多使用高通表示。

### 3. 理论保证

**定理 1**：在异常随机块模型（ASBM）下，当低通和高通滤波器分别应用于同质性和异质性节点时，存在参数使所有节点线性可分（概率 $1-o_d(1)$）。

## 实验

### 实验设置
- **10 个 GADBench 数据集**：Reddit, Weibo, Amazon, Yelp, T-Finance, Elliptic 等
- **半监督设置**：仅 100 个标注节点（20 异常 + 80 正常）
- **指标**：AUPRC, AUROC, Rec@K

### 主实验（AUPRC）

| 模型 | Reddit | Weibo | Amazon | T-Fin | 平均 |
|------|--------|-------|--------|-------|------|
| GCN | 4.2 | 86.0 | 32.8 | 60.5 | 29.3 |
| BWGNN | 4.2 | 80.6 | 81.7 | 60.9 | - |
| BernNet | 4.9 | 66.6 | 81.2 | 51.8 | 31.1 |
| **APF** | **最佳/次佳** | **最佳/次佳** | **最佳/次佳** | **最佳/次佳** | **最高** |

### 消融实验关键发现
1. Rayleigh 商引导的子图选择显著提升异常感知能力
2. 双滤波器比单一滤波器表现更好
3. 门控融合网络优于直接参数优化
4. 异常感知正则化在类别级差异大的数据集上效果更明显

## 亮点
1. **创新的无标签异常度量**：Rayleigh 商作为预训练阶段的异常信号
2. **双粒度设计**：从预训练时的节点级到微调时的节点+维度级自适应
3. **理论支撑**：ASBM 模型下的线性可分性证明
4. **10 个数据集的全面验证**

## 局限性
1. 预训练依赖 DGI 框架，可能不是所有场景的最优选择
2. Rayleigh 商假设异常表现为谱能量右移，对某些类型异常可能不敏感
3. 需要手动设定 $p^a, p^n$ 的值
4. 标注数据极少时正则化损失的优化可能不稳定

## 相关工作
- **图异常检测**: PCGNN, AMNet, BWGNN — 全局同质性处理
- **图预训练**: DGI, GraphMAE, BGRL — 任务无关语义
- **谱方法**: BernNet, ChebNet — 可学习谱滤波器

## 评分
- **创新性**: ⭐⭐⭐⭐ — Rayleigh 商 + 双滤波器预训练的组合很有洞察
- **实验充分性**: ⭐⭐⭐⭐⭐ — 10 个数据集全面评比
- **写作质量**: ⭐⭐⭐⭐ — 理论与实践结合紧密
- **实用性**: ⭐⭐⭐⭐ — 标签稀缺场景下有实际价值

<!-- RELATED:START -->

## 相关论文

- [Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning](../../CVPR2026/llm_evaluation/weakly_supervised_video_anomaly_detection_with_anomaly-connected_components_and_.md)
- [Normal-Abnormal Guided Generalist Anomaly Detection](../../NeurIPS2025/llm_evaluation/normal-abnormal_guided_generalist_anomaly_detection.md)
- [RefineVAD: Semantic-Guided Feature Recalibration for Weakly Supervised Video Anomaly Detection](../../AAAI2026/llm_evaluation/refinevad_semantic-guided_feature_recalibration_for_weakly_supervised_video_anom.md)
- [AD-LLM: Benchmarking Large Language Models for Anomaly Detection](../../ACL2025/llm_evaluation/ad-llm_benchmarking_large_language_models_for_anomaly_detection.md)
- [On the Robustness Tradeoff in Fine-Tuning](../../ICCV2025/llm_evaluation/on_the_robustness_tradeoff_in_fine-tuning.md)

<!-- RELATED:END -->
