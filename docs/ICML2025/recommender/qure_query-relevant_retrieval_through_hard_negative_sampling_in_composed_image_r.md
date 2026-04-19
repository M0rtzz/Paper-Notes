---
title: >-
  [论文解读] QuRe: Query-Relevant Retrieval through Hard Negative Sampling in Composed Image Retrieval
description: >-
  [ICML2025][Composed Image Retrieval] 提出 QuRe，通过基于相关性分数陡降的硬负样本采样策略和奖励模型优化目标，在组合图像检索(CIR)中同时召回目标图像和其他相关图像，从而提升用户满意度。
tags:
  - ICML2025
  - Composed Image Retrieval
  - 硬负样本采样
  - 奖励模型
  - Bradley-Terry
  - 人类偏好对齐
---

# QuRe: Query-Relevant Retrieval through Hard Negative Sampling in Composed Image Retrieval

**会议**: ICML2025  
**arXiv**: [2507.12416](https://arxiv.org/abs/2507.12416)  
**代码**: [jackwaky/QuRe](https://github.com/jackwaky/QuRe)  
**领域**: 图像检索  
**关键词**: Composed Image Retrieval, 硬负样本采样, 奖励模型, Bradley-Terry, 人类偏好对齐

## 一句话总结

提出 QuRe，通过基于相关性分数陡降的硬负样本采样策略和奖励模型优化目标，在组合图像检索(CIR)中同时召回目标图像和其他相关图像，从而提升用户满意度。

## 研究背景与动机

组合图像检索 (Composed Image Retrieval, CIR) 利用参考图像和文本描述共同检索目标图像。现有方法存在一个关键局限：

- **仅关注目标图像召回**：数据集通常每条 query 只标注一个 target，其余全部被视为负样本
- **对比学习导致假负样本**：batch 内除 target 外全部作为负样本训练，会将与 query 高度相关但未标注的图像错误地推开
- **用户满意度被忽视**：即使 target 被检索到，其余 top-k 结果中充斥不相关图像，用户体验仍然很差

核心问题：标准的 Recall@k 只衡量 target 是否出现在 top-k 中，无法反映整体检索集的质量。

## 方法详解

### 整体框架

QuRe 基于 BLIP-2 架构（ViT-L 图像编码器 + Q-Former），主要包含两个创新：

1. **奖励模型训练目标**：用 Bradley-Terry 偏好模型替代传统对比损失
2. **硬负样本采样策略**：基于相关性分数的双陡降定位硬负样本区间

### 相关性分数

对语料库中每张图像 $I$，定义相关性分数为双模态 query 嵌入与图像嵌入的内积：

$$s(x_I, x_T, I) = \frac{Q(E_{img}(x_I), x_T) \cdot Q(E_{img}(I))}{\tau}$$

其中 $E_{img}$ 为 BLIP-2 图像编码器，$Q$ 为 Q-Former，$\tau$ 为可学习温度参数。

### 训练目标（Bradley-Terry 偏好模型）

不同于对比学习中将 batch 内所有非目标作为负样本，QuRe 采用奖励模型目标，每次仅配对一个正样本和一个负样本：

$$p^*(I_p \succ I_n \mid x_I, x_T) = \sigma(s(x_I, x_T, I_p) - s(x_I, x_T, I_n))$$

目标函数为最小化负对数似然（等价于 KL 散度最小化）：

$$\mathcal{L} = -\mathbb{E}_{(x_I, x_T, I_p, I_n) \sim \mathbb{D}^*} [\log(p^*(I_p \succ I_n \mid x_I, x_T))]$$

其中 $I_p = y_I$（目标图像），$I_n$ 从硬负样本集 $\mathbb{H}$ 中采样。

### 硬负样本集采样（核心贡献）

**两个条件**：

- C1：负样本应比目标图像与 query 的相关性更低
- C2：负样本的相关性分数应与目标图像相近（具有挑战性）

**具体步骤**：

1. 将语料库中所有图像按相关性分数降序排列：$\mathbb{S}_i = \{s_{i,1}, \ldots, s_{i,N_{img}}\}$
2. 取分数低于 target 的子集：$\mathbb{S}_i^{<targ} = \{s_{i,j} \mid s_{i,j} < s(x_{I_i}, x_{T_i}, y_i)\}$
3. 找到该子集中**相邻分数差最大的两个位置** $k_1, k_2$（即两次最陡分数下降点）
4. 硬负样本集定义为两个陡降之间的图像：

$$\mathbb{H}_i = \{I_j \mid j \in [\min(k_1,k_2)+1,\ \max(k_1,k_2)],\ s_{i,j} < s(x_{I_i}, x_{T_i}, y_i)\}$$

**直觉**：第一次陡降将假负样本（与 target 高度相似）排除在外，第二次陡降将太容易的负样本排除在外，中间区域恰好是"在至少一个关键属性（如颜色、形状）上与 query 不同"的硬负样本。

### 训练细节

- 每 $\lfloor n_{epoch} / n_{def} \rfloor$ 个 epoch 重新更新一次硬负样本集（$n_{def}=6$）
- 初始 warm-up 阶段：硬负样本集包含除 target 外的全部语料库
- 每个 epoch 从 $\mathbb{H}$ 中均匀采样一个负样本，保证多样性

## 实验关键数据

### FashionIQ 验证集（Recall@10 / Recall@50）

| 方法 | Dress R@10 | Shirt R@10 | Toptee R@10 | 平均 R@10 | 平均 Avg |
|------|-----------|-----------|------------|----------|---------|
| CLIP4CIR | 38.32 | 44.31 | 47.27 | 43.30 | 55.03 |
| SPRC | 45.71 | 51.37 | 55.48 | 50.86 | 62.13 |
| **QuRe** | **46.80** | **53.53** | **57.47** | **52.60** | **63.04** |

### CIRR 测试集

| 方法 | R@1 | R@5 | R@10 | R_s@1 | R@5+R_s@1 |
|------|-----|-----|------|-------|-----------|
| SPRC | 50.75 | 80.58 | 88.72 | 79.57 | 80.07 |
| **QuRe** | **52.22** | **82.53** | **90.31** | 78.51 | **80.52** |

- FashionIQ 上 R@10 平均比 SPRC 提升 **+1.74%**
- CIRR 上 R@1 提升 **+1.47%**，R@5 提升 **+1.95%**

### HP-FashionIQ 人类偏好对齐

QuRe 在新构建的 HP-FashionIQ 数据集上展示了最佳的人类偏好对齐能力，表明其检索结果整体更符合用户期望。

## 亮点与洞察

1. **问题定义精准**：首次在 CIR 中明确提出"不止检索 target，还要让其他 top-k 结果也相关"的目标
2. **硬负样本策略巧妙**：利用相关性分数的双陡降点自适应定位硬负样本区间，无需额外标注
3. **奖励模型目标**：从 RLHF 借鉴 Bradley-Terry 模型，每次仅对比一对正负样本，自然避免了假负样本问题
4. **HP-FashionIQ 数据集**：填补了 CIR 领域缺乏人类偏好评估基准的空白（61 名参与者，2715 有效 query）
5. **资源高效**：单卡 RTX 3090 即可训练，实用性强

## 局限与展望

1. **R_s@K 指标略低于 SPRC**：由于 QuRe 允许假负样本获得高分，在子集召回指标上有轻微退步
2. **硬负样本集需周期性重建**：每 $\lfloor n/n_{def} \rfloor$ 个 epoch 需对全部语料排序计算陡降点，计算开销随语料库增大而增长
3. **双陡降假设的鲁棒性**：当相关性分数分布平滑、无明显陡降时，硬负样本区间的定义可能不稳定
4. **仅在时尚和通用域验证**：缺少医学、遥感等复杂场景的实验验证
5. **HP-FashionIQ 规模有限**：仅覆盖 shirts 和 toptee 两个品类，泛化性待验证

## 相关工作与启发

- **CoVR-BLIP** / **SPRC**：现有最强 CIR 基线，均采用对比学习
- **HCL** (Robinson et al., 2020)：硬负样本经典定义——类别不同 + 嵌入接近
- **RLHF** (Ouyang et al., 2022)：Bradley-Terry 偏好模型的灵感来源
- **FNC** (Huynh et al., 2022)：用阈值过滤假负样本，QuRe 通过陡降点实现自适应替代

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将 RLHF 的偏好模型引入 CIR 训练 + 双陡降硬负采样均为新颖思路
- 实验充分度: ⭐⭐⭐⭐ — 两个标准数据集 + 新建的人类偏好数据集，消融实验完整
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，公式推导严谨，图示直观
- 价值: ⭐⭐⭐⭐ — 单卡可训，开源代码，在用户满意度维度推进了 CIR 研究

<!-- RELATED:START -->

## 相关论文

- [CroPS: Improving Dense Retrieval with Cross-Perspective Positive Samples in Short-Video Search](../../AAAI2026/recommender/crops_improving_dense_retrieval_with_cross-perspective_positive_samples_in_short.md)
- [VisualLens: Personalization through Task-Agnostic Visual History](../../NeurIPS2025/recommender/visuallens_personalization_through_task-agnostic_visual_history.md)
- [LOTUS: A Leaderboard for Detailed Image Captioning from Quality to Societal Bias and User Preferences](../../ACL2025/recommender/lotus_a_leaderboard_for_detailed_image_captioning_from_quality_to_societal_bias_.md)
- [Beyond Single Labels: Improving Conversational Recommendation through LLM-Powered Data Augmentation](../../ACL2025/recommender/beyond_single_labels_improving_conversational_recommendation_through_llm-powered.md)
- [AID-AppEAL: Automatic Image Dataset and Algorithm for Content Appeal Enhancement and Assessment Labeling](../../ECCV2024/recommender/aid-appeal_automatic_image_dataset_and_algorithm_for_content_appeal_enhancement_.md)

<!-- RELATED:END -->
