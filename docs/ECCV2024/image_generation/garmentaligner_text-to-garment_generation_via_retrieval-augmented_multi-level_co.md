---
title: >-
  [论文解读] GarmentAligner: Text-to-Garment Generation via Retrieval-augmented Multi-level Corrections
description: >-
  [ECCV 2024][图像生成][服装生成] 针对文本到服装图像生成中的细粒度语义错位（组件数量、位置和相互关系），提出 GarmentAligner，通过自动组件提取管线获取空间-数量信息，并结合检索增强对比学习和多级校正损失，实现服装组件的视觉、空间和数量级别的精确对齐。
tags:
  - ECCV 2024
  - 图像生成
  - 服装生成
  - 扩散模型
  - 检索增强
  - 对比学习
  - 细粒度对齐
---

# GarmentAligner: Text-to-Garment Generation via Retrieval-augmented Multi-level Corrections

**会议**: ECCV 2024  
**arXiv**: [2408.12352](https://arxiv.org/abs/2408.12352)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 服装生成, 扩散模型, 检索增强, 对比学习, 细粒度对齐

## 一句话总结

针对文本到服装图像生成中的细粒度语义错位（组件数量、位置和相互关系），提出 GarmentAligner，通过自动组件提取管线获取空间-数量信息，并结合检索增强对比学习和多级校正损失，实现服装组件的视觉、空间和数量级别的精确对齐。

## 研究背景与动机

文本到服装生成（Text-to-Garment）作为 T2I 的下游任务，在时尚产业有巨大的商业价值。然而，即使最先进的 T2I 模型（如 Midjourney、SDXL）在服装生成方面仍然表现不佳：

**核心挑战**：

**文本语义差异**：服装描述具有特定的文字结构和专业修饰词（如"带褶皱的V领"），通用 T2I 模型难以准确理解

**组件细粒度对齐困难**：服装组件（纽扣、口袋、领子等）具有独特属性和复杂相互关系，现有方法主要关注视觉语义，忽视了**组件定位**和**数量对齐**

作者展示了 Midjourney 在服装生成中的典型失败案例：
- 纽扣数量不对（描述5个但生成3个或7个）
- 口袋位置偏移（应在左胸但出现在右侧）
- 组件关系关系混乱（拉链和衣领的连接方式不正确）

**关键洞察**：需要在多个语义层级上挖掘服装图像和描述中的组件信息，从整体感知到细粒度细节全面提升生成质量。

## 方法详解

### 整体框架

GarmentAligner 基于预训练的潜在扩散模型（SD v2.1），通过**检索增强多级校正**训练策略进行领域适配。包含三个核心组件：

1. **自动组件提取管线**：从服装图像中提取空间和数量信息
2. **检索增强对比学习**：通过语义相似性排名构建正负样本对进行对比训练
3. **多级校正损失**：从视觉、空间和数量三个角度进行细粒度校正

### 关键设计

#### 1. 自动组件提取管线

在 CM-Fashion 数据集基础上，利用开放域检测和分割模型提取组件级信息：

**步骤流程**：
1. **组件检测**：用 GroundingDINO 从服装图像中获取目标组件的 bounding boxes
2. **数量统计**：通过计数 bounding boxes 确定每种组件的数量
3. **位置定位**：计算 boxes 的几何中心作为组件的空间位置
4. **组件分割**：

    - 先用服装解析模型进行初步分割
    - 再用 SAM 结合 bounding boxes 增强分割（处理初步分割遗漏的组件）
5. **文本增强**：将提取的数量和空间信息与原始文本描述对齐，丰富标注

最终产出：每张服装图像配备详细描述 + 组件分割掩码 + 组件位置 + 组件数量。

#### 2. 检索增强对比学习

为了弥补服装数据集规模不足的问题，通过检索增强扩展训练样本。

**语义相似性排名**：对样本对 $(x, y)$，第 $i$ 个组件的相似度分数：
$$S(x, y, i) = \frac{1}{|q_i^x - q_i^y| + Jaro(t_i^x, t_i^y)}$$

其中 $q_i$ 为组件数量，$t_i$ 为组件文本描述，$Jaro$ 为 Jaro 字符串距离。

总体相似度跨所有 $k$ 个组件求和并减去全属句相似度的惩罚：
$$S(x, y) = \sum_{i=1}^k S(x, y, i) - \alpha \cdot Jaro(t_x, t_y)$$

**正负样本构建**：
- 在 $N$ 个样本的随机子集中按相似度排名检索
- 高相似度 + 高审美/人类偏好分数 → 正样本
- 低相似度 + 低审美/偏好分数 → 负样本
- 每个样本扩展为 $N_p \times N_n$ 个样本对

**对比损失**：
$$\mathcal{L}_{RACL} = \|\hat{x} - x_p\|^2 + 1 - \|\hat{x} - x_n\|^2$$

最小化生成结果与正样本的距离，最大化与负样本的距离。

#### 3. 多级校正损失

三种组件级校正损失从不同角度增强细粒度对齐：

**视觉校正**（Text-Image 一致性）：
$$\mathcal{L}_{visual} = \sum_{i=1}^k \frac{1}{CLIPScore(m_i \odot \hat{X}, t_i)}$$

使用组件掩码 $m_i$（来自GT）裁剪生成图像中的组件区域，计算与组件描述 $t_i$ 的 CLIP Score 作为奖励函数。

**空间校正**（组件位置对齐）：
$$\mathcal{L}_{spatial} = \sum_{i=1}^k \sum_{j=1}^l \|a_i^j - I_j(m_i)\|^2$$

从cross-attention中提取组件描述对应的空间注意力图 $A_i$，与GT组件掩码 $m_i$ 进行 MSE 对齐。

**数量校正**（组件计数对齐）：
$$\mathcal{L}_{quantitative} = \sum_{i=1}^k |q_i - \hat{q}_i|$$

使用组件检测器（GroundingDINO）检测生成结果中的组件数量 $\hat{q}_i$，与GT数量 $q_i$ 对比。

### 损失函数 / 训练策略

**总损失函数**：
$$\mathcal{L} = \omega_v \cdot \mathcal{L}_{visual} + \omega_s \cdot \mathcal{L}_{spatial} + \omega_q \cdot \mathcal{L}_{quantitative} + \omega_r \cdot \mathcal{L}_{RACL}$$

**训练配置**：
- 基座模型：SD v2.1
- 预测类型：混合预测（噪声+图像），替代纯噪声预测
- 硬件：8x Tesla V100，batch size 32
- 学习率：$1 \times 10^{-6}$
- 训练时长：40 epochs，约70小时
- 数据集：CM-Fashion（50万张 512×512 服装图像 + 描述）

## 实验关键数据

### 主实验

与多种基线的定量比较（CM-Fashion 数据集）：

| 方法 | FID ↓ | CLIPScore ↑ | AestheticScore ↑ | HPSv2 ↑ |
|------|-------|------------|-----------------|---------|
| DALL·E | 13.249 | 0.6423 | 4.8592 | 0.2137 |
| ARMANI | 12.336 | 0.6988 | 5.3585 | 0.2237 |
| SD v1.5 | 9.368 | 0.8911 | 5.2807 | 0.2419 |
| SD v2.1 | 9.157 | 0.8818 | 5.3881 | 0.2426 |
| DiffCloth | 9.201 | 0.8974 | 5.3957 | 0.2440 |
| SDXL | 9.091 | 0.8756 | 5.4299 | 0.2450 |
| **GarmentAligner** | **8.735** | **0.9245** | **5.8776** | **0.2648** |

GarmentAligner 在所有指标上均取得最优，FID 降低至 8.735，CLIPScore 提升至 0.9245。

**组件级准确率**（1000条描述×100张图像）：
- 数量准确率：GarmentAligner 超越其他方法 **20~45%**
- 空间准确率：同样显著领先

用户研究（110位参与者）：GarmentAligner 获得超过 **28%** 的偏好率。

### 消融实验

各组件的贡献分析：

| 变体 | FID ↓ | CLIPScore ↑ | AestheticScore ↑ | HPSv2 ↑ |
|------|-------|------------|-----------------|---------|
| [V] 视觉校正 | 8.975 | 0.9136 | 5.4081 | 0.2459 |
| [S] 空间校正 | 9.143 | 0.8976 | 5.4003 | 0.2447 |
| [C] 数量校正 | 9.091 | 0.8840 | 5.3912 | 0.2433 |
| [V+S+C] 三种校正 | 8.924 | 0.9183 | 5.4190 | 0.2462 |
| [R] 检索对比 | 8.802 | 0.8984 | 5.7443 | 0.2639 |
| **[V+S+C+R] 完整** | **8.735** | **0.9245** | **5.8776** | **0.2648** |

### 关键发现

1. **检索增强对比学习贡献最大**：对 FID、aesthetic score 和 HPSv2 的提升最显著，主要提升图像真实感和整体质量
2. **多级校正主要提升文本-图像一致性**：对 CLIPScore 的贡献最显著
3. **各组件互补**：检索对比学习提升全局感知，多级校正增强细粒度详细，组合产生叠加效果
4. **数量对齐比空间对齐更难**：消融中 [C] 的独立 CLIPScore 最低，说明数量对齐最具挑战性
5. **预测类型改变有效**：混合预测（噪声+图像）比纯噪声预测提升了生成质量

## 亮点与洞察

1. **问题定义精准**：聚焦于服装组件的"数量+位置+关系"三重对齐，是此前work忽略的细粒度维度
2. **自动化管线可迁移**：组件提取管线基于 GroundingDINO + SAM，可应用于任何服装数据集
3. **检索增强解决数据稀缺**：通过组件级相似性检索构建正负样本，有效利用有限数据
4. **多损失协同设计**：视觉（CLIP反馈）、空间（attention map对齐）、数量（检测计数）三个正交维度的校正，覆盖了服装生成的主要失败模式

## 局限与展望

1. **依赖提取管线精度**：组件信息完全依赖 GroundingDINO 和 SAM 的准确性，大规模数据中不可避免存在错误
2. **预训练模型偏见**：继承了 SD 模型的固有偏差，可能导致输出缺乏鲁棒性和用户友好性
3. **训练成本**：需要8GPU训练70小时，且需要先运行组件提取管线
4. **仅针对单品服装**：CM-Fashion 是单品服装数据集，搭配生成或穿搭场景未涉及
5. **数量校正的梯度问题**：检测器输出的计数是离散的，梯度传播可能不连续

## 相关工作与启发

- **ARMANI / DiffCloth**：此前的服装生成方法，使用分割描述和语义分割，但忽略了定位和数量
- **Attend-and-Excite**：注意力调制方法，GarmentAligner 在空间校正中使用了类似的 attention map 引导思路
- **CLIP 反馈用于生成**：将 CLIP Score 作为训练信号用于生成质量反馈，可泛化到其他细粒度生成任务
- **启发**：检索增强+对比学习的训练策略可能对其他具有丰富结构化信息的领域（如建筑设计、机械制图）同样有效

## 评分

- **创新性**：★★★★☆ — 多级校正+检索对比的组合设计原创性强，组件提取管线实用
- **实用性**：★★★★☆ — 直接解决商业场景的痛点问题，但代码未开源限制了复现
- **实验充分度**：★★★★☆ — 多维度指标+用户研究+详细消融，数量准确率评估新颖
- **写作质量**：★★★★☆ — 结构清晰，可视化对比效果好

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [\[AAAI 2026\] TruthfulRAG: Resolving Factual-level Conflicts in Retrieval-Augmented Generation with Knowledge Graphs](../../AAAI2026/image_generation/truthfulrag_resolving_factual-level_conflicts_in_retrieval-augmented_generation_.md)
- [\[ECCV 2024\] OMG: Occlusion-friendly Personalized Multi-concept Generation in Diffusion Models](omg_occlusion-friendly_personalized_multi-concept_generation_in_diffusion_models.md)
- [\[ECCV 2024\] MultiGen: Zero-Shot Image Generation from Multi-modal Prompts](multigen_zero-shot_image_generation_from_multi-modal_prompts.md)
- [\[ECCV 2024\] Latent Guard: a Safety Framework for Text-to-Image Generation](latent_guard_a_safety_framework_for_text-to-image_generation.md)

<!-- RELATED:END -->
