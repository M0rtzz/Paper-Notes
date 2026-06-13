---
title: >-
  [论文解读] Demo: Generative AI helps Radiotherapy Planning with User Preference
description: >-
  [NeurIPS 2025 (GenAI for Health Workshop) Best Paper][医学图像][放疗规划] 提出 Flexible Dose Proposer (FDP)，通过两阶段训练框架（VQ-VAE 预训练 + 多条件编码）实现基于滑块的用户偏好交互式 3D 剂量分布预测，并集成到 Eclipse 临床治疗计划系统中，在头颈部癌症放疗场景中超越 Varian RapidPlan。
tags:
  - "NeurIPS 2025 (GenAI for Health Workshop) Best Paper"
  - "医学图像"
  - "放疗规划"
  - "剂量预测"
  - "用户偏好交互"
  - "VQ-VAE"
  - "生成式AI"
---

# Demo: Generative AI helps Radiotherapy Planning with User Preference

**会议**: NeurIPS 2025 (GenAI for Health Workshop) Best Paper  
**arXiv**: [2512.08996](https://arxiv.org/abs/2512.08996)  
**代码**: [Demo Video](https://huggingface.co/Jungle15/DoseProposerDemo)  
**领域**: 医学图像  
**关键词**: 放疗规划, 剂量预测, 用户偏好交互, VQ-VAE, 生成式AI

## 一句话总结
提出 Flexible Dose Proposer (FDP)，通过两阶段训练框架（VQ-VAE 预训练 + 多条件编码）实现基于滑块的用户偏好交互式 3D 剂量分布预测，并集成到 Eclipse 临床治疗计划系统中，在头颈部癌症放疗场景中超越 Varian RapidPlan。

## 研究背景与动机

**领域现状**：放疗规划是复杂的临床流程，不同机构和规划师之间存在显著差异。深度学习方法已在剂量预测、fluence map 生成、MLC 叶片排序等环节取得进展。

**现有痛点**：
   - RapidPlan 等知识驱动系统依赖 DVH 预测，无法捕捉空间剂量细节；基于 PCA 回归的 pipeline 仅用数十个计划训练，泛化能力有限
   - 现有深度学习剂量预测模型忽略了用户偏好交互——不同规划师对 OAR 保护与 PTV 覆盖的权衡有不同需求
   - 剂量预测本身不是可交付计划，与临床 TPS 系统的集成研究仍然不足

**核心矛盾**：单一模型无法适配多样化的规划风格，且训练过程容易被参考计划的特定风格所偏倚

**切入角度**：借鉴生成式 AI 的条件控制思路，通过"preference flavors"滑块让用户实时自定义 OAR-PTV 权衡

**核心 idea**：用 VQ-VAE 预训练稳定剂量解码器 + 用户偏好编码作为条件输入，实现可交互的个性化剂量预测

## 方法详解

### 整体框架
输入为 CT 图像、RT 结构（PTV/OAR 轮廓）、beam/angle plates 和用户偏好滑块值。Pipeline 分两阶段：Stage I 预训练 VQ-VAE 剂量解码器，Stage II 训练带多条件输入的灵活剂量预测模型。输出为 3D 剂量分布，之后通过目标函数提取转化为 Eclipse 中的可交付治疗计划。

### 关键设计

1. **Stage I: VQ-VAE 基础剂量解码器预训练**

    - 功能：用 31K 剂量数据预训练 VQ-VAE，学习真实剂量分布的潜在表示
    - 核心思路：不同于 latent diffusion 用 VAE 做压缩加速，本文用预训练来**稳定 Stage II 训练**。损失函数为：
      $\mathcal{L}_{\text{stage1}} = \underbrace{\mathbb{E}_i[\|x_i - \hat{x}_i\|]}_{\text{Reconstruction}} + \beta L_{vq} + L_{adv}(x, \hat{x}) + \underbrace{\lambda \cdot \log(\mathbb{E}_{i<j}[\exp(-t\|\hat{z}_i - \hat{z}_j\|^2)])}_{\text{Uniformity}}$
    - 设计动机：引入均匀性损失（uniformity loss）正则化潜在空间，避免模式坍塌；对抗损失保证生成剂量的真实性。没有预训练时，Stage II 在复杂条件下训练不稳定，会出现 PTV/OAR 边界伪影

2. **Stage II: 多条件灵活剂量预测**

    - 功能：以 CT+RT 结构为图像输入，用户偏好+beam plates 为条件输入，预测个性化 3D 剂量
    - 核心思路：图像编码器采用 MedNeXt 架构处理多通道输入；用户偏好和 beam/angle 通过 AdaIN（Adaptive Instance Normalization）编码注入。损失函数：
      $\mathcal{L}_{\text{stage2}}^{(i)} = \|x_i - \hat{x}_i\| + \|z_i - \hat{z}_i\| + L_{adv}(x_i, \hat{x}_i) + \mathcal{L}_{\text{obj}}^{(i)}$
    - 目标一致性损失 $\mathcal{L}_{\text{obj}}$ 确保预测与用户偏好对齐：
      $\mathcal{L}_{\text{obj}}^{(i)} = \|\tilde{h} - \hat{h}\| + \|p - \hat{p}\| + \|\tilde{w} \cdot u_{\text{oar}} - \hat{u}_{\text{oar}}\|$
      其中 $\tilde{h}$ 是用户偏好的 HI 值，$\tilde{w}$ 是 OAR 保护偏好权重

3. **训练时偏好随机采样**

    - 功能：训练时滑块值 $\{\tilde{h}, \tilde{w}\}$ 在预定义范围内随机采样
    - 设计动机：让模型学会响应不同偏好组合，而非过拟合到单一规划风格

4. **临床集成**

    - 预测的 3D 剂量通过目标函数提取转化为 Eclipse 治疗计划系统的优化目标，实现可交付计划

### 损失函数 / 训练策略
- Stage I：重建 + VQ 量化 + 对抗 + 均匀性，用 31K 剂量训练
- Stage II：图像空间重建 + 潜在空间重建 + 对抗 + 目标一致性损失，用 6 个 cohort（共 ~820 训练样本）
- 采用单步生成（GAN），推理约 30ms，不用扩散模型的迭代采样

## 实验关键数据

### 主实验：患者内 DVH 差异（VMAT 计划）

| OAR | RapidPlan std | FDP std | RapidPlan mean | FDP mean |
|-----|--------------|---------|----------------|----------|
| SpinalCord05 | 3.02 | **1.18** | 3.25 | **1.69** |
| Larynx-PTV | 3.49 | **1.78** | 6.29 | **2.94** |
| ParotidCon-PTV | 3.38 | **1.22** | 4.51 | **1.92** |
| PosteriorNeck | 4.30 | **0.92** | 8.15 | **1.19** |
| Trachea | 3.40 | **1.68** | 3.42 | **2.01** |
| **Better count** | 0 | **15** | 1 | **14** |

FDP 在 15/15 个 OAR 的 std 指标上优于 RapidPlan，14/15 个 OAR 的 mean 指标上更优。

### 消融实验

| 配置 | MAE (↓) | 说明 |
|------|---------|------|
| w/o Stage I pretrain | 2.63 | 出现边界伪影 |
| w/ Stage I pretrain | **2.56** | 剂量分布更真实 |

### 关键发现
- Stage I 预训练不仅降低 MAE，更重要的是消除了 PTV/OAR 边界的伪影，说明预训练提供了剂量分布的先验约束
- 用户偏好滑块能有效控制 OAR sparing vs PTV homogeneity 的权衡：P1（侧重 OAR 保护）和 P2（侧重 PTV）在实际 Eclipse 计划中产生了不同的剂量分布
- 模型推理仅 30ms，加可视化约 1.5s，接近实时交互

## 亮点与洞察
- **首个带交互滑块的剂量预测模型**：将生成式 AI 的条件控制范式引入放疗规划，实现用户偏好的实时调整，这个思路可迁移到其他需要个性化输出的医学影像任务
- **两阶段训练策略巧妙**：用大规模（31K）低质量数据预训练解码器，再用小规模高质量数据做条件微调，解决了医学数据稀缺下的训练稳定性问题
- **端到端临床验证**：不只停留在剂量预测，还通过 Eclipse 集成验证了可交付计划质量，这在学术研究中较为少见

## 局限与展望
- 仅在头颈部癌症场景评估，未验证其他治疗部位（肺、腹部等）的泛化能力
- RapidPlan 使用小数据集传统方法训练，与深度学习模型的比较存在训练范式差异，公平性有待讨论
- 用户偏好目前只有 HI 和 OAR 权重两个维度，缺少对 CI 和更细粒度 DVH 目标的控制
- 基于 GAN 的单步生成可能限制了输出多样性，扩散模型可能在条件控制上提供更精细的结果

## 相关工作与启发
- **vs RapidPlan**: RapidPlan 基于 PCA+小数据传统 ML，只预测 DVH；FDP 直接预测 3D 剂量，有 voxel 级空间感知能力
- **vs DoseDiff**: 扩散模型做剂量预测精度高但推理慢（迭代采样），FDP 用 GAN 单步生成实现实时交互
- **vs OpenKBP 系列**: 只关注预测精度，不考虑用户偏好交互和临床 TPS 集成

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个交互式偏好剂量预测，但技术组件（VQ-VAE, AdaIN）均非新提出
- 实验充分度: ⭐⭐⭐ 数据集规模有限，仅单一治疗部位，缺少与更多深度学习方法的对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，临床动机阐述到位，demo 展示直观
- 价值: ⭐⭐⭐⭐ 临床应用导向强，解决了真实的规划师需求痛点

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Rethinking Bias in Generative Data Augmentation for Medical AI: a Frequency Recalibration Approach](../../AAAI2026/medical_imaging/rethinking_bias_in_generative_data_augmentation_for_medical_ai_a_frequency_recal.md)
- [\[NeurIPS 2025\] The Boundaries of Fair AI in Medical Image Prognosis: A Causal Perspective](the_boundaries_of_fair_ai_in_medical_image_prognosis_a_causal_perspective.md)
- [\[NeurIPS 2025\] Doctor Approved: Generating Medically Accurate Skin Disease Images through AI-Expert Feedback](doctor_approved_generating_medically_accurate_skin_disease_images_through_ai-exp.md)
- [\[CVPR 2025\] Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-Supervised Medical Image Segmentation](../../CVPR2025/medical_imaging/enhancing_sam_with_efficient_prompting_and_preference_optimization_for_semi-supe.md)
- [\[NeurIPS 2025\] EEGReXferNet: A Lightweight Gen-AI Framework for EEG Subspace Reconstruction via Cross-Subject Transfer Learning and Channel-Aware Embedding](eegrexfernet_a_lightweight_gen-ai_framework_for_eeg_subspace_reconstruction_via_.md)

</div>

<!-- RELATED:END -->
