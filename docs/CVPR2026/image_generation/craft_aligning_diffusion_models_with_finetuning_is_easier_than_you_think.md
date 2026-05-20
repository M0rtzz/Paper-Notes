---
title: >-
  [论文解读] CRAFT: Aligning Diffusion Models with Fine-Tuning Is Easier Than You Think
description: >-
  [CVPR 2026][图像生成][扩散模型对齐] CRAFT 提出一种超轻量的扩散模型对齐方法：通过组合奖励过滤(CRF)策略自动构建高质量训练集，然后执行增强版 SFT，理论证明 CRAFT 实际优化的是分组强化学习的下界，仅用 100 个样本就超越了需要数千偏好对的 SOTA 方法…
tags:
  - "CVPR 2026"
  - "图像生成"
  - "扩散模型对齐"
  - "人类偏好"
  - "组合奖励过滤"
  - "监督微调"
  - "数据高效"
---

# CRAFT: Aligning Diffusion Models with Fine-Tuning Is Easier Than You Think

**会议**: CVPR 2026  
**arXiv**: [2603.18991](https://arxiv.org/abs/2603.18991)  
**代码**: 无  
**领域**: 图像生成 / 扩散模型对齐  
**关键词**: 扩散模型对齐, 人类偏好, 组合奖励过滤, 监督微调, 数据高效

## 一句话总结

CRAFT 提出一种超轻量的扩散模型对齐方法：通过组合奖励过滤(CRF)策略自动构建高质量训练集，然后执行增强版 SFT，理论证明 CRAFT 实际优化的是分组强化学习的下界，仅用 100 个样本就超越了需要数千偏好对的 SOTA 方法，且训练速度快 11-220 倍。

## 研究背景与动机

1. **领域现状**：扩散模型的后训练对齐主要有三条路线——SFT（需要高质量数据）、DPO 风格的偏好优化（需要大规模偏好对）、在线 RL 方法（计算开销大）。
2. **现有痛点**：SFT 依赖难以获取的高质量图像；DPO 方法如 Diff-DPO 依赖大规模偏好数据集且质量不一致；在线方法如 SPO 需要反复采样和评估，计算极其昂贵。
3. **核心矛盾**：数据效率与计算效率的双重挑战——现有方法要么需要大量数据，要么需要大量计算，两者难以兼得。
4. **本文目标**：设计一种既数据高效又计算轻量的微调方法。
5. **切入角度**：不需要外部高质量数据或偏好对，模型自己生成候选图像并通过多维奖励筛选最优样本。
6. **核心 idea**：组合多个奖励模型进行数据过滤 + 优势加权的 SFT，理论上等价于分组 RL 的下界优化。

## 方法详解

### 整体框架

CRAFT 分三个阶段：(1) 数据构建：从 HPD 数据集采样 10000 个提示，用 Qwen-Plus 扩展为多个变体，用基础模型生成候选图像；(2) 组合奖励过滤：用多个奖励模型联合筛选高质量样本；(3) 加权 SFT 微调：用分组优势计算权重，只对通过过滤的样本计算损失。

### 关键设计

1. **组合奖励过滤 (CRF)**:

    - 功能：从模型自生成的候选图像中自动筛选高质量训练数据
    - 核心思路：使用三个互补的奖励模型——HPSv2.1（人类偏好）、PickScore（拣选偏好）、AES（美学评分）。设计多级过滤策略：单奖励过滤 $\mathcal{I}_\xi$（任一奖励提升即保留）、双奖励过滤 $\mathcal{I}_{ha}$（两个同时提升）、三重过滤 $\mathcal{I}_{hpa}$（三个都提升，最严格）。对每个原始提示，如果其扩展版本生成的图像在所有奖励上都优于原始版本，则保留这批样本
    - 设计动机：自动数据策展避免了依赖外部高质量数据集或强模型蒸馏，组合多维奖励确保数据一致性

2. **分组优势加权 SFT**:

    - 功能：根据样本质量自适应调整梯度贡献
    - 核心思路：对每组样本计算归一化优势 $\hat{A}^{(i,j)} = (r^{(i,j)}_{\text{total}} - \text{mean}) / (\text{std} + \epsilon)$，然后以优势值加权标准 SFT 损失 $\|\epsilon_\theta(x^{(i,j)}_t, t, c) - \epsilon^{(i,j)}_t\|^2$，并用指示函数只对通过过滤的样本计算梯度
    - 设计动机：质量好的样本获得更大梯度贡献，差的样本被抑制，实现隐式的奖励引导

3. **理论保证 (Theorem 3.1)**:

    - 功能：建立 SFT 与强化学习之间的理论联系
    - 核心思路：在小学习率假设下，证明 CRAFT 的损失实际上优化的是分组强化学习目标 $\hat{J}(\theta)$ 的下界。具体来说，优势加权的 SFT 损失与 RL 目标之间存在精确的数学关系
    - 设计动机：为"用选择性数据做 SFT 就能实现 RL 级别对齐"提供理论基础，不再是纯经验方法

### 损失函数 / 训练策略

损失函数为优势加权的噪声预测 MSE 损失。使用 AdamW 优化器对 UNet 进行全参数微调。SD1.5 训练 120 步，SDXL 训练 200 步，batch size 128，学习率 5e-5。总训练仅需约 4 GPU 小时（SDXL on H100）。

## 实验关键数据

### 主实验

| 基准/指标 | SDXL 基线 | Diff-DPO | SPO | CRAFT | 提升 vs SPO |
|-----------|----------|----------|-----|-------|------------|
| HPDv2 HPSv2.1↑ | 27.93 | 29.76 | 32.32 | **32.67** | +0.35 |
| HPDv2 ImgReward↑ | 0.819 | 1.037 | 1.103 | **1.312** | +0.209 |
| HPDv2 MPS↑ | 14.35 | 14.70 | 15.36 | **15.62** | +0.26 |
| Parti HPS↑ | 27.32 | 28.74 | 30.54 | **31.10** | +0.56 |

CRAFT 在所有指标和数据集上全面领先，且 ImageReward 和 MPS 未在训练中使用，证明泛化能力。

### 消融实验

| 配置 | HPSv2.1 | 训练数据量 | GPU 时间 |
|------|---------|-----------|---------|
| CRAFT ($\mathcal{I}_{hpa}$) | 32.67 | 100 | ~4h |
| CRAFT ($\mathcal{I}_{ha}$) | 32.45 | ~300 | ~4h |
| CRAFT ($\mathcal{I}_h$) | 32.12 | ~1000 | ~4h |
| 无过滤 SFT | 31.80 | 10000 | ~4h |

### 关键发现

- 最严格的三重过滤 $\mathcal{I}_{hpa}$ 效果最好，说明数据质量远比数量重要
- CRAFT 仅用 100 个样本即超越需要 4000 样本的 SPO，数据效率提升 40 倍
- 训练速度比 SPO 快 19.7 倍（SDXL），比 SmPO 快 60.1 倍
- 在 GenEval 组合推理基准上也表现优异，说明对齐能力迁移到了指令跟随
- 在未训练的奖励指标上同样领先，说明不是过拟合训练奖励

## 亮点与洞察

- **极致数据效率**：100 个样本超越数千偏好对的方法，颠覆了"对齐需要大量偏好数据"的认知
- **自策展数据管线**：不需要外部数据，模型自己生成、自己筛选、自己训练，完全自包含
- **理论优雅**：证明选择性 SFT 等价于 RL 下界优化，建立了两种范式的理论桥梁
- **即时落地价值**：4 GPU 小时就能对齐 SDXL，极大降低了扩散模型后训练的门槛

## 局限与展望

- 依赖奖励模型的质量，如果奖励模型本身有偏差会传递到微调模型
- 仅在 SD1.5 和 SDXL 上验证，未在更新的架构（如 DiT/FLUX）上测试
- 理论证明需要小学习率假设，大学习率下可能不成立
- 未来可探索在视频扩散模型或 3D 生成上的应用

## 相关工作与启发

- **vs Diff-DPO**: DPO 需要大量偏好对且效率低，CRAFT 用 SFT 达到更好效果
- **vs SPO**: SPO 需要在线采样和评估，CRAFT 完全离线且快 20 倍
- **vs RLHF/GRPO**: CRAFT 理论证明与 RL 等价但实现简单得多

## 评分

- 新颖性: ⭐⭐⭐⭐ 组合奖励过滤新颖，理论联系有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 多基准、多指标、多基线对比全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，理论和实验结合好
- 价值: ⭐⭐⭐⭐⭐ 极高实用价值，大幅降低扩散模型对齐成本

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Aligning Text to Image in Diffusion Models is Easier Than You Think](../../NeurIPS2025/image_generation/aligning_text_to_image_in_diffusion_models_is_easier_than_you_think.md)
- [\[CVPR 2026\] Memory-Efficient Fine-Tuning Diffusion Transformers via Dynamic Patch Sampling and Block Skipping](memory-efficient_fine-tuning_diffusion_transformers_via_dynamic_patch_sampling_a.md)
- [\[CVPR 2026\] RewardFlow: Generate Images by Optimizing What You Reward](rewardflow_generate_images_by_optimizing_what_you_reward.md)
- [\[CVPR 2026\] Low-Resolution Editing is All You Need for High-Resolution Editing](low-resolution_editing_is_all_you_need_for_high-resolution_editing.md)
- [\[CVPR 2026\] Cycle-Consistent Tuning for Layered Image Decomposition](cycle-consistent_tuning_for_layered_image_decomposition.md)

</div>

<!-- RELATED:END -->
