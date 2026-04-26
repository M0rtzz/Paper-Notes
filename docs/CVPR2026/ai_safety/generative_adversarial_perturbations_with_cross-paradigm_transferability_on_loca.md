---
title: >-
  [论文解读] Generative Adversarial Perturbations with Cross-paradigm Transferability on Localized Crowd Counting
description: >-
  [CVPR 2026][AI安全][对抗攻击] 提出首个跨范式（密度图 + 点回归）对抗攻击框架 CrowdGen，利用轻量级 UNet 生成器和多任务损失（logit 抑制 + 密度抑制 + GradCAM 引导 + 频域约束），在保持视觉隐蔽性（~19dB PSNR）的同时实现对七个 SOTA 人群计数模型的高迁移率（TR 最高 1.69），攻击 MAE 平均提升 7 倍。
tags:
  - CVPR 2026
  - AI安全
  - 对抗攻击
  - 人群计数
  - 跨范式可迁移性
  - 生成对抗扰动
  - 黑盒攻击
---

# Generative Adversarial Perturbations with Cross-paradigm Transferability on Localized Crowd Counting

**会议**: CVPR 2026  
**arXiv**: [2603.24821](https://arxiv.org/abs/2603.24821)  
**代码**: https://github.com/simurgh7/CrowdGen (有)  
**领域**: AI Safety / 对抗攻击  
**关键词**: 对抗攻击, 人群计数, 跨范式可迁移性, 生成对抗扰动, 黑盒攻击

## 一句话总结
提出首个跨范式（密度图 + 点回归）对抗攻击框架 CrowdGen，利用轻量级 UNet 生成器和多任务损失（logit 抑制 + 密度抑制 + GradCAM 引导 + 频域约束），在保持视觉隐蔽性（~19dB PSNR）的同时实现对七个 SOTA 人群计数模型的高迁移率（TR 最高 1.69），攻击 MAE 平均提升 7 倍。

## 研究背景与动机
定位化人群计数广泛应用于公共安全、零售分析和流行病跟踪等场景，目前主流方案分为两大范式：**密度图方法**（如 SASNet、FIDTM）通过回归空间密度分布再后处理提取定位，以及**点回归方法**（如 P2PNet、PET）端到端输出坐标和置信度。

现有对抗攻击存在以下痛点：

**攻击强度与隐蔽性矛盾**：PAP、GE-AdvGAN 视觉质量好（PSNR≥22dB）但攻击弱（MAE<120）；DiffAttack 攻击强（MAE=414）但视觉塌陷（PSNR=11.5dB）

**单范式局限**：已有可迁移攻击（APAM、PAP）仅在密度图方法间迁移，未考虑跨范式（密度图 ↔ 点回归）

**黑盒场景需求**：实际部署的人群计数系统通常是黑盒的，需要基于代理模型的迁移攻击

核心 idea：利用两种范式共享的 backbone 特征空间（如 VGG-16, ResNet-50）的归纳偏置，通过范式特有的攻击损失 + 范式无关的感知约束，学习一个统一的生成式扰动器。

## 方法详解

### 整体框架
框架包含三个核心部分：一个 3 层 UNet 生成器 $G_\theta$ 将输入图像映射为有界扰动 $\delta$，总损失由**范式特有损失** $\mathcal{L}_{model}$ 和**跨范式扰动损失** $\mathcal{L}_{pert}$ 组成。训练时使用代理模型攻击，推理时单次前向即可生成对抗样本，无需逐图优化。

### 关键设计

1. **Logit 抑制损失（针对点回归模型）**：

    - 针对高置信检测 $\mathcal{P}_{high} = \{i : s_i^{(h)} > \tau\}$ 进行攻击
    - 密集场景（$C_{gt} > C_{sparse}$）：直接最小化高置信区域 logit 值
    - 稀疏场景（$C_{gt} \le C_{sparse}$）：对置信边界附近的检测加权惩罚
    - 自适应阈值衰减：$\tau(t) = \max(\tau_{min}, \tau_{max} - \nu \cdot t / T_{max})$，随训练推进降低阈值，扩大攻击面

2. **密度抑制损失（针对密度图模型）**：

    - **Heatmap 抑制** $\mathcal{L}_{hmap}$：同时攻击显著峰值和阈值附近区域，通过 3×3 max-pooling 检测局部极大值，自适应阈值 $\phi = \phi' \cdot \max(\mathcal{D})$ 分离前景
    - **峰值抑制** $\mathcal{L}_{peak}$：额外加入峰值prominence（峰值与局部邻域差值），侧重孤立的高密度聚集区
    - 通过孤立比率（5×5 窗口内无邻近峰的比例 > 0.7）自动选择使用哪个损失

3. **跨范式扰动损失**：

    - **频域约束** $\mathcal{L}_{freq}$：通过 FFT 抑制高频分量，利用人群场景低频主导的统计特性提升迁移性
    - **GradCAM 引导** $\mathcal{L}_{cam}$：将扰动集中到共享 backbone 认为语义重要的区域，最小化注意力区域外的扰动
    - **幅度约束** $\mathcal{L}_{hinge}$：L2 范数限制扰动能量
    - **空间平滑正则** $\mathcal{L}_{tv}$：总变差正则化减少扰动伪影

### 损失函数 / 训练策略
总损失 $\mathcal{L}_{attack} = \alpha \cdot \mathcal{L}_{model} + \beta \cdot \mathcal{L}_{hinge} + \gamma \cdot \mathcal{L}_{tv} + \zeta \cdot \mathcal{L}_{freq} + \kappa \cdot \mathcal{L}_{cam}$

- 扰动上限 $\epsilon = 8/255$，图像尺寸 512×512
- 使用 cosine annealing 调整学习率
- 超参通过验证集网格搜索：$\beta=0.01, \gamma=0.05, \zeta=0.01, \kappa=0.5$

## 实验关键数据

### 主实验（跨模型迁移性，SHHA 数据集）

| 代理模型 → 目标模型 | MAE / TR | 说明 |
|---------|---------|------|
| HMoDE → P2PNet | 420.71 / **1.69** | 跨范式超迁移：比白盒自身还强 |
| FIDTM → P2PNet | 426.89 / 1.64 | 密度图→点回归 强迁移 |
| SASNet → APGCC | 397.96 / 1.32 | 密度图→点回归 |
| P2PNet → SASNet | 281.00 / 0.89 | 点回归→密度图 |
| APGCC → HMoDE | 171.53 / **0.55** | 最弱迁移但 MAE 仍翻倍 |
| Clean baseline | 28-75 | 干净图像计数误差 |

### 消融实验（损失组合，SHHA 数据集）

| 损失组合 | Miss Rate(%) | PSNR(dB) | 说明 |
|---------|-------------|----------|------|
| $\mathcal{L}_{hmap} + \mathcal{L}_{hinge}$（基线） | 45.35 | 17.67 | 仅基础密度攻击 |
| + $\mathcal{L}_{cam}$ | 59.47 | 17.67 | GradCAM带来+14%MR |
| + $\mathcal{L}_{freq}$ | 60.46 | 17.75 | 频域约束也显著提升 |
| 全组合（密度图） | **60.89** | 17.47 | 最优攻击力 |
| $\mathcal{L}_{logit} + \mathcal{L}_{hinge}$（基线） | 45.15 | 19.09 | 仅基础logit攻击 |
| + $\mathcal{L}_{cam}$ | **45.61** | **19.10** | 点回归最优权衡 |

### 关键发现
- **跨范式超迁移**：CNN-based HMoDE 攻击 Transformer-based PET 的 TR 达 1.60（UCF-QNRF），证实共享 backbone 的归纳偏置假设
- 损失组件的影响**取决于范式**：频域约束对密度图至关重要，GradCAM 引导对点回归更有益
- 密集场景下 miss rate 达 58%，隐藏了大部分人群，而此前方法仅 15-31%

## 亮点与洞察
- 首次揭示人群计数模型的跨范式对抗脆弱性，TR>1 的"超迁移"现象表明黑盒攻击者可能比白盒攻击者更有效
- 场景密度自适应的 logit 抑制策略（密集 vs 稀疏分支）精巧地处理了不同场景特性
- 生成式单次前向攻击比迭代优化方法更实用，推理效率高

## 局限与展望
- 仅验证了数字域攻击，未考虑物理世界（打印、投影）场景
- 攻击主要采用 under-counting 策略，over-counting（幻造人群）方向未探索
- 扰动上限 $\epsilon = 8/255$ 比较标准，更小扰动下的效果待验证
- 缺乏对抗防御方法的对抗实验

## 相关工作与启发
- 可为人群计数系统的鲁棒性评估提供标准化 benchmark
- GradCAM 引导的扰动分配思路可推广到其他密集预测任务的对抗攻击
- 跨范式共享 backbone 脆弱性的发现对模型部署的安全策略有重要参考

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个跨范式人群计数对抗攻击，但生成式对抗扰动框架本身不算全新
- 实验充分度: ⭐⭐⭐⭐ 7个模型×2个数据集的迁移矩阵+9种baseline对比+消融，较完整
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，公式完整，但部分符号较重
- 价值: ⭐⭐⭐⭐ 对安全关键的人群分析系统的脆弱性有重要揭示

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models](multi-paradigm_collaborative_adversarial_attack_against_multi-modal_large_langua.md)
- [\[NeurIPS 2025\] Boosting Adversarial Transferability with Spatial Adversarial Alignment](../../NeurIPS2025/ai_safety/boosting_adversarial_transferability_with_spatial_adversarial_alignment.md)
- [\[ICCV 2025\] Enhancing Adversarial Transferability by Balancing Exploration and Exploitation with Gradient-Guided Sampling](../../ICCV2025/ai_safety/enhancing_adversarial_transferability_by_balancing_exploration_and_exploitation_.md)
- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](../../AAAI2026/ai_safety/rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[CVPR 2026\] V-Attack: Targeting Disentangled Value Features for Controllable Adversarial Attacks on LVLMs](v-attack_targeting_disentangled_value_features_for_controllable_adversarial_atta.md)

<!-- RELATED:END -->
