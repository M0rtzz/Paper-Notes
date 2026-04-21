---
title: >-
  [论文解读] ScaleDreamer: Scalable Text-to-3D Synthesis with Asynchronous Score Distillation
description: >-
  [ECCV 2024][图像生成][文本到3D] 本文提出异步分数蒸馏（ASD），通过将扩散时间步前移（而非微调扩散模型）来降低噪声预测误差、对齐渲染图像分布，解决了VSD微调破坏文本理解能力的问题，实现了稳定训练且可扩展至10万条文本提示的prompt-amortized 3D生成器训练。
tags:
  - ECCV 2024
  - 图像生成
  - 文本到3D
  - 分数蒸馏
  - 扩散模型
  - 异步时间步
  - 大规模3D生成
---

# ScaleDreamer: Scalable Text-to-3D Synthesis with Asynchronous Score Distillation

**会议**: ECCV 2024  
**arXiv**: [2407.02040](https://arxiv.org/abs/2407.02040)  
**代码**: https://github.com/theEricMa/ScaleDreamer  
**领域**: 图像生成  
**关键词**: 文本到3D, 分数蒸馏, 扩散模型, 异步时间步, 大规模3D生成

## 一句话总结
本文提出异步分数蒸馏（ASD），通过将扩散时间步前移（而非微调扩散模型）来降低噪声预测误差、对齐渲染图像分布，解决了VSD微调破坏文本理解能力的问题，实现了稳定训练且可扩展至10万条文本提示的prompt-amortized 3D生成器训练。

## 研究背景与动机

**领域现状**：文本到3D方法利用预训练2D扩散模型的先验知识，通过分数蒸馏（Score Distillation）优化3D表示。Prompt-specific方法（SDS、VSD）每个提示需单独优化数小时；Prompt-amortized方法训练一个文本到3D生成网络后可在秒级推理，但依赖高效的分数蒸馏。

**现有痛点**：
   - **SDS**：假设渲染图像是Dirac分布，导致梯度极大（需CFG=100），训练数值不稳定，3DConv-Net等深层网络几千步就崩溃
   - **CSD**：用无条件扩散项替代，但无条件项不依赖文本，无法为不同提示提供有效梯度
   - **VSD**：微调扩散模型来对齐渲染图像分布，但微调**破坏了预训练模型对文本的理解能力**，在大规模提示下导致模式坍塌

**核心矛盾**：VSD的核心目标是降低噪声预测误差来对齐分布，但它通过微调扩散模型来实现这一目标，形成了一个双层优化问题（类似GAN的交替训练），在大量提示下不稳定且损害文本理解。

**本文目标** 能否**不改变扩散模型权重**就降低噪声预测误差？

**切入角度**：作者发现扩散模型在**更早的时间步**（t更大，更接近纯噪声）上天然有更低的噪声预测误差。这是因为当 $t \to T_{max}$ 时，$\mathbf{x}_t \to \boldsymbol{\epsilon}$，模型可以"复制输入作为输出"来准确预测噪声。

**核心 idea**：通过将扩散时间步前移 $\Delta t$ 来近似VSD微调的效果——不需改变扩散模型权重，保持其强大的文本理解能力，即可实现分布对齐。

## 方法详解

### 整体框架
ASD与SDS/VSD共享相同的文本到3D流程：文本→3D生成器→渲染图像→加噪→扩散模型预测噪声→计算梯度→更新3D生成器。关键区别在于梯度计算方式。ASD在两个异步时间步（$t$ 和 $t + \Delta t$）分别对渲染图像加噪并预测噪声，用两个预测的差作为梯度。

### 关键设计

1. **异步分数蒸馏目标函数**:

    - 功能：用时间步前移替代VSD的扩散模型微调
    - 核心思路：$\nabla_\theta \mathcal{L}_{ASD} = \mathbb{E}_{t, \boldsymbol{\epsilon}} [\omega(t)(\boldsymbol{\epsilon}_\phi(\mathbf{x}_t; t, y) - \boldsymbol{\epsilon}_\phi(\mathbf{x}_{t+\Delta t}; t+\Delta t, y)) \frac{\partial \mathbf{x}}{\partial \theta}]$
    - 对比：SDS用 $\boldsymbol{\epsilon}$（真实噪声），CSD用 $\boldsymbol{\epsilon}_\phi(\mathbf{x}_t; t)$（无条件预测），VSD用 $\boldsymbol{\epsilon}_{\phi'}$（微调模型预测），ASD用 $\boldsymbol{\epsilon}_\phi(\mathbf{x}_{t+\Delta t}; t+\Delta t, y)$（前移时间步的预测）
    - 优势：冻结扩散模型权重，无双层优化，保持文本理解能力

2. **时间步偏移 $\Delta t$ 的设置**:

    - 功能：为不同时间步动态设置前移量
    - 核心思路：$\Delta t \sim \mathcal{U}[0, \eta(t - T_{min})]$，均匀随机采样
    - 设计动机：(a) $\Delta t$ 随 $t$ 增大而增大——因为越接近 $T_{max}$ 误差曲线越平坦，需要更大偏移才能匹配微调模型的误差；(b) 随机采样而非确定性——因为不同训练迭代、不同图像、不同提示下最优偏移不同
    - 超参数 $\eta = 0.1$：$\eta$ 过大会退化为SDS，过小则没有充分利用前移带来的降噪效果

3. **与多种3D生成器的兼容性**:

    - 功能：验证ASD作为通用分数蒸馏方法可与不同3D架构配合
    - 三种生成器：Hyper-iNGP（超网络+哈希编码）、3DConv-Net（3D卷积体素）、Triplane-Transformer（三平面+Transformer）
    - 两种扩散模型：Stable Diffusion 和 MVDream（多视角扩散）
    - 设计动机：证明ASD不依赖特定架构，是真正通用的分数蒸馏方法

### 损失函数
无额外损失，核心就是ASD梯度。CFG设为7.5（SDS需要100），梯度范围与VSD相当。

## 实验关键数据

### Prompt-Specific vs Prompt-Amortized (Hyper-iNGP, MG15)

| 设置 | 方法 | CLIP Sim↑ | R@1↑ |
|------|------|-----------|------|
| Prompt-Specific (iNGP) | SDS | 0.288 | 1.000 |
| Prompt-Specific (iNGP) | VSD | 0.276 | 0.932 |
| Prompt-Specific (iNGP) | **ASD** | **0.289** | **1.000** |
| Prompt-Amortized (Hyper-iNGP) | ATT3D(SDS) | 0.195 | 0.468 |
| Prompt-Amortized (Hyper-iNGP) | SDS | 0.257 | 0.918 |
| Prompt-Amortized (Hyper-iNGP) | VSD | 0.259 | 0.987 |
| Prompt-Amortized (Hyper-iNGP) | **ASD** | **0.284** | **1.000** |

ASD在prompt-specific和prompt-amortized上都取得最优，且从specific到amortized几乎没有性能下降。

### 大规模可扩展性 (3DConv-Net)

| 提示集 | 方法 | Sim↑ | R@1↑ |
|--------|------|------|------|
| DF415 | SDS | ×(崩溃) | ×(崩溃) |
| DF415 | CSD | 0.176 | 0.062 |
| DF415 | VSD | 0.158 | 0.002 |
| DF415 | **ASD** | **0.237** | **0.276** |
| CP100k | CSD | 0.195 | 0.108 |
| CP100k | VSD | 0.103 | 0.000 |
| CP100k | **ASD** | **0.199** | **0.117** |

VSD在10万提示下完全崩溃（R@1=0.000），ASD保持有效。SDS在3DConv-Net上训练即崩溃。

### 消融实验（$\Delta t$ 设置）

| 设置 | Sim↑ | R@1↑ | 说明 |
|------|------|------|------|
| $\eta=0$（无前移） | 0.235 | 0.267 | Janus问题严重 |
| 确定性 $\Delta t = 0.1(t-T_{min})$ | 0.214 | 0.178 | 几何颜色不准确 |
| 确定性 $\Delta t = 0.2(t-T_{min})$ | 0.214 | 0.180 | 类似上行 |
| 随机 $\Delta t \sim \mathcal{U}[0, 0.1(t-T_{min})]$ | **0.237** | **0.276** | **最优** |
| 随机 $\Delta t \sim \mathcal{U}[0, 0.2(t-T_{min})]$ | 0.229 | 0.237 | 形状偏大偏圆 |

### 关键发现
- **时间步偏移是必要的**：$\eta=0$ 条件下Janus问题严重（青蛙多眼、孔雀前后都有尾巴），因为未对齐的扩散模型倾向于生成每个视角都像正面的内容
- **随机采样比确定性好得多**：确定性偏移R@1仅0.178，随机采样达0.276
- **$\eta$ 不能太大**：$\eta=0.2$ 时形状偏大偏圆，因为过大偏移退化为SDS
- **VSD在大规模下模式坍塌**：10万提示时R@1=0.000，所有输出几乎相同，验证了微调破坏文本理解的假设
- **SDS对深层网络致命**：3DConv-Net中SDS始终崩溃，印证了大梯度对深层网络的破坏性
- ASD与MVDream配合也优于SDS*，生成更自然的几何和纹理

## 亮点与洞察
- **时间步前移的洞察极具启发性**：扩散模型在更早时间步有更低噪声预测误差是一个简单但深刻的观察。用这个性质优雅地替代VSD的微调，理论和实践上都很漂亮
- **冻结扩散模型保持文本理解**：这个设计选择在大规模训练中至关重要。VSD的双层优化在小规模work，但扩展到10万提示时彻底失败——这为分数蒸馏的可扩展设计提供了重要教训
- **ASD是真正通用的分数蒸馏**：与三种3D生成器、两种扩散模型都兼容，代码改动极小（只是改采样时间步），实用价值高
- **CP100k提示集**：首次在10万提示规模上评估分数蒸馏，为后续研究提供了基准

## 局限性
- 对于形状规则的人造物体（椅子、飞机），性能仍落后于数据驱动方法，因为缺乏3D训练数据
- 10万提示下的绝对质量仍有提升空间（R@1仅0.117），距离实际应用有距离
- $\eta$ 需要手动调节，且最优值可能在不同扩散模型或生成器间不同
- 没有对比ISM（Interval Score Matching）等其他近期分数蒸馏方法
- 3DConv-Net和Triplane-Transformer的渲染分辨率限制在64×64，实际效果有限

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 时间步前移的思路新颖简洁，理论动机清晰优美
- 实验充分度: ⭐⭐⭐⭐⭐ 三种生成器、两种扩散模型、从15到10万提示的全面评估
- 写作质量: ⭐⭐⭐⭐⭐ 问题分析和方法推导逻辑严密，误差曲线图直觉性强
- 价值: ⭐⭐⭐⭐⭐ 解决了分数蒸馏在大规模训练中的核心瓶颈，对文本到3D领域有重要推动

<!-- RELATED:START -->

## 相关论文

- [Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)
- [ShapeFusion: A 3D Diffusion Model for Localized Shape Editing](shapefusion_a_3d_diffusion_model_for_localized_shape_editing.md)
- [Editable Image Elements for Controllable Synthesis](editable_image_elements_for_controllable_synthesis.md)
- [Ponymation: Learning Articulated 3D Animal Motions from Unlabeled Online Videos](ponymation_learning_articulated_3d_animal_motions_from_unlabeled_online_videos.md)
- [Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis](rejection_sampling_imle_designing_priors_for_better_few-shot_image_synthesis.md)

<!-- RELATED:END -->
