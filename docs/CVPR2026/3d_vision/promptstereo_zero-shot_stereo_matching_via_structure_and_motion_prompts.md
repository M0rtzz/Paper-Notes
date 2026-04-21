---
title: >-
  [论文解读] PromptStereo: Zero-Shot Stereo Matching via Structure and Motion Prompts
description: >-
  [CVPR 2026][3D视觉][零样本立体匹配] 提出 Prompt Recurrent Unit (PRU)，将单目深度基础模型的 DPT 解码器作为迭代精炼模块（替代 GRU），通过 Structure Prompt 和 Motion Prompt 将单目结构和立体运动线索以残差方式注入，在不破坏单目先验的情况下实现零样本 SOTA 立体匹配（Middlebury 2021 上误差降低近50%）。
tags:
  - CVPR 2026
  - 3D视觉
  - 零样本立体匹配
  - 单目深度先验
  - 提示学习
  - DPT解码器
  - 仿射不变融合
---

# PromptStereo: Zero-Shot Stereo Matching via Structure and Motion Prompts

**会议**: CVPR 2026  
**arXiv**: [2603.01650](https://arxiv.org/abs/2603.01650)  
**代码**: [GitHub](https://github.com/Windsrain/PromptStereo)  
**领域**: 3D视觉  
**关键词**: 零样本立体匹配, 单目深度先验, Prompt迭代精炼, DPT解码器, 仿射不变融合

## 一句话总结
提出 Prompt Recurrent Unit (PRU)，将单目深度基础模型的 DPT 解码器作为迭代精炼模块（替代 GRU），通过 Structure Prompt 和 Motion Prompt 将单目结构和立体运动线索以残差方式注入，在不破坏单目先验的情况下实现零样本 SOTA 立体匹配（Middlebury 2021 上误差降低近50%）。

## 研究背景与动机
**领域现状**：零样本立体匹配近年受到越来越多关注。得益于 Depth Anything V2 等单目深度基础模型的强泛化能力，最新方法通过适配预训练特征来提升泛化性能。

**现有痛点**：
   - 现有方法（MonSter、DEFOM-Stereo、BridgeDepth）主要利用单目模型提取鲁棒特征构建代价体积和初始化视差，但**迭代精炼阶段**仍依赖传统 GRU，这一阶段对零样本泛化同样至关重要却被忽视
   - GRU 的三个根本局限：(a) 独立于视觉基础模型训练，无法继承强先验；(b) 隐状态被限制在窄范围内（tanh/sigmoid），难以处理极端视差变化；(c) 通过直接卷积融合输入和隐状态，既扭曲原始状态信息又压缩外部输入

**核心矛盾**：如何让迭代精炼模块也能继承单目深度基础模型的强先验，同时有效融合立体匹配特有的运动线索

**切入角度**：注意到 DPT 解码器也是多尺度的精炼结构，与 GRU 的粗到细更新有结构相似性。可以直接用预训练的 DPT 解码器作为迭代单元

**核心 idea**：用预训练 DPT 解码器替代 GRU 作为迭代精炼单元，通过 prompt（残差加法）注入立体匹配特有的结构和运动线索，不修改解码器架构即可继承单目先验

## 方法详解

### 整体框架
以 MonSter 为基线。输入立体图像对 → Depth Anything V2 提取单目特征+相对深度 → MonSter 特征编码器提取多尺度立体特征 → 代价体积构建+初始视差回归 → Affine-Invariant Fusion 融合初始视差和单目深度 → PRU 迭代精炼 → 输出最终视差。

### 关键设计

1. **仿射不变融合（Affine-Invariant Fusion, AIF）**

    - 功能：将初始视差 $\mathbf{d}_0$ 和单目相对深度 $\mathbf{d}_M$ 在归一化尺度下融合
    - 核心思路：对两个深度/视差分别做仿射不变归一化 $\hat{\mathbf{d}} = (\mathbf{d} - t(\mathbf{d})) / s(\mathbf{d})$，其中 $t = \text{median}$，$s = \text{MAD}$。将归一化后的单目深度投影到视差空间：$\mathbf{d}_M' = s(\mathbf{d}_0) \cdot \hat{\mathbf{d}}_M + t(\mathbf{d}_0)$。用 $\mathbf{d}_0$ warp 右特征，与左特征拼接后预测逐像素置信图 $\mathbf{c}$：$\mathbf{d}_F = \mathbf{c} \odot \mathbf{d}_0 + (1-\mathbf{c}) \odot \mathbf{d}_M'$
    - 设计动机：代价体积初始视差有局部匹配精度但缺乏全局一致性；单目深度有全局结构但存在仿射歧义。归一化后融合可以互补各自优势

2. **Prompt Recurrent Unit (PRU)**

    - 功能：替代 GRU 作为迭代精炼的核心单元
    - 核心思路：采用 Depth Anything V2 的 DPT 精炼层作为多分辨率架构（4级），用预训练权重初始化，直接继承单目深度先验。隐状态初始化为左右特征的拼接（经 $\mathbf{d}_0$ warp 右特征），比传统 GRU 仅用左特征初始化更早学习立体对应
    - 更新策略：去掉 GRU 的 reset gate，仅保留 update gate $\mathbf{z}_k = \sigma(\text{ConvBlock}([\cdot]))$。隐状态更新 $\mathbf{h}_{k+1}^i = (1-\mathbf{z}_k) \odot \mathbf{h}_k^i + \mathbf{z}_k \odot \hat{\mathbf{h}}_k^i$，**不限制隐状态的值域范围**
    - 设计动机：GRU 的 tanh 限制了隐状态范围，在极端视差场景下表达力不足。PRU 的 DPT 架构天然支持多分辨率，且预训练权重提供了强初始化

3. **Structure Prompt (SP)**

    - 功能：将冻结的单目深度特征 $\mathbf{F}_M$ 和结构差异信息以 prompt 方式注入 PRU
    - 核心思路：计算当前视差与单目深度的仿射不变差异 $\mathbf{D} = |\hat{\mathbf{d}}_k - \hat{\mathbf{d}}_M|$，与 $\mathbf{F}_M$ 一起编码为结构 prompt $\mathbf{P}_S$，以残差加法注入隐状态：$\mathbf{h} = \mathbf{h} + \text{ConvBlock}(\mathbf{P}_S)$
    - 设计动机：直接卷积融合会扭曲 DPT 继承的单目先验。残差加法作为 feature-level prompt 引导隐状态而不破坏已有表示

4. **Motion Prompt (MP)**

    - 功能：将立体匹配特有的运动线索（局部代价体积和当前视差）注入 PRU
    - 核心思路：$\mathbf{P}_M^k = \text{Encoder}(\mathbf{V}_k, \mathbf{d}_k)$，同样以残差加法注入：$\mathbf{h} = \mathbf{h} + \text{ConvBlock}(\mathbf{P}_M^k)$
    - 设计动机：DPT 解码器只有单目先验，缺乏立体匹配的运动信息。Motion Prompt 自适应补充立体对应关系

### 损失函数 / 训练策略
- 跟随 IGEV-Stereo：$\mathcal{L} = \|\mathbf{d}_0 - \mathbf{d}_{gt}\|_{\text{smooth}} + \sum_{k=1}^K \gamma^{K-k} \|\mathbf{d}_k - \mathbf{d}_{gt}\|_1$，$\gamma = 0.9$
- 训练16次迭代，推理32次
- 冻结 DINOv2 编码器+单目特征分支，保持单目先验不被破坏
- 4×RTX 4090，AdamW，one-cycle LR 2e-4

## 实验关键数据

### 主实验——零样本泛化基础基准（SceneFlow 训练）

| 方法 | KITTI12 EPE↓ | KITTI15 Bad3↓ | Midd-T Bad2↓ | Midd-2021 Bad2↓ | ETH3D Bad1↓ |
|------|-------------|-------------|-------------|----------------|------------|
| RAFT-Stereo | 0.90 | 5.68 | 11.07 | 11.11 | 2.61 |
| IGEV-Stereo | 1.03 | 6.03 | 9.95 | 10.00 | 4.05 |
| MonSter | 0.93 | 5.52 | 8.97 | 15.55 | 3.20 |
| BridgeDepth | 0.83 | 4.69 | 7.84 | 15.92 | 1.26 |
| DEFOM-Stereo | 0.83 | 4.99 | 6.77 | 8.62 | 2.40 |
| **PromptStereo** | **0.79** | **4.59** | **6.03** | **8.26** | **1.56** |

### 主实验——无限训练集

| 方法 | Midd-T Bad2↓ | Midd-2021 Bad2↓ | ETH3D Bad1↓ |
|------|-------------|----------------|------------|
| FoundationStereo† | 3.11 | 7.14 | 0.67 |
| MonSter | 5.51 | 12.43 | 1.25 |
| BridgeDepth | 3.36 | 13.66 | 1.22 |
| **PromptStereo** | **3.90** | **5.97** | **0.97** |

### 关键发现
- 相比基线 MonSter，PromptStereo 在 Middlebury 2021 上误差降低近 50%（15.55→8.26，SceneFlow设定；12.43→5.97，无限设定），这是最具挑战的数据集（手机拍摄、不完美矫正）
- 在 SceneFlow 训练设定下几乎所有指标排名第一，在无限训练设定下 Midd-2021 和 ETH3D 上超越了 FoundationStereo（后者需要显著更多的计算资源训练不可公平比较）
- PRU 继承了 DPT 预训练权重，提供了 GRU 无法获得的视觉理解能力和表示容量
- Prompt（残差加法）方式注入信息不破坏预训练先验，优于直接卷积融合

## 亮点与洞察
- **用解码器当循环单元的思路非常巧妙**：DPT 解码器和多级 GRU 在结构上都是多分辨率精炼，这个类比使得替换成为自然设计。预训练权重赋予了 PRU 视觉 foundation model 的表征能力
- **Prompt 式信息注入**：残差加法 $\mathbf{h} = \mathbf{h} + \text{ConvBlock}(\mathbf{P})$ 比直接卷积融合更温和，不会扭曲已有表示。Structure Prompt 中仿射不变差异的使用避免了尺度歧义
- **去掉 reset gate 并放开隐状态范围**：简化了 GRU 结构同时提供了更灵活的表示空间，这在极端视差场景（近距离物体）下尤其重要
- **AIF 的仿射不变归一化**：用 median + MAD 做归一化是鲁棒统计的经典做法，应用于视差-深度融合非常自然

## 局限与展望
- PRU 使用 DPT 解码器，推理速度是否与 GRU 可比？论文称"comparable or faster"但未给出详细的逐模块时间对比
- Booster 数据集（反射/透明表面）上仅用 SceneFlow 训练时效果有限，说明 PRU 的泛化能力仍受训练数据覆盖度限制
- 仅冻结了 DINOv2 编码器和单目分支，DPT 解码器本身被微调——如果训练数据有偏差可能破坏预训练先验
- 结构和运动 prompt 仅在最高分辨率层注入，其他层是否也能从 prompt 中受益未探索

## 相关工作与启发
- **vs MonSter**：MonSter 用 Depth Anything V2 做特征提取和初始化，但迭代精炼仍用 GRU。PromptStereo 将单目先验扩展到迭代精炼阶段，在 Midd-2021 上误差减半
- **vs BridgeDepth**：BridgeDepth 也用单目先验引导 GRU 迭代，但受限于 GRU 的表达能力。PRU 直接用 DPT 解码器替代 GRU，容量和先验都更强
- **vs FoundationStereo**：FoundationStereo 依赖大规模数据集和大模型训练（不可公平比较），PromptStereo 在可比设置下达到同等甚至更好的泛化性能

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用预训练DPT解码器替代GRU是一个具有范式意义的设计，prompt注入方式优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 5+数据集、多训练设定、消融充分、代码开源
- 写作质量: ⭐⭐⭐⭐ 问题分析深入，GRU局限性的三点总结很精准
- 价值: ⭐⭐⭐⭐⭐ 指明了零样本立体匹配中迭代精炼的新方向，成果显著

<!-- RELATED:START -->

## 相关论文

- [Lite Any Stereo: Efficient Zero-Shot Stereo Matching](lite_any_stereo_efficient_zero-shot_stereo_matching.md)
- [FoundationStereo: Zero-Shot Stereo Matching](../../CVPR2025/3d_vision/foundationstereo_zero-shot_stereo_matching.md)
- [WMGStereo: What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching?](what_makes_good_synthetic_training_data_for_zerosh.md)
- [What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching?](what_makes_good_synthetic_training_data_for_zero-shot_stereo_matching.md)
- [PIP-Stereo: Progressive Iterations Pruner for Iterative Optimization based Stereo Matching](pip-stereo_progressive_iterations_pruner_for_iterative_optimization_based_stereo.md)

<!-- RELATED:END -->
