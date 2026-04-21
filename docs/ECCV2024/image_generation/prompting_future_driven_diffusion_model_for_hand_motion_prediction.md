---
title: >-
  [论文解读] Prompting Future Driven Diffusion Model for Hand Motion Prediction
description: >-
  [ECCV 2024][图像生成][手部运动预测] 本文提出PromptFDDM，一个基于prompt的未来驱动扩散模型用于手部运动预测，通过空间-时间提取网络(STEN)结合Ground Truth提取网络(GTEN)和参考数据生成网络(RDGN)的引导机制，以及交互式prompt增强，在第一和第三人称视角的手部运动预测中达到SOTA。
tags:
  - ECCV 2024
  - 图像生成
  - 手部运动预测
  - 扩散模型
  - 提示学习
  - 第一人称/第三人称
  - 未来驱动
---

# Prompting Future Driven Diffusion Model for Hand Motion Prediction

**会议**: ECCV 2024  
**arXiv**: N/A  
**代码**: 无  
**领域**: 扩散模型 / 手部运动预测  
**关键词**: 手部运动预测, 扩散模型, 提示学习, 第一人称/第三人称, 未来驱动

## 一句话总结
本文提出PromptFDDM，一个基于prompt的未来驱动扩散模型用于手部运动预测，通过空间-时间提取网络(STEN)结合Ground Truth提取网络(GTEN)和参考数据生成网络(RDGN)的引导机制，以及交互式prompt增强，在第一和第三人称视角的手部运动预测中达到SOTA。

## 研究背景与动机

**领域现状**：手部运动预测在AR/VR用户体验增强和远程机器人臂安全控制中至关重要。之前的工作主要集中在人体全身运动预测或手部轨迹预测上，直接预测手部骨架运动（关节角度和位置）的工作相对较少。手部运动预测面临独特的挑战：手部骨架紧凑（关节数量多但空间范围小），微小的预测误差会导致严重的姿态失真。

**现有痛点**：(1) 手部骨架紧凑性问题：相比全身骨架，手部关节在空间上高度密集，信噪比低，传统的轨迹预测方法难以捕捉精细的手指运动；(2) 现有方法大多只关注第三人称视角，忽略了AR/VR中更重要的第一人称视角；(3) 确定性预测方法无法建模手部运动的多义性——相同的观察序列可能对应多种合理的未来运动。

**核心矛盾**：手部运动具有高度的不确定性和多义性（一个手势可以有多种后续运动），但同时需要精确预测（因为骨架紧凑，容错空间小）。简单使用扩散模型进行概率预测会导致多样性过高而精度不足。

**本文目标** (1) 如何在手部运动预测中平衡多样性和精度；(2) 如何利用未来信息引导扩散模型学习更精确的预测；(3) 如何通过prompt机制增强模型对观测运动的理解。

**切入角度**：在训练阶段利用ground truth未来运动作为引导信号，训练一个参考数据生成网络(RDGN)来模拟这种引导。推理时RDGN生成"替代未来数据"来引导预测。同时从观测运动中提取交互式prompt，提供额外的运动上下文。

**核心 idea**：通过"未来驱动"机制（训练时用GT引导，推理时用生成的参考数据替代）和基于观测的交互式prompt来引导扩散模型进行精确的手部运动预测。

## 方法详解

### 整体框架
PromptFDDM包含三个主要网络：(1) 空间-时间提取网络(STEN)，核心的预测网络，使用扩散过程生成未来手部运动；(2) Ground Truth提取网络(GTEN)，训练时从真实未来运动中提取引导特征；(3) 参考数据生成网络(RDGN)，推理时生成替代未来数据来代替不可得的GT。此外还有一个prompt生成模块从观测运动中提取交互式prompt。

### 关键设计

1. **空间-时间提取网络(Spatial-Temporal Extractor Network, STEN)**:

    - 功能：核心预测网络，基于扩散过程在引导信号和prompt条件下预测未来手部运动
    - 核心思路：STEN接收三种输入：(a) 噪声化的未来运动 $x_t$（扩散过程中的噪声样本），(b) 来自GTEN（训练时）或RDGN（推理时）的引导特征 $g$，(c) 交互式prompt $p$。网络内部使用时空Transformer架构，空间维度上捕捉不同关节间的依赖关系，时间维度上建模运动序列的动态特征。引导特征通过cross-attention注入，prompt通过adaptive layer normalization注入。训练目标为去噪——预测 $x_t$ 中添加的噪声 $\epsilon$
    - 设计动机：时空双维度的建模对手部运动至关重要：空间上手指间有强耦合（抓取时所有手指协调运动），时间上运动有平滑性约束。引导和prompt的双重条件化使模型能利用更丰富的上下文

2. **Ground Truth提取网络(GTEN)与参考数据生成网络(RDGN)**:

    - 功能：GTEN在训练时从真实未来运动提取引导信号；RDGN在推理时生成替代引导信号
    - 核心思路：**GTEN**是一个编码器网络，输入真实的未来手部运动序列 $y$，输出引导特征 $g_{gt} = \text{GTEN}(y)$。这个引导特征包含了"未来运动的概要信息"，帮助STEN更准确地去噪。**RDGN**是一个生成网络，输入观测序列 $x_{obs}$，生成"参考未来数据" $\hat{y} = \text{RDGN}(x_{obs})$，然后通过GTEN编码为引导特征 $g_{ref} = \text{GTEN}(\hat{y})$。RDGN使用简单的GRU网络实现，训练时以MSE损失直接回归未来运动。虽然RDGN的预测可能不够精确，但它提供了合理的"方向性引导"
    - 设计动机：这种"训练时用GT，推理时用生成替代"的策略解决了一个根本矛盾——训练时我们有未来信息可以引导学习，但推理时没有。RDGN桥接了这一差距。即使RDGN的输出不完美，它提供的粗略方向信息也足以帮助STEN在正确的区域内进行精细去噪

3. **交互式Prompt生成**:

    - 功能：从观测到的手部运动序列中提取上下文信息，作为额外的条件注入STEN
    - 核心思路：prompt生成模块分析观测序列中的运动模式，提取关键特征如运动速度、方向趋势、周期性模式等。具体地，对观测序列进行多尺度时间编码——短时窗口捕捉局部动态（如手指弯曲速率），长时窗口捕捉全局趋势（如手臂移动方向）。这些多尺度特征被拼接并通过MLP映射为prompt向量。prompt通过adaptive layer norm注入STEN的每一层
    - 设计动机：观测序列包含了关于未来运动的重要线索——当前的运动趋势很大程度上决定了近期的运动。prompt机制使STEN能够"先理解当前在做什么"再"预测接下来会做什么"

### 损失函数 / 训练策略
STEN使用标准扩散去噪损失 $\mathcal{L}_{STEN} = \mathbb{E}\|\epsilon - \epsilon_\theta(x_t, t, g, p)\|^2$。RDGN使用MSE重建损失 $\mathcal{L}_{RDGN} = \|y - \text{RDGN}(x_{obs})\|^2$。训练分两阶段：先训练RDGN，再固定RDGN训练STEN+GTEN。推理时RDGN和GTEN前向一次生成引导特征，STEN进行迭代去噪。

## 实验关键数据

### 主实验

| 数据集 | 视角 | 指标 (MPJPE↓) | PromptFDDM | 之前SOTA | 提升 |
|--------|------|--------------|-----------|----------|------|
| FPHA | 第一人称 | MPJPE@80ms | **8.2** | 9.7 | -15.5% |
| FPHA | 第一人称 | MPJPE@400ms | **32.1** | 38.6 | -16.8% |
| HO3D | 第三人称 | MPJPE@80ms | **5.4** | 6.8 | -20.6% |
| HO3D | 第三人称 | MPJPE@400ms | **24.8** | 29.3 | -15.4% |

### 消融实验

| 配置 | MPJPE@400ms (FPHA) | 说明 |
|------|-------------------|------|
| Full PromptFDDM | **32.1** | 完整模型 |
| w/o GTEN+RDGN引导 | 39.2 | 无未来驱动引导，掉22.1% |
| w/o Prompt | 35.7 | 无观测prompt，掉11.2% |
| w/o RDGN (推理无引导) | 37.8 | 推理时缺少引导信号 |
| RDGN替代为直接预测 | 36.4 | 不通过GTEN编码 |

### 关键发现
- 未来驱动引导(GTEN+RDGN)是最关键的模块——去掉后性能下降超过20%，说明引导信号对扩散模型的去噪方向至关重要
- Prompt的贡献在长期预测（400ms）中更显著——观测运动的趋势信息对短期预测帮助有限但对长期规划很重要
- 在第一人称视角下性能改善更大，这可能因为第一人称的运动更复杂（受相机自身运动影响）
- RDGN生成的参考数据质量虽然不够精确（MSE较高），但作为引导信号已经足够有效——说明扩散模型只需要"大致方向"的引导

## 亮点与洞察
- **"训练看答案，推理靠估计"的引导策略**很有实际意义。GTEN-RDGN的组合巧妙地解决了信息泄露问题——训练时利用GT引导学习高质量的去噪策略，推理时用生成的替代品近似GT的引导效果。这种teacher-student式的推理策略可以迁移到其他条件扩散模型
- **Prompt注入方式**比简单的条件拼接更灵活——通过adaptive layer norm在每一层都注入prompt，使模型在不同抽象层级上都能利用观测信息
- 手部运动预测这一应用场景选择得好——手部骨架的紧凑性使得传统方法误差放大，但扩散模型的概率性可以有效处理多义性

## 局限与展望
- 只在FPHA和HO3D两个数据集上验证，未涉及更大规模的手部运动数据集
- 未考虑手-物交互的约束——手抓取物体时的运动受物体几何约束，当前方法忽略了这一点
- RDGN作为简单的GRU网络，其生成的参考数据质量有限，可以用更强的模型替代
- 推理速度可能受扩散迭代步数限制，不确定是否满足AR/VR的实时需求
- 可以探索结合手部图像信息（如纹理、深度图）来增强运动预测

## 相关工作与启发
- **vs LTD**: LTD使用离散余弦变换进行人体运动预测，但未针对手部骨架紧凑性做优化。PromptFDDM通过扩散模型更好地处理了手部运动的多义性
- **vs MotionDiff**: MotionDiff使用扩散模型进行全身运动预测，但没有未来驱动引导和prompt机制。PromptFDDM在这两点上的创新使其更适合手部运动预测
- **vs MDM**: MDM使用classifier-free guidance进行运动生成，PromptFDDM的未来驱动引导提供了一种不同的引导范式——基于未来状态而非语义标签

## 评分
- 新颖性: ⭐⭐⭐⭐ 未来驱动引导和交互式prompt的组合新颖
- 实验充分度: ⭐⭐⭐ 只在两个数据集上验证，实时性分析缺失
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，各模块的动机阐释到位
- 价值: ⭐⭐⭐ 手部运动预测场景较窄但对AR/VR有实际价值

<!-- RELATED:START -->

## 相关论文

- [SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)
- [NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model](nl2contact_natural_language_guided_3d_hand-object_contact_modeling_with_diffusio.md)
- [Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction](learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)
- [EMDM: Efficient Motion Diffusion Model for Fast and High-Quality Motion Generation](emdm_efficient_motion_diffusion_model_for_fast_and_high.md)

<!-- RELATED:END -->
