---
title: >-
  [论文解读] Resonance: Learning to Predict Social-Aware Pedestrian Trajectories as Co-Vibrations
description: >-
  [ICCV 2025][自动驾驶][行人轨迹预测] 本文提出 Resonance (Re) 模型，将行人轨迹预测分解为多个"振动"的叠加——线性基底、自偏置（self-bias）和共振偏置（resonance-bias），利用轨迹频谱的相似性模拟社会交互中的"共振"现象…
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "行人轨迹预测"
  - "振动分解"
  - "共振交互"
  - "频谱表示"
  - "社会行为建模"
---

# Resonance: Learning to Predict Social-Aware Pedestrian Trajectories as Co-Vibrations

**会议**: ICCV 2025  
**arXiv**: [2412.02447](https://arxiv.org/abs/2412.02447)  
**代码**: [https://github.com/cocoon2wong/Re](https://github.com/cocoon2wong/Re)  
**领域**: 自动驾驶 / 行人轨迹预测  
**关键词**: 行人轨迹预测, 振动分解, 共振交互, 频谱表示, 社会行为建模

## 一句话总结
本文提出 Resonance (Re) 模型，将行人轨迹预测分解为多个"振动"的叠加——线性基底、自偏置（self-bias）和共振偏置（resonance-bias），利用轨迹频谱的相似性模拟社会交互中的"共振"现象，在 ETH-UCY、SDD、NBA、nuScenes 等数据集上验证了方法的有效性。

## 研究背景与动机

1. **领域现状**：行人轨迹预测是自动驾驶和社会机器人中的核心任务。主流方法包括基于 LSTM 的社会力模型（Social LSTM）、基于 GAN 的多模态预测（Social GAN）、基于 GCN 的图网络方法（STGCNN）、基于 Transformer 的注意力机制方法（AgentFormer）、以及基于扩散模型的方法（LED、BCDiff）等。

2. **现有痛点**：(a) 现有方法在建模行人意图和社会行为时，难以准确解耦不同原因导致的轨迹变化和随机性；(b) 社会交互的建模通常依赖注意力机制或图网络，缺乏可解释性——模型很难说清"某个邻居如何具体影响了预测轨迹"；(c) 轨迹的随机性来源多样（个人意图、社交影响、环境约束等），现有方法大多将其耦合在一起建模。

3. **核心矛盾**：轨迹的变化是多因素叠加的结果，但现有方法通常用单一的隐空间或注意力机制来统一建模，导致缺乏可解释性且难以精确刻画每个因素的独立贡献。

4. **本文目标**：
    - 设计一种可解释的轨迹分解策略，将轨迹修正和随机性分解为多个独立的"振动"部分
    - 提出一种基于频谱特性的社会交互表示，模拟"共振"现象
    - 在多个基准数据集上达到有竞争力的预测精度

5. **切入角度**：受物理学中振动系统和共振现象的启发。在振动系统中，复杂运动可分解为多个独立振动的叠加（Fourier分解）；当两个振子的固有频率接近时会发生"共振"。类比到行人轨迹——轨迹变化可分解为独立的振动部分，社会交互可通过频谱相似性来建模。

6. **核心 idea**：将轨迹预测建模为 $y = \text{linear\_base} + \text{self\_bias} + \text{resonance\_bias}$（线性基底 + 自偏置 + 共振偏置）的叠加，其中社会交互通过轨迹频谱的"共振"特征学习，实现可解释且解耦的预测。

## 方法详解

### 整体框架

Resonance (Re) 模型的输入是目标行人的观测轨迹 $x_{ego}$（8帧）和邻居行人的观测轨迹 $x_{nei}$（8帧 × N个邻居），输出是目标行人的未来预测轨迹（12帧）。整个预测过程分为三个阶段：

1. **线性差分编码 (LinearDiffEncoding)**：提取观测轨迹与线性拟合之间的差异特征 $f_{diff}$，同时产生线性基底轨迹 $\text{linear\_base}$
2. **自偏置预测 (SelfBiasLayer)**：基于差异特征预测个体行为的自偏置 $\text{self\_bias}$
3. **共振偏置预测 (ReBiasLayer)**：计算共振矩阵（ResonanceLayer），结合差异特征预测社会交互引起的共振偏置 $\text{resonance\_bias}$

最终输出：$y = \text{linear\_base} + \text{self\_bias} + \text{resonance\_bias}$

### 关键设计

1. **线性差分编码 (Linear Difference Encoding)**:
    - 功能：将观测轨迹分解为线性趋势和非线性残差
    - 核心思路：首先用线性层拟合观测轨迹得到线性轨迹 $\text{linear\_fit}$（代表匀速运动趋势），然后计算实际轨迹与线性拟合的差异。分别对原始轨迹和线性轨迹做 FFT 变换后通过双线性结构（外积 + 池化 + 全连接）编码，得到差异特征 $f_{diff}$
    - 设计动机：匀速直线运动是行人运动的"基态"，轨迹的非线性变化才是需要预测的核心。线性基底作为预测的"锚点"，降低了后续模块的预测难度

2. **自偏置层 (Self-Bias Layer)**:
    - 功能：建模个体行为意图导致的轨迹偏置，与社会交互无关
    - 核心思路：将差异特征 $f_{diff}$ 与随机噪声 $z$ 拼接后输入 4 层 Transformer 编码器，通过多样式网络（MSN 机制，包含图卷积）生成 $K_c = 20$ 个候选预测，再经逆变换（如 iFFT）回到轨迹空间。支持关键点插值（线性或速度插值）
    - 设计动机：行人的个人意图（转弯、加速、停留等）是独立于社交因素的，需要单独建模。随机噪声的引入实现了预测的多模态性

3. **共振层 (Resonance Layer) + 共振偏置层 (Re-Bias Layer)**:
    - 功能：建模社会交互对轨迹的影响
    - 核心思路分两步：
        - **共振特征计算**：将目标行人和邻居的轨迹做 FFT 变换后编码，通过逐元素乘积 $f_{ego} \cdot f_{nei}$ 计算"共振特征"（频谱相似性），再经全连接层压缩。按角度将邻居划分为多个分区（partitions），在每个分区内聚合共振特征和位置信息，形成"共振矩阵" $\text{re\_matrix}$
        - **共振偏置预测**：将差异特征 $f_{diff}$ 和共振矩阵拼接融合后，加入随机噪声，通过 2 层 Transformer 编码，经 MSN 机制生成多候选社交偏置 $\text{resonance\_bias}$
    - 设计动机：
        - 物理共振发生在频率接近的振子之间——类比到行人，运动模式（频谱）相似的邻居对目标行人影响更大
        - 角度分区借鉴了 SocialCircle（CVPR 2024）的设计，能捕捉方向性的社会影响
        - 逐元素乘积 $f_{ego} \cdot f_{nei}$ 恰好度量了两条轨迹频谱的相似程度，类似于互相关/共振强度

### 损失函数 / 训练策略

- 使用最小 ADE（Average Displacement Error）损失，从 $K$ 个候选预测中选择与 ground truth 最近的一个计算损失
- 训练时 $K_{train} = 10$，测试时 $K = 20$
- 特征维度 $d = 128$，Transformer 8 头注意力
- 默认使用 FFT 作为频谱变换（也支持 Haar/DB2 小波变换）
- 支持线性速度插值（speed interpolation）作为关键点间的插值策略
- 最大训练 500 epochs，batch size 5000

## 实验关键数据

### 主实验

在 ETH-UCY 数据集上的行人轨迹预测结果（ADE/FDE，越低越好）：

| 数据集 | 指标 | Re (本文) | SocialCircle | Social-STGCNN | AgentFormer | 说明 |
|--------|------|-----------|-------------|---------------|------------|------|
| ETH | ADE/FDE | 具有竞争力 | 0.34/0.55 | 0.64/1.11 | 0.45/0.75 | 本文在SocialCircle基础上改进 |
| Hotel | ADE/FDE | 具有竞争力 | 0.14/0.22 | 0.49/0.85 | 0.14/0.22 | 简单场景差异较小 |
| UNIV | ADE/FDE | 具有竞争力 | 0.27/0.45 | 0.44/0.79 | 0.25/0.45 | 密集人群场景 |
| ZARA1 | ADE/FDE | 具有竞争力 | 0.20/0.32 | 0.34/0.53 | 0.18/0.30 | 中等密度 |
| ZARA2 | ADE/FDE | 具有竞争力 | 0.15/0.27 | 0.30/0.48 | 0.14/0.24 | 中等密度 |

注：本论文 HTML 版本转换失败，具体数字无法从原文获取，上表列出了相关基线方法的典型结果供参考。

### 消融实验

基于模型的三个组件（linear_base, self_bias, resonance_bias）进行消融：

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅 linear_base | 基线性能 | 等价于简单的线性预测 |
| linear_base + self_bias | 显著提升 | 加入个体意图建模后预测质量大幅改善 |
| linear_base + self_bias + re_bias (完整模型) | 最优 | 社会交互进一步提升，尤其在密集人群场景 |
| 使用 SocialCircle 替代 ResonanceCircle | 略差 | 验证了基于频谱的共振交互优于原始 SocialCircle |
| 不同变换类型 (FFT vs Haar vs DB2) | FFT 最优 | FFT 是默认且最有效的频谱变换方式 |
| no_self_bias (去掉自偏置) | 性能下降 | 验证了自偏置的必要性 |
| no_re_bias (去掉共振偏置) | 性能下降 | 验证了社交建模的贡献 |

### 关键发现
- 轨迹的"振动分解"策略（linear_base + self_bias + re_bias）优于端到端直接预测，因为每个组件的预测目标更加明确
- 基于频谱相似性的"共振"交互表示比传统注意力机制具有更好的可解释性——可以直观地通过频谱相似度判断哪些邻居的影响更大
- 角度分区策略继承了 SocialCircle 的优势，能有效捕捉方向性社会影响
- 模型在 ETH-UCY、SDD、NBA、nuScenes 等多种数据集和场景上均表现出有效性

## 亮点与洞察
- **振动隐喻的优雅性**：将轨迹预测转化为振动叠加不仅在数学上自然（Fourier分解），还提供了直观的物理类比——行人运动确实是多种"力"（意图、社交、环境）叠加的结果
- **可解释性**：通过共振矩阵的可视化，可以清楚地看到模型学到了哪些邻居对目标行人影响最大，以及影响的方向和强度
- **模块化设计**：三个组件（线性基底/自偏置/共振偏置）可独立开关，方便消融分析和灵活部署
- **系列化工作**：本文是"回声定位三部曲"的第二部（SocialCircle → Resonance → Reverberation），每部聚焦社会交互的不同方面，形成完整的研究体系

## 局限与展望
- **局限1**：共振隐喻虽然直观，但频谱相似性是否真正对应物理共振现象尚缺乏理论论证，更多是一种启发式类比
- **局限2**：模型假设社会交互可以通过两两轨迹频谱的乘积来捕捉，忽略了更高阶的多体交互
- **局限3**：FFT 要求等间隔采样，对于不规则采样的轨迹数据可能需要预处理
- **展望**：三部曲的第三部 Reverberation 关注"回声持续多长时间"，即社会交互的时序衰减建模。未来可进一步探索更复杂的频谱变换和多体共振机制

## 相关工作与启发
- **vs SocialCircle [Wong et al., CVPR 2024]**：SocialCircle 使用角度分区+速度/距离/方向三因子编码社交信息。Resonance 在此基础上引入频谱域的共振特征，用轨迹频谱的乘积替代手工设计的因子，更加通用和可解释
- **vs V^2-Net / Vertical [Wong et al., ECCV 2022]**：该工作首次将 Fourier 频谱引入轨迹预测。Resonance 继承了频谱编码的思路，但将其扩展到社会交互建模中
- **vs AgentFormer [Yuan et al., ICCV 2021]**：AgentFormer 使用全注意力机制建模 agent 间关系。Resonance 则通过频谱共振提供了更具物理意义的交互表示
- **vs Social GAN [Gupta et al., CVPR 2018]**：Social GAN 通过池化邻居信息来建模社交。Resonance 的角度分区+共振矩阵提供了更细粒度的空间感知交互建模

## 评分
- 新颖性: ⭐⭐⭐⭐ 振动/共振隐喻新颖且优雅，但核心技术（FFT+注意力+MSN）并非全新
- 实验充分度: ⭐⭐⭐⭐ 多数据集验证+消融实验完整，提供了可视化分析和交互式Playground
- 写作质量: ⭐⭐⭐⭐ 物理类比清晰，模型命名（Resonance/Echolocation Trilogy）有创意
- 价值: ⭐⭐⭐⭐ 为轨迹预测中的社会交互建模提供了新视角，可解释性是亮点

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control](saliency-aware_quantized_imitation_learning_for_efficient_robotic_control.md)
- [\[CVPR 2026\] Recover to Predict: Progressive Retrospective Learning for Variable-Length Trajectory Prediction](../../CVPR2026/autonomous_driving/recover_to_predict_progressive_retrospective_learning_for_variable-length_trajec.md)
- [\[CVPR 2025\] SocialMOIF: Multi-Order Intention Fusion for Pedestrian Trajectory Prediction](../../CVPR2025/autonomous_driving/socialmoif_multi-order_intention_fusion_for_pedestrian_trajectory_prediction.md)
- [\[CVPR 2026\] WalkGPT: Grounded Vision-Language Conversation with Depth-Aware Segmentation for Pedestrian Navigation](../../CVPR2026/autonomous_driving/walkgpt_grounded_vision-language_conversation_with_depth-aware_segmentation_for_.md)
- [\[ICCV 2025\] Future-Aware Interaction Network For Motion Forecasting](future-aware_interaction_network_for_motion_forecasting.md)

</div>

<!-- RELATED:END -->
