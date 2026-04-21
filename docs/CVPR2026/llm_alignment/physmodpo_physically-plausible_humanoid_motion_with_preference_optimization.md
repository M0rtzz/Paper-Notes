---
title: >-
  [论文解读] PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization
description: >-
  [CVPR 2026][LLM对齐][DPO] 将 DPO 偏好优化引入扩散运动生成模型的后训练阶段，通过物理仿真控制器自动构造偏好数据对，使生成的人体运动既符合文本/空间控制指令又满足物理约束，并成功零样本迁移到 Unitree G1 真实机器人。
tags:
  - CVPR 2026
  - LLM对齐
  - DPO
  - 人体运动生成
  - 物理仿真
  - 扩散模型
  - 人形机器人
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization

**会议**: CVPR 2026  
**arXiv**: [2603.13228](https://arxiv.org/abs/2603.13228)  
**代码**: 有 ([项目页面](https://mael-zys.github.io/PhysMoDPO/))  
**领域**: 对齐RLHF / 人体运动  
**关键词**: DPO, 人体运动生成, 物理仿真, 扩散模型, 人形机器人

## 一句话总结

将 DPO 偏好优化引入扩散运动生成模型的后训练阶段，通过物理仿真控制器自动构造偏好数据对，使生成的人体运动既符合文本/空间控制指令又满足物理约束，并成功零样本迁移到 Unitree G1 真实机器人。

## 研究背景与动机

**领域现状**：基于扩散模型的文本驱动人体运动生成取得了显著进展（MDM、MoMask、MotionStreamer 等），可以根据文本描述或空间约束生成运动序列。

**痛点**：这些模型在运动学空间中训练和评估，但部署到物理仿真或真实机器人时会暴露严重缺陷——脚部滑动、质心不稳、动力学不合理等，导致生成的运动无法直接执行。

**核心矛盾**：当通过 Whole-Body Controller (WBC) 将运动映射到可执行轨迹时，控制器会大幅修改不合理的运动以满足物理约束，导致实际执行轨迹与生成运动偏差巨大。即「运动学指标好 ≠ 部署效果好」。

**要解决的问题**：如何让扩散运动生成器直接输出物理上可行、同时忠于文本/空间条件的运动？

**现有方案的不足**：(a) 后处理/投影方法（PhysPT、PhysDiff）会改变输出分布，损害任务性能；(b) 用手工设计奖励（脚滑、漂浮惩罚）做 RL 微调（ReinDiffuse、HY-Motion）难以覆盖复杂动力学。

**核心 idea**：将预训练的物理追踪控制器 (DeepMimic) 直接集成到训练流程中——用它执行候选运动、在执行后的轨迹上计算物理和任务奖励、自动构造 DPO 偏好数据对来微调生成器，形成 generate–finetune 的迭代优化循环。

## 方法详解

### 整体框架

PhysMoDPO 的流程分为三步循环：

1. **生成阶段**：对每个条件 $C$（文本或文本+空间控制）用扩散生成器采样 $K$ 个候选运动 $X_k$
2. **仿真阶段**：将每个候选送入固定的物理追踪策略 $\mathcal{T}$（DeepMimic），得到仿真轨迹 $X_k' = \mathcal{T}(X_k)$
3. **偏好优化阶段**：在仿真轨迹上计算奖励，构造偏好对 $(X_{\text{win}}, X_{\text{lose}})$，用 DPO + SFT 联合损失微调生成器

整个过程可迭代多轮：每轮更新生成器后，重新采样候选并构造新偏好数据。

### 关键设计

**物理算子 $\mathcal{T}$ 的引入**：

- **功能**：定义从运动学空间到物理可行空间的映射 $\mathcal{T}: \mathcal{X}_{\text{kin}} \to \mathcal{X}_{\text{phys}}$，用 DeepMimic 追踪策略实现
- **核心思路**：不再在运动学空间评估，而是在部署空间（经 WBC 执行后）评估所有指标
- **设计动机**：直接衡量运动的物理可部署性，用追踪偏差 $\Delta(X) = \|X' - X\|_2^2$ 度量物理一致性

**偏好对构造的 Pareto 支配规则**：

- **功能**：要求 $X_{\text{win}}$ 在所有奖励维度上都优于 $X_{\text{lose}}$
- **核心思路**：对 $K$ 个候选的仿真结果，只有当一个样本在追踪、滑动、文本匹配（和空间控制）所有奖励上都更优时，才构成有效偏好对
- **设计动机**：避免引入敏感的奖励权重超参，保持不同目标间的一致性

**迭代训练**：

- **功能**：每轮 DPO 更新后，用改进的模型重新生成候选并重建偏好对
- **核心思路**：随着模型改进，旧偏好数据不再反映当前模型的失败模式，需要刷新
- **设计动机**：类似 online DPO，持续针对模型当前弱点做优化

### 损失函数 / 训练策略

总训练目标：

$$\mathcal{L} = \mathcal{L}_{\text{DPO}}(X_{\text{win}}, X_{\text{lose}}) + \lambda_{\text{SFT}} \mathcal{L}_{\text{SFT}}(X_{\text{win}})$$

四个奖励函数：

| 奖励 | 公式/含义 | 作用 |
|------|----------|------|
| $\mathcal{R}_{\text{track}}$ | $-\|X' - X\|_2^2$，追踪偏差的负值 | 鼓励运动经物理执行后变化尽量小 |
| $\mathcal{R}_{\text{slide}}$ | 脚部接地时水平速度的惩罚 | 减少脚部滑动伪影 |
| $\mathcal{R}_{\text{M2T}}$ | TMR 编码器中文本和运动的余弦相似度 | 保持文本语义一致性 |
| $\mathcal{R}_{\text{control}}$ | 控制关节处的加权 MSE | 满足空间控制约束（仅空间控制任务） |

训练细节：对 MotionStreamer 每个训练 prompt 生成 12 个样本；仅更新扩散头部；使用 SMPL 关节旋转表示替代 HumanML3D 表示。

## 实验关键数据

### 主实验

**表1：文本驱动运动生成（SMPL 仿真，HumanML3D 数据集）**

| 方法 | MM-Dist ↓ | R@3 ↑ | FID ↓ | Jerk ↓ |
|------|-----------|-------|-------|--------|
| MaskedMimic | 19.73 | 0.6305 | 73.79 | 66.08 |
| MotionStreamer | 17.17 | 0.8310 | 49.14 | 46.75 |
| SFT | 17.23 | 0.8355 | 49.22 | 48.30 |
| **PhysMoDPO** | **16.95** | **0.8517** | **48.29** | **43.60** |

**表2：空间-文本控制（Cross 设置，SMPL 仿真，HumanML3D）**

| 方法 | Err. ↓ | MM-Dist ↓ | FID ↓ | Jerk ↓ |
|------|--------|-----------|-------|--------|
| OmniControl (Cross) | 0.0938 | 3.086 | 0.75 | 64.07 |
| SFT | 0.0972 | 3.075 | 0.68 | 61.22 |
| **PhysMoDPO** | **0.0923** | 3.099 | **0.66** | **58.02** |

**零样本迁移 G1 机器人**（文本驱动，Table 4）：PhysMoDPO 在 M2T (0.7919)、R@1 (0.4707)、R@3 (0.7640)、FID (0.3029) 上均为最优，且 Jerk 从 MotionStreamer 的 95.08 大幅降至 90.14。

### 消融实验

**迭代轮数消融**（OmniControl，HumanML3D）：

| 轮数 | Err. ↓ | FID ↓ | Jerk ↓ |
|------|--------|-------|--------|
| 1 | 0.1421 | 1.17 | 72.13 |
| 2 | 0.1324 | 0.97 | 63.55 |
| 3 | **0.1298** | **0.93** | **62.31** |

**奖励组件消融**：

| 奖励组合 | Err. ↓ | FID ↓ | Jerk ↓ |
|---------|--------|-------|--------|
| 仅 Tracking | 0.1467 | 1.61 | 74.76 |
| + Control | 0.1447 | 1.45 | 74.47 |
| + Sliding | 0.1422 | 1.21 | **68.43** |
| + M2T | **0.1421** | **1.17** | 72.13 |

### 关键发现

- MaskedMimic 虽然是端到端物理策略，但文本遵循能力极差（R@3 仅 0.6305 vs PhysMoDPO 0.8517）
- 迭代训练 3 轮比 1 轮在 FID 上提升了 20%（1.17→0.93），表明刷新偏好数据至关重要
- 滑动奖励对 Jerk 改善最大（74.76→68.43），M2T 奖励对 FID 改善最大（1.21→1.17）
- 在 OOD 数据集 OMOMO 上，PhysMoDPO 同样大幅领先，证明泛化能力
- 用户研究中，PhysMoDPO 在文本遵循、运动流畅性和稳定性三项指标上一致优于 OmniControl 和 MaskedMimic

## 亮点与洞察

1. **评估范式转变**：把评估从运动学空间转移到物理部署空间（WBC 执行后），这个视角切换非常关键——暴露了之前论文只看运动学指标的盲区
2. **DPO 用于连续运动的巧妙适配**：在仿真轨迹上算奖励、用运动学样本做 DPO 训练——绕过了物理仿真器不可微分的难题
3. **Pareto 支配构造偏好对**：不用权重加和，而是要求所有维度都胜出才算偏好对——既避免了调权重的麻烦又保证了信号一致性
4. **真机部署验证**：不只是仿真实验，实际在 Unitree G1 上零样本部署并录制视频，说服力很强

## 局限与展望

1. **场景单一**：目前仅在平地上验证，尚未涉及阶梯、斜坡、不平整地形等复杂场景
2. **依赖固定 WBC**：偏好对质量受限于追踪策略的能力，如果 WBC 本身有系统偏差，会传导到训练信号中
3. **迭代训练成本**：每轮需要为大量 prompt 跑 K 次采样 + 物理仿真，计算开销不小
4. **SMPL 表示限制**：需要将数据从 HumanML3D 格式转为 SMPL 关节旋转重新训练，增加了模型切换门槛
5. **缺少与 Morph 的直接对比**：作为同类方法，Morph 用物理模型细化再微调，但实验中未直接比较

## 相关工作与启发

- **DPO for diffusion**：借鉴了 Diffusion-DPO（图像）和 VideoDPO（视频）的框架，首次将其扩展到人体运动这一结构化序列生成任务
- **ReinDiffuse / HY-Motion**：用 PPO/GRPO + 手工奖励微调运动生成器，PhysMoDPO 用基于仿真器的自动奖励替代，覆盖面更广
- **BeyondMimic / HOVER**：这些是改进 WBC 鲁棒性的工作，PhysMoDPO 反过来——保持 WBC 固定，改进生成器
- 核心启发：**把下游执行器作为评估信号反馈给上游生成器**，这个模式在其他生成-部署链条（如代码生成→编译执行、文本→TTS）中同样适用

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 把 DPO 对齐框架从 LLM/图像迁移到物理运动生成是一个很好的跨领域 idea，Pareto 支配构造偏好对的设计也有新意
- **实验充分度**: ⭐⭐⭐⭐⭐ — 两个任务 × 三个机器人 × 多个数据集 × OOD 测试 × 消融 × 用户研究 × 真机部署，非常全面
- **写作质量**: ⭐⭐⭐⭐ — 公式化严谨，逻辑清晰，Related Work 表格对比很直观，略微冗长但不影响阅读
- **价值**: ⭐⭐⭐⭐ — 为运动生成和机器人部署之间的 gap 提供了一条可扩展的技术路线，真机验证增加了实际意义

<!-- RELATED:START -->

## 相关论文

- [GlyphPrinter: Region-Grouped Direct Preference Optimization for Glyph-Accurate Visual Text Rendering](glyphprinter_region-grouped_direct_preference_optimization_for_glyph-accurate_vi.md)
- [Rethinking Direct Preference Optimization in Diffusion Models](../../AAAI2026/llm_alignment/rethinking_direct_preference_optimization_in_diffusion_models.md)
- [Margin-aware Preference Optimization for Aligning Diffusion Models without Reference](../../AAAI2026/llm_alignment/margin-aware_preference_optimization_for_aligning_diffusion_models_without_refer.md)
- [LocalDPO: Direct Localized Detail Preference Optimization for Video Diffusion Models](mind_the_generative_details_direct_localized_detail_preference_optimization_for_.md)
- [MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization](mod-dpo_towards_mitigating_cross-modal_hallucinations_in_omni_llms_using_modalit.md)

<!-- RELATED:END -->
