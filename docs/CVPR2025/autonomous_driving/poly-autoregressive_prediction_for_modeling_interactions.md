---
title: >-
  [论文解读] PAR: Poly-Autoregressive Prediction for Modeling Interactions
description: >-
  [CVPR 2025][自动驾驶][多智能体交互] PAR（Poly-Autoregressive）提出了一种简洁统一的多智能体行为预测框架，通过将交互中其他智能体的状态序列作为条件，结合同智能体下一时间步预测和学习的智能体ID嵌入，在社交行为预测、自动驾驶轨迹预测和手-物交互三个截然不同的任务上均优于单智能体自回归基线。
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "多智能体交互"
  - "自回归预测"
  - "Transformer"
  - "轨迹预测"
  - "行为预测"
---

# PAR: Poly-Autoregressive Prediction for Modeling Interactions

**会议**: CVPR 2025  
**arXiv**: [2502.08646](https://arxiv.org/abs/2502.08646)  
**代码**: 已开源  
**领域**: 自动驾驶（轨迹预测）  
**关键词**: 多智能体交互, 自回归预测, Transformer, 轨迹预测, 行为预测

## 一句话总结

PAR（Poly-Autoregressive）提出了一种简洁统一的多智能体行为预测框架，通过将交互中其他智能体的状态序列作为条件，结合同智能体下一时间步预测和学习的智能体ID嵌入，在社交行为预测、自动驾驶轨迹预测和手-物交互三个截然不同的任务上均优于单智能体自回归基线。

## 研究背景与动机

在多智能体交互场景中预测某个智能体的未来行为是一个核心问题。与语言中的自回归建模不同，物理世界中的交互受到物理定律和智能体内部状态的双重约束，且多个智能体的状态**同时**变化。

现有方法的问题：
- **标准自回归（AR）不足**：只关注单个智能体的历史状态序列，忽略了其他智能体的影响。例如，预测一个人会继续说话，而实际上对方已经开始说话，他应该转为倾听
- **多智能体方法各自为政**：不同交互场景（社交行为、驾驶、手物交互）各自设计专用方案，缺乏统一框架
- **朴素的多智能体AR适得其反**：简单地将多智能体token排成序列做next-token预测会混淆模型——因为下一个token是另一个智能体的同一时间步状态，而非同一智能体的下一时间步状态

核心洞察：在交互场景中，ego智能体的未来取决于自身历史**和**其他智能体的当前/过去状态。需要"同智能体下一时间步"而非"序列下一token"的预测范式。

## 方法详解

### 整体框架

PAR框架将$N$个智能体在$T$个时间步的状态表示为$N \times T$个token的扁平序列。Transformer decoder学习给定所有智能体的历史状态后，预测ego智能体的下一时间步状态。框架无需修改架构即可应用于不同任务，仅需调整数据预处理和token化方式。

### 关键设计

**设计一：同智能体下一时间步预测 — 替代标准next-token预测**

- **功能**：确保每次预测时模型利用了同一时间步所有智能体的状态信息
- **核心思路**：在扁平化的$N \times T$序列中，标准AR的next-token预测会从智能体$k$在时间$t$预测智能体$k+1$在时间$t$（同一时间步的不同智能体）。PAR改为预测智能体$k$在时间$t+1$的状态（同一智能体的下一时间步）。训练时联合计算所有$N$个智能体的损失
- **设计动机**：next-token预测违反了因果关系——用一个智能体的状态预测同时刻另一个智能体的状态没有物理意义。同智能体下一时间步预测才是正确的时序因果关系

**设计二：学习的智能体ID嵌入 — 区分多智能体身份**

- **功能**：使模型知道每个token属于哪个智能体
- **核心思路**：将整数智能体ID映射为hidden dim大小的向量，与token嵌入相加。使模型在处理混合序列时能区分不同智能体的状态
- **设计动机**：消融实验表明，缺少智能体ID嵌入的多智能体模型性能比单智能体AR更差，说明模型在混淆不同智能体的状态

**设计三：统一框架 — 支持离散/连续token和多种任务**

- **功能**：无需修改架构即可处理不同类型的多智能体交互预测
- **核心思路**：离散token（如动作类别）使用标准embedding+交叉熵损失；连续token（如位置坐标）使用learned投影层+回归损失。数据来源为视频，通过数据集标注或CV技术提取各智能体的状态序列。可选的位置编码（如轨迹预测中的LPE）叠加提供空间信息
- **设计动机**：不同交互任务的状态表示差异巨大（60维动作概率 vs 2D位置 vs 6DoF位姿），但交互建模的核心框架应该是通用的。统一框架降低了迁移到新领域的成本

### 损失函数

- 社交行为预测：60维动作token上的MSE回归损失
- 车辆轨迹预测：离散速度/加速度token上的交叉熵分类损失
- 手物交互：6DoF位姿上的回归损失

## 实验关键数据

### 主实验：三个案例研究

| 任务 | 指标 | AR | PAR | 提升 |
|------|------|-----|-----|------|
| AVA社交行为预测 | mAP ↑ | 40.7 | **42.6** | +1.9 |
| AVA 2人交互类 | mAP ↑ | 36.3 | **39.8** | +3.5 |
| nuScenes轨迹预测 | ADE ↓ | 基线 | **-6.3%** | 相对 |
| nuScenes轨迹预测 | FDE ↓ | 基线 | **-6.4%** | 相对 |
| DexYCB物体旋转 | 误差 ↓ | 基线 | **-8.9%** | 相对 |
| DexYCB物体平移 | 误差 ↓ | 基线 | **-41.0%** | 相对 |

### 消融实验：PAR组件的贡献（AVA数据集）

| 方法 | 时间步预测 | ID嵌入 | mAP ↑ |
|------|-----------|--------|-------|
| 1-agent AR | N/A | N/A | 40.7 |
| 2-agent AR | ✗ | ✗ | 38.0 |
| 2-agent PAR* | ✗ | ✓ | 40.2 |
| 2-agent PAR* | ✓ | ✗ | 40.0 |
| **2-agent PAR** | **✓** | **✓** | **42.6** |

### 关键发现

- 朴素的多智能体AR（行2）反而比单智能体AR差2.7 mAP，证明了next-token预测在多智能体场景的失败
- 同智能体下一时间步预测和智能体ID嵌入**缺一不可**——两者都是必要条件
- 在2人交互类别上（kiss +8.3, listen +7.0, hug +5.7, fight/hit +5.7），PAR的提升尤为显著
- DexYCB中平移预测的41%相对改进表明，物体运动高度依赖于手的状态
- 仅4.4M参数的小型Transformer即可展示PAR的优势

## 亮点与洞察

1. **极致的简洁性**：同一个4M参数Transformer，不修改架构，仅调数据预处理和token化即可处理三个截然不同的任务
2. **深刻的失败分析**：展示了朴素多智能体AR为何失败，next-token vs next-timestep的区分非常有启发性
3. **定性分析生动**：talk-listen轮换预测的例子直观展示了PAR捕获交互动态的能力

## 局限与展望

- 当前仅用4M小型Transformer验证概念，大规模实验有待进行
- 三个任务中只考虑2个智能体的交互（ego + 1 other），更多智能体的扩展性需验证
- 推理时需要其他智能体的ground-truth未来状态（或准确预测），实际应用中这是额外约束
- 未来可与更大规模的Transformer和数据集结合

## 相关工作与启发

- **MotionLM**：多智能体轨迹预测使用learned agent ID嵌入，启发了PAR的设计
- **SinGAN/GPT系列**：单示例/自回归建模的成功，PAR将其扩展到多智能体场景
- **Flamingo/LLaVa**：多模态自回归模型，PAR关注于物理交互而非视觉-语言
- 启发：**问题的关键不在于模型复杂度，而在于正确的序列化和因果关系建模**——next-token和next-timestep的区别看似微小，但带来了本质的差异

## 评分

⭐⭐⭐⭐ — 框架的简洁统一性令人赞赏，"一个框架三个任务"的验证有力。朴素AR失败的分析和PAR修正的逻辑链条清晰完整。限于小规模验证是不足之处，但概念验证充分。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] DriveGPT: Scaling Autoregressive Behavior Models for Driving](../../ICML2025/autonomous_driving/drivegpt_scaling_autoregressive_behavior_models_for_driving.md)
- [\[CVPR 2025\] ModeSeq: Taming Sparse Multimodal Motion Prediction with Sequential Mode Modeling](modeseq_taming_sparse_multimodal_motion_prediction_with_sequential_mode_modeling.md)
- [\[CVPR 2025\] Certified Human Trajectory Prediction](certified_human_trajectory_prediction.md)
- [\[ICCV 2025\] Epona: Autoregressive Diffusion World Model for Autonomous Driving](../../ICCV2025/autonomous_driving/epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)
- [\[CVPR 2025\] Physical Plausibility-aware Trajectory Prediction via Locomotion Embodiment](physical_plausibility-aware_trajectory_prediction_via_locomotion_embodiment.md)

</div>

<!-- RELATED:END -->
