---
title: >-
  [论文解读] Self-Improving Loops for Visual Robotic Planning
description: >-
  [ICLR 2026][图像生成][视觉规划] 提出 SILVR 框架，通过迭代更新域内视频生成模型在自收集的在线轨迹上进行微调，实现视觉机器人规划器在未见任务上的持续自我改进，在 MetaWorld 和真实机器人上实现高达 285% 的性能提升。
tags:
  - ICLR 2026
  - 图像生成
  - 视觉规划
  - 自我改进
  - 视频生成模型
  - 逆动力学模型
  - 在线经验
---

# Self-Improving Loops for Visual Robotic Planning

**会议**: ICLR 2026  
**arXiv**: [2506.06658](https://arxiv.org/abs/2506.06658)  
**代码**: [https://diffusion-supervision.github.io/silvr/](https://diffusion-supervision.github.io/silvr/)  
**领域**: 机器人规划 / 视频生成  
**关键词**: 视觉规划, 自我改进, 视频生成模型, 逆动力学模型, 在线经验

## 一句话总结

提出 SILVR 框架，通过迭代更新域内视频生成模型在自收集的在线轨迹上进行微调，实现视觉机器人规划器在未见任务上的持续自我改进，在 MetaWorld 和真实机器人上实现高达 285% 的性能提升。

## 研究背景与动机

视频生成模型作为文本条件的视觉规划器已展现出强大的机器人任务规划能力。然而，对未见任务的泛化仍是挑战。现有方法主要依赖离线数据（预收集的演示或互联网视频），缺乏从在线自收集行为中持续改进的能力。

核心问题：**能否设计一个在线自我改进的视觉规划智能体？**

## 方法详解

### 整体框架

SILVR 的核心循环：
1. 视频模型生成视觉规划 → 2. 逆动力学模型转换为动作 → 3. 环境交互收集轨迹 → 4. 过滤成功轨迹 → 5. 微调视频模型 → 重复

### 关键设计一：视频模型作为视觉规划器

基于 UniPi 框架：
- 文本到视频模型预测未来帧序列作为任务规划
- 逆动力学模型 (IDM) 将连续帧对转换为可执行动作
- 支持基于 AVDC 的域内视频模型 + 文本交叉注意力

### 关键设计二：逆概率适配 (IPA)

将域内视频模型与互联网预训练视频先验（AnimateDiff，~2B 参数）通过得分组合集成：

$$\tilde{\epsilon}_{\text{inv}} = \epsilon_{\text{general}}(\tau_t, t) + \alpha(\epsilon_{\text{general}}(\tau_t, t|\text{text}) + \gamma\epsilon_\theta(\tau_t, t|\text{text}) - \epsilon_{\text{general}}(\tau_t, t))$$

- $\gamma$：先验强度
- $\alpha$：文本引导尺度
- 域内模型提供环境特定视觉知识
- 预训练模型提供文本泛化和运动先验

### 关键设计三：自我改进循环

每轮迭代（Algorithm 1）：
1. 可选地与互联网视频先验适配
2. 执行 $N$ 次视觉规划交互
3. 使用过滤函数 $f_r$（成功信号）过滤轨迹
4. 累积数据微调域内视频模型
5. 可选地蒸馏为轻量策略

## 实验

### MetaWorld 结果（12 个未见任务，平均成功率）

| 方法 | 迭代 0 | 迭代 1 | 迭代 2 | 迭代 3 | 迭代 4 |
|------|--------|--------|--------|--------|--------|
| DSRL（GT 过滤） | 9.4 | 8.3 | 7.4 | 7.5 | 7.7 |
| BCIL（GT 过滤） | 5.6 | 12.3 | 20.9 | 23.3 | 23.2 |
| **SILVR（GT 过滤）** | **14.7** | **27.7** | **33.5** | **43.5** | **44.2** |
| SILVR（VLM 过滤） | 17.0 | 24.4 | 28.7 | 34.4 | 38.4 |

SILVR 在迭代 4 后成功率达到 44.2%，远超 BCIL（23.2%）和 DSRL（7.7%）。

### 真实机器人实验

- **推杯任务**：在未见颜色上持续改进
- **开抽屉任务**：SILVR + 互联网视频先验成功引导自我改进
- 没有互联网视频先验时，真实世界实验中改进趋势停滞或恶化

### 蒸馏

SILVR 最终视频规划器蒸馏为 Diffusion Policy 后性能进一步略有提升（44.2% → 49.2%）。

### 消融实验

| 设置 | 结论 |
|------|------|
| 无数据过滤 | MetaWorld 上改进缓慢；真实世界仍能改进 |
| VLM 代替 GT 过滤 | Gemini-2.5-Pro 效果最佳，仍能实现自我改进 |
| 次优初始数据 | SILVR 仍然能持续改进 |
| 10 轮迭代 | 第 5 轮后趋于饱和 |

### 关键发现

- 视觉规划的解耦设计（动力学建模 vs 动作预测）使泛化更容易
- 互联网视频先验对真实世界实验至关重要
- 无过滤时次优经验仍能通过得分组合传递有用信息
- SILVR 比 RL 微调方法更具样本效率

## 亮点

- 首个系统性的视觉规划自我改进框架
- 将离线数据和在线经验有机结合
- 互联网视频先验的引入优雅地解决了真实世界泛化问题
- 对过滤信号的鲁棒性强（GT/VLM/无过滤均可工作）
- 蒸馏方案平衡了规划质量和推理速度

## 局限性

- 假设初始模型有合理的基础成功率（冷启动问题）
- 大规模预训练视频模型的选择引入效率/质量权衡
- 10 轮后趋于饱和，可能陷入策略局部最优
- 视频生成的推理速度仍然是部署瓶颈
- 未探索如何引入"探索"机制突破单模态行为

## 相关工作

- **视频规划**：UniPi、AVDC 等视频模型用于决策
- **自我改进模型**：LLM 中的自我改进、VideoAgent 等
- **RL 微调 BC 策略**：DPPO、DSRL、ResIP 等

## 评分

- 新颖性：⭐⭐⭐⭐ — 视觉规划 + 自我改进循环的结合新颖
- 实验：⭐⭐⭐⭐⭐ — 仿真+真实、多消融、多基线的全面评估
- 实用性：⭐⭐⭐⭐ — 真实机器人验证+蒸馏方案兼顾部署
- 完整性：⭐⭐⭐⭐ — 对各种设计决策的消融研究充分

<!-- RELATED:START -->

## 相关论文

- [Image Generation as a Visual Planner for Robotic Manipulation](../../CVPR2026/image_generation/image_generation_as_a_visual_planner_for_robotic_manipulation.md)
- [Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards](../../CVPR2026/image_generation/improving_text-to-image_generation_with_intrinsic_self-confidence_rewards.md)
- [SOLACE: Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards](../../CVPR2026/image_generation/solace_self_confidence_rewards_t2i.md)
- [SafeFlowMatcher: Safe and Fast Planning using Flow Matching with Control Barrier Functions](safeflowmatcher_safe_and_fast_planning_using_flow_matching_with_control_barrier_.md)
- [Steer Away From Mode Collisions: Improving Composition In Diffusion Models](steer_away_from_mode_collisions_improving_composition_in_diffusion_models.md)

<!-- RELATED:END -->
