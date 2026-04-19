---
title: >-
  [论文解读] Horizon Imagination: Efficient On-Policy Rollout in Diffusion World Models
description: >-
  [ICLR 2026][图像恢复][扩散世界模型] 提出 Horizon Imagination (HI)，通过在去噪中途采样动作并行处理多个未来帧，将扩散世界模型的 on-policy 想象计算量降至每帧不到一次完整去噪，同时保持控制性能。
tags:
  - ICLR 2026
  - 图像恢复
  - 扩散世界模型
  - on-policy rollout
  - 强化学习
  - 样本效率
  - Atari
---

# Horizon Imagination: Efficient On-Policy Rollout in Diffusion World Models

**会议**: ICLR 2026  
**arXiv**: [2602.08032](https://arxiv.org/abs/2602.08032)  
**代码**: https://github.com/leor-c/horizon-imagination  
**领域**: 强化学习 / 世界模型  
**关键词**: 扩散世界模型, on-policy rollout, 强化学习, 样本效率, Atari

## 一句话总结
提出 Horizon Imagination (HI)，通过在去噪中途采样动作并行处理多个未来帧，将扩散世界模型的 on-policy 想象计算量降至每帧不到一次完整去噪，同时保持控制性能。

## 研究背景与动机

**领域现状**：世界模型通过学习环境动力学来生成模拟数据，扩散世界模型（如 DIAMOND）因出色的生成保真度受到关注，但多步去噪开销巨大。

**现有痛点**：on-policy 想象需要在每步生成后根据当前策略采样动作决定下一步，形成严格串行依赖，无法利用扩散模型的并行去噪能力。

**核心矛盾**：扩散生成质量高但计算量大，而 on-policy RL 的串行需求进一步放大了这个问题。

**本文目标** 在保持 on-policy 想象质量的同时大幅降低扩散世界模型的计算开销。

**切入角度**：观察到可以在去噪过程中间步骤采样动作，将策略选择嵌入扩散过程。

**核心 idea**：在去噪中途（而非完成后）进行动作采样，使多帧想象的前半段去噪可以完全并行。

## 方法详解

### 整体框架
传统做法是"完整去噪→动作采样→完整去噪"串行过程。HI 改为同时初始化未来 $h$ 帧的噪声潜变量，在部分去噪（$k$ 步，$k < K$ 总步数）后从中间结果采样动作，然后用这些动作继续去噪剩余步骤。

### 关键设计

1. **中途动作采样 (Intermediate Action Sampling)**:

    - 做什么：在去噪中间步骤确定 on-policy 动作
    - 核心思路：设总去噪步数 $K$，在第 $k$ 步从部分去噪的潜变量 $\mathbf{z}^k_t$ 中通过策略网络 $\pi(\mathbf{z}^k_t)$ 采样动作。前 $k$ 步无动作条件，可并行处理所有帧
    - 设计动机：部分去噪的帧虽噪声较大，但已包含足够语义信息支持策略决策

2. **帧并行 Horizon 去噪**:

    - 做什么：同时去噪多个未来时间步的潜变量
    - 核心思路：将 $h$ 个未来帧 $\{\mathbf{z}^0_{t+1}, ..., \mathbf{z}^0_{t+h}\}$ 打包为一个批次送入去噪网络，共享每次前向传播的开销，总去噪预算降至每帧不到一次完整去噪
    - 设计动机：扩散模型的去噪网络天然支持批处理，并行化带来直接的计算节省

3. **生成稳定性策略**:

    - 做什么：防止长程想象中的生成退化
    - 核心思路：引入动作一致性正则化，鼓励 $\pi(\mathbf{z}^k_t) \approx \pi(\mathbf{z}^K_t)$，减小中间采样与完整去噪采样的分布偏移
    - 设计动机：直接从噪声状态采样可能引入偏差，在 Craftium 等复杂 3D 环境中尤为关键

### 损失函数 / 训练策略
世界模型使用标准扩散去噪损失训练。策略和价值函数通过 actor-critic 在想象轨迹上优化。动作一致性正则化作为辅助损失。

## 实验关键数据

### 主实验
| 环境 | 方法 | 人类归一化分数 | 每帧去噪步数 |
|------|------|--------------|-------------|
| Atari 100K | DIAMOND | ~1.0x | 10 |
| Atari 100K | HI (k=3) | ~1.0x (匹配) | <1 |
| Craftium | DIAMOND | 基线 | 10 |
| Craftium | HI | 匹配 | <1 |

### 消融实验
| 配置 | 性能 (相对) | 说明 |
|------|-----------|------|
| HI 完整 (k=3) | 1.00 | 最优中途采样点 |
| 无动作一致性正则 | 0.93 | 动作偏差累积 |
| k=1 (极早采样) | 0.87 | 信息不足 |
| k=K (顺序) | 1.00 | 无效率增益 |

### 关键发现
- HI 在保持同等控制性能的同时将每帧计算量降至不到一次去噪
- 中途采样点 $k$ 存在甜蜜点：太早信息不足，太晚失去效率优势
- 动作一致性正则化对长程 rollout 稳定性至关重要

## 亮点与洞察
- **去噪-决策耦合设计**：将策略选择嵌入扩散过程，实现生成与决策的并行，是扩散世界模型效率的重要突破
- **部分去噪即足够**：证明策略决策不需要完全清晰的观测

## 局限与展望
- 仅在离散动作空间验证，连续控制适用性未知
- 中途采样点需要调节
- 更复杂环境可能需要更多稳定性策略

## 相关工作与启发
- **vs DIAMOND**: HI 在 DIAMOND 之上实现效率优化，不改架构只改想象过程
- **vs IRIS/TWM**: Transformer 世界模型不需多步去噪，但生成质量通常不如扩散

## 评分
- 新颖性: ⭐⭐⭐⭐ 中途采样思路新颖自然
- 实验充分度: ⭐⭐⭐ 覆盖尚可但缺连续控制
- 写作质量: ⭐⭐⭐⭐ 动机清晰
- 价值: ⭐⭐⭐⭐ 对扩散世界模型实际部署有重要意义

<!-- RELATED:START -->

## 相关论文

- [wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models](wd1_weighted_policy_optimization_for_reasoning_in_diffusion_language_models.md)
- [Activation Steering for Masked Diffusion Language Models](activation_steering_for_masked_diffusion_language_models.md)
- [Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerabilities](toward_safer_diffusion_language_models_discovery_and_mitigation_of_priming_vulne.md)
- [Are Deep Speech Denoising Models Robust to Adversarial Noise?](are_deep_speech_denoising_models_robust_to_adversarial_noise.md)
- [RefiDiff: Progressive Refinement Diffusion for Efficient Missing Data Imputation](../../AAAI2026/image_restoration/refidiff_progressive_refinement_diffusion_for_efficient_missing_data_imputation.md)

<!-- RELATED:END -->
