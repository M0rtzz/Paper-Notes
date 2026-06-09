---
title: >-
  [论文解读] Horizon Imagination: Efficient On-Policy Rollout in Diffusion World Models
description: >-
  [ICLR 2026][图像恢复][扩散世界模型] 提出 Horizon Imagination (HI)，通过在去噪中途采样动作并行处理多个未来帧，将扩散世界模型的 on-policy 想象计算量降至每帧不到一次完整去噪，同时保持控制性能。
tags:
  - "ICLR 2026"
  - "图像恢复"
  - "扩散世界模型"
  - "on-policy rollout"
  - "强化学习"
  - "样本效率"
  - "Atari"
---

# Horizon Imagination: Efficient On-Policy Rollout in Diffusion World Models

**会议**: ICLR 2026  
**arXiv**: [2602.08032](https://arxiv.org/abs/2602.08032)  
**代码**: [https://github.com/leor-c/horizon-imagination](https://github.com/leor-c/horizon-imagination)  
**领域**: 图像复原  
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
HI 把扩散世界模型的 on-policy 想象从"完整去噪一帧 → 用清晰帧采样动作 → 再完整去噪下一帧"这个严格串行的链条，改写成一个尽量并行的批量去噪过程：它同时初始化未来 $h$ 帧的噪声潜变量，让所有帧共享前半段的无条件去噪，只在去噪进行到中途（第 $k$ 步，$k<K$）时才从尚带噪声的帧里采样动作并注入条件，再把剩余几步去噪补完。核心赌注是——策略决策并不需要一张完全清晰的预测帧，部分去噪的语义就够用，于是省下的去噪步数直接转化为算力节省。

### 关键设计

**1. 中途动作采样：把动作决策提前到去噪没做完的时候，打破串行依赖。**

on-policy 想象之所以慢，是因为它要求"先看清这一帧、再决定动作、动作又决定下一帧怎么去噪"，每一帧都得等上一帧彻底跑完 $K$ 步去噪。HI 的破局点是不等去噪结束。设总去噪步数为 $K$，HI 在中途第 $k$ 步停一下，从此时部分去噪的潜变量 $\mathbf{z}^k_t$ 上直接调用策略网络采样动作 $a_t \sim \pi(\mathbf{z}^k_t)$，再把这个动作作为条件喂回去噪网络跑完剩下的 $K-k$ 步。前 $k$ 步因为还没有动作条件，所有帧的去噪是同质的、互不依赖的，可以一口气并行算掉。这之所以成立，是因为扩散去噪在前期就已经勾勒出场景的粗结构和语义，足以支撑策略做出和看清晰帧时几乎一致的决策；消融里把采样点压到极早的 $k=1$ 时性能掉到 0.87，说明太早确实"信息不足"，而 $k=3$ 是信息量与节省量的甜蜜点。

**2. 帧并行 Horizon 去噪：让一次前向同时推进多帧，把每帧成本摊薄到不足一次去噪。**

既然中途采样切断了帧与帧之间前半段的依赖，HI 就顺势把未来 $h$ 帧的潜变量 $\{\mathbf{z}_{t+1}, \dots, \mathbf{z}_{t+h}\}$ 打包成一个批次送进同一个去噪网络，让它们共享每次前向传播的算力。扩散去噪网络本就天然支持批处理，这一步几乎是零额外代价地把横向的多帧并行接到了纵向的部分去噪上。两者叠加的结果是，原本每帧需要 $K=10$ 步完整去噪，现在被压缩到每帧平均不到一次完整去噪（实验中标注为 $<1$），而控制性能与基线 DIAMOND 持平。

**3. 动作一致性正则化：补上"从噪声帧采样会偏"的窟窿，稳住长程想象。**

提前采样省了算力，但代价是 $\mathbf{z}^k_t$ 比完整去噪的 $\mathbf{z}^K_t$ 更糊，从糊的状态上采动作会引入分布偏移，长程 rollout 里这种偏差会逐帧累积、最终拖垮想象质量。HI 用一个动作一致性正则化把这个缝补上，约束中途采样得到的策略分布逼近完整去噪后的策略分布，即 $\pi(\mathbf{z}^k_t) \approx \pi(\mathbf{z}^K_t)$，作为辅助损失加进训练。它在 Craftium 这类复杂 3D 环境里尤其关键——消融中去掉该正则化后性能从 1.00 跌到 0.93，正是动作偏差随 rollout 累积的直接体现。

世界模型本身仍以标准扩散去噪损失训练，策略与价值函数则用 actor-critic 在想象轨迹上优化，动作一致性正则化作为附加项叠加，三者不改 DIAMOND 的网络架构、只重写想象过程。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Activation Steering for Masked Diffusion Language Models](activation_steering_for_masked_diffusion_language_models.md)
- [\[NeurIPS 2025\] Encoder-Decoder Diffusion Language Models for Efficient Training and Inference](../../NeurIPS2025/image_restoration/encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)
- [\[ICML 2026\] Consistent Diffusion Language Models](../../ICML2026/image_restoration/consistent_diffusion_language_models.md)
- [\[ICLR 2026\] Are Deep Speech Denoising Models Robust to Adversarial Noise?](are_deep_speech_denoising_models_robust_to_adversarial_noise.md)
- [\[ICLR 2026\] SoFlow: Solution Flow Models for One-Step Generative Modeling](soflow_solution_flow_models_for_one-step_generative_modeling.md)

</div>

<!-- RELATED:END -->
