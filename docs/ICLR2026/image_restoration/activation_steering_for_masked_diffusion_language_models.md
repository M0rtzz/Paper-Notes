---
title: >-
  [论文解读] Activation Steering for Masked Diffusion Language Models
description: >-
  [ICLR 2026][图像恢复][activation steering] 首次将激活引导（activation steering）应用于 Masked Diffusion 语言模型（MDLM），发现 MDLM 的拒绝行为也受单一低维方向控制，通过在去噪过程中全局投影可完全绕过安全对齐，且与自回归模型不同…
tags:
  - "ICLR 2026"
  - "图像恢复"
  - "activation steering"
  - "扩散模型"
  - "safety"
  - "refusal direction"
  - "LLaDA"
---

# Activation Steering for Masked Diffusion Language Models

**会议**: ICLR 2026  
**arXiv**: [2512.24143](https://arxiv.org/abs/2512.24143)  
**代码**: 有  
**领域**: 图像复原  
**关键词**: activation steering, masked diffusion LM, safety, refusal direction, LLaDA  

## 一句话总结
首次将激活引导（activation steering）应用于 Masked Diffusion 语言模型（MDLM），发现 MDLM 的拒绝行为也受单一低维方向控制，通过在去噪过程中全局投影可完全绕过安全对齐，且与自回归模型不同，有效方向可从指令前的 token 中提取——反映了扩散模型的非因果并行处理特性。

## 研究背景与动机
**领域现状**：激活引导在自回归 LLM 中已被广泛研究（Arditi et al. 的拒绝方向消除等），但 MDLM（如 LLaDA）生成机制完全不同——通过迭代去 mask 而非逐 token 生成。MDLM 是否存在类似的低维控制方向完全未知。

**现有痛点**：(1) MDLM 的推理时控制方法仅有采样级引导（如 DIJA），没有表示级控制；(2) 自回归的 GCG、PAIR 等越狱攻击对扩散模型效果差——因为模型架构和生成方式不同。

**核心矛盾**：MDLM 采用非因果注意力（所有 token 互相可见），这与自回归的因果注意力完全不同——自回归中只有最后一个 token 能看到完整输入，但 MDLM 中每个 token 都能。激活引导的位置和时间选择需要重新理解。

**本文目标** MDLM 是否存在拒绝方向？在扩散的哪个阶段、哪些层、哪些 token 位置引导最有效？

**切入角度**：直接适配 Arditi et al. 的对比方向提取方法到 MDLM，但探索 MDLM 特有的 token 位置（含指令前 token）和去噪步骤。

**核心 idea**：将自回归 LLM 的激活引导原语移植到 MDLM，揭示扩散特有的表示特性。

## 方法详解

### 整体框架
这篇论文要回答一个此前没人验证过的问题：自回归 LLM 里那套"拒绝行为由单一低维方向控制、把它投影掉就能绕过安全对齐"的激活引导范式，搬到 Masked Diffusion 语言模型（MDLM）上还成不成立。整条流程分三步且无需任何训练：先从一批配对好的有害 / 无害 prompt 里算出候选引导方向；再在验证集上把"哪一层、哪个 token 位置"的组合 sweep 一遍挑出最有效的那个；最后在反向扩散的每一步、每一层、每个 token 位置上，把激活向量里这个方向的分量全部投影掉，从而消除拒绝行为。难点在于 MDLM 用的是非因果注意力（所有 token 互相可见），位置和时机的选择不能照搬自回归的直觉。

### 关键设计

**1. 方向提取：用有害与无害 prompt 的激活均值差锚定"拒绝方向"**

要找到那条控制拒绝的方向，本文沿用 Arditi et al. 的对比思路：准备 128 个有害 prompt 与 128 个无害 prompt，分别跑前向、收集激活，再对每个（层 $\ell$, token 位置 $i$）算两组激活均值之差并归一化，得到一个候选方向 $v_i^{(\ell)} = \text{normalize}(\mu_{+,i}^{(\ell)} - \mu_{-,i}^{(\ell)})$。这样得到的候选方向有成百上千个（层 × 位置），再在验证集上逐一 sweep、选消除拒绝最彻底的那个 $(\ell, i)$ 组合。真正的扩散特性体现在位置的选择上：自回归里只有指令之后的 token 才看得到完整输入，所以有效方向只能从 post-instruction token 提取；而 MDLM 的非因果注意力让**每个** token 都携带完整指令信息，于是连 **pre-instruction token** 也能提取出同样有效的引导方向——这是自回归架构下根本不可能的现象。

**2. 方向应用：在反向扩散全程做正交投影，把拒绝分量逐步剥离**

定下方向 $v$ 后，干预方式是投影而非加偏置：对每个激活向量，减去它落在 $v$ 上的分量，留下与 $v$ 正交的部分，

$$\tilde{h}_i^{(\ell)} = h_i^{(\ell)} - \langle h_i^{(\ell)}, v \rangle v$$

关键是这个投影不是只在某一处做一次，而是**全局**铺开——所有层 × 所有 token 位置 × 所有去噪步骤都施加，等于在整个反向扩散轨迹上持续把拒绝方向压成零。这样做不触碰扩散采样过程本身（采样器、噪声调度都原样不动），只在表示层动手，因此属于轻量的表示级控制，区别于此前 MDLM 上仅有的采样级引导（如 DIJA）。

**3. 扩散特有的敏感区域：早期步骤、中晚层最吃干预，且跨模型一致**

把投影限定在不同的扩散步骤 / 层上做消融，会暴露出 MDLM 独有的结构。时间维度上，早期去噪步骤的影响不成比例地大——第一个扩散 block 贡献最突出，呼应了扩散模型"浅层对齐"的现象；层维度上，中到晚期的 transformer 层最敏感。把这些敏感度画成热力图，LLaDA-8B 与 LLaDA-1.5 的灵敏区域高度重合，说明这种局部化是同族模型共享的稳定结构；但换到 MMaDA 上模式就变了——干预带来的是更广泛的退化、没有清晰的局部化区域，提示该方法并非对所有 MDLM 都同样干净地起效。

### 损失函数 / 训练策略
无需训练（training-free）。仅需一次前向传播提取方向，推理时全局投影即可，不引入任何额外参数或微调。

## 实验关键数据

### 主实验（JailbreakBench 100 个有害指令）

| 方法 | LLaDA-8B Refusal↓ | LLaDA-8B Safety↓ | LLaDA-1.5 Refusal↓ |
|------|-------------------|-------------------|---------------------|
| Direct | ~98% | ~100% | ~98% |
| GCG (suffix优化) | ~95% | ~98% | - |
| PAIR (自动越狱) | ~70% | ~85% | - |
| Slice (前缀初始化) | ~50% | ~65% | - |
| **Activation Steering (post)** | **0-16%** | **16-25%** | **~低** |
| **Activation Steering (pre)** | **~类似** | **~类似** | **~类似** |

### 消融发现

| 消融维度 | 发现 |
|---------|------|
| Token 位置 | Pre-instruction 和 post-instruction 同样有效 |
| 去噪步骤 | 早期步骤影响最大（第一个 block 贡献最大） |
| Transformer 层 | 中到晚期层最敏感 |
| 跨语言迁移 | 英/中之间强迁移 |
| 跨架构迁移 | MDLM→AR 迁移失败 |

### 关键发现
- **MDLM 的拒绝行为确实受单一低维方向控制**——与自回归 LLM 类似的现象在完全不同的架构中再现
- Pre-instruction token 可以提取有效方向——这是自回归模型中不可能的（因果注意力限制），凸显了扩散模型的非因果特性
- GCG 对 MDLM 几乎无效——为 MDLM 设计的攻击需要新方法
- 跨架构迁移失败说明安全表示是架构依赖的——同一个概念在 AR 和 MDLM 中的编码方式不同
- 早期去噪步骤最关键——与 A2D 论文的发现一致，扩散模型的"浅层对齐"问题

## 亮点与洞察
- **架构间的安全表示对比**：同样的"拒绝"概念在 AR 和 MDLM 中编码不同，但在同一 MDLM 的中英文间共享——说明架构比语言更影响表示结构
- **非因果注意力的安全含义**：MDLM 中每个 token 都能看到完整输入，这使得从任何位置都能提取安全方向——也意味着攻击面更广
- **与 A2D 形成呼应**：A2D 从防御角度做 token 级 [EOS] 对齐，本文从攻击角度做激活引导——共同揭示了 MDLM 安全的新前沿

## 局限与展望
- 仅以安全拒绝为案例研究，其他控制目标（如毒性、风格）未验证
- 对 MMaDA 的引导模式与 LLaDA 不同，说明方法可能不完全通用
- 全局投影可能影响 MDLM 在正常任务上的性能——utility 评估不充分
- 方向提取需要有害 prompt 数据集，实际部署中可能受限

## 相关工作与启发
- **vs Arditi et al. (AR LLM 拒绝方向)**: 本文将其方法论移植到 MDLM，发现类似现象但有扩散特有的结构差异
- **vs A2D / DIJA**: A2D 做防御、DIJA 做攻击——都在 token/采样级操作；本文在表示级操作，更轻量
- **vs AlphaSteer**: AlphaSteer 用零空间约束精确引导 AR LLM；本文揭示了 MDLM 中类似的低维控制结构

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将激活引导应用到 MDLM，揭示扩散特有的表示特性
- 实验充分度: ⭐⭐⭐⭐⭐ 3 个 MDLM + 跨架构迁移、详细消融（层/token/步骤）
- 写作质量: ⭐⭐⭐⭐ 方法清晰，消融图示丰富
- 价值: ⭐⭐⭐⭐⭐ 为 MDLM 安全研究开辟了新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models](../../ICML2026/image_restoration/plan_for_speed_dilated_scheduling_for_masked_diffusion_language_models.md)
- [\[ICML 2026\] Consistent Diffusion Language Models](../../ICML2026/image_restoration/consistent_diffusion_language_models.md)
- [\[NeurIPS 2025\] Encoder-Decoder Diffusion Language Models for Efficient Training and Inference](../../NeurIPS2025/image_restoration/encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)
- [\[ICLR 2026\] Horizon Imagination: Efficient On-Policy Rollout in Diffusion World Models](horizon_imagination_efficient_on-policy_rollout_in_diffusion_world_models.md)
- [\[ICLR 2026\] Trust but Verify: Adaptive Conditioning for Reference-Based Diffusion Super-Resolution](trust_but_verify_adaptive_conditioning_for_reference-based_diffusion_super-resol.md)

</div>

<!-- RELATED:END -->
