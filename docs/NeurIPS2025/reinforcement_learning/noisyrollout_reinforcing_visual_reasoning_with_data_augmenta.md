---
title: >-
  [论文解读] NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation
description: >-
  [NeurIPS 2025][视觉推理] 提出NoisyRollout，一种零额外训练成本的数据增强方法，在GRPO训练VLM时混合来自干净和适度扰动图像的rollout以增强策略探索多样性，仅用2.1K样本在5个域外基准上达到开源RL微调模型SOTA。
tags:
  - NeurIPS 2025
  - 视觉推理
  - 策略探索
  - 数据增强
  - GRPO
  - 噪声退火
---

# NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation

**会议**: NeurIPS 2025  
**arXiv**: [2504.13055](https://arxiv.org/abs/2504.13055)  
**代码**: [GitHub](https://github.com/NoisyRollout)  
**领域**: 强化学习 / VLM推理  
**关键词**: 视觉推理, 策略探索, 数据增强, GRPO, 噪声退火

## 一句话总结

提出NoisyRollout，一种零额外训练成本的数据增强方法，在GRPO训练VLM时混合来自干净和适度扰动图像的rollout以增强策略探索多样性，仅用2.1K样本在5个域外基准上达到开源RL微调模型SOTA。

## 研究背景与动机

- 通过强化学习扩展测试时计算（推理）是增强模型智能的重要方向，但VLM面临独特挑战：
    - **策略探索不足**：传统提高温度等方法引入的是表面多样性，无法引导策略发现更鲁棒的行为
    - **视觉感知缺陷**：VLM经常出现感知错误，进而影响后续推理过程
- 现有VLM-RL工作多直接移植LLM领域的方法，未考虑视觉感知的特殊挑战
- 核心洞察：**如果在扰动图像上也能成功推理，说明推理路径更鲁棒；干净/扰动图像上的奖励差异可作为隐式对比信号改善感知**

## 方法详解

### 整体框架

对每个训练样本 $(I, \mathbf{q})$，老策略生成两组rollout：$n_1$ 个来自干净图像、$n_2$ 个来自扰动图像 $\tilde{I} = T_{\alpha_t}(I)$。所有rollout混合计算奖励基线和优势值。**关键**：策略更新仅以干净图像为条件，扰动图像仅用于收集多样化rollout。噪声退火调度逐渐降低扰动强度。

### 关键设计

1. **混合Rollout策略**:
    - 功能：将干净和扰动图像的推理轨迹混合用于GRPO优化
    - 核心思路：$n_1$ 个clean rollout + $n_2$ 个noisy rollout共同组成一个group，计算统一的奖励均值和标准差作为归一化基准
    - 设计动机：
        - 扰动图像上的成功轨迹提供了替代的、更鲁棒的推理路径
        - 干净/扰动之间的奖励差异暴露感知脆弱性，起到隐式对比学习的作用

2. **噪声退火调度**:
    - 功能：训练过程中逐渐降低图像扰动强度
    - 核心思路：使用sigmoid形退火 $\alpha_t = \alpha_0 \cdot (1 - \frac{1}{1 + e^{-\lambda(t-\gamma)/t_{max}}})$
    - 设计动机：早期高噪声鼓励探索，后期低噪声减少分布偏移确保稳定收敛

3. **策略更新仅条件于干净输入**:
    - 功能：虽然rollout来自扰动图像，但策略梯度计算使用 $\frac{\pi_\theta(\mathbf{o}_i | I, \mathbf{q})}{\pi_{\theta_{old}}(\mathbf{o}_i | I, \mathbf{q})}$
    - 设计动机：避免让策略学习依赖噪声的行为，确保推理时在干净输入上表现最优

### 损失函数 / 训练策略

$$\mathcal{J}(\theta) = \mathbb{E}\left[\frac{1}{n_1+n_2}\sum_{i=1}^{n_1+n_2} \min\left(\frac{\pi_\theta(\mathbf{o}_i | I, \mathbf{q})}{\pi_{\theta_{old}}(\mathbf{o}_i | I, \mathbf{q})}\hat{A}_i, \text{clip}(\cdot, 1-\epsilon, 1+\epsilon)\hat{A}_i\right)\right]$$

- 使用规则奖励（正确=1，错误=0），无KL散度约束
- 默认配置：Gaussian噪声，$n_1=6, n_2=6$（总rollout数=12不变）
- 冻结视觉编码器，学习率1e-6

## 实验关键数据

### 主实验（表格）

Qwen2.5-VL-7B-Instruct，仅2.1K Geometry3K样本：

| 方法 | MathVerse | MathVision | MathVista | WeMath | HallusionBench |
|------|-----------|------------|-----------|--------|----------------|
| Qwen2.5-VL-7B (base) | 46.2 | 25.0 | 67.5 | 63.1 | 64.6 |
| + Vanilla GRPO | 50.8 | 27.3 | 70.5 | 67.4 | 69.8 |
| + **NoisyRollout** | **53.2** | **28.5** | **72.6** | **69.6** | **72.1** |

### 消融实验

- **Rollout多样性分析**：NoisyRollout在训练早期显著提升rollout余弦距离多样性，效果类似提高温度到1.2
- **温度对比**：NoisyRollout（温度1.0）在所有基准上一致超越vanilla GRPO在任何温度（0.8–1.4），说明提供了更有针对性的多样性
- **噪声类型**：高斯噪声和旋转均有效，高斯噪声略优
- **比例实验**：$n_1=6, n_2=6$（50%噪声rollout）是最优比例
- **32B模型**：NoisyRollout同样有效（MathVision 41.6 vs GRPO 40.0）

### 关键发现

- 仅2.1K训练样本即可超越使用15K–260K样本的竞品（如OpenVLThinker、R1-VL），数据效率极高
- HallusionBench上的提升（+2.3%）表明NoisyRollout不仅改善推理，还改善了视觉感知
- 噪声退火是稳定训练的关键——固定噪声强度会导致后期不稳定
- 不同数据集（Geometry3K vs MMK12）和模型规模（7B vs 32B）上均有一致提升

## 亮点与洞察

- 设计极其简洁（"free lunch"）：无额外训练成本、不修改RL目标、不增加总rollout数
- 将视觉扰动作为策略探索工具的思路新颖——利用VLM的视觉感知特性提供有意义的多样性
- 隐式对比学习机制精妙：干净/扰动之间的奖励差异自然约束了感知行为
- 数据效率惊人，2.1K样本在5个域外基准上达SOTA

## 局限与展望

- 扰动类型（高斯噪声、旋转）相对简单，未探索更复杂的增强（如遮挡、风格迁移）
- 裁剪（cropping）等策略不成功的原因未深入分析
- 噪声退火调度的超参数（$\alpha_0, \lambda, \gamma$）选择较为手动
- 对非视觉推理任务（如纯文本推理）的适用性未讨论

## 相关工作与启发

- 与DeepVideo-R1等同期工作互补：NoisyRollout改进探索策略，DeepVideo-R1改进优化目标
- 混合rollout思想可推广到其他RL调优场景（如代码生成、数学推理）
- 噪声退火与curriculum learning理念一致：从宽探索逐渐过渡到窄利用

## 评分

- ⭐⭐⭐⭐⭐ — 方法简洁高效、效果显著、泛化性强，是VLM-RL领域的实用贡献

<!-- RELATED:START -->

## 相关论文

- [Zero-Shot Generalization of Vision-Based RL Without Data Augmentation](../../ICML2025/reinforcement_learning/zero-shot_generalization_of_vision-based_rl_without_data_augmentation.md)
- [RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning](rl_tango_reinforcing_generator_and_verifier_together_for_lan.md)
- [Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning](open_vision_reasoner_transferring_linguistic_cognitive_behavior_for_visual_reaso.md)
- [MergeMix: A Unified Augmentation Paradigm for Visual and Multi-Modal Understanding](../../ICLR2026/reinforcement_learning/mergemix_a_unified_augmentation_paradigm_for_visual_and_multi-modal_understandin.md)
- [AbstRaL: Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking](../../ICLR2026/reinforcement_learning/abstral_augmenting_llms_reasoning_by_reinforcing_abstract_thinking.md)

<!-- RELATED:END -->
