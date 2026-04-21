---
title: >-
  [论文解读] Distillation of Discrete Diffusion through Dimensional Correlations (Di4C)
description: >-
  [ICML 2025][图像生成][离散扩散模型] 提出Di4C方法，通过"mixture"模型捕获维度间相关性，配合一致性损失函数，将多步离散扩散模型蒸馏为少步模型，同时在图像和语言任务上展示了有效性。
tags:
  - ICML 2025
  - 图像生成
  - 离散扩散模型
  - 知识蒸馏
  - 维度相关性
  - Mixture模型
  - 少步采样
---

# Distillation of Discrete Diffusion through Dimensional Correlations (Di4C)

**会议**: ICML 2025  
**arXiv**: [2410.08709](https://arxiv.org/abs/2410.08709)  
**代码**: [sony/di4c](https://github.com/sony/di4c)  
**领域**: 扩散模型 / 离散生成模型  
**关键词**: 离散扩散模型, 知识蒸馏, 维度相关性, Mixture模型, 少步采样

## 一句话总结

提出Di4C方法，通过"mixture"模型捕获维度间相关性，配合一致性损失函数，将多步离散扩散模型蒸馏为少步模型，同时在图像和语言任务上展示了有效性。

## 研究背景与动机

离散扩散模型在减少采样步数方面面临连续模型不具有的独特挑战：

**维度独立性限制**：传统离散扩散模型使用"product模型"，各维度独立建模采样分布，即 $p_{s|t}(x_s|x_t) = \prod_{d=1}^D p_{s|t}^d(x_s^d|x_t)$。这虽然保证了可扩展性（输出长度从 $O(D^{|S|})$ 降到 $O(D|S|)$），但忽略了维度间的依赖关系。

**少步采样的根本障碍**：在极端情况下（如masked diffusion的单步去噪），product模型完全无法逼近复杂的联合分布。本文Theorem 1证明了 $N$ 步product模型的总变差距离有 $\Omega(1/N)$ 的下界，说明减少步数必须建模维度相关性。

**训练损失的局限**：连续时间score-based离散扩散的损失函数仅需要边际分布，即使模型有能力表达维度相关性也无法学习到。

核心洞察：**多步product模型的复合（composition）可以隐式捕获维度相关性**，即使每一步都是维度独立的。

## 方法详解

### 整体框架

Di4C的核心思想是将多步teacher（product模型）蒸馏为少步student（mixture模型）：
$$p_{0|t_n}^\theta \approx p_{0|t_1}^\psi \circ \cdots \circ p_{t_{n-1}|t_n}^\psi$$

### 关键设计

#### 1. Mixture模型（Section 3.2）

为捕获维度相关性，提出mixture模型：
$$p_{s|t}^\theta(x_s|x_t) = \mathbb{E}_\lambda[p_{s|t}^\theta(x_s|x_t;\lambda)]$$
$$\text{其中 } p_{s|t}^\theta(x_s|x_t;\lambda) = \prod_{d=1}^D p_{s|t}^{\theta,d}(x_s^d|x_t;\lambda)$$

- 每个条件于 $\lambda$ 的分量仍是product模型（维度独立），但通过对 $\lambda$ 取期望实现维度相关
- **Proposition 1**证明了mixture模型的**通用逼近性**：任意离散分布都可以表示为product分布的混合
- 推理时几乎无额外计算开销，只需采样并插入 $\lambda$

#### 2. 蒸馏损失函数（Section 3.3）

**蒸馏损失**——在小时间步 $\delta$ 处逼近teacher：
$$\mathcal{L}_{\text{distil}}(\theta;\psi,r_\delta,\delta) = \mathbb{E}_{x_\delta \sim r_\delta}[D_{\text{KL}}(p_{0|\delta}^\psi(\cdot|x_\delta) \| p_{0|\delta}^\theta(\cdot|x_\delta))]$$

**一致性损失**——学习teacher复合体现的维度相关性：
$$\mathcal{L}_{\text{consis}}(\theta;\psi,r_t,s,u,t) = \mathbb{E}_{x_t \sim r_t}[D_{\text{KL}}(p_{s|u}^\theta \circ p_{u|t}^\psi(\cdot|x_t) \| p_{s|t}^\theta(\cdot|x_t))]$$

- 一致性损失的关键在于混合使用student denoiser和teacher denoiser
- 通过Monte Carlo或控制变量近似计算

### 损失函数 / 训练策略

- 小 $\delta$ 处使用蒸馏损失（teacher单步足够），维度相关性主要通过一致性损失引入
- 参考分布 $r_t$ 可以使用从数据生成的 $q_t$ 或多步teacher去噪得到的分布
- 实际中 $\lambda$ 可以用额外的随机噪声向量输入网络实现

## 实验关键数据

### 主实验

**1. CIFAR-10 像素级离散高斯扩散**
- 使用Campbell et al. (2022)作为teacher
- 少步采样（2-4步）下显著改善teacher的FID指标

**2. ImageNet 类条件生成（Masked Diffusion）**
- Teacher: Besnier & Chen (2023)
- 实现2倍加速，同时保持与teacher相当的生成质量

**3. 语言建模（OpenWebText, Masked Diffusion LM）**
- 在已蒸馏模型（Deschenaux & Gulcehre, 2025）基础上进一步蒸馏
- 通过捕获维度相关性进一步提升，同时不大幅损害采样多样性

### 消融实验

- Mixture模型的混合数量对质量的影响
- 蒸馏步数与一致性损失时间步选择
- 一致性损失vs纯蒸馏损失的效果对比

### 关键发现

| 理论结果 | 内容 |
|---------|------|
| Theorem 1 上界 | $N$ 步product模型的TV距离 = $O(1/N)$ |
| Theorem 1 下界 | 即使 $|S|=D=2$ 的简单例子，下界为 $\Omega(1/N)$ |
| Theorem 2 | Di4C损失可上界teacher与student输出分布的距离 |
| Lemma 1 | 单步product模型的TV距离 = $O(\epsilon^2)$ |

## 亮点与洞察

1. **理论与实践的优美结合**：Theorem 1严格证明了product模型需要 $\Omega(1/\epsilon)$ 步，为维度相关性建模提供了理论必要性
2. **Mixture模型的优雅简洁**：通过简单的随机变量 $\lambda$ 索引product模型族，实现了通用的维度相关性表达，推理开销极小
3. **跨模态通用性**：方法在图像（像素级和masked diffusion）和语言（masked diffusion LM）上都有效
4. **与连续域一致性模型的联系**：一致性损失的设计思想类似连续域的consistency model，但作用在条件概率的复合上
5. **每步近似误差为 $O(\epsilon^2)$**：利用前向过程的维度因子化性质，证明了product模型在小时间间隔内的良好近似性

## 局限与展望

1. **一致性损失的计算**：精确计算需要对高维离散空间求和，Monte Carlo近似引入方差
2. **Mixture模型容量**：$\lambda$ 的分布和维度选择需要经验调优，理论上需要的混合数量可能很大
3. **teacher质量依赖**：蒸馏效果上界取决于teacher的多步采样质量
4. **连续时间设置**：理论分析在连续时间框架下，但实际训练使用离散时间步
5. **扩展到更大规模**：ImageNet实验使用的模型规模相对有限，大规模文本生成的效果有待进一步验证

## 相关工作与启发

- **连续域蒸馏**：Salimans & Ho (2022), Kim et al. (2024) 的consistency distillation
- **离散扩散基础**：Austin et al. (2021) D3PM, Campbell et al. (2022) 连续时间离散扩散
- **并发工作**：Park et al. (2025), Liu et al. (2024), Xu et al. (2025) 也指出了product模型忽略维度相关性的问题
- **启发**：mixture模型的思想可能启发其他需要高效建模高维离散联合分布的场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次系统性地解决离散扩散的少步蒸馏问题，理论+方法均有创新
- 实验充分度: ⭐⭐⭐⭐ — 跨模态验证（图像+语言），多种离散扩散变体
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，Figure 1直观展示了核心问题
- 价值: ⭐⭐⭐⭐⭐ — 对离散扩散模型的加速有重要理论和实践意义

<!-- RELATED:START -->

## 相关论文

- [Learnable Sampler Distillation for Discrete Diffusion Models](../../NeurIPS2025/image_generation/learnable_sampler_distillation_for_discrete_diffusion_models.md)
- [Shallow Diffuse: Robust and Invisible Watermarking through Low-Dimensional Subspaces in Diffusion Models](../../NeurIPS2025/image_generation/shallow_diffuse_robust_and_invisible_watermarking_through_low-dimensional_subspa.md)
- [Constrained Discrete Diffusion](../../NeurIPS2025/image_generation/constrained_discrete_diffusion.md)
- [Information-Theoretic Discrete Diffusion](../../NeurIPS2025/image_generation/information-theoretic_discrete_diffusion.md)
- [Annealing Flow Generative Models Towards Sampling High-Dimensional and Multi-Modal Distributions](annealing_flow_generative_models_towards_sampling_high-dimensional_and_multi-mod.md)

<!-- RELATED:END -->
