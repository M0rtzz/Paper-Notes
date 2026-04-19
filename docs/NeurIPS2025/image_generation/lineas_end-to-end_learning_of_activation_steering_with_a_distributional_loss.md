---
title: >-
  [论文解读] LinEAS: End-to-end Learning of Activation Steering with a Distributional Loss
description: >-
  [NeurIPS 2025][图像生成][Activation Steering] 提出 LinEAS（Linear End-to-end Activation Steering），通过端到端优化跨层仿射变换映射，利用 1D Wasserstein 分布损失进行全局激活值对齐，仅需 32 个无配对样本即可高效控制 LLM 毒性和 T2I 模型概念生成。
tags:
  - NeurIPS 2025
  - 图像生成
  - Activation Steering
  - optimal transport
  - Toxicity Mitigation
  - text-to-image
  - Sparse Regularization
---

# LinEAS: End-to-end Learning of Activation Steering with a Distributional Loss

**会议**: NeurIPS 2025  
**arXiv**: [2503.10679](https://arxiv.org/abs/2503.10679)  
**代码**: [github.com/apple/ml-lineas](https://github.com/apple/ml-lineas) (有)  
**领域**: 图像生成  
**关键词**: Activation Steering, optimal transport, Toxicity Mitigation, text-to-image, Sparse Regularization

## 一句话总结

提出 LinEAS（Linear End-to-end Activation Steering），通过端到端优化跨层仿射变换映射，利用 1D Wasserstein 分布损失进行全局激活值对齐，仅需 32 个无配对样本即可高效控制 LLM 毒性和 T2I 模型概念生成。

## 研究背景与动机

**领域现状**：生成模型（LLM、T2I）广泛部署后需要高效的行为控制机制——降低毒性、风格调整、概念删除等。现有 alignment 方法（RLHF、LoRA）计算成本高，需大量标注数据。

**现有痛点**：
   - 基于配对数据的方法（CAA、ReFT）需要反事实数据，很多场景下不存在。
   - 逐层独立优化的方法（ITI-c、Lin-AcT）不考虑上下游层的交互，导致因果不一致——在一层的干预会在后续层产生意外偏移。
   - 无法自动选择需要干预的层和神经元。

**核心矛盾**：控制效果与模型实用性（utility）之间的权衡——干预太多损害通用能力，干预太少效果不足。

**本文目标**：用极少量无配对数据（32 个源 + 32 个目标样本）实现精确、低成本的激活值引导。

**切入角度**：将激活值引导视为最优传输问题——用全局分布损失端到端联合优化所有层的仿射变换。

**核心 idea**：联合优化所有层的逐坐标仿射映射，使源分布的激活值在每一层都对齐到目标分布，配合稀疏正则化自动选择关键神经元。

## 方法详解

### 整体框架

给定预训练模型 $f_1, \ldots, f_{L+1}$，在 $L$ 个中间层之间插入仿射变换 $T_\ell$：
$$\mathbf{o} = f_{L+1} \circ T_L \circ f_L \circ \cdots \circ T_1 \circ f_1(\mathbf{x})$$

每个 $T_\ell(z) = \omega_\ell \odot z + b_\ell$，参数仅为逐元素的缩放 $\omega_\ell$ 和偏移 $b_\ell$。

### 关键设计

#### 1. 分布损失函数
- **功能**：衡量经变换后的源分布激活值与目标分布激活值在每层的差异。
- **核心思路**：使用 1D Wasserstein 距离沿每个激活维度独立计算：
  $$\Delta(U, V) = \sum_{j=1}^d W_2^2(U_{\cdot j}, V_{\cdot j}) = \frac{1}{n}\sum_{j=1}^d \|\tilde{U}_{\cdot j} - \tilde{V}_{\cdot j}\|^2$$
  其中 $\tilde{U}, \tilde{V}$ 为排序后的激活值矩阵。
- **总损失**：$\mathcal{C} = \sum_{\ell \leq L} \Delta((\xi_\ell^i)_i, (\eta_\ell^j)_j)$。
- **设计动机**：高维低样本场景（$d_\ell \gg N$）下，多维 Wasserstein 不稳定；1D 边际 Wasserstein 更鲁棒。

#### 2. 端到端联合优化
- **功能**：所有 $T_\ell$ 同时优化，考虑层间因果依赖。
- **与 Lin-AcT 的区别**：Lin-AcT 逐层独立闭式求解，每层假设其他层冻结，忽略了干预在下游层的传播效应。LinEAS 通过反向传播同时更新所有层参数。
- **优化方法**：Proximal SGD，带余弦学习率衰减，1K 步。

#### 3. 稀疏正则化
- **功能**：自动选择需要干预的层和神经元子集。
- **Sparse Group Lasso**：
  $$\mathcal{R} = \lambda_1 \sum_\ell (\|\omega_\ell - \mathbf{1}\|_1 + \|b_\ell\|_1) + \lambda_G \sum_\ell \sqrt{d_\ell}(\|\omega_\ell - \mathbf{1}\|_2 + \|b_\ell\|_2)$$
  - $\ell_1$ 范数促进层内神经元稀疏；$\ell_2$ 范数促进层间选择。
  - 空间开销极小：identity 变换（$\omega = \mathbf{1}, b = \mathbf{0}$）即为不干预。
- **效果**：将干预 support 从 100% 降至约 1% 仍保持毒性缓解效果，且 utility 更好。

#### 4. 连续可控强度
- 引入缩放因子 $\lambda \in [0, 1]$：$T_\ell^\lambda(z) = (\mathbf{1} + \lambda(\omega_\ell - \mathbf{1})) \odot z + \lambda b_\ell$。
- $\lambda = 0$ 无干预，$\lambda = 1$ 完全干预，中间值平滑过渡。

### 损失函数/训练策略

$$\mathcal{L}(\mathbf{w}, \mathbf{b}) = \mathbb{E}_{(\mathbf{x}^i) \sim p, (\mathbf{y}^j) \sim q}[\mathcal{C}(\mathbf{w}, \mathbf{b}; (\mathbf{x}^i)_i, (\mathbf{y}^j)_j)] + \gamma \mathcal{R}(\mathbf{w}, \mathbf{b})$$

- 目标分布 $q$ 的激活值可预计算（不经过 $T_\ell$），节省计算。
- 源分布 $p$ 的激活值需在线计算（经过 $T_\ell$ 后传播）。
- Training: SGD, lr=0.1, 1K steps, batch size = N (32)。

## 实验关键数据

### 主实验：LLM 毒性缓解

在 3 个模型上用 N=32 无配对样本的毒性率（%，越低越好）：

| 模型 | 方法 | Tox(RTP) ↓ | Tox(TET) ↓ | PPL(WIK) ↓ | MMLU ↑ |
|------|------|------------|------------|------------|--------|
| Qwen2.5-1.5B | 无干预 | 3.00 | 23.09 | 13.67 | 60.95 |
| | Lin-AcT | 1.50 | 13.88 | 13.88 | 60.09 |
| | **LinEAS** | **1.07** | **12.70** | 14.10 | 59.97 |
| Gemma2-2B | 无干预 | 4.00 | 13.39 | 14.79 | 53.03 |
| | Lin-AcT | 1.60 | 7.76 | 14.78 | 52.43 |
| | **LinEAS** | **0.73** | **4.02** | 15.46 | 52.22 |
| Qwen2.5-7B | 无干预 | 3.92 | 25.16 | 10.67 | 74.26 |
| | Lin-AcT | 2.72 | 21.64 | 11.42 | 72.18 |
| | **LinEAS** | **1.95** | **14.95** | 10.91 | 73.67 |

LinEAS 在 Gemma2-2B 上将毒性降低 **5.5×**，接近使用 oracle 标签的 LoFIT-RL。

### T2I 概念删除（DMD2 模型）

| 方法 | 用户偏好 ↑ | IMGScore ↑ | CLIPScore ↓ |
|------|-----------|------------|-------------|
| ITI-c | 12.4% | 0.24 | 0.19 |
| Lin-AcT | 24.4% | 0.45 | 0.18 |
| **LinEAS** | **63.3%** | **0.66** | 0.18 |

### 消融实验

| 消融维度 | 结果 |
|---------|------|
| 数据量（N=1→1024）| N=32 即达到接近饱和的毒性缓解效果 |
| 稀疏度 γ（0→0.1）| support 降至 1%，毒性不变，PPL↓ MMLU↑ |
| 训练步数（100→10K）| 1K 步最优，100 步缓解不足，10K 步 utility 下降 |
| 干预层选择 | layernorm 层最优；对层类型选择鲁棒 |

### 关键发现

- 端到端优化比逐层独立优化（Lin-AcT）泛化性更好，对层选择更鲁棒。
- CAA 和 ReFT 虽有更强监督信号，但在非配对设置下严重损害 utility（MMLU 下降 20+）。
- **逆向映射** $T_\ell^{-1}$ 可将概念删除变为概念注入，说明激活空间的强结构性。

## 亮点与洞察

- **极低数据需求**：32 个无配对样本即可学到有效干预，远低于 fine-tuning 方法。
- **理论根基**：基于最优传输理论的分布对齐，$\lambda$ 控制强度有连续且有界的理论保证。
- **稀疏自动选择**：Sparse Group Lasso 同时实现层选择和神经元选择，100× 支撑压缩。
- **模态无关**：同一框架适用于 LLM（毒性缓解）和 T2I（概念控制）。

## 局限与展望

- **可组合性**：同时应用多个干预（如同时删除两个概念）效果有限——仅 19% 同时成功。
- **干预选择性**：当前对所有 token 统一应用干预，无法逐 token 选择性施加。
- **推理开销**：虽然仿射变换成本极低，但需要存储每个控制目标的参数。
- 需要访问模型内部激活值，不适用于 API-only 的模型。

## 相关工作与启发

- **Lin-AcT**：LinEAS 的直接前身，逐层独立闭式求解仿射映射。LinEAS 的端到端优化弥补了逐层累积误差。
- **ITI-c**：用线性分类器找引导向量，但不考虑跨层效应。
- **ReFT**：需要配对数据的低秩表示微调，在无配对设置下退化。
- **AurA**：按分类能力衰减激活值，不如仿射变换灵活。
- **启发**：分布对齐 + 稀疏正则化是一种强大的轻量级模型控制范式，可推广到更多生成模型控制场景。

## 评分

⭐⭐⭐⭐⭐ (5/5)

方法简洁优雅，理论根基扎实（OT + Sparse Lasso），实验覆盖 LLM 和 T2I 两个模态且均领先。极低数据需求（32 样本）和极少参数（<0.25M）使其高度实用。可组合性是尚待克服的主要挑战。

<!-- RELATED:START -->

## 相关论文

- [End-to-End Multi-Modal Diffusion Mamba](../../ICCV2025/image_generation/end-to-end_multi-modal_diffusion_mamba.md)
- [REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers](../../ICCV2025/image_generation/repa-e_unlocking_vae_for_end-to-end_tuning_of_latent_diffusion_transformers.md)
- [REPA-E: Unlocking VAE for End-to-End Tuning of Latent Diffusion Transformers](../../ICCV2025/image_generation/repae_unlocking_vae_for_endtoend_tuning_of_latent_diffusion.md)
- [On Optimal Steering to Achieve Exact Fairness](on_optimal_steering_to_achieve_exact_fairness.md)
- [Energy Loss Functions for Physical Systems](energy_loss_functions_for_physical_systems.md)

<!-- RELATED:END -->
