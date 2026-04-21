---
title: >-
  [论文解读] Laplacian Multi-scale Flow Matching for Generative Modeling
description: >-
  [ICLR 2026][图像生成][多尺度生成] 提出 LapFlow，将图像分解为拉普拉斯金字塔残差，通过混合 Transformer（MoT）架构和因果注意力并行建模不同尺度，在减少计算量的同时提升生成质量。
tags:
  - ICLR 2026
  - 图像生成
  - 多尺度生成
  - Laplacian金字塔
  - Flow Matching
  - Transformer
  - 因果注意力
---

# Laplacian Multi-scale Flow Matching for Generative Modeling

**会议**: ICLR 2026  
**arXiv**: [2602.19461](https://arxiv.org/abs/2602.19461)  
**代码**: [GitHub](https://github.com/sjtuytc/gen)  
**领域**: 扩散模型 / Flow Matching  
**关键词**: 多尺度生成, Laplacian金字塔, Flow Matching, Mixture-of-Transformers, 因果注意力

## 一句话总结
提出 LapFlow，将图像分解为拉普拉斯金字塔残差，通过混合 Transformer（MoT）架构和因果注意力并行建模不同尺度，在减少计算量的同时提升生成质量。

## 研究背景与动机
- 扩散模型和 Flow Matching 在图像合成中取得SOTA，但随着分辨率增加，可扩展性成为关键挑战
- 现有多尺度方法（Cascaded Diffusion、EdifyImage、Pyramidal Flow）各有局限：需要训练多个独立网络，或在像素空间操作导致推理慢，或从零训练效果不佳
- 需要一种既能提升生成质量又能加速采样、可扩展到高分辨率的多尺度框架

## 方法详解

### 整体框架
LapFlow 将图像分解为拉普拉斯金字塔残差（3 个尺度），通过统一的 MoT 模型并行处理不同尺度。采用渐进式生成策略：先去噪最粗尺度，再逐步条件化更细尺度。

### 关键设计

1. **拉普拉斯分解**: 将图像分解为三个尺度的残差：
$$\mathbf{x}_1^{(2)} = \text{Down}(\text{Down}(\mathbf{x}_1)), \quad \mathbf{x}_1^{(1)} = \text{Down}(\mathbf{x}_1) - \text{Up}(\mathbf{x}_1^{(2)})$$
$$\mathbf{x}_1^{(0)} = \mathbf{x}_1 - \text{Up}(\text{Down}(\mathbf{x}_1))$$
   重建：$\mathbf{x}_1 = \mathbf{x}_1^{(0)} + \text{Up}(\mathbf{x}_1^{(1)}) + \text{Up}(\text{Up}(\mathbf{x}_1^{(2)}))$

2. **多尺度噪声过程**: 不同尺度在不同时间范围内训练。两个关键时间点 $T_1, T_2$：最小尺度 $k=2$ 在 $[0,1]$ 训练，中间尺度 $k=1$ 在 $[T_2,1]$ 训练，最大尺度 $k=0$ 在 $[T_1,1]$ 训练。每尺度的噪声插值：
$$\mathbf{x}_t^{(k)} = \alpha_t^{(k)} \mathbf{x}_1^{(k)} + \sigma_t^{(k)} \mathbf{x}_0^{(k)}$$

3. **MoT 架构与因果注意力**: 使用尺度特定的 QKV 投影和共享全局注意力。因果掩码确保信息从低分辨率单向流向高分辨率：
$$\text{MaskedGlobalAttn}(Q,K,V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d}} + M_c\right)V$$
   其中 $M_c$ 为块因果掩码，确保尺度 $k$ 只能关注 $k' \geq k$ 的尺度。

### 损失函数 / 训练策略
多尺度条件 Flow Matching 损失：
$$\mathcal{L}_{mv} = \sum_{k=2}^{s} w_k \mathbb{E}_{t,q,p_t} \|\mathbf{v}_t^{(k)} - \mathbf{u}_t^{(k)}(\mathbf{x}_t^{(k)}|\mathbf{x}_1^{(k)})\|^2$$
渐进式训练策略：每次采样阶段 $s$，训练所有 $k \geq s$ 的尺度。

## 实验关键数据

### 主实验

| 方法 | 数据集 | 分辨率 | FID↓ | GFLOPs | 推理时间(s) |
|------|--------|--------|------|--------|-------------|
| LFM | CelebA-HQ | 256 | 5.26 | 22.1 | 1.70 |
| Pyramidal Flow | CelebA-HQ | 256 | 11.20 | 14.2 | 1.85 |
| **LapFlow (Ours)** | CelebA-HQ | **256** | **3.53** | **16.5** | **1.51** |
| LFM | CelebA-HQ | 512 | 6.35 | 43.5 | 2.90 |
| **LapFlow (Ours)** | CelebA-HQ | **512** | **4.04** | **41.7** | **2.60** |
| LFM | CelebA-HQ | 1024 | 8.12 | 154.8 | 4.20 |
| **LapFlow (Ours)** | CelebA-HQ | **1024** | **5.51** | **148.2** | **3.30** |

### 消融实验

| 配置 | FID (256×256) | GFLOPs | 说明 |
|------|--------------|--------|------|
| Separate Model | 3.60 | 38.9 | 各尺度独立模型 |
| **MoT (默认)** | **3.53** | **16.5** | 共享参数+专家 |
| SDVAE | 4.37 | - | 标准 VAE |
| **EQVAE (默认)** | **3.53** | - | 等变 VAE |

### 关键发现
- LapFlow 在 CelebA-HQ 256 上 FID=3.53，远优于 LFM 的 5.26
- MoT 设计将 GFLOPs 从 38.9 降至 16.5，同时 FID 还略有提升
- 因果掩码至关重要：无掩码或仅自注意力均导致性能下降
- 有效扩展到 1024×1024 高分辨率，保持较低计算开销

## 亮点与洞察
- 利用拉普拉斯金字塔天然的多尺度特性，将不同频率成分分开建模
- MoT 架构巧妙结合尺度特定处理和全局共享注意力，实现参数高效计算
- 因果注意力强制了自然的信息流：从结构到细节的层次化生成
- 时间加权复杂度分析证明了渐进式多尺度设计的注意力成本理论上低于 DiT

## 局限与展望
- 目前仅在 CelebA-HQ 和 ImageNet 上验证，缺少文本引导生成的评估
- 拉普拉斯分解在 latent space 中的适用性可能不如像素空间直观
- 关键时间点 $T_1, T_2$ 需手动设定
- 未与最新的 text-to-image 大模型比较

## 相关工作与启发
- 多尺度思想从 LapGAN 开始，经 Cascaded Diffusion、Pyramidal Flow 发展，LapFlow 通过消除显式桥接机制实现并行
- MoT 思想源自 Mixture-of-Experts，首次应用于多尺度视觉生成
- 为高分辨率视觉生成提供了更高效的替代方案

## 技术细节补充
- 3 个尺度的拉普拉斯分解在 latent space 中操作（VAE 下采样因子 8），最大潜在尺寸 32×32
- 训练：CelebA-HQ 使用 DiT-L/2，ImageNet 支持 DiT-B/2 和 DiT-XL/2
- 采样使用 Dormand-Prince 方法（dopri5）ODE solver
- GVP 路径通常优于线性路径（消融中验证）
- 支持 classifier-free guidance，ImageNet 上使用 CFG
- 时间加权复杂度分析证明有效注意力成本低于 DiT
- EQVAE 对 LapFlow 有益但对 LFM 无益（Table 2a），说明多尺度框架更能利用高质量 VAE
- ImageNet 256 上 LapFlow 超越单尺度和多尺度基线，同时 GFLOPs 更低
- 支持 ImageNet 类条件生成和 CelebA-HQ 无条件生成

## 评分
- 新颖性: ⭐⭐⭐⭐ 拉普拉斯金字塔+MoT+因果注意力的组合有创意
- 实验充分度: ⭐⭐⭐⭐ 在两个数据集上全面评估和消融，但缺少 T2I 实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，算法描述完善
- 价值: ⭐⭐⭐⭐ 在效率和质量之间取得良好平衡，对多尺度生成有指导意义

<!-- RELATED:START -->

## 相关论文

- [Multi-agent Coordination via Flow Matching](multi-agent_coordination_via_flow_matching.md)
- [SoFlow: Solution Flow Models for One-Step Generative Modeling](soflow_solution_flow_models_for_one-step_generative_modeling.md)
- [Flow2GAN: Hybrid Flow Matching and GAN with Multi-Resolution Network for Few-step High-Fidelity Audio Generation](flow2gan_hybrid_flow_matching_and_gan_with_multi-resolution_network_for_few-step.md)
- [GenCP: Towards Generative Modeling Paradigm of Coupled Physics](gencp_towards_generative_modeling_paradigm_of_coupled_physics.md)
- [SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation](senseflow_scaling_distribution_matching_for_flow-based_text-to-image_distillatio.md)

<!-- RELATED:END -->
