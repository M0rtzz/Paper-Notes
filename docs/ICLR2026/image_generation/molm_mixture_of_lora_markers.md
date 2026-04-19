---
title: >-
  [论文解读] MOLM: Mixture of LoRA Markers
description: >-
  [ICLR 2026][图像生成][水印] 提出 MOLM 水印框架，将 LoRA 适配器重新解释为水印载体，通过二进制密钥驱动的路由机制在冻结生成模型中嵌入可验证、鲁棒的水印，无需逐密钥重训练。
tags:
  - ICLR 2026
  - 图像生成
  - 水印
  - LoRA
  - 扩散模型
  - 路由机制
  - 鲁棒性
---

# MOLM: Mixture of LoRA Markers

**会议**: ICLR 2026  
**arXiv**: [2510.00293](https://arxiv.org/abs/2510.00293)  
**代码**: 未公开  
**领域**: 图像水印 / 扩散模型安全  
**关键词**: 水印, LoRA, 扩散模型, 路由机制, 鲁棒性

## 一句话总结

提出 MOLM 水印框架，将 LoRA 适配器重新解释为水印载体，通过二进制密钥驱动的路由机制在冻结生成模型中嵌入可验证、鲁棒的水印，无需逐密钥重训练。

## 研究背景与动机

- 扩散模型生成的高质量图像引发真实性和归属权担忧
- 现有水印方法面临三大挑战：
  1. **脆弱性**：对抗攻击（再生攻击、平均攻击）易破解水印
  2. **质量冲突**：提升鲁棒性通常引入可见退化
  3. **高成本**：更换水印密钥需要昂贵的重训练（如 Stable Signature 需逐密钥训练）

## 方法详解

### 通用水印框架

将水印形式化为冻结生成模型的密钥依赖参数扰动：

$$\tilde{\mathbf{x}} = \mathcal{G}_{\Phi + \Delta\Phi(\kappa)}(\mathbf{q}, \mathbf{t})$$

其中 $\Delta\Phi(\kappa)$ 为密钥 $\kappa$ 决定的参数扰动。

### MOLM 路由机制

1. **结构设计**：在 $L$ 个预选块中各添加 $P$ 个 LoRA 适配器
2. **密钥映射**：$M$ 位二进制密钥分为 $L$ 个不重叠块 $\kappa_\ell$，每块 $\log_2 P$ 位
3. **路由选择**：每块 $\kappa_\ell$ 转换为十进制索引 $s_\ell \in [P]$，激活对应适配器

块 $\ell$ 的操作：

$$\boldsymbol{h}_\ell = \mathcal{F}_\ell(\boldsymbol{h}_{\ell-1}) + \alpha \mathcal{A}_\ell^{(s_\ell)}(\boldsymbol{h}_{\ell-1})$$

默认配置：$L=14$ 个 ResNet 块（VAE 解码器），$P=4$ 适配器/块，总密钥 $M = 14 \times 2 = 28$ 位。

### 训练损失

感知不可见性损失：

$$\mathcal{L}_{\text{imp}} = \mathbb{E}_{\kappa} \frac{1}{N} \sum_{n=1}^N \sum_{k=1}^K w_k \|\varphi_k(\mathcal{G}_{\Phi+\Psi(\kappa)}(\mathbf{q}, \mathbf{t}_n)) - \varphi_k(\mathcal{G}_\Phi(\mathbf{q}, \mathbf{t}_n))\|_2^2$$

可验证性损失（二元交叉熵）：

$$\mathcal{L}_{\text{ver}} = \mathbb{E}_{T \sim \Pi} \frac{1}{NM} \sum_{n,m} [-\kappa_m \log \sigma(u_m) - (1-\kappa_m)\log(1-\sigma(u_m))]$$

总目标：$\min_{\Psi, \eta} [\mathcal{L}_{\text{ver}} + \lambda \mathcal{L}_{\text{imp}}]$

## 实验关键数据

### 检测与鲁棒性对比（Stable Diffusion v1.5, MS-COCO）

| 方法 | FID(↓) | SSIM(↑) | Clean | Crop | Rot | Resize | Bright | JPEG | 密钥大小 |
|------|--------|---------|-------|------|-----|--------|--------|------|---------|
| Stable Signature | 29.5 | 0.85 | 0.99 | 0.97 | 0.56 | 0.72 | 0.95 | 0.89 | 48 |
| AquaLoRA | 30.5 | 0.63 | 0.95 | 0.91 | 0.45 | 0.91 | 0.72 | 0.94 | 48 |
| WOUAF | 27.8 | 0.73 | 0.98 | 0.96 | 0.85 | 0.71 | 0.98 | 0.98 | 32 |
| **MOLM** | **27.7** | 0.77 | 0.98 | 0.91 | **0.84** | **0.90** | 0.95 | 0.89 | 28 |

### 对抗攻击鲁棒性（增强训练后）

| 攻击类型 | 参数 | Bit Acc. | FID |
|---------|------|----------|-----|
| Cheng2020 压缩 | q=1/3/6 | 0.94/0.95/0.97 | 30.1/28.9/28.7 |
| 扩散再生 | steps=30/60/100 | 0.85/0.85/0.82 | 30.2/29.9/31.2 |
| PGD 对抗 | ε=10⁻³/10⁻²/10⁻¹ | 1.00/0.99/0.96 | 28.4/28.6/29.0 |
| 平均攻击(5000 图) | k=5000 | ≥0.96 | - |

### 关键发现

1. MOLM 在更小密钥（28 位 vs 48 位）下实现了综合最优鲁棒性
2. 平均攻击下 MOLM 维持 ≥0.96 精度（5000 图），WOUAF 降至 <0.90
3. 伪造攻击下 MOLM 保持随机猜测水平（≈0.5），有效防伪
4. 训练仅需约 1 天（单 A100），推理无额外开销

## 亮点与洞察

1. **概念创新**：将 LoRA 从模型适配工具重新定义为水印载体，思路新颖
2. **无需逐密钥训练**：容量通过路由层数和适配器数量自然扩展
3. **分布式冗余编码**：映射分析表明密钥在多个块之间冗余编码，增强鲁棒性
4. **采样无关性**：不依赖特定采样器（不同于 Tree-Ring 等需要确定性采样的方法）

## 局限性

- UNet 路由实验导致生成质量下降，密钥大小和保真度需权衡
- 仅在 SD v1.5 和 FLUX 上验证，更多架构需要进一步测试
- 28 位密钥容量可能不足以支撑大规模用户归属
- 攻击者独立重训练模型时水印不可迁移（设计预期）

## 相关工作

- **编码-解码方法**：Hidden, Stable Signature
- **后门方法**：DreamBooth 微调, SleeperMark
- **生成过程方法**：Tree-Ring, Gaussian Shading, ROBIN
- **LoRA 混合专家**：MoLE

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — LoRA-as-watermark 的概念转换非常巧妙
- 技术深度：⭐⭐⭐⭐ — 框架设计完整，攻击评估全面
- 实验完整性：⭐⭐⭐⭐ — 多种攻击、多数据集、多架构验证
- 实用价值：⭐⭐⭐⭐ — 高效可部署的水印方案

<!-- RELATED:START -->

## 相关论文

- [Implicit Style-Content Separation using B-LoRA](../../ECCV2024/image_generation/implicit_style-content_separation_using_b-lora.md)
- [T-LoRA: Single Image Diffusion Model Customization Without Overfitting](../../AAAI2026/image_generation/t-lora_single_image_diffusion_model_customization_without_overfitting.md)
- [MoFRR: Mixture of Diffusion Models for Face Retouching Restoration](../../ICCV2025/image_generation/mofrr_mixture_of_diffusion_models_for_face_retouching_restoration.md)
- [Gaussian Mixture Flow Matching Models](../../ICML2025/image_generation/gaussian_mixture_flow_matching_models.md)
- [Flat-LoRA: Low-Rank Adaptation over a Flat Loss Landscape](../../ICML2025/image_generation/flat-lora_low-rank_adaptation_over_a_flat_loss_landscape.md)

<!-- RELATED:END -->
