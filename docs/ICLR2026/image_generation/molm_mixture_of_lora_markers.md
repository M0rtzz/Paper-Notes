---
title: >-
  [论文解读] MOLM: Mixture of LoRA Markers
description: >-
  [ICLR 2026][图像生成][水印] 提出 MOLM 水印框架，将 LoRA 适配器重新解释为水印载体，通过二进制密钥驱动的路由机制在冻结生成模型中嵌入可验证、鲁棒的水印，无需逐密钥重训练。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "水印"
  - "LoRA"
  - "扩散模型"
  - "路由机制"
  - "鲁棒性"
---

# MOLM: Mixture of LoRA Markers

**会议**: ICLR 2026  
**arXiv**: [2510.00293](https://arxiv.org/abs/2510.00293)  
**代码**: 未公开  
**领域**: 图像生成  
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

### 整体框架

MOLM 把水印问题统一为对冻结生成模型施加密钥依赖的参数扰动 $\tilde{\mathbf{x}} = \mathcal{G}_{\Phi + \Delta\Phi(\kappa)}(\mathbf{q}, \mathbf{t})$，其中 $\Delta\Phi(\kappa)$ 由二进制密钥 $\kappa$ 决定。区别于以往把整套扰动绑死到单一密钥的做法，MOLM 在模型的若干块中预置一批可选 LoRA 适配器，让密钥的每一段去“路由”激活哪个适配器，从而把切换水印密钥变成切换路由选择，而不再需要重训练。训练时只优化这批 LoRA 与一个解码头，使生成图既保持感知不可见，又能从中可靠恢复出密钥。

### 关键设计

**1. 密钥依赖的参数扰动统一视角：把水印归约为可学习的权重偏移。** 现有水印方法形态各异（编码-解码、后门、采样阶段），难以横向比较和复用。MOLM 先用一个统一形式概括它们：在冻结主干 $\Phi$ 上叠加由密钥决定的扰动 $\Delta\Phi(\kappa)$，生成过程改写为 $\tilde{\mathbf{x}} = \mathcal{G}_{\Phi + \Delta\Phi(\kappa)}(\mathbf{q}, \mathbf{t})$。这一视角的价值在于把“水印强度/容量/可切换性”都落到了对 $\Delta\Phi$ 的结构设计上，为后续用 LoRA 实例化扰动、用路由实现密钥切换铺平了道路。

**2. LoRA 标记的路由机制：用密钥分段索引适配器，实现免重训练的密钥切换。** MOLM 在 $L$ 个预选块中各放置 $P$ 个并列的 LoRA 适配器作为“标记”。一个 $M$ 位密钥被切成 $L$ 个互不重叠的段 $\kappa_\ell$，每段 $\log_2 P$ 位，将该段转换为十进制索引 $s_\ell \in [P]$ 后即选中块 $\ell$ 内的第 $s_\ell$ 个适配器。块的前向运算为 $\boldsymbol{h}_\ell = \mathcal{F}_\ell(\boldsymbol{h}_{\ell-1}) + \alpha \mathcal{A}_\ell^{(s_\ell)}(\boldsymbol{h}_{\ell-1})$，即在原块输出上加一条被选中适配器的低秩支路。默认配置取 VAE 解码器中 $L=14$ 个 ResNet 块、每块 $P=4$ 个适配器，于是单块编码 $2$ 位、总密钥 $M=14\times 2=28$ 位。由于密钥只决定“激活哪条支路”，更换密钥时所有适配器都已训练好、无需任何重训练，这正是 MOLM 相对 Stable Signature 等逐密钥训练方法的核心优势；同时密钥被分摊到多个块上分布式编码，单点被攻击破坏也不致整体失效，带来天然的鲁棒冗余。

**3. 不可见性与可验证性的联合训练目标：在不退化画质的前提下保证密钥可恢复。** 优化只针对适配器参数 $\Psi$ 和解码头 $\eta$，由两项损失约束。感知不可见性损失对比加水印图与原图在多层特征上的差异 $\mathcal{L}_{\text{imp}} = \mathbb{E}_{\kappa} \frac{1}{N} \sum_{n=1}^N \sum_{k=1}^K w_k \|\varphi_k(\mathcal{G}_{\Phi+\Psi(\kappa)}(\mathbf{q}, \mathbf{t}_n)) - \varphi_k(\mathcal{G}_\Phi(\mathbf{q}, \mathbf{t}_n))\|_2^2$，把水印带来的视觉退化压到最低；可验证性损失用二元交叉熵让解码头从生成图恢复出每一位密钥 $\mathcal{L}_{\text{ver}} = \mathbb{E}_{T \sim \Pi} \frac{1}{NM} \sum_{n,m} [-\kappa_m \log \sigma(u_m) - (1-\kappa_m)\log(1-\sigma(u_m))]$，其中 $T\sim\Pi$ 表示训练中随机采样的图像变换，使解码在裁剪、旋转、压缩等扰动下仍成立。总目标 $\min_{\Psi, \eta} [\mathcal{L}_{\text{ver}} + \lambda \mathcal{L}_{\text{imp}}]$ 以权重 $\lambda$ 平衡可恢复性与画质，这也是论文里“提升鲁棒性常引入可见退化”这一冲突被显式建模、而非靠经验权衡的地方。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Implicit Style-Content Separation using B-LoRA](../../ECCV2024/image_generation/implicit_style-content_separation_using_b-lora.md)
- [\[AAAI 2026\] T-LoRA: Single Image Diffusion Model Customization Without Overfitting](../../AAAI2026/image_generation/t-lora_single_image_diffusion_model_customization_without_overfitting.md)
- [\[ICCV 2025\] MoFRR: Mixture of Diffusion Models for Face Retouching Restoration](../../ICCV2025/image_generation/mofrr_mixture_of_diffusion_models_for_face_retouching_restoration.md)
- [\[ICML 2025\] Gaussian Mixture Flow Matching Models](../../ICML2025/image_generation/gaussian_mixture_flow_matching_models.md)
- [\[ICML 2025\] Flat-LoRA: Low-Rank Adaptation over a Flat Loss Landscape](../../ICML2025/image_generation/flat-lora_low-rank_adaptation_over_a_flat_loss_landscape.md)

</div>

<!-- RELATED:END -->
