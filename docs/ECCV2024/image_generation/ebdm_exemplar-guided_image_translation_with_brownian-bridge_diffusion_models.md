---
title: >-
  [论文解读] EBDM: Exemplar-guided Image Translation with Brownian-bridge Diffusion Models
description: >-
  [ECCV 2024][图像生成][样例引导图像翻译] 提出 EBDM 框架，将样例引导的图像翻译建模为随机布朗桥扩散过程，从结构控制直接翻译为真实感图像，通过 Global Encoder、Exemplar Network 和 Exemplar Attention Module 三个组件有效整合样例的全局风格和细节纹理信息。
tags:
  - ECCV 2024
  - 图像生成
  - 样例引导图像翻译
  - 布朗桥扩散模型
  - 纹理迁移
  - 条件图像生成
  - 风格控制
---

# EBDM: Exemplar-guided Image Translation with Brownian-bridge Diffusion Models

**会议**: ECCV 2024  
**arXiv**: [2410.09802](https://arxiv.org/abs/2410.09802)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 样例引导图像翻译, 布朗桥扩散模型, 纹理迁移, 条件图像生成, 风格控制

## 一句话总结

提出 EBDM 框架，将样例引导的图像翻译建模为随机布朗桥扩散过程，从结构控制直接翻译为真实感图像，通过 Global Encoder、Exemplar Network 和 Exemplar Attention Module 三个组件有效整合样例的全局风格和细节纹理信息。

## 研究背景与动机

样例引导的图像翻译（Exemplar-guided Image Translation）旨在生成同时符合结构控制（语义分割图、边缘图、姿态关键点）和风格样例的真实感图像，在用户可控的风格操作中具有重要应用价值。

现有方法面临三大挑战：

**1. 密集对应的局限**：主流方法（CoCosNet、RABIT 等）依赖建立跨域输入间的密集对应关系，但这带来二次方的内存和计算开销，且在稀疏对应场景（如语义分割图到真实图像）中匹配质量差，导致局部扭曲和语义不一致。

**2. 文本提示的不足**：虽然扩散模型在文本到图像生成中表现出色，但很难用文本准确描述图像的每一个细节（特别是纹理、颜色等视觉属性），且 CLIP 嵌入不足以捕获所有视觉细节。

**3. 多条件的敏感性**：现有基于扩散的方法（如 ControlNet + IP-Adapter 组合）同时使用结构控制和风格条件时，对引导尺度等超参数极其敏感，难以稳定生成。

EBDM 的核心创新在于：利用布朗桥扩散过程将结构控制作为扩散的固定起点，直接翻译为真实感图像，无需额外的结构条件注入机制。这使得网络可以专注于学习样例风格信息的融合，训练和推理更加稳健。

## 方法详解

### 整体框架

EBDM 基于 Stable Diffusion 框架的布朗桥扩散模型（BBDM），包含三个核心组件：

1. **去噪 U-Net**：基于布朗桥过程直接学习从结构控制到真实图像的翻译
2. **Global Encoder**：使用 DINOv2 提取样例图像的全局风格信息
3. **Exemplar Network + Exemplar Attention Module**：提取并融合样例图像的细节纹理信息

**布朗桥扩散 vs 标准扩散的关键区别**：
- 标准 DDPM：$x_T \sim \mathcal{N}(0, I)$（终点是纯高斯噪声）
- 布朗桥：$(x_T, x_0) \sim q_{\text{data}}(\mathcal{X}, \mathcal{Y})$（两个端点都是固定数据点）

具体来说，$x_T = z_\mathcal{X}$ 是结构控制的潜码，$x_0 = z_{\mathcal{X} \to \mathcal{Y}}$ 是目标图像的潜码。前向过程为：

$$q(x_t | x_0, y) = \mathcal{N}(x_t; (1-m_t)x_0 + m_t y, \delta_t I)$$

其中 $m_t = t/T$，$\delta_t = 2(m_t - m_t^2)$。这意味着去噪 U-Net 直接学习从结构控制到图像的翻译，**无需显式的结构条件注入**。

### 关键设计

**1. Global Encoder（全局风格编码）**

选择 DINOv2（而非 CLIP）作为全局风格编码器，因为：
- DINOv2 的自监督学习策略使其在捕获语义特征方面优于 CLIP
- 本方法不需要文本-图像对齐，CLIP 的文本对齐能力在此场景无优势

处理方式：
$$\tau_\theta(I_\mathcal{Y}) = \text{Linear}(\text{DINO}(I_\mathcal{Y})_{[\text{CLS}]}) \in \mathbb{R}^c$$

提取 DINO 的 [CLS] token 通过线性层映射，作为全局风格信息通过交叉注意力机制注入去噪过程。

**2. Exemplar Network（细节纹理网络）**

Global Encoder 受限于输入分辨率（$224^2$），无法保留细粒度纹理细节。因此引入 Exemplar Network $\psi_\theta$：

- 采用与去噪 U-Net 类似的 siamese 结构，移除冗余层以提高效率
- 将样例图像 $z_\mathcal{Y}$ 编码为多层特征图 $\{F_1^l\}_{l=0}^N$
- 在每个块中通过交叉注意力接收全局风格信息

**3. Exemplar Attention Module（样例注意力模块）**

由于样例图像和目标控制不是空间对齐的，简单的拼接或相加不适用。提出空间注意力融合方案：

- 将样例特征 $F_1^l$ 和去噪特征 $F_2^l$ 在空间维度拼接：$F_{\text{in}}^l = \text{concat}(F_1^l, F_2^l) \in \mathbb{R}^{C \times H \times 2W}$
- 对拼接特征施加自注意力，使去噪特征能够查询样例中的相关纹理
- 通过 Chunk 操作提取对应去噪特征部分作为输出

$$Q = \phi_q^l(F_{\text{in}}^l), \quad K = \phi_k^l(F_{\text{in}}^l), \quad V = \phi_v^l(F_{\text{in}}^l)$$

$$F_{\text{EA}}^l = W^l \text{Softmax}(QK^T / \sqrt{V}) V + F_{\text{in}}^l$$

这种设计既避免了密集对应匹配的高开销，又允许模型自适应地从样例中选取相关纹理。

### 损失函数 / 训练策略

**两阶段训练**：

- **第一阶段**：训练去噪 U-Net + Global Encoder 的交叉注意力，学习从控制到图像的翻译 + 粗略样例风格融合。使用重建任务（目标图像本身作为样例），冻结 VAE 和 Global Encoder 预训练参数
- **第二阶段**：引入 Exemplar Network 和 Exemplar Attention Module，冻结第一阶段的参数，专注训练细节纹理整合。使用预定义的样例-目标对

**训练目标**：

$$\mathbb{E}_{x_0, y, I_\mathcal{Y}, \epsilon}[c_{\epsilon t} \| m_t(x_T - x_0) + \sqrt{\delta_t}\epsilon - \epsilon_\theta(x_t, t, \tau_\theta(I_\mathcal{Y}), \psi_\theta(z_\mathcal{Y}, \tau_\theta(I_\mathcal{Y}))) \|^2]$$

**推理**：使用确定性 ODE 采样器，从结构控制起点开始逐步去噪，仅需单一的样例条件。

## 实验关键数据

### 主实验

图像质量对比（FID ↓ / SWD ↓ / LPIPS ↑，三个任务）：

| 方法 | DeepFashion FID | CelebA-HQ Edge FID | CelebA-HQ Mask FID |
|------|----------------|--------------------|--------------------|
| CoCosNet | 14.40 | 14.30 | 21.83 |
| CoCosNetv2 | 12.81 | 12.85 | 20.64 |
| RABIT | 12.58 | 11.67 | 20.44 |
| MIDMs | 10.89 | 15.67 | N/A |
| **EBDM (Ours)** | **10.62** | **11.84** | **12.21** |

与 SOTA 扩散方法对比（CelebA-HQ Edge）：

| 方法 | SSIM ↑ | PSNR ↑ |
|------|--------|--------|
| ControlNet | 0.882 | 35.30 |
| ControlNet+CLIP | 0.894 | 35.94 |
| **EBDM (Ours)** | **0.901** | **36.40** |

### 消融实验

Global Encoder 选择对比（CelebA-HQ Edge）：

| 配置 | SSIM ↑ | FID ↓ | Sem. ↑ |
|------|--------|-------|--------|
| Baseline (无全局编码) | 0.831 | 16.31 | 0.531 |
| + CLIP | 0.632 | 23.42 | 0.752 |
| + DINO | 0.754 | 21.32 | 0.786 |
| **完整方法 (EBDM)** | **0.901** | **11.84** | **0.920** |

CLIP 作全局编码器反而大幅降低 SSIM（从 0.831 降到 0.632），因为其文本对齐特性在此任务中并非优势。DINOv2 配合完整框架效果最优。

### 关键发现

1. **布朗桥扩散的根本优势**：将结构控制作为扩散端点而非额外条件，使模型天然保持结构一致性，释放条件容量给风格融合
2. **DINOv2 优于 CLIP 做视觉风格编码**：自监督学习特征在细粒度视觉相似性上显著优于对比学习特征
3. **在 mask-to-photo 任务上优势最大**（FID 12.21 vs 次优 20.44）：因为匹配方法在语义分割图上难以建立有效对应，而扩散方法通过迭代去噪自然处理
4. **单一条件的鲁棒性**：相比 ControlNet + IP-Adapter 的多条件组合，EBDM 仅用样例条件就实现更好效果，且无超参敏感问题

## 亮点与洞察

1. **布朗桥扩散的巧妙应用**：将图像翻译自然表述为两个固定端点间的随机过程，避免了结构条件注入的复杂性，是比 ControlNet 更优雅的结构保持方案
2. **摆脱密集对应匹配范式**：证明了扩散框架可以完全替代"先匹配后生成"的传统流水线，计算效率和生成质量同时提升
3. **Global + Local 的互补设计**：DINOv2 [CLS] 抓全局风格，Exemplar Network 抓局部纹理，双路径互补
4. **Exemplar Attention 的空间自注意力融合**：解决了非对齐特征整合的核心挑战

## 局限与展望

1. 仅在人脸（CelebA-HQ）和时装（DeepFashion）数据集上验证，未测试更复杂的场景级任务
2. 语义一致性分数（Tab. 4）在部分指标上不如 DynaST 等方法，可能因为布朗桥的随机性引入了一些变异
3. 两阶段训练增加了实现复杂度
4. Exemplar Attention Module 的空间拼接在高分辨率下显存开销可能较大
5. 未探索视频序列的时间一致性扩展

## 相关工作与启发

- **BBDM**：本文首次将布朗桥扩散模型应用于样例引导图像翻译，证明了该扩散范式在跨域翻译中的优势
- **ControlNet / IP-Adapter**：虽然灵活，但多条件组合的超参敏感性是实际部署的瓶颈，EBDM 的单条件设计更稳健
- **CoCosNet 系列**：建立了"匹配-生成"范式，但 EBDM 证明了不需要显式匹配也能实现甚至超越其效果
- **启发**：布朗桥扩散过程可推广到其他跨域翻译任务（语义 → 真实、草图 → 照片、白天 → 夜晚等）

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 布朗桥在样例引导翻译中的首次应用，框架设计优雅
- 技术深度：⭐⭐⭐⭐ — 三组件各有章法，消融实验清晰验证每个设计选择
- 实验充分度：⭐⭐⭐⭐ — 三个任务 + 多个基线对比 + 消融
- 实用价值：⭐⭐⭐⭐ — 虚拟试衣、人脸编辑等场景有直接应用
- 总体推荐：⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] Diffusion-based Image-to-Image Translation by Noise Correction via Prompt Interpolation](diffusion-based_image-to-image_translation_by_noise_correction_via_prompt_interp.md)
- [\[NeurIPS 2025\] Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge](../../NeurIPS2025/image_generation/towards_general_modality_translation_with_contrastive_and_predictive_latent_diff.md)
- [\[ECCV 2024\] LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)
- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [\[ECCV 2024\] Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models](source_prompt_disentangled_inversion_for_boosting_image_editability_with_diffusi.md)

<!-- RELATED:END -->
