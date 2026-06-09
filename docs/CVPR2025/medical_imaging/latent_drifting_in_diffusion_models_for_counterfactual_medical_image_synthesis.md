---
title: >-
  [论文解读] Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis
description: >-
  [CVPR 2025][医学图像][反事实图像生成] 本文提出 Latent Drifting (LD)，通过在扩散模型的前向和反向过程中引入一个标量偏移参数 δ 来弥合预训练自然图像模型与医学图像目标分布之间的差距，显著提升了多种微调方案下的医学图像生成和反事实图像合成效果。
tags:
  - "CVPR 2025"
  - "医学图像"
  - "反事实图像生成"
  - "扩散模型微调"
  - "隐空间漂移"
  - "医学影像合成"
  - "分布迁移"
---

# Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis

**会议**: CVPR 2025  
**arXiv**: [2412.20651](https://arxiv.org/abs/2412.20651)  
**代码**: [https://latentdrifting.github.io/](https://latentdrifting.github.io/) (项目页)  
**领域**: 医学图像 / 扩散模型  
**关键词**: 反事实图像生成, 扩散模型微调, 隐空间漂移, 医学影像合成, 分布迁移

## 一句话总结
本文提出 Latent Drifting (LD)，通过在扩散模型的前向和反向过程中引入一个标量偏移参数 δ 来弥合预训练自然图像模型与医学图像目标分布之间的差距，显著提升了多种微调方案下的医学图像生成和反事实图像合成效果。

## 研究背景与动机

1. **领域现状**：预训练扩散模型（如 Stable Diffusion）在自然图像生成上表现卓越，医学领域希望利用这些模型的强大生成能力。现有微调方法（Textual Inversion、DreamBooth、Custom Diffusion）允许用少量样本为模型引入新概念。
2. **现有痛点**：医学图像与自然图像的分布差异巨大（如脑 MRI 背景必须全黑、骨性结构必须保持形状），直接微调预训练模型难以适应这种分布偏移。少量医学样本无法有效调整模型学到的自然图像分布。同时，从头训练医学扩散模型面临数据隐私、成本和稀有疾病等限制。
3. **核心矛盾**：预训练模型的隐空间噪声分布 $z_T \sim \mathcal{N}(0, I)$ 是为自然图像设计的，但医学图像的最优采样分布可能与 $\mathcal{N}(0, I)$ 有偏移。微调只调整模型参数 θ，但从不改变隐空间分布。
4. **本文目标** (1) 如何高效地将预训练扩散模型适配到医学图像域；(2) 如何实现高质量的医学反事实图像生成（如疾病添加/移除、年龄变化、性别转换）。
5. **切入角度**：将隐空间的终态变量 $z_T$ 视为另一个条件因子而非固定假设，通过一个简单的标量偏移 δ 修改隐空间均值来匹配目标分布。
6. **核心 idea**：在扩散过程的每个时间步为均值添加一个全局偏移 δ，将隐空间分布从自然图像域「漂移」到医学图像域。

## 方法详解

### 整体框架
Latent Drifting 是一个通用的插件方法，可以与任何扩散模型微调方案结合。给定预训练的 Stable Diffusion 和目标医学数据集，LD 在微调时同时修改前向和反向过程的分布，在推理时修改反向过程的分布。方法将反事实图像生成形式化为一个 min-max 优化问题，在保持与原图相似性（Counterfactual Fidelity）的同时最大化结果变化（Desired Outcome Fidelity）。

### 关键设计

1. **Latent Drifting 机制**:

    - 功能：通过引入标量偏移 δ，修改扩散过程的隐空间分布以匹配目标医学图像域。
    - 核心思路：在反向过程的转移核中添加偏移 $p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t) + \delta, \Sigma_\theta(x_t, t))$。δ 是一个有符号标量，在前向扩散时也相应偏移 $z_T$。通过网格搜索（遍历 δ 从 -0.2 到 0.2）找到使生成分布 $\mathcal{D}_\theta$ 与目标分布 $\mathcal{D}_{GT}$ 的 L1 距离最小的 δ 值。实验发现 δ=0.1 在脑 MRI 上效果最佳。
    - 设计动机：传统微调假设 θ 的更新足以覆盖分布迁移，但实际上隐空间分布 $\mathcal{N}(\mu, \sigma)$ 从未被调整——未加 LD 时微调后的隐空间分布方差很大且不稳定，加了 LD 后分布达到稳定点，对分布偏移更鲁棒。

2. **反事实生成的形式化框架**:

    - 功能：将医学反事实图像生成统一建模为约束优化问题。
    - 核心思路：目标函数 $L(x, x', y', \lambda) = \min_{\ell_o}[\lambda \cdot \ell_o(\hat{f}(x'), y')] + \min_{\ell_{in}}[\ell_{in}(x, x')]$。其中 $\ell_{in}$ 保证反事实图像 $x'$ 与原图 $x$ 的相似性，$\ell_o$ 保证反事实结果符合目标标签 $y'$。两者互为约束：$\ell_{in} \propto 1/\ell_o$。当 $\lambda=0$ 退化为标准微调（$z=z'$），当 $\lambda>0$ 引入 LD 的 δ 偏移来增强条件控制。
    - 设计动机：反事实图像生成本质上是在"改变目标特征"和"保留原始特征"之间的平衡问题，这个优化框架自然地将 LD 纳入条件控制。

3. **与多种微调方案的结合**:

    - 功能：证明 LD 作为通用插件可以适配不同的微调策略。
    - 核心思路：对四种微调方法分别结合 LD：(1) Textual Inversion——仅微调文本编码器的嵌入空间；(2) DreamBooth——微调去噪 U-Net 并用类先验保留损失；(3) Custom Diffusion——仅微调 U-Net 中的交叉注意力层权重；(4) Basic FT——微调整个去噪 U-Net。每种方法都简单地在扩散过程中加入 δ 偏移即可。对于 image-to-image 的反事实生成，还与 Pix2Pix Zero 和 InstructPix2Pix 结合。
    - 设计动机：LD 是在扩散过程层面的修改（改变均值），与模型参数的微调方式正交，因此可以无缝嵌入任何微调方案，无需修改其内部机制。

### 损失函数 / 训练策略
基础训练损失为标准去噪目标 $\mathbb{E}_{x,c,\epsilon,t}[w_t\|\hat{x}_\theta(\alpha_t x + \sigma_t \epsilon, c) - x\|_2^2]$，在此基础上 LD 仅修改采样分布。使用 SD-v1.4 预训练模型，δ 通过网格搜索在 [-0.2, 0.2] 范围内确定，使用 L1 归一化距离作为评价指标。Text-to-image 用 200 样本评估，image-to-image 用纵向数据集评估。

## 实验关键数据

### 主实验

| 微调方法 | LD | FID (脑MR)↓ | KID (脑MR)↓ | AUC (脑MR)↑ | FID (胸片)↓ | AUC (胸片)↑ |
|---------|-----|-----------|-----------|-----------|-----------|-----------|
| SD + Basic FT | ✗ | 92.13 | 0.071 | 0.704 | 112 | 0.672 |
| SD + Basic FT | ✓ | **49.68** | **0.035** | **0.724** | **84** | **0.746** |
| Textual Inversion | ✗ | 120.63 | 0.098 | 0.600 | 171.77 | 0.600 |
| Textual Inversion | ✓ | 67.56 | 0.065 | 0.670 | 133.18 | 0.640 |
| DreamBooth | ✗ | 130.92 | 0.125 | 0.500 | 188 | 0.567 |
| DreamBooth | ✓ | 92.37 | 0.099 | 0.512 | 177 | 0.582 |
| Real + Synthetic | ✓ | - | - | **0.883** | - | **0.892** |

LD 在所有微调方法上都带来显著改进，Basic FT + LD 在脑 MRI 上 FID 从 92.13 降至 49.68（降 46%），且合成+真实数据训练的分类器 AUC 甚至超过纯真实数据（0.883 vs 0.870）。

### 消融实验

| 配置 | FID (aging)↓ | SSIM↑ | LPIPS↓ | PSNR↑ |
|------|-------------|-------|--------|-------|
| InstructPix2Pix (Binned) + SD + Basic FT + LD | 15.39 | 0.74 | 0.13 | 32.77 |
| InstructPix2Pix (Word) + SD + Basic FT + LD | **15.25** | 0.75 | 0.13 | 32.78 |
| InstructPix2Pix (Numerical) + SD + Basic FT + LD | 15.37 | **0.76** | **0.12** | **32.83** |
| InstructPix2Pix + SD + CD + LD (Numerical) | 24.05 | 0.32 | 0.23 | 30.70 |

Prompt 格式对照实验表明简单的 Diverse + Patient Info 组合最佳（FID 51.35, KID 0.0351），数值型年龄条件在 image-to-image 任务中综合最优。

### 关键发现
- **LD 在所有微调方案中一致有效**：无论是只调文本嵌入（Textual Inversion）还是调整 U-Net（Basic FT），LD 都能大幅降低 FID/KID。效果最好的是 Basic FT + LD。
- **合成数据增强超越真实数据**：用 50% LD 合成 + 50% 真实数据训练分类器，AUC 超过 100% 真实数据（脑 MRI: 0.883 vs 0.870），验证了合成数据的实用价值。
- **视觉改善明显**：加 LD 后脑 MRI 背景从灰色杂质变为纯黑，脑部结构更逼真，白质灰质边界更清晰。
- Prompt 中包含患者信息（年龄、性别、诊断）显著优于通用 prompt。
- δ 的最优值在 0.05-0.1 范围内，对不同微调方法较为稳定。

## 亮点与洞察
- **极简但有效**：仅一个标量参数 δ 就能弥合自然图像和医学图像的分布差距，实现成本几乎为零。这个发现揭示了扩散模型中隐空间分布是一个被忽视的关键自由度。
- **方法无关性**：作为插件可以嵌入任何微调方案，且都有效——这种正交于模型架构的改进方式非常优雅，可以直接应用于未来新出现的微调方法。
- **反事实生成的统一框架**：将疾病添加/移除、年龄变化、性别转换等多种医学场景统一到一个反事实优化框架下，从 min-max 的角度理解条件生成，对该领域有理论贡献。
- **合成数据增强的证据**：成功证明了 LD 生成的合成数据可以作为数据增强手段提升下游分类性能，为数据稀缺的医学 AI 提供了可行路径。

## 局限与展望
- **δ 的确定方式**：目前通过网格搜索确定 δ，对新的目标域需要重新搜索。可以考虑自动化地根据源域和目标域的分布差异估计 δ。
- **全局标量的局限**：δ 是全局的、各通道相同的偏移，对于不同空间区域或通道可能需要不同的偏移量。可以探索空间自适应或通道自适应的 LD。
- **2D 切片处理**：实验仅在 2D 脑 MRI 切片上进行，未处理 3D 体积数据。扩展到 3D 扩散模型需要验证 LD 在更高维空间的有效性。
- **反事实评估困难**：缺乏真正的反事实 ground-truth（如"这个人如果得了阿尔茨海默病，MRI 应该长什么样"），评估主要依赖 FID/KID 等分布指标和下游分类 AUC。
- 可以尝试将 LD 与 ControlNet 等条件控制方法结合，实现更精细的医学图像编辑。

## 相关工作与启发
- **vs DreamBooth**: DreamBooth 通过类先验保留损失微调 U-Net引入新概念，但在脑 MRI 上 FID 高达 130.92；加上 LD 后降至 92.37，说明仅微调参数不够，还需要调整隐空间分布。
- **vs Textual Inversion**: TI 仅调文本嵌入是最轻量的方案，但对医学图像理解不足；LD 将其 FID 从 120.63 降至 67.56，且 AUC 从 0.600 提升到 0.670。
- **vs 从头训练的医学扩散模型（如 Khader et al., Pinaya et al.）**: 这些方法需要大量医学数据训练，LD 利用预训练模型的先验知识仅需少量样本微调。
- LD 的思路可以推广到其他存在分布偏移的领域适配场景，如遥感图像、工业检测等。

## 评分
- 新颖性: ⭐⭐⭐⭐ 隐空间偏移的思路简洁独到，将隐变量视为可调条件是新视角
- 实验充分度: ⭐⭐⭐⭐ 覆盖多种微调方法、多个医学数据集、多种生成任务
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，可视化丰富
- 价值: ⭐⭐⭐⭐ 简单有效的插件方法对医学图像合成有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Noise-Consistent Siamese-Diffusion for Medical Image Synthesis and Segmentation](noise-consistent_siamese-diffusion_for_medical_image_synthesis_and_segmentation.md)
- [\[CVPR 2025\] SeaLion: Semantic Part-Aware Latent Point Diffusion Models for 3D Generation](sealion_semantic_part-aware_latent_point_diffusion_models_for_3d_generation.md)
- [\[NeurIPS 2025\] Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology](../../NeurIPS2025/medical_imaging/semantic_and_visual_crop-guided_diffusion_models_for_heterogeneous_tissue_synthe.md)
- [\[CVPR 2025\] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation?](are_general-purpose_vision_models_all_we_need_for_2d_medical_image_segmentation_.md)
- [\[CVPR 2025\] Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)

</div>

<!-- RELATED:END -->
