---
title: >-
  [论文解读] MoE-DiffIR: Task-customized Diffusion Priors for Universal Compressed Image Restoration
description: >-
  [ECCV 2024][图像恢复][压缩图像复原] 提出 MoE-DiffIR，首个基于扩散模型的通用压缩图像复原（CIR）框架，通过混合专家（MoE）Prompt 模块从 Stable Diffusion 中挖掘任务定制化的扩散先验，结合 Visual-to-Text 适配器激活 SD 的跨模态生成先验，并构建了覆盖 7 种编解码器 × 3 个压缩级别共 21 种退化的首个通用 CIR 基准数据集。
tags:
  - ECCV 2024
  - 图像恢复
  - 压缩图像复原
  - 混合专家
  - 提示学习
  - 扩散模型
  - 通用图像修复
---

# MoE-DiffIR: Task-customized Diffusion Priors for Universal Compressed Image Restoration

**会议**: ECCV 2024  
**arXiv**: [2407.10833](https://arxiv.org/abs/2407.10833)  
**代码**: [项目页面](https://renyulin-f.github.io/MoE-DiffIR.github.io/)  
**领域**: 图像修复 / 压缩图像复原  
**关键词**: 压缩图像复原, 混合专家, Prompt学习, Stable Diffusion, 通用图像修复

## 一句话总结

提出 MoE-DiffIR，首个基于扩散模型的通用压缩图像复原（CIR）框架，通过混合专家（MoE）Prompt 模块从 Stable Diffusion 中挖掘任务定制化的扩散先验，结合 Visual-to-Text 适配器激活 SD 的跨模态生成先验，并构建了覆盖 7 种编解码器 × 3 个压缩级别共 21 种退化的首个通用 CIR 基准数据集。

## 研究背景与动机

**压缩图像复原的多样性挑战**：现实中存在大量不同的图像编解码器（JPEG、WebP、VVC、HEVC、学习型编解码器等），每种在低比特率下产生截然不同的压缩伪影——JPEG 倾向于块效应（blocking artifacts），学习型编解码器如 $C_{PSNR}$ 倾向于模糊（blur），WebP 产生色偏等。现有 CIR 方法大多针对单一编解码器（通常是 JPEG）设计，缺乏通用性。

**低比特率下的纹理生成能力不足**：传统 CNN/Transformer 基础的 CIR 方法（如 QGAC、FBCNN、HAT）优化 PSNR/SSIM 等失真指标，但在极低比特率下生成的图像过于平滑，缺乏纹理细节和感知质量。

**扩散模型先验的利用不充分**：现有 SD-based 修复方法（StableSR、DiffBIR）通过 ControlNet 或特征适配器复用生成先验，但使用**共享的调制参数**处理所有退化类型，无法为不同压缩任务提供定制化的调制方案。此外，大多数方法将 SD 的文本条件设为空字符串，**浪费了丰富的 text-to-image 跨模态先验**。

**现有 Prompt 学习方法的局限**：单 Prompt（如 ProRes）难以建模多任务的复杂关系；多 Prompt 加权融合（如 PromptIR、DACLIP）容易产生"均值特征"问题——各 Prompt 学习到相似的特征，缺乏多样性，降低了对不同退化类型的调制能力。

**MoE 启发**：MoE 框架通过路由机制为不同输入选择和激活不同的专家子集，非常适合多任务场景。将此思想引入 Prompt 学习，可以让每个 Prompt 作为专家感知不同退化类型，通过路由器为每个压缩任务动态选择最优的 Prompt 组合。

**跨模态先验的潜力**：SD 在大规模 text-image 数据上训练，存储了丰富的文本到图像的生成先验。如果能将低质量图像的视觉信息转化为文本嵌入作为 SD 的条件输入，可以激活这些跨模态先验，生成更一致、更合理的纹理。

## 方法详解

### 整体框架

MoE-DiffIR 基于 Stable Diffusion 2.1-base 构建，采用两阶段微调策略：

- **第一阶段**：冻结 VAE 和 UNet，仅训练 MoE-Prompt 模块。低质量（LQ）图像特征通过 MoE-Prompt Module 提取多尺度特征，经 SPADE 层调制到 SD UNet 的多尺度输出上，使用标准扩散损失 $\mathcal{L}_{SD}$ 训练，总共 0.4M 迭代步
- **第二阶段**：冻结所有模块，仅微调 VAE Decoder Compensator。使用第一阶段权重生成 70,000 张 latent 图像，以 LPIPS 感知损失训练补偿器校正结构保真度，0.1M 迭代步

输入图像统一缩放到 $256 \times 256$，batch size=32，4 块 RTX 3090 GPU 训练。

### 关键设计

#### 1. MoE-Prompt 模块（混合专家 Prompt 生成器）

区别于已有 Prompt 方案，MoE-Prompt 将每个 Prompt 视为一个退化专家，通过路由器动态选择 Top-K 个 Prompt 组合以挖掘任务定制化的扩散先验：

- **退化先验提取**：利用预训练的 DACLIP 编码器从 LQ 图像提取退化先验（Degradation Prior, DP），DP 通过交叉注意力机制与输入特征交互后送入路由器
- **Noisy Top-K 路由**：路由器使用带噪声的 Top-K 函数自适应选择 Prompt 组合：

$$G(x) = \text{Top-K}\big(\text{Softmax}(xW_g + \mathcal{N}(0,1) \cdot \text{Softplus}(xW_{\text{noise}}))\big)$$

其中 $W_g$ 是全局特征权重矩阵，$W_{\text{noise}}$ 引入随机性以鼓励 Prompt 选择的鲁棒性和多样性。选出 $K$ 个 Prompt 后通过矩阵乘法与输入特征交互。

- **与已有方法的对比**：单 Prompt 难以建模多任务；多 Prompt 软权重融合导致"均值特征"（各 Prompt 学到相似特征）；MoE-Prompt 通过稀疏路由让每个 Prompt 专注于感知不同退化，实现参数的高效复用。

#### 2. Visual-to-Text 适配器（V2T Adapter）

为激活 SD 丰富的 text-to-image 跨模态先验（通常被现有方法忽略），设计 V2T 适配器将视觉信息转入文本域：

- **质量增强前处理**：LQ 图像先经过若干 Transformer blocks 作为质量增强器，避免严重压缩伪影损害后续视觉特征的质量
- **视觉特征提取**：增强后的图像通过 CLIP 图像编码器提取视觉嵌入
- **域转换**：若干 MLP 层（即 V2T Adapter）将 CLIP 视觉嵌入转化为 SD 文本空间中的嵌入，作为 SD 的文本条件指导生成
- **与 PASD 的区别**：PASD 直接用 BLIP 从 LQ 图像提取文本特征，但在极低比特率下图像严重受损，提取的文本特征质量差；MoE-DiffIR 先增强再编码，再做域转换

#### 3. Decoder Compensator（解码器补偿器）

SD 的预训练 VAE 解码器与 CIR 任务不完全对齐（高压缩率导致重建阶段信息丢失），因此在第二阶段引入低质量信息对解码器进行补偿微调：

$$L_{\text{Decoder}} = \mathcal{L}_{\text{lpips}}[z_{lq}, z_0, hr]$$

其中 $z_0$ 为 UNet 去噪输出，$z_{lq}$ 为低质量图像的 latent 变量，$hr$ 为高质量参考图像。LPIPS 感知损失确保结构保真度。

### 损失函数 / 训练策略

- **第一阶段损失**：标准扩散去噪损失 $\mathcal{L}_{SD} = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,1)}[\|\epsilon - \epsilon(z_t, t)\|_2^2]$
- **第二阶段损失**：LPIPS 感知损失，保证结构保真度
- **训练数据**：DF2K 数据集（3450 张图像 × 21 种压缩任务 = 72,450 张训练图像）
- **优化器**：Adam（$\beta_1=0.9$, $\beta_2=0.999$）
- **学习率**：第一阶段 $5 \times 10^{-5}$（固定），第二阶段 $1 \times 10^{-4}$
- **数据增强**：随机翻转和旋转
- **最终超参数**：$N=7$ 个基本 Prompt，Top-K 中 $K=3$

## 实验关键数据

### 主实验

在 7 种编解码器上的平均性能对比（每种编解码器平均 3 个压缩级别，在 LIVE1 数据集上测试）：

| 方法 | 类型 | JPEG LPIPS↓/FID↓ | VVC LPIPS↓/FID↓ | HEVC LPIPS↓/FID↓ | WebP LPIPS↓/FID↓ | $C_{PSNR}$ LPIPS↓/FID↓ | $C_{SSIM}$ LPIPS↓/FID↓ | HIFIC LPIPS↓/FID↓ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| PromptIR | All-in-one | 0.213/111.2 | 0.305/168.4 | 0.314/172.6 | 0.250/149.4 | 0.234/141.4 | 0.328/160.1 | 0.121/82.8 |
| DiffBIR | Diffusion | 0.100/67.2 | 0.169/95.5 | 0.179/104.1 | 0.147/91.4 | 0.106/85.4 | 0.136/82.5 | 0.060/45.1 |
| StableSR | Diffusion | 0.107/68.2 | 0.168/98.5 | 0.185/106.0 | 0.121/75.1 | 0.108/73.9 | 0.154/90.7 | 0.071/46.7 |
| SUPIR | Diffusion | 0.125/71.0 | 0.147/99.7 | 0.161/108.7 | 0.142/95.6 | 0.107/90.1 | 0.136/87.0 | 0.074/48.4 |
| **MoE-DiffIR** | **MoE+Diff** | **0.096/62.7** | **0.144/88.8** | **0.162/98.7** | **0.110/70.6** | **0.100/72.2** | **0.135/80.2** | **0.059/43.6** |

MoE-DiffIR 在所有 7 种编解码器上的感知指标（LPIPS、FID）均达到最优。相比 SUPIR，LPIPS 平均降低 10.9%，FID 平均改善 5.4；相比 StableSR，PSNR 平均提升 0.41dB。在 HIFIC 上的优势尤其明显（LPIPS=0.059，FID=43.6）。

### 消融实验

| 消融条件 | LIVE1 PSNR/SSIM | LIVE1 LPIPS/FID | BSDS500 LPIPS/FID | DIV2K LPIPS/FID |
|:---|:---:|:---:|:---:|:---:|
| No Prompt | 28.73/0.806 | 0.134/85.9 | 0.159/96.5 | 0.130/79.5 |
| Single Prompt | 28.86/0.806 | 0.127/79.5 | 0.153/89.6 | 0.114/71.3 |
| Multiple Prompt (软权重) | 28.98/0.810 | 0.121/77.1 | 0.148/89.3 | 0.112/71.7 |
| **MoE-Prompt (Ours)** | **29.02/0.811** | **0.118/75.9** | **0.143/88.1** | **0.107/68.9** |
| MoE-Prompt + V2T Adapter | 29.03/0.812 | 0.115/74.1 | 0.137/86.8 | — |
| MoE-Prompt + DP | 29.07/0.814 | 0.115/76.6 | 0.141/88.0 | — |
| **MoE-Prompt + V2T + DP** | **29.10/0.814** | **0.114/73.6** | **0.136/86.8** | — |

消融结论：（1）MoE-Prompt 较 Multiple Prompt 在 LPIPS 上改善 5%、FID 降低约 4 点；（2）V2T Adapter 主要提升感知质量（LPIPS 降 3-5%）；（3）Degradation Prior 主要提升保真度（PSNR +0.07dB）。

### 关键发现

1. **首个通用 CIR 基准**：覆盖 4 种传统编解码器（JPEG、VVC、HEVC、WebP）+ 3 种学习型编解码器（$C_{PSNR}$、$C_{SSIM}$、HIFIC），各 3 个压缩级别，共 21 种退化类型，为后续研究提供了标准评测基准。
2. **MoE 路由有效避免"均值特征"**：通过 Top-K 稀疏选择而非全 Prompt 软权重融合，让不同 Prompt 专注于不同退化模式，提升了参数复用效率和多样性。
3. **跨模态先验的价值**：V2T Adapter 将视觉信息转入文本域作为 SD 条件，在极低比特率下可有效抑制 SD 将严重压缩伪影误生成为噪点的问题。
4. **Prompt 数量和 K 值的影响**：$N=7$ 时性能最优且参数经济；$K=1$ 虽然 PSNR 更高但感知质量差，$K=3$ 在失真与感知之间取最佳平衡。
5. **对未见退化的泛化能力**：在 Cross-Degree（VVC 未见 QP [32,52]）和 Cross-Type（未训练的 AVC 编解码器）两类 unseen 测试中，MoE-DiffIR 均优于其他 Prompt 方案，表现出良好的泛化性。
6. **两阶段训练的必要性**：第二阶段的 Decoder Compensator 微调对结构保真度提升至关重要——预训练 VAE 解码器与 CIR latent 不完全对齐的问题需要专门的补偿机制。
7. **在失真与感知指标间取得平衡**：MoE-DiffIR 不仅在 LPIPS/FID 上全面领先，PSNR 也保持竞争力（平均比 StableSR 高 0.41dB），兼顾了保真度和感知质量。

## 亮点与洞察

- **MoE + Prompt 的创新结合**：将 MoE 的路由思想引入 Prompt 学习，让每个 Prompt "专家"通过数据驱动的方式自动分化，避免了传统多 Prompt 方案的"均值特征"退化问题，是一个具有普适意义的技术贡献
- **跨模态先验的挖掘策略**：现有 SD 修复方法普遍使用空文本作为条件输入，MoE-DiffIR 通过"质量增强→CLIP 视觉编码→MLP 域转换"的流水线将视觉信息桥接到文本域，充分利用了 SD 的 text-to-image 生成能力
- **首个通用 CIR 基准的构建**：21 种退化任务的基准数据集填补了该领域的评测空白，包含传统和学习型编解码器的多层级压缩，有很好的引领价值
- **极低比特率下的视觉改善**：在 JPEG QF=5、VVC QP=47 等极端条件下，MoE-DiffIR 能生成比 DiffBIR、SUPIR 更准确和一致的纹理，有效避免错误纹理生成

## 局限与展望

1. 在极低比特率下，复原图像与 ground truth 之间仍存在明显差距，生成纹理的准确性有待进一步提升
2. 基于 Stable Diffusion 的推理速度较慢（需多步去噪），不适合实时应用场景
3. 训练数据仅使用 DF2K，规模有限；更大规模、更多样化的压缩数据集训练可能进一步提升效果
4. DACLIP 提供的退化先验质量依赖于该预训练模型的覆盖范围，对极端或罕见退化类型的泛化性待验证
5. $256 \times 256$ 的训练分辨率在高分辨率图像修复中可能不够
6. 未探索与最新的 SDXL 或 SD 3.0 等更强扩散模型的结合

## 相关工作与启发

- **StableSR / DiffBIR (2023)**：开创了用 SD 生成先验做图像修复的范式，MoE-DiffIR 在此基础上引入任务自适应的调制机制和跨模态条件
- **PromptIR (NIPS 2023)**：首个将 Prompt 学习引入 all-in-one 图像修复的工作，采用多 Prompt 加权方式；MoE-DiffIR 通过 MoE 路由改进了 Prompt 交互方式
- **DACLIP (2023)**：在大规模退化数据上训练的 CLIP 变体，MoE-DiffIR 复用其编码器提取退化先验
- **MoE 框架 (Sparsely-Gated MoE)**：MoE 的稀疏路由和专家选择机制被创造性地应用于 Prompt 学习场景
- **PASD (2023)**：尝试用 BLIP 提取文本引导 SD 修复，但未做 LQ 图像预增强，在高压缩场景下效果受限

## 评分

| 维度 | 分数 (1-10) | 说明 |
|:---|:---:|:---|
| 创新性 | 8 | MoE+Prompt 的结合、V2T 适配器、首个通用 CIR 基准数据集，三项贡献各有新意 |
| 实验充分度 | 9 | 7 种编解码器 × 3 级压缩 × 5 个测试集 × 多种指标 × 对比 8 个 SOTA × 详细消融 |
| 实用性 | 7 | 感知质量提升显著，但 SD 推理速度偏慢限制了实际部署 |
| 写作质量 | 7 | 框架图清晰，实验表格详尽，但部分公式和符号的排版需要改进 |
| 总评 | 8 | 系统性很强的工作，问题定义清晰，技术方案完整，基准数据集有持续影响力 |

<!-- RELATED:START -->

## 相关论文

- [Exploiting Diffusion Prior for Task-driven Image Restoration](../../ICCV2025/image_restoration/exploiting_diffusion_prior_for_task-driven_image_restoration.md)
- [Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration](seeing_the_unseen_a_frequency_prompt_guided_transformer_for_image_restoration.md)
- [UniRes: Universal Image Restoration for Complex Degradations](../../ICCV2025/image_restoration/unires_universal_image_restoration_for_complex_degradations.md)
- [Complexity Experts are Task-Discriminative Learners for Any Image Restoration](../../CVPR2025/image_restoration/complexity_experts_are_task-discriminative_learners_for_any_image_restoration.md)
- [Disentangled Textual Priors for Diffusion-based Image Super-Resolution](../../CVPR2026/image_restoration/disentangled_textual_priors_for_diffusion-based_image_super-resolution.md)

<!-- RELATED:END -->
