---
title: >-
  [论文解读] Hierarchical Image Tokenization for Multi-Scale Image Super Resolution
description: >-
  [ICML 2026][图像恢复][VAR] H-VAR 把"残差量化做多尺度生成"的 VAR 范式重新切片成层次化的图像 tokenization (HIT)，让一个 310M 的小模型只跑一次前向就能输出 128 / 256 / 512 三个有意义的中间分辨率…
tags:
  - "ICML 2026"
  - "图像恢复"
  - "VAR"
  - "残差量化"
  - "多尺度超分"
  - "层次化分词"
  - "DPO 正则"
---

# Hierarchical Image Tokenization for Multi-Scale Image Super Resolution

**会议**: ICML 2026  
**arXiv**: [2605.14891](https://arxiv.org/abs/2605.14891)  
**代码**: 无  
**领域**: 模型压缩 / 图像超分 / 视觉自回归  
**关键词**: VAR, 残差量化, 多尺度超分, 层次化分词, DPO 正则

## 一句话总结
H-VAR 把"残差量化做多尺度生成"的 VAR 范式重新切片成层次化的图像 tokenization (HIT)，让一个 310M 的小模型只跑一次前向就能输出 128 / 256 / 512 三个有意义的中间分辨率，再配一个不需要外部奖励模型的 DPO 正则项推动输出偏向 HR，在标准 ISR 数据上对打 1B 参数的 VARSR。

## 研究背景与动机

**领域现状**：图像超分的强 baseline 长期被 GAN（Real-ESRGAN）和扩散模型（StableSR、SeeSR、ResShift）占据；近期 next-scale 预测的 VAR 因为天然按尺度残差展开，被 VARSR、PURE、VARestorer 拿来做 ISR——pretraining 与 downstream 的对齐度比扩散更好。

**现有痛点**：现有 AR-based 超分两大短板。其一，原版 RQ-VAE 把图像分成 $L$ 个不断加细的残差，但前几级残差里并没有"低分辨率语义"，只是高频细节的随机分配，所以中间阶段不能解码成有意义的低分图；要做 $\times 4$ 就只能一次跑完整条 token 序列，无法顺带产 $\times 2$。其二，为了追上 SOTA，VARSR 必须用 1B 大模型 + classifier-free guidance + 海量带标注数据，PURE 直接套 7B Lumina-mGPT。

**核心矛盾**：VAR 的 token 序列是"通用残差堆"——压缩效率最高，但缺少"尺度语义"这条强约束；想要多尺度有意义，就必须把"按尺度可解"硬塞进 tokenization，但这又会让单尺度的重建变差，存在一对显式 trade-off。

**本文目标**：(a) 设计一种 tokenization，使前 $k$ 个 token 能确定地解码出该尺度的有效图像，且尺度间共享 token；(b) 在不堆数据、不加 VLM 的前提下，把"VAR 输出 HR 而不是 LR"的偏好硬编码进训练目标。

**切入角度**：作者观察到，next-scale prediction 之所以能压缩冗余，是因为下一尺度的预测要依赖上一尺度的全部 token；如果把"下采样—量化—升采样"做成在每个目标尺度上独立闭环并强制 token 复用，就能既保多尺度可解性，又保留 VAR 的序列预测格式。

**核心 idea**：用 HIT（Hierarchical Image Tokenization）把 RQ-VAE 的残差按目标尺度切片复用 token，加上一个用 $p(z_{HR})/p(z_{LR})$ 比值的 DPO 正则项，做一个 310M 的多尺度 H-VAR。

## 方法详解

### 整体框架
端到端两件事：(1) Hierarchical RQ-VAE：在 Switti 预训练 RQ-VAE 基础上 finetune vocabulary + decoder，让 token 序列被切成 $N$ 个嵌套片段 $\{s_1, s_2, \dots, s_N\}$，每个 $s_i$ 都能独立解码到该尺度图像；(2) Hierarchical VAR：一个 16 层 GPT-2 风格 transformer（310M），以 RQ-VAE encoder 编码的 LR 特征为 condition，用 cross-entropy + DPO 联合训练，按 next-scale 预测整条 token 序列。推理时一次前向给出 $\times 1 / \times 2 / \times 4$ 三个分辨率，重用 KV-cache。

### 关键设计

1. **Hierarchical Image Tokenization (HIT)**:

    - 功能：把残差量化按目标尺度分段，使前 $k$ 个 token 真正对应"$\times k$ 上采样后的有效图像"。
    - 核心思路：定义目标尺度 $s_1 < s_2 < \dots < s_N$（论文取 $(0.25, 0.5, 1)$ 对应 $\times 1/\times 2/\times 4$）。对每个尺度 $n$，先把输入图下采样到 $s_n \rho_L$ 编码出 $\mathbf{Z}_n$；然后在该尺度上量化残差，量化到的 token 既会被记到 $s_n$ 子序列，也会作为下一尺度的"起点 token"被复用；之后切换到尺度 $s_{n+1}$，把前一尺度的 token 上采到当前残差空间扣掉，再继续量化新增的残差。最终一张图被切成嵌套结构 $z = \{\{\{z_1,\dots\}_{s_1},\dots\}_{s_2}, \dots\}_{s_N}$。同时 finetune RQ-VAE 的 vocabulary 和 decoder：保持 decoder 冻结，用 encoder 特征与 token 嵌入的 $\ell_2$ 距离梯度去更新 vocabulary。
    - 设计动机：原版 RQ-VAE 的前几级残差没有任何"低分辨率对应"的约束，这是 VAR 无法产中间尺度的根因；HIT 把"前 $k$ 个 token 必须能重建尺度 $k$"做成训练时的硬约束，相当于在表示空间里注入一条很强的归纳偏置。作者发现这条偏置非常贵——它顺带把 transformer 从 1B 砍到 310M 还能维持 SOTA，因为 token 序列本身的"路径搜索空间"被大幅压缩了。

2. **DPO 正则项推动 HR 偏好**:

    - 功能：阻止 VAR 偷懒去预测和 LR 高度重叠的 token，强制它输出 HR 序列。
    - 核心思路：观察到 HR 与 LR 在低尺度的 token 严重重合，模型容易直接复读 LR；作者把上采到 512 的 LR 也跑一遍 HIT 拿到 $z_{LR}$，然后定义 $\mathcal{L}_{DPO} = -\log\sigma\left(\beta \log \frac{p(z_{HR})}{p(z_{LR})}\right)$，与标准 cross-entropy 等权相加。$\beta = 0.2$；过小损失项几乎恒定无效，过大会训练不稳。这里既不需要"参考策略"也不需要"外部奖励模型"，因为 LR 自带"负样本"角色。
    - 设计动机：传统 DPO 必须有 pair-wise preference + reference policy，在生成式 ISR 里通常要训外部 reward 模型；这里作者发现 ISR 的 LR/HR 天然就是一对 preference pair，AR 模型又恰好能算出两个序列各自的 log-likelihood（diffusion 没这能力），所以把 DPO 退化成一个"无监督正则"——成本几乎为零却显著锐化结果。

3. **多尺度的位置编码与条件注入**:

    - 功能：让单个 transformer 处理 $\sum_l \rho_l^2 = 3452$ 个不同尺度位置的 token，并以 LR 为条件。
    - 核心思路：用一份"过参数化的可学习位置嵌入"——按最大尺度声明一张大表，然后对每个目标分辨率 $\rho_l$ 下采样到对应尺寸去用；与 VARSR 用 ControlNet 编码 LR 不同，作者直接把 LR 双线性上采到 512 再过 RQ-VAE encoder 拿到 1024 个 conditioning token，节省了一个独立分支。
    - 设计动机：多尺度训练里位置嵌入是最容易出 BUG 的地方，统一一张可下采样的大表既避免维护多套权重，又能让模型在不同尺度间共享位置归纳偏置；用 encoder 特征当 condition 也消除了 ControlNet 与主干尺度不匹配的麻烦。

### 损失函数 / 训练策略
- RQ-VAE 微调：$\mathcal{L}_{RQVAE} = \ell_2 + 5\, \mathcal{L}_{LPIPS}$，AdamW、batch 384、lr 0.00025、25K 步、24 张 A100、约 24 小时；按 HART 方式以 50% 概率丢掉量化直接通 decoder，让 vocabulary 不过拟合。
- H-VAR 训练：cross-entropy + $\mathcal{L}_{DPO}$ 等权重；从 VAR d-16 官方 checkpoint 初始化、24 张 A100、200 epochs、batch 384、lr 1e-3、AdamW betas $(0.9, 0.95)$，约 13 小时完成。
- 训练数据完全标准：DIV2K + DIV8K + Flickr2K + OST + 10K FFHQ，用 Real-ESRGAN degradation 合成 LR-HR，不依赖任何专有数据集。

## 实验关键数据

### 主实验

| 数据集 | 指标 | StableSR | ResShift | VARSR (1B) | VARSR-d16 | H-VAR (310M, ours) |
|---|---|---|---|---|---|---|
| DIV2K-Val | LPIPS ↓ | 0.323 | 0.428 | 0.326 | 0.495 | **0.317** |
| DIV2K-Val | FID ↓ | 28.32 | 30.79 | 35.51 | 45.96 | **28.86** |
| RealSR | LPIPS ↓ | 0.300 | 0.346 | 0.350 | 0.413 | **0.256** |
| DRealSR | LPIPS ↓ | 0.333 | 0.401 | 0.354 | 0.409 | **0.259** |
| DRealSR | FID ↓ | 148.2 | 159.8 | 155.9 | 244.7 | **145.1** |

| 模型 | 参数量 | FLOPs | 推理时间 | DIV2K-Val FID (LPIPS) |
|---|---|---|---|---|
| H-VAR (Ours) | 310M | 0.921T | 0.25s | 28.86 (0.317) |
| VARSR | 1B | 3.071T | 0.93s | 35.51 (0.326) |
| ResShift | 173M | 2.651T | 0.17s | 30.79 (0.428) |
| StableSR | 919M | 79.94T | 5.51s | 28.32 (0.323) |

### 消融实验

| 数据集 | 配置 | PSNR@128 | PSNR@256 | PSNR@512 | LPIPS@512 |
|---|---|---|---|---|---|
| RealSR | w/o DPO | 20.56 | 23.09 | 25.72 | 0.310 |
| RealSR | w/ DPO | **22.09** | **24.41** | 25.55 | **0.256** |
| DRealSR | w/o DPO | 23.03 | 26.38 | 28.61 | 0.335 |
| DRealSR | w/ DPO | **25.26** | **27.65** | **28.73** | **0.259** |

| 配置 (RealSR LPIPS@512) | 128 | 256 | 512 |
|---|---|---|---|
| VARSR (1B) | 0.618 | 0.450 | 0.350 |
| Baseline (RQ-VAE 但无 HIT) | 0.686 | 0.491 | 0.311 |
| H-VAR (HIT) | **0.199** | **0.236** | **0.256** |

### 关键发现
- 在中间尺度 128 / 256，没有 HIT 的 baseline 几乎不可用（LPIPS > 0.4），HIT 直接把分数砍到 0.2 段，验证它是真正在中间尺度产可读图，不是噱头。
- HIT 当 inductive bias 极强：把 transformer 从 1B 砍到 310M、把训练数据从 VARSR 的专有集换成标准公开集，最终 FID/LPIPS 仍能并列或超过 VARSR；说明很多看似要靠"堆数据/堆参数"解决的问题，本质是 token 表示没对齐。
- DPO 正则在所有数据集和所有尺度上几乎都涨点，且不需要外部 reward 模型，是一个"成本几乎为零"的免费午餐。
- 副作用：因为前几级残差被强制分给低分辨率，最终 512 分辨率重建会有轻微退化；$L=10 \to 11$ 能补回来但推理成本飙升 24%，作者老实承认这是 trade-off。

## 亮点与洞察
- "把多尺度可解性写进 tokenization"是这篇最值得记住的一招——它不是改 transformer 架构、不是加 loss，而是在更上游的 vocabulary 上做约束；上游一旦改对，下游模型可以小一个数量级。
- 用 LR 自身当 DPO 的负样本是非常聪明的"自监督 preference learning"，省掉了 reward model；这个 trick 可以直接迁到任何"有自然劣化对"的生成任务（去模糊、去噪、风格弱化）。
- 论文老实揭露 trade-off：HIT 在最高分辨率上会折损一点重建质量，必须靠加更多 token 步数补回来——这种"利弊都摊在桌面上"的写作非常加分。
- 单次前向给三个分辨率，对实际产品（手机端、缩略图预览）非常友好，是一个真正能落地的工程优势而不仅是 paper 指标。

## 局限与展望
- 多尺度被硬切成 3 段离散尺度，想要任意倍率上采（$\times 1.5, \times 3$）还需要重新设计 $\rho_l$ 分配；这是 tokenization 范式天生的离散性。
- DPO 用 LR 当负样本默认 LR 是 "差答案"，但当输入本身就是 close-to-HR 的轻度退化时，这条偏好可能反而把模型推过头去产生 over-sharpening。
- 实验全部在 $\times 4$ 标准设置下，未在 $\times 8 / \times 16$ 上验证 HIT 是否仍保持效率优势；高倍率下中间尺度更多，token 序列展开后是否仍能压在小模型里，需要进一步检验。
- 与扩散类强 baseline（如 PASD、SUPIR）的对比未覆盖，主要对手仍是同门 VARSR；如果要把 SOTA 帽子戴得更稳，建议补这些对比。

## 相关工作与启发
- **vs VARSR**：同样把 VAR 用到 ISR，但 VARSR 用原版 RQ-VAE，中间尺度无意义、必须用 1B 模型和大量私有数据；H-VAR 用 HIT 把这两个短板一次性解决，且不需要 ControlNet 这种额外分支。
- **vs PURE**：PURE 用 7B Lumina-mGPT，把图像和退化描述都塞进 vocabulary；H-VAR 走相反方向——靠"上游 token 设计 + 简单 DPO"，证明 ISR 不一定要堆多模态大模型。
- **vs diffusion-based ISR（StableSR / ResShift）**：DM 推理慢、不能写出序列 likelihood，所以做不了原生 DPO；H-VAR 的 AR 形式带来这两个好处，也是论文选 VAR 而不是 DM 的关键理由。
- 启发："把任务结构当 inductive bias 注入 tokenization"是一个被低估的方向，下一个值得做的是 video AR（按时间尺度切 token）、医学影像 AR（按解剖层次切 token）。

## 评分
- 新颖性: ⭐⭐⭐⭐ HIT 是 VAR ISR 里首个支持多尺度的方案，DPO 用 LR 当负样本也是新做法；但底层范式仍是 RQ-VAE+VAR。
- 实验充分度: ⭐⭐⭐⭐ 三类 baseline、多个数据集、$L/\rho_l$ 敏感性、复杂度全有；缺与扩散 SOTA（PASD/SUPIR）对比。
- 写作质量: ⭐⭐⭐⭐⭐ 算法伪代码、图示、消融、限制讨论都做得很干净。
- 价值: ⭐⭐⭐⭐ 用 310M 打平 1B，且一次前向出 3 个分辨率，对工业部署有直接价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] UniBlendNet: Unified Global, Multi-Scale, and Region-Adaptive Modeling for Ambient Lighting Normalization](../../CVPR2026/image_restoration/uniblendnet_unified_global_multi_scale_and_region_adaptive_modeling_for_ambient_lighting_normalization.md)
- [\[CVPR 2026\] Disentangled Textual Priors for Diffusion-based Image Super-Resolution](../../CVPR2026/image_restoration/disentangled_textual_priors_for_diffusion-based_image_super-resolution.md)
- [\[CVPR 2026\] Bridging the Perception Gap in Image Super-Resolution Evaluation](../../CVPR2026/image_restoration/bridging_the_perception_gap_in_image_super-resolution_evaluation.md)
- [\[CVPR 2026\] SAT: Selective Aggregation Transformer for Image Super-Resolution](../../CVPR2026/image_restoration/sat_selective_aggregation_transformer_for_image_super_resolution.md)
- [\[ICML 2026\] Image Restoration via Diffusion Models with Dynamic Resolution](image_restoration_via_diffusion_models_with_dynamic_resolution.md)

</div>

<!-- RELATED:END -->
