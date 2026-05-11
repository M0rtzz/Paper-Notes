---
title: >-
  [论文解读] SPDMark: Selective Parameter Displacement for Robust Video Watermarking
description: >-
  [CVPR 2026][图像生成][视频水印] SPDMark 提出了一种基于选择性参数位移（SPD）的视频扩散模型内嵌水印框架，通过在解码器中学习低秩基 shift 字典并根据水印密钥选择组合，实现了逐帧水印嵌入、不可感知、高鲁棒性和低计算开销，同时支持时序篡改检测与定位。
tags:
  - CVPR 2026
  - 图像生成
  - 视频水印
  - 参数位移
  - LoRA
  - 扩散模型
  - 鲁棒性
---

# SPDMark: Selective Parameter Displacement for Robust Video Watermarking

**会议**: CVPR 2026  
**arXiv**: [2512.12090](https://arxiv.org/abs/2512.12090)  
**代码**: 有（论文中提及）  
**领域**: 扩散模型 / 视频水印  
**关键词**: 视频水印, 参数位移, LoRA, 扩散模型, 鲁棒性

## 一句话总结
SPDMark 提出了一种基于选择性参数位移（SPD）的视频扩散模型内嵌水印框架，通过在解码器中学习低秩基 shift 字典并根据水印密钥选择组合，实现了逐帧水印嵌入、不可感知、高鲁棒性和低计算开销，同时支持时序篡改检测与定位。

## 研究背景与动机

1. **领域现状**：高质量视频生成模型（如 Sora、SVD）的出现使得 AI 生成视频的溯源问题日益严峻。EU AI Act 和美国 AI 行政令均建议对 AI 生成内容添加水印。视频水印需同时满足不可感知性、鲁棒性和计算效率三个要求。

2. **现有痛点**：(a) 后处理方法（如 VideoSeal）增加延迟且无法利用生成先验；(b) 噪声空间方法（如 VideoShield）通过 DDIM inversion 解码，计算开销大且易受扰动影响；(c) 模型微调方法（如 LVMark）统一调制所有层限制了逐帧控制，VidSig 只嵌入单一固定签名无法检测时序篡改。三类方法在不可感知性、鲁棒性和效率之间存在此消彼长。

3. **核心矛盾**：如何在不牺牲视频质量的前提下，实现高效的多密钥逐帧水印嵌入，且能检测帧级时序篡改？

4. **本文目标** 设计一种 in-generation 视频水印方案，支持任意密钥、逐帧水印、时序篡改检测，且计算开销可忽略。

5. **切入角度**：不扰动像素或噪声，而是通过学习一组低秩基 shift 的字典，根据水印密钥选择性地位移生成模型的参数来嵌入水印。

6. **核心 idea**：学习一个固定的 LoRA 基 shift 字典，每个帧的水印密钥决定每层选择哪个基 shift，从而在解码器参数空间中嵌入逐帧水印，无需推理开销也无需逐密钥重训。

## 方法详解

### 整体框架
SPDMark 的 pipeline：(1) 给定视频级密钥 $K_{base}$，通过密码学哈希函数为每帧生成唯一水印消息 $\kappa_t$；(2) 每个 $\kappa_t$ 映射为二进制 mask $\mathbf{b}(\kappa_t)$，选择解码器每层的一个 LoRA 基 shift；(3) 用位移后的解码器生成水印视频 $\tilde{\mathbf{x}}$；(4) 逐帧提取水印后，用最大二部图匹配和假设检验验证水印有效性并定位时序篡改。

### 关键设计

1. **选择性参数位移框架（Selective Parameter Displacement）**:

    - 功能：将水印密钥编码为生成模型的参数位移
    - 核心思路：将生成模型参数分为不修改部分 $\Phi_U$ 和待修改部分 $\Phi_M$（仅解码器）。$\Phi_M$ 跨 $L$ 层，每层有 $P$ 个基 shift $\zeta_{\ell,p}$，位移为 $\Delta\phi_\ell = \sum_{p=1}^P b_{\ell,p} \zeta_{\ell,p}$。密钥到 mask 的映射：将 $M = L\log_2 P$ 位密钥分为 $L$ 个 chunk，每个 chunk 的十进制值决定选择该层哪个基 shift。实际上每层只选一个基 shift，位移 $\Delta\Phi_M(\kappa) = [\zeta_{1,i_1+1}, \ldots, \zeta_{L,i_L+1}]^T$。
    - 设计动机：全参数位移空间太大不可学习，通过分解为层级基 shift 的选择问题大幅降低搜索空间。用固定字典支持任意密钥而无需逐密钥重训。

2. **基于 LoRA 的参数高效实现**:

    - 功能：参数高效地实现基 shift
    - 核心思路：每个基 shift $\zeta_{\ell,p} = A_{\ell,p} B_{\ell,p}$，其中 $A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times d}, r \ll d$（论文中 $r=32$）。位移后的层输出为 $\mathbf{h}_\ell = \mathcal{F}_{\phi_\ell}(\mathbf{h}_{\ell-1}) + \alpha \mathcal{F}_{\Delta\phi_\ell}(\mathbf{h}_{\ell-1})$。具体应用于解码器的 $L=14$ 个空间 ResNet 块，每块 $P=4$ 个 LoRA，共 $\log_2 4 = 2$ bit/层，每帧 payload 为 28 bit。
    - 设计动机：直接学习全秩 shift 参数量太大，LoRA 低秩分解既保证表达力又极大降低参数，使方案可部署在大模型上。

3. **逐帧水印与时序篡改检测**:

    - 功能：嵌入帧级唯一水印消息，支持帧级篡改定位
    - 核心思路：用 HMAC-SHA256 从基密钥和帧号生成帧级消息 $\kappa_t = \text{Trunc}_M(\mathcal{H}(K_{base}, t))$。提取时用 ResNet-50 逐帧提取 28 维 logits。验证时构建参考消息 $\mathbf{K}$ 和提取消息 $\hat{\mathbf{K}}$ 的二部图，边权重为 Hamming 相似度 $\bar{S}_{m,n} = 1 - \psi(\kappa_m, \hat{\kappa}_n)/M$，用 Hungarian 算法做最大权重匹配。然后通过二项分布假设检验（帧级阈值 $\tau_f$ 和视频级阈值 $\tau_v$）判断水印有效性。未匹配帧就是被篡改的帧。
    - 设计动机：逐帧唯一消息使得帧级别的删除、交换、插入都能通过匹配失败被检测到，这是此前仅嵌入单一签名的方法做不到的。

### 损失函数 / 训练策略

总损失 $\min_{\zeta,\eta} \mathcal{L}_{imp}(\mathbf{x}, \tilde{\mathbf{x}}) + \mathcal{L}_{rec}(\mathcal{V}_\eta(\tilde{\mathbf{x}}), \kappa)$。消息恢复损失用 BCElogits；不可感知性损失 $\mathcal{L}_{imp} = \lambda_{ps} \mathbb{E}_t[\text{LPIPS}(x_t, \tilde{x}_t)] + \lambda_{tc} \mathbb{E}_t[\|\delta y_t - \delta \tilde{y}_t\|_1]$，其中 LPIPS 保证感知相似度，时序一致性损失（亮度差的 L1）防止闪烁。训练在 OpenVid-1M 的 10000 个视频上进行，对 $\kappa, \mathbf{c}, \mathbf{z}$ 取期望优化。提取器用 ResNet-50（ImageNet 预训练），推理时对测试视频的所有帧做 batch normalization 以稳定预测。

## 实验关键数据

### 主实验（视频质量 + 水印检测）

**SVD-XT 模型**:

| 方法 | Payload | Bit Acc↑ | SC↑ | BC↑ | MS↑ | IQ↑ |
|------|---------|----------|-----|-----|-----|-----|
| VideoShield | 512 | 0.979 | 0.954 | 0.954 | 0.956 | **0.695** |
| VideoSeal | 256 | **0.999** | 0.955 | 0.950 | 0.961 | 0.682 |
| VidSig | 48 | 0.958 | 0.951 | 0.953 | 0.956 | 0.693 |
| **SPDMark** | 28×25 | 0.995 | **0.966** | **0.958** | **0.975** | 0.690 |

### 鲁棒性实验（SVD-XT 平均 Bit Acc）

| 方法 | 光度攻击 | 时序攻击 | 后处理 | 平均 |
|------|---------|---------|--------|------|
| VideoShield | ~0.82 | ~0.94 | ~0.83 | 0.833 |
| VideoSeal | ~0.94 | ~1.00 | ~0.82 | 0.912 |
| VidSig | ~0.66 | ~0.96 | ~0.53 | 0.685 |
| **SPDMark** | ~0.94 | ~0.99 | ~0.89 | **0.935** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Full SPDMark | Avg Bit Acc 0.935 | 完整模型 |
| SPDMark 在 ModelScope 上 | Avg Bit Acc 高 | 跨架构（UNet→DiT）泛化 |
| 时序篡改定位 | 高 Precision/Recall/F1 | 帧删除/插入/交换均可检测 |

### 关键发现
- SPDMark 在视频质量指标（SC/BC/MS）上一致优于所有对比方法，说明参数位移方式对视觉质量影响最小
- 在鲁棒性方面平均 Bit Acc 达 0.935，超越 VideoSeal（0.912）和 VideoShield（0.833）
- 在 Screen Recording 攻击下 SPDMark 达 0.837 远超 VideoSeal 的 0.598，说明生成式水印比后处理水印更鲁棒
- 在 Crop&Drop 复合攻击下 SPDMark（0.856）显著优于其他方法（0.458-0.513）
- 逐帧水印使得时序篡改（帧删除、交换、插入）均可被检测和定位

## 亮点与洞察
- **参数空间水印是一个巧妙的范式转换**：不在像素或噪声空间操作，而是在模型参数空间嵌入水印，天然继承了模型的生成质量，开销极低
- **LoRA 基 shift 字典支持无限密钥**：一次训练字典后，任意新密钥只需选择不同组合，无需重训。这比 per-key fine-tuning 高效得多
- **密码学哈希生成帧级消息 + Hungarian 匹配验证**：将密码学工具与图匹配算法结合，优雅地解决了时序篡改检测问题，这个框架可以推广到其他需要序列完整性验证的场景

## 局限与展望
- 每帧仅 28 bit payload，容量有限（14 层 × 2 bit/层），增加位深需要更多 LoRA 基或更多层
- 仅在解码器上做水印，如果攻击者替换解码器则水印失效（但这在 API 控制场景下不太可能）
- 提取器使用 ResNet-50 相对简单，对极端攻击（如高压缩比 H.265）可能不够鲁棒
- 训练需要成对的 watermarked/non-watermarked 视频，数据成本较高

## 相关工作与启发
- **vs VideoShield**: 基于噪声空间+DDIM inversion，计算开销大且 Crop 攻击下 Bit Acc 仅 0.521。SPDMark 避免了 inversion
- **vs VideoSeal**: 后处理方法，Screen Recording 下严重退化（0.598）。SPDMark 利用生成先验更鲁棒
- **vs VidSig**: 冻结 PAS 层+时序对齐，但只嵌入固定签名无法检测时序篡改。SPDMark 的逐帧机制更灵活
- **vs AQuaLoRA**: 图像级 LoRA 水印，SPDMark 扩展到视频并加入时序一致性和篡改检测

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 参数位移框架和 LoRA 基 shift 字典的设计非常新颖，时序篡改检测机制优雅
- 实验充分度: ⭐⭐⭐⭐ 覆盖两种生成架构和多种攻击类型，但消融实验可以更详细
- 写作质量: ⭐⭐⭐⭐ 形式化推导清晰，但符号较多需要仔细阅读
- 价值: ⭐⭐⭐⭐⭐ 高度实用的视频水印方案，直接可部署到视频生成 API 服务中

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Towards Robust Content Watermarking Against Removal and Forgery Attacks](towards_robust_content_watermarking_against_removal_and_forgery_attacks.md)
- [\[CVPR 2026\] TRACE: Structure-Aware Character Encoding for Robust and Generalizable Document Watermarking](trace_structure-aware_character_encoding_for_robust_and_generalizable_document_w.md)
- [\[CVPR 2026\] Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking](editing_away_the_evidence_diffusion-based_image_manipulation_and_the_failure_mod.md)
- [\[CVPR 2026\] Rel-Zero: Harnessing Patch-Pair Invariance for Robust Zero-Watermarking Against AI Editing](rel-zero_harnessing_patch-pair_invariance_for_robust_zero-watermarking_against_a.md)
- [\[ECCV 2024\] Robust-Wide: Robust Watermarking against Instruction-driven Image Editing](../../ECCV2024/image_generation/robust-wide_robust_watermarking_against_instruction-driven_image_editing.md)

</div>

<!-- RELATED:END -->
