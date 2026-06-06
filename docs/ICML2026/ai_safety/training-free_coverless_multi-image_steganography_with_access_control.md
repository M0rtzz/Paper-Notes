---
title: >-
  [论文解读] Training-Free Coverless Multi-Image Steganography with Access Control
description: >-
  [ICML 2026][AI安全][无载体隐写] 提出 MIDAS，一种基于预训练扩散模型的 training-free 无载体多图隐写框架，用 Random Basis 正交随机基替代传统 Noise Flip 实现按私钥的细粒度访问控制，配合 Latent Vector Fusion 消除拼接边界…
tags:
  - "ICML 2026"
  - "AI安全"
  - "无载体隐写"
  - "多图隐写"
  - "访问控制"
  - "扩散模型"
  - "Random Basis"
---

# Training-Free Coverless Multi-Image Steganography with Access Control

**会议**: ICML 2026  
**arXiv**: [2603.09390](https://arxiv.org/abs/2603.09390)  
**代码**: https://github.com/Minyeol/MIDAS  
**领域**: AI 安全 / 信息隐藏 / 扩散模型  
**关键词**: 无载体隐写、多图隐写、访问控制、扩散模型、Random Basis

## 一句话总结
提出 MIDAS，一种基于预训练扩散模型的 training-free 无载体多图隐写框架，用 Random Basis 正交随机基替代传统 Noise Flip 实现按私钥的细粒度访问控制，配合 Latent Vector Fusion 消除拼接边界，在不传输任何与秘密相关的附加信息的前提下实现多图隐藏 + 抗隐写分析。

## 研究背景与动机

**领域现状**：图像隐写主流分两路。**Modification-based**（如 Baluja, HiNet, DeepMIH, IIS, AIS）把秘密图像直接编码到 cover 图像的像素/小波系数上，质量很高但 cover 一旦泄露就被隐写分析轻易识破；**Coverless Image Steganography (CIS)** 则用生成模型直接合成 stego 图像（不存在被改动的 cover），天然抗隐写分析，CRoSS / DiffStega / DStyleStego 是代表性的 training-free CIS 方案。

**现有痛点**：现有 training-free CIS 方法几乎都不支持 access control，把它们朴素扩展到多图场景会出现两类失效——(1) 单图设计被强行复制 N 份后**重建质量崩塌**；(2) 把 N 个秘密的 noisy latent 直接 concat，扩散逆过程**无法平滑跨边界**，stego 图像会出现一道明显的拼接缝（论文 Fig.1 显示 DiffStega* / CRoSS* 都有这个问题）。更严重的是，即便 DiffStega 用错的 K_priv 也能解出大致能看的图像，**完全没有 access control 安全性**。

**核心矛盾**：CIS 想"训练免费 + 多图 + 访问控制 + 不传辅助信息"四个能力同时具备。已有工作要么训练成本高（Chen 2025、Qin 2025 自训生成器），要么需要为每次通信传额外的与秘密相关的 side information（DStyleStego、HIS），要么访问控制保证不住——四个条件没有任何方法能全占。

**本文目标**：在公开扩散模型上构造一个真正的 training-free 多图访问控制 CIS：(a) N 个秘密图像融合到 1 张 stego，(b) 只有持有正确 K_priv_i 的用户能恢复第 i 张，错 key 必须解出"无意义"图像，(c) 不传输任何与秘密相关的附加信息，(d) 抗隐写分析。

**切入角度**：作者观察到，过往的私钥机制 Noise Flip 用的是 $M_d = \text{diag}(e), e\in\{-1,1\}^d$ 这种简单对角符号翻转矩阵——它的搜索空间太规整，且对 noisy latent 内部的结构残留信息抑制不充分。如果换成**用 seed 派生的随机正交基** $Q_d(\mathcal{K},\gamma)$，既保持可逆性（$Q^T Q = I$ 完美还原），又能从信息论角度证明信息泄漏率 $R_L$ 随强度 $\gamma\to 1$ 趋于零。

**核心 idea**：用 seed-driven 随机正交矩阵 (Random Basis) 加密 + 共享 reference latent 的 Latent Vector Fusion 替代朴素 concat，从而把"加密"和"消拼接缝"两件事统一在同一个数学结构里。

## 方法详解

### 整体框架
MIDAS 完全跑在预训练的 Stable Diffusion v1.5 latent 空间（$C\times H\times W$）上，分 Hiding 和 Reconstruction 两阶段。

**Hiding Stage**（发送端）：N 张秘密图像 $I_{sec}^i$ → 各自下采样后做 DDIM forward 拿到 noisy latent $\mathbf{z}_{sec}^i \in \mathbb{R}^{C\times H/N_1\times W/N_2}$（$N_1 N_2 = N$，$N=2$ 时 $N_1=2,N_2=1$）→ 用私钥 $\mathcal{K}_{priv}^i$ 经 Random Basis 加密成 $\mathbf{z}_{prot}^i$ → 拼接得 $\mathbf{z}_{prot}\in\mathbb{R}^{C\times H\times W}$ → Latent Vector Fusion 用公钥 $\mathcal{K}_{pub}$ 把它和参考 latent $\mathbf{z}_{ref}$ 混合成 $\mathbf{z}_{pub}$ → DDIM reverse + 公开 prompt $\mathcal{P}_{pub}$ 渲染出 stego 图像 $I_{stego}$ 发出去。其中 $\mathbf{z}_{ref}$ 由一个 Reference Generator (RefGen) 从 $(\mathcal{K}_{pub}, \mathcal{P}_{pub})$ 确定性生成，所以不需要单独传。

**Reconstruction Stage**（接收端）：收到 $\tilde{I}_{stego}$（可能被信道污染）→ DDIM inversion 得 $\tilde{\mathbf{z}}_{pub}$ → 用公钥反 Latent Vector Fusion 拿到 $\hat{\mathbf{z}}_{prot}$ → 用自己的私钥 $\mathcal{K}_{priv}^i$ 解 N 段（只有第 i 段会解出有意义的 latent，其他段是噪声）→ 整段 joint denoise 后再切分 → VAE decode 得 $\hat{I}_{sec}^i(i)$。

### 关键设计

1. **Random Basis 私钥加密机制**:

    - 功能：把每个秘密 latent 用 seed 派生的正交矩阵打散，既实现加密又抑制 noisy latent 中残留的结构信息。
    - 核心思路：对任意 d 维向量 $\mathbf{z}$，加密为 $\mathbf{z}_{enc} = M_d \mathbf{z}$，其中 $M_d = Q_d(\mathcal{K},\gamma)$ 是由种子 $\mathcal{K}$ 和强度 $\gamma$ 派生的正交矩阵（$\gamma$ 控制被该正交变换影响的元素比例，其余元素恒等）。由于正交性 $\mathbf{z} = M_d^T \mathbf{z}_{enc}$ 可完美还原。论文给出 Theorem 3.1：信息泄漏率满足 $R_L \approx O\left(\frac{-\log\Delta+\log m}{m} + (1-\gamma)(-\log\Delta+1)\right)$，在 $m\approx 10^6$（512×512×3 图像）、$\Delta\approx 10^{-7}$（float32）下第一项可忽略，第二项随 $\gamma\to 1$ 趋零，**即便 $\gamma=0.4$ 实测就已经足够安全**。
    - 设计动机：替换 DiffStega 用的 Noise Flip（$M_d = \text{diag}(e), e\in\{-1,1\}^d$，搜索空间只有 $2^d$ 且无法打乱空间结构）。Random Basis 用真正的旋转打散 latent 的空间相关性，从信息论上有可证的泄漏控制。

2. **Latent Vector Fusion 拼接融合**:

    - 功能：在拼接好的多图 latent 上再施一层**整体**的正交变换并和参考 latent 混合，从根本上消除"拼接缝"。
    - 核心思路：定义 $\mathbf{z}_{pub} = \sqrt{\alpha}\, M_D \mathbf{z}_{prot} + \sqrt{1-\alpha}\, \mathbf{z}_{ref}$，其中 $M_D = Q_D(\mathcal{K}_{pub}, \gamma_{fuse})$ 维度 $D = C\times H\times W$，$\mathbf{z}_{ref}$ 是 RefGen 用 $(\mathcal{K}_{pub}, \mathcal{P}_{pub})$ 确定性生成的参考图像对应的 noisy latent。$M_D$ 把多个子段的空间信息**全局打散**，破坏拼接边界；和 $\mathbf{z}_{ref}$ 加权混合则注入"自然图像"先验。接收端做严格逆 $\hat{\mathbf{z}}_{prot} = M_D^T\left(\frac{\tilde{\mathbf{z}}_{pub} - \sqrt{1-\alpha}\mathbf{z}_{ref}}{\sqrt{\alpha}}\right)$。
    - 设计动机：作者发现 noisy latent 即便经 DDIM forward 也保留秘密图像的残留结构（引 Greenberg 2025），直接 concat → DDIM reverse 模型无法跨段平滑，因此必须主动把空间结构打散并替入自然图像分布。这一步是 stego 图像视觉质量从崩塌到 SOTA 的关键。

3. **RefGen 无 control image 的参考图生成**:

    - 功能：把参考图像 $I_{ref}$ 完全由公开资源 $(\mathcal{K}_{pub}, \mathcal{P}_{pub})$ 确定性生成，发送端和接收端各自本地复现，无需任何传输。
    - 核心思路：用一个独立的预训练扩散模型（论文用 PicX_real）以 $\mathcal{K}_{pub}$ 为初始高斯噪声种子、$\mathcal{P}_{pub}$ 为 prompt 跑确定性采样得到 $I_{ref}$；由于扩散在固定种子下完全可复现，双方拿到同一份 $I_{ref}$，再 forward 一次得 $\mathbf{z}_{ref}$。**显式拒绝使用 ControlNet 类的 control image 条件**。
    - 设计动机：DiffStega 用 ControlNet + OpenPose / 分割图做条件，但这意味着 control image 要公开传输 → 直接泄漏秘密图像的结构信息；且刚性条件会限制生成多样性。MIDAS 把 ControlNet 这条路径砍掉，用 Random Basis + Latent Vector Fusion 同时承担"高质量条件生成"和"嵌入秘密信息"两个角色。

### 损失函数 / 训练策略
**完全 training-free**，不更新任何模型参数。整套流程跑在 SD v1.5 + EDICT 精确反演 + DDIM 采样器上；超参主要是 $\gamma_{priv}, \gamma_{fuse}, \alpha$ 三个。Reconstruction Stage 在 DDIM backward 时采用 joint denoise（对完整 $\hat{\mathbf{z}}_{sec}(i)$ 一次性去噪后再切分），实测优于"先切分再分别去噪"。

## 实验关键数据

### 主实验
数据集：Stego260 (Yu 2023) + UniStega (Yang 2024)，主表 $N=2$。指标分四组：stego 质量 (MANIQA↑)、stego 多样性（与秘密图相似度，PSNR↓/SSIM↓/LPIPS↑/CLIP Score↑）、正确 key 重建质量、错 key 重建质量（PSNR↓/SSIM↓/LPIPS↑ 表示越差越安全）。

| 设定 | 方法 | MANIQA↑ | Stego-PSNR↓ | CLIP↑ | 正确 key 重建 PSNR↑ | 错 key 重建 PSNR↓ |
|---|---|---|---|---|---|---|
| N=2 | CRoSS* | 0.406 | 15.55 | 26.07 | 17.61 | 15.27 |
| N=2 | DiffStega* | 0.399 | 17.07 | 26.95 | 21.91 | 18.14 |
| N=2 | **MIDAS** | **0.434** | **9.89** | **30.13** | **23.90** | **9.96** |
| N=4 | CRoSS* | 0.418 | 13.45 | 24.60 | 13.19 | 12.73 |
| N=4 | DiffStega* | 0.364 | 16.16 | 27.37 | 19.23 | 17.53 |
| N=4 | **MIDAS** | **0.479** | **9.00** | **30.17** | **22.28** | **9.40** |

注意 N=4 时 MIDAS 的 stego 质量 (MANIQA 0.479) 反而比 N=2 (0.434) 还略高，而 CRoSS*/DiffStega* 随 N 上升明显下滑——**多图扩展性**才是 MIDAS 的核心优势。同时正确 key 与错 key 重建之间出现 ~14 dB 的 PSNR 鸿沟（23.9 vs 9.96），是真正可用的 access control 强度。

### 鲁棒性 & 抗隐写分析

| 干扰 | IIS | AIS | CRoSS* | DiffStega* | **MIDAS** |
|---|---|---|---|---|---|
| Clean (PSNR↑) | 41.36 | 30.77 | 17.61 | 21.91 | **23.90** |
| Gaussian noise σ=5 | 12.44 | 14.65 | 16.30 | 20.08 | **20.05** |
| JPEG Q=70 | 10.05 | 9.44 | 16.93 | **20.42** | 19.92 |
| Gaussian blur σ=2 | 10.68 | 10.00 | 15.63 | 19.38 | **19.69** |

modification-based 方法 (IIS/AIS) 在 clean 设置下数据漂亮，但加任何信道噪声就崩到 10 dB 出头；MIDAS 在三类常见干扰下都保持 ~20 dB。

**抗隐写分析**（XuNet / SiaStegNet 训练在 cover-stego 对上）：IIS/AIS 检出率 >90%（修改痕迹明显），CRoSS*/DiffStega* >85%（artefact 暴露），MIDAS 的检出率比所有 baseline 低约 20%——直接逼近随机猜测水平。

### 关键发现
- **Random Basis vs Noise Flip**：Appendix D.1 消融显示同一框架下 Random Basis 在 stego 质量和重建质量上都显著优于 Noise Flip，验证了"正交旋转 > 符号翻转"。
- **Latent Vector Fusion 是消拼接缝的命门**：去掉这一步直接退化成 DiffStega* 的拼接缝问题，stego 质量崩塌。
- **$\gamma_{priv}=0.4$ 已足够安全**：理论分析的 $(1-\gamma)$ 项虽不为零，但实测此时错 key 重建质量已掉到 10 dB 左右，信息已不可恢复。
- **N=8 极端容量**：Appendix C.3 验证 MIDAS 在 8 图共享 1 张 stego 时仍保持可用，是 scalable 的。

## 亮点与洞察
- **把"加密"和"消拼接缝"统一在正交矩阵代数下**：Random Basis 既做 access control 又做空间打散，Latent Vector Fusion 在整张 latent 上又施一次正交变换并融入参考图像——两个机制共享同一个数学骨架 $Q_d(\mathcal{K},\gamma)$，工程上极优雅。
- **信息论 + 高维概率给出可证泄漏界**：Theorem 3.1 给出 $R_L$ 随 $\gamma$ 和维度 $m$ 的渐近形式，让"隐写到底安不安全"这件事不再只看实验曲线，而是有可解释的 scaling 行为。
- **彻底砍掉 ControlNet 依赖**：之前 DiffStega 需要公开传输 control image，这本身就是泄漏。MIDAS 的"公开资源完全确定性生成 + 私钥仅用作正交矩阵 seed"是一种更"密码学纯净"的设计哲学。
- **可迁移到任何 latent 生成模型**：方法只依赖 (a) 有 forward/backward 的扩散模型，(b) latent 空间为欧氏向量。Stable Diffusion v1.5 是举例，换成 SD3 / Flux 应该直接可用，且无需重训。

## 局限与展望
- **作者承认**：DDIM inversion + EDICT 推理延迟较高，未来需要做 sampling acceleration（few-step diffusion / consistency model）。
- **assumption**：N1·N2 = N 要求把秘密图像数限定在能切成规整网格的数（N=2, 4, 8, …），任意 N 需要更灵活的 patch packing 方案。
- **公开 prompt $\mathcal{P}_{pub}$ 选取的影响**：实验中用 target caption，若 $\mathcal{P}_{pub}$ 与秘密图像内容语义冲突太大，CRoSS*/DiffStega* 会崩；MIDAS 似乎更稳，但论文没系统量化 $\mathcal{P}_{pub}$ 失配下的退化。
- **改进方向**：Theorem 3.1 给的是上界，能否反过来推导信息论意义上的"完全保密 $\gamma^*$"？另外 Latent Vector Fusion 中的 $\alpha$ 现在是固定超参，按 timestep 调度可能更好。

## 相关工作与启发
- **vs CRoSS (Yu 2023)**：CRoSS 单图，私钥是 prompt，私钥要每次会话传输；MIDAS 私钥是 seed（短）+ 多图 + 不传额外信息。
- **vs DiffStega (Yang 2024)**：DiffStega 用 ControlNet + Noise Flip + 控制图；MIDAS 移除 ControlNet、把 Noise Flip 升级为 Random Basis、加入 Latent Vector Fusion 解决多图拼接，且具备真正的 access control。
- **vs HIS (Xu 2025)**：HIS 用 CRoSS 生成 stego 再做 modification-based 多图隐写，每次仍要传秘密相关 side info；MIDAS 完全 coverless 且不传 side info。
- **vs IIS / AIS (Zhou 2025)**：modification-based 多图访问控制方案，clean 设置 PSNR 高但抗隐写分析能力弱、抗信道噪声差；MIDAS 用 coverless + diffusion 路线把两个 trade-off 都翻过来。

## 评分
- 新颖性: ⭐⭐⭐⭐ Random Basis + Latent Vector Fusion 的组合在 CIS 领域是干净的新设计，信息论分析也合理。
- 实验充分度: ⭐⭐⭐⭐ 两个数据集 × N∈{1,2,4,8} × 多个 baseline × 鲁棒性 × 抗隐写分析 × 消融，覆盖较完整。
- 写作质量: ⭐⭐⭐⭐ 动机层层递进、表 1 一眼能看出方法定位、定理陈述清楚。
- 价值: ⭐⭐⭐⭐ training-free + 多图 + 访问控制 + 不传 side info 四个条件第一次被同时满足，在多用户隐私通信场景有直接落地价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] One-to-More: High-Fidelity Training-Free Anomaly Generation with Attention Control](../../CVPR2026/ai_safety/one-to-more_high-fidelity_training-free_anomaly_generation_with_attention_control.md)
- [\[ICML 2026\] TimeGuard: Channel-wise Pool Training for Backdoor Defense in Time Series Forecasting](timeguard_channel-wise_pool_training_for_backdoor_defense_in_time_series_forecas.md)
- [\[ICML 2026\] Demystifying the Optimal Fair Classifier in Multi-Class Classification](demystifying_the_optimal_fair_classifier_in_multi-class_classification.md)
- [\[ICML 2026\] Hidden in Plain Tokens: Simply Robust, Gradient-Free Watermark for Synthetic Audio](hidden_in_plain_tokens_simply_robust_gradient-free_watermark_for_synthetic_audio.md)
- [\[ICML 2026\] How Hard Can It Be? Hardness-Aware Multi-Objective Unlearning](how_hard_can_it_be_hardness-aware_multi-objective_unlearning.md)

</div>

<!-- RELATED:END -->
