---
title: >-
  [论文解读] SURF: Separation via Unsupervised Remixing Flow
description: >-
  [ICML 2026][图像生成][单通道源分离] SURF 把监督流匹配 FLOSS 与无监督的 ReMixIT / Self-Remixing 教师-学生重混合训练拼到一起，让一个生成式 flow matching 分离器**完全从混合观测**（没有任何干净源样本）训练出来…
tags:
  - "ICML 2026"
  - "图像生成"
  - "单通道源分离"
  - "流匹配"
  - "无监督学习"
  - "教师-学生蒸馏"
  - "Wake-Sleep"
---

# SURF: Separation via Unsupervised Remixing Flow

**会议**: ICML 2026  
**arXiv**: [2606.04921](https://arxiv.org/abs/2606.04921)  
**代码**: 待确认  
**领域**: 音频语音 / 源分离 / 生成模型  
**关键词**: 单通道源分离, 流匹配, 无监督学习, 教师-学生蒸馏, Wake-Sleep

## 一句话总结
SURF 把监督流匹配 FLOSS 与无监督的 ReMixIT / Self-Remixing 教师-学生重混合训练拼到一起，让一个生成式 flow matching 分离器**完全从混合观测**（没有任何干净源样本）训练出来，在 MNIST/CIFAR10 图像分离和 LibriSpeech / FUSS 音频分离上几乎追平有监督 flow 的指标，刷新无监督 SOTA。

## 研究背景与动机

**领域现状**：单通道源分离（给一个混合，恢复 K 个底层源）是一个高度病态的反问题，深度学习时代分两派——一派是判别式回归（Conv-TasNet、TF-Locoformer 等），直接把 mixture 映到 source；另一派是生成式（diffusion / flow matching），把分离视为条件生成，靠学一个 clean source 的强先验来约束解空间。后者最近用 FLOSS（Scheibler et al., 2025）做出了 SOTA。

**现有痛点**：所有生成式方法都假设**能拿到干净源样本**来训练先验。但在生物声学、高光谱、引力波检测等场景，"录到一个单独的源"本身就是不可能的；即便能采到，训练-测试 domain shift 仍然是顽疾。无监督派的代表 MixIT / ReMixIT / Self-Remixing 用"教师产估计→打乱重组成新混合→学生学回"的自监督把判别式分离器训出来了，但这些方法只适用于回归式分离器（直接出 $\hat{x}$），**没人把它和 flow matching 这种"学速度场"的生成式模型对上**。

**核心矛盾**：Flow matching 训练目标是回归速度 $v_\theta(x_t, t, m)$，ReMixIT 类自监督方法的目标是直接回归源 $\hat{x}$；两者的输出语义不同，PIT 配对、mixture consistency 的注入方式都不一样，简单拼接训不动。

**本文目标**：把 FLOSS 这种生成式 flow 分离器**完全从混合观测训练**出来，且不能掉太多点。

**切入角度**：作者发现一个简单恒等式——flow matching 路径 $x_t = (1-t)x_0 + t x_1$ 配合 $\boldsymbol{u}(x,t)=\mathbb{E}[x_1-x_0|x_t,m]$ 可推出 $\mathbb{E}[x_1|x_t, m] \approx x_t + (1-t)v_\theta(x_t, t, m)$。这就把速度场和 ReMixIT 需要的"clean source 估计"在概念上对上了，可以把 ReMixIT / Self-Remixing 的 loss 嫁接到 flow 上。

**核心 idea**：以 EMA 教师 flow 模型生成伪源 → 跨 batch 打乱重组成新混合 → 用 FLOSS 的"PIT 选最优置换 + 在选后路径上回归速度"训练学生 flow → 学生再 EMA 回写教师；ReMixIT 把目标设为教师伪源、Self-Remixing 把目标设为原始混合本身。

## 方法详解

### 整体框架
SURF 用两个同结构的 flow matching 网络 $v_{\theta_\mathcal{T}}$（teacher）和 $v_\theta$（student），都按 FLOSS 训练。两者初始化来自 MixIT 预热的回归分离器。每个迭代步：

1. **教师推断**：从真实混合批次 $\boldsymbol{M}=[\boldsymbol{m}_1,\dots,\boldsymbol{m}_B]$ 出发，教师以 flow ODE 采样得到伪源 $\mathcal{X} \in \mathbb{R}^{BK\times d}$。
2. **重混合**：在 $S_{BK}$ 上均匀采样置换 $\boldsymbol{\Pi}$ 打乱所有 $BK$ 行，得到 $\tilde{\boldsymbol{X}}_1 = \boldsymbol{\Pi}\mathcal{X}$，再求和成新混合 $\tilde{\boldsymbol{M}}=(\boldsymbol{I}_B\otimes \mathbf{1}^\top)\tilde{\boldsymbol{X}}_1$。
3. **构造 FM 路径**：噪声端 $\tilde{\boldsymbol{X}}_0 = \tfrac{1}{K}(\boldsymbol{I}_B\otimes \mathbf{1})\tilde{\boldsymbol{M}} + (\boldsymbol{I}_B\otimes \boldsymbol{P}^\perp)\boldsymbol{Z}$（保证 mixture consistency），用学生的 $t=0$ 速度对每个 mixture 做 PIT 选最优置换 $\boldsymbol{\Upsilon}$，得到插值 $\tilde{\boldsymbol{X}}_t^{\boldsymbol{\Upsilon}}=(1-t)\tilde{\boldsymbol{X}}_0 + t\boldsymbol{\Upsilon}\tilde{\boldsymbol{X}}_1$。
4. **损失计算**：定义残差 $\boldsymbol{R}_t = v_\theta(\tilde{\boldsymbol{X}}_t^{\boldsymbol{\Upsilon}}, t, \tilde{\boldsymbol{M}}) - (\boldsymbol{\Upsilon}\tilde{\boldsymbol{X}}_1 - \tilde{\boldsymbol{X}}_0)$，分两条路径监督。
5. **EMA 更新教师**：$\theta_\mathcal{T} \leftarrow \alpha \theta_\mathcal{T} + (1-\alpha)\theta$。

整个过程**完全不需要 clean source**——所有"目标"要么来自教师自己（ReMixIT 变体），要么来自原始观测混合（Self-Remixing 变体）。

### 关键设计

1. **Velocity-to-Denoiser 桥接（让 FM 能接 ReMixIT）**:

    - 功能：把 flow matching 学到的速度 $v_\theta$ 和 ReMixIT/Self-Remixing 需要的"clean source 估计"在数学上对接起来。
    - 核心思路：基于 $\boldsymbol{u}(x,t)=\mathbb{E}[x_1-x_0|x_t,m]$ 和路径 $x_1-x_0=(x_1-x_t)/(1-t)$，得到 $\mathbb{E}[x_1|x_t,m]=x_t+(1-t)\boldsymbol{u}(x_t,t,m)\approx x_t+(1-t)v_\theta(x_t,t,m)$。于是任意一步 $t$ 都可以拿这个量当"时间相关的去噪估计"，直接代进 Self-Remixing 那种"用估计源重构原始 mixture"的回归式 loss 里。
    - 设计动机：FM 与回归式自监督本来"语义不同"，简单拼接会让自监督信号和 FM 训练目标脱节。这个恒等式让一份模型同时给出"速度场"和"去噪源"两种视角，整套 ReMixIT 数学不必动结构就能搬过来。

2. **ReMixIT-FM 与 Self-Remixing-FM 双损失（监督 / 反思两种选择）**:

    - 功能：在重混合后的合成数据上给学生 flow 提供训练信号。
    - 核心思路：ReMixIT-FM 直接用 FLOSS 风格的 PIT-FM 损失 $\mathcal{L}_{\text{RM-FM}}=\mathbb{E}\|\boldsymbol{R}_t\|^2$，把教师伪源当 ground truth。Self-Remixing-FM 反而退一步——只要求学生的估计 sum 回去等于原始 mixture：$\mathcal{L}_{\text{SR-FM}}=\mathbb{E}\|(\boldsymbol{I}_B\otimes\mathbf{1}^\top)\boldsymbol{\Pi}^{-1}\boldsymbol{\Upsilon}^{-1}\boldsymbol{R}_t\|^2$。两条 loss 共享同一个 $\boldsymbol{R}_t$ 但施加不同投影。
    - 设计动机：ReMixIT 训练信号更密集但会"继承"教师误差；Self-Remixing 不直接惩罚伪源错误、只要求 mixture 重构一致，作者在 Appendix 证明其 gradient 是 $2(K-1)\mathbb{E}[\beta_t(\nabla_\theta \delta^{(1)}_{\theta,t})^\top \delta^{(2)}_{\theta,t}]$，当不同源的系统误差弱相关时这一项接近零，主导梯度回到 pseudo-supervised FM。实验显示两者性能接近、Self-Remixing 在音频上稍好。

3. **Wake-Sleep 解释 + EMA 教师（让"自我蒸馏"理论闭环）**:

    - 功能：解释为什么"教师生成 → 重混合 → 学生学回 → EMA 回写教师"这个看似 hacky 的循环能收敛，并指导教师的更新方式。
    - 核心思路：把教师定义的边缘 $\bar{p}_{\theta_\mathcal{T}}(\bar{\boldsymbol{x}})=\prod_k \bar{p}_{\theta_\mathcal{T}}(\bar{\boldsymbol{x}}^{(k)})$ 当成隐式 prior，把学生 $p_\theta(\bar{\boldsymbol{x}}|m)$ 当成 inference network。Sleep 阶段（更新学生）等价于在合成对 $(\bar{\boldsymbol{x}}, m)\sim \bar{p}_{\theta_\mathcal{T}}(\bar{\boldsymbol{x}})p(m|\bar{\boldsymbol{x}})$ 上做极大似然，正好就是 ReMixIT；Wake 阶段（更新教师）的精确解需要不可得的 aggregate posterior，于是退化成"把 $\theta_\mathcal{T}$ 朝 $\theta$ 移动"，并用 EMA $\theta_\mathcal{T} \leftarrow \alpha\theta_\mathcal{T} + (1-\alpha)\theta$ 实现稳定的"慢追快"。
    - 设计动机：以前 ReMixIT 的成功更多是经验观察，没有清晰的概率框架。这里把它落到 Wake-Sleep 上，既给了"为什么能 bootstrap"的解释，也告诉你 EMA 不是随便加的稳定 trick，而是 Wake 步的实际替代品。

### 损失函数 / 训练策略
联合训练目标可以是 $\mathcal{L}_{\text{RM-FM}}$ 或 $\mathcal{L}_{\text{SR-FM}}$（论文给出两个变体）。教师 EMA 衰减 $\alpha$ 是关键超参；学生网络结构沿用 FLOSS（permutation-equivariant + mixture-consistency 投影层）。每次学生更新一步后教师 EMA 同步。初始化用 MixIT 预训练的回归式分离器（无监督起点）。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 (SURF-RM) | 之前无监督 SOTA | 监督 Flow 上界 |
|--------|------|------|----------|------|
| MNIST 2-source | PSNR ↑ | **37.26** | 23.13 (Self-Remixing) | 37.44 |
| MNIST 2-source | FID ↓ | **19.57** | 28.14 | 19.47 |
| CIFAR10 2-source | PSNR ↑ | **19.73** | 17.51 | 20.38 |
| CIFAR10 2-source | FID ↓ | **14.83** | 28.44 | 9.60 |
| LibriSpeech+FUSS (2 src) | SI-SDR ↑ | 14.98 / **15.23** (SR) | 14.81 (Self-Remixing) | 18.21 |
| FUSS 1-src | SI-SDR ↑ | **32.67** | 19.83 (ReMixIT) | 38.79 |

> 重点：图像分离上 SURF 把 PSNR 从无监督派的 23 拉到 37，**追平监督 flow**；CIFAR10 FID 从 28 砍到 14，比之前 BASIS（diffusion prior，需要 clean data）还低。

### 消融实验

| 配置 | MNIST PSNR | 说明 |
|------|---------|------|
| MixIT (初始化用) | 21.90 | 回归式起点 |
| Regression-ReMixIT | 22.81 | 回归式自监督 |
| SURF (ReMixIT-FM) | **37.26** | flow + ReMixIT |
| SURF (Self-Remixing-FM) | 37.03 | flow + Self-Remixing |
| Supervised Flow | 37.44 | 上界（需 clean data） |

### 关键发现
- "flow + 自监督"远好于"回归 + 自监督"（37 vs 23 PSNR），说明生成式先验对去除回归 artifacts 至关重要——这是把 FM 接进来的核心收益。
- ReMixIT-FM 与 Self-Remixing-FM 性能接近，但 Self-Remixing 在音频 LibriSpeech 上略胜（SI-SDR 15.23 vs 14.98），符合理论分析中"自监督误差与教师误差解耦"的预期。
- SURF 在 CIFAR10 FID 上甚至**低于**监督回归（14.83 vs 25.44），印证生成式 prior 在分布匹配上的优势（哪怕样本级 PSNR 略低）。
- 在 FUSS 多源情境（3、4 源），SURF 与监督 flow 仍有 2-3 dB 差距，说明源数变多时 PIT 配对/EMA 稳定性是后续可以攻的方向。

## 亮点与洞察
- **velocity-denoiser 恒等式是真正的关键 plug**：$\mathbb{E}[x_1|x_t, m]\approx x_t+(1-t)v_\theta$ 这个一行推导让所有依赖"clean source 估计"的自监督 loss（ReMixIT、Self-Remixing、以及未来可能的 cycle consistency）都能无痛接到任意 flow / diffusion 分离器上，是个可迁移到 inverse problems 一大类问题的通用桥梁。
- **Wake-Sleep 视角的归约**让 ReMixIT 这种工程经验上升到隐变量生成模型的标准训练范式，"EMA 教师 = 不可解的 Wake 步的代理"这个解释把先前的启发式 trick 接到了变分推断框架里，对设计新的 self-training 算法有方法论意义。
- **图像分离做对照实验**是个聪明设计：源分离传统是音频领域，作者用 MNIST/CIFAR10 给出量化的 PSNR/FID/LPIPS 让方法对比有像视觉社区那样清晰的"SOTA 表"，比纯音频 SI-SDR 的可解释性强很多，也方便后续工作复现对照。

## 局限与展望
- 作者承认理论分析建立在 $B\to \infty$ 的 population limit + 简化假设上，有限批次下的偏差未刻画；EMA 衰减 $\alpha$ 实际选取仍属经验调参。
- 整套训练依赖 MixIT 预热得到的初始分离器作为种子；如果种子非常差（domain 极冷启动），bootstrap 是否还能收敛、收敛到哪个 mode 没有给出可行性证据。
- 在 FUSS 3-4 源混合场景指标和监督 flow 还有 2-3 dB 差，说明随 K 增大 PIT 配对的组合复杂度和教师误差累积会放大，需要更稳的配对策略或层次化 source factorization。
- 自然延伸：把这个"velocity-denoiser 桥接 + Wake-Sleep 教师 EMA"迁移到其他 inverse problem（去噪、去模糊、超分），尤其在干净训练数据稀缺的医学/科学成像场景潜力很大。

## 相关工作与启发
- **vs FLOSS (Scheibler et al., 2025)**: FLOSS 是监督 flow 分离的当前 SOTA，依赖 clean source；SURF 把它的 PIT-FM 损失结构原样借过来，只是把"真 source"换成"教师伪源 / 原始混合"，可以视为 FLOSS 的无监督版本。
- **vs ReMixIT / Self-Remixing (Tzinis et al., 2022; Saijo & Ogawa, 2023)**: 老 ReMixIT 只能驱动回归式分离器（Conv-TasNet 那种），输出是确定性的 $\hat{x}$；SURF 把同样的"重混合 → 学生学回"循环搬到学习速度场的生成式 flow 上，借生成式 prior 解决回归 artifacts。
- **vs BASIS / 基于 diffusion prior 的分离 (Jayaram & Thickstun, 2020; Mariani et al., 2024)**: 这些方法用 clean source 训练 prior + 测试时 guidance 做分离；SURF 不需要 clean source 训练 prior，因此可适用于无法采集干净源的领域（生物声学、引力波）。
- **vs Rozet et al., 2024 / Hosseintabar et al., 2025（无监督 diffusion prior with EM）**: 这些工作也尝试从混合数据学 diffusion prior，但需在每步训练条件 diffusion 来近似 unconditional，计算量极大且尚未应用于单通道分离；SURF 避开了 prior 显式建模，直接在条件分离器上做 bootstrap。

## 评分
- 新颖性: ⭐⭐⭐⭐ velocity-denoiser 桥接虽是简单恒等式但 unlock 了 FM + 自监督的整条 pipeline，Wake-Sleep 解释也补齐了理论缺口
- 实验充分度: ⭐⭐⭐⭐ 横跨图像（MNIST/CIFAR10）+ 音频（FUSS/LibriSpeech/Libri2Mix）四个 benchmark，对照监督上界 + 多种无监督 baseline 都给出
- 写作质量: ⭐⭐⭐⭐ 公式繁多但 Algorithm 1 红绿配色把两种 loss 分支讲得很清楚，理论部分给出了几个有用的 decomposition
- 价值: ⭐⭐⭐⭐⭐ 给"无干净源训练生成式分离器"提供了一条可行路径，对生物声学、医学成像、引力波等数据稀缺领域可直接借鉴

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Cinematic Audio Source Separation Using Visual Cues](../../CVPR2026/image_generation/cinematic_audio_source_separation_using_visual_cues.md)
- [\[ECCV 2024\] Implicit Style-Content Separation using B-LoRA](../../ECCV2024/image_generation/implicit_style-content_separation_using_b-lora.md)
- [\[ICLR 2026\] From Parameters to Behaviors: Unsupervised Compression of the Policy Space](../../ICLR2026/image_generation/from_parameters_to_behaviors_unsupervised_compression_of_the_policy_space.md)
- [\[ICML 2025\] Unsupervised Learning for Class Distribution Mismatch (UCDM)](../../ICML2025/image_generation/unsupervised_learning_for_class_distribution_mismatch.md)
- [\[ICML 2026\] Adversarial Flow Models](adversarial_flow_models.md)

</div>

<!-- RELATED:END -->
