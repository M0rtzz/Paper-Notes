---
title: >-
  [论文解读] TAG: Tangential Amplifying Guidance for Hallucination-Resistant Sampling
description: >-
  [ICML 2026][幻觉检测][切向放大] TAG 把每一步扩散更新沿当前潜变量方向分解为"径向 + 切向"两个分量，只对切向分量额外乘一个 $\eta \ge 1$ 的放大系数，从一阶 Taylor 展开上证明这等价于单调提升对数似然增益，从而把样本拉向数据流形高密度区…
tags:
  - "ICML 2026"
  - "幻觉检测"
  - "切向放大"
  - "推理时引导"
  - "几何感知采样"
  - "幻觉抑制"
  - "CFG 增强"
---

# TAG: Tangential Amplifying Guidance for Hallucination-Resistant Sampling

**会议**: ICML 2026  
**arXiv**: [2510.04533](https://arxiv.org/abs/2510.04533)  
**代码**: 有（论文中以 Project Page 形式给出）  
**领域**: 幻觉检测  
**关键词**: 切向放大、推理时引导、几何感知采样、幻觉抑制、CFG 增强

## 一句话总结
TAG 把每一步扩散更新沿当前潜变量方向分解为"径向 + 切向"两个分量，只对切向分量额外乘一个 $\eta \ge 1$ 的放大系数，从一阶 Taylor 展开上证明这等价于单调提升对数似然增益，从而把样本拉向数据流形高密度区，几乎零额外算力地缓解扩散模型的语义幻觉。

## 研究背景与动机

**领域现状**：扩散模型在图像生成上达到 SOTA，但常出现"幻觉"——多手指、错混物体、违反 prompt 的结构。主流补救手段是推理时引导（CFG 及其变体 PAG、SEG、CFG++ 等），通过对条件/无条件残差做标量缩放或对网络做小修改来把样本推离低密度区。

**现有痛点**：现有引导本质上是"几何无感"的——只对 cond–uncond 残差做一个标量乘法，并不考虑当前样本在该噪声水平下数据流形的局部方向结构；过强的缩放又会扰动 ODE/SDE 求解器、压缩多样性，甚至引入新的伪影。

**核心矛盾**：扩散采样轨迹既要沿噪声调度走"径向"（半径要按调度收缩），又要沿数据流形走"切向"（把样本细节修到真实分布上）。一个标量同时放大两个方向会破坏径向时间表，导致过平滑；而完全不区分则错失了沿流形上爬的机会。

**本文目标**：设计一种不改架构、不重训、和任意现成扩散主干即插即用的引导方法，仅利用轨迹本身可得的几何信号，定向放大"对样本质量真正有用的方向"。

**切入角度**：作者从 Tweedie 恒等式和高斯环面（Gaussian annulus）定理出发，论证在噪声水平 $t_k$ 下数据集中在一个薄球壳 $\mathcal{M}_k$ 附近；把每一步更新 $\Delta_{k+1}$ 在 $\mathcal{M}_{k+1}$ 上做正交分解后发现：径向分量是噪声调度规定好的"减半径"动作，切向分量才承载语义结构（图 2 可视化显示切向部分对应清晰的语义图，径向部分像噪声）。

**核心 idea**：保留径向不动，只把切向分量按系数 $\eta \ge 1$ 等比放大，单步更新即可在不改变噪声调度的前提下，沿数据流形多走一步、爬向更高密度区。

## 方法详解

### 整体框架
TAG 想解决的是扩散采样中"语义幻觉"——多手指、错混物体、违反 prompt 的结构。它的做法不是去改网络或重训，而是把每一步求解器算出的更新量当成一个向量，沿当前样本所在的球壳做正交分解，只把承载语义的那个分量放大一点。具体说，它套在标准 DDIM/EDM/Flow Matching 等基础求解器外层，给定第 $k+1$ 步状态 $\bm{x}_{k+1}$，先用主干 $\epsilon_\theta$ 算出原始更新 $\Delta_{k+1} = \tilde{\bm{x}}_k - \bm{x}_{k+1}$，再用样本自身的归一化方向把这个更新拆成"径向 + 切向"，保留径向、放大切向后再走出这一步。整个过程不需要额外的 denoiser 前向、不改任何参数，还能和 CFG/PAG/SEG/SSG 等已有引导直接叠加。

### 关键设计

**1. 切向放大的更新规则：让语义方向走得更远，噪声调度方向不动**

扩散轨迹其实背着两个不同性质的任务：径向方向是噪声调度规定好的"按时间收缩半径"，切向方向才是模型自由发挥、往数据流形上修细节的地方。已有引导对整条残差做一个标量缩放，等于同时放大这两个方向，把噪声调度也一起扰动了。TAG 的关键是把它们分开处理：用当前样本自身作投影基 $\hat{\bm{x}}_{k+1} = \bm{x}_{k+1}/\|\bm{x}_{k+1}\|_2$，定义径向投影 $\bm{P}_{\mathcal{M}_{k+1}} = \hat{\bm{x}}_{k+1}\hat{\bm{x}}_{k+1}^\top$ 和切向投影 $\bm{P}^\perp_{\mathcal{M}_{k+1}} = I - \bm{P}_{\mathcal{M}_{k+1}}$，然后只对切向分量乘一个 $\eta \ge 1$ 的系数：

$$\Delta^{\text{TAG}}_{k+1} = (\bm{P}_{\mathcal{M}_{k+1}} + \eta\,\bm{P}^\perp_{\mathcal{M}_{k+1}})\Delta_{k+1},\qquad \bm{x}_k = \bm{x}_{k+1} + \Delta^{\text{TAG}}_{k+1}.$$

把这个增量代回 DDIM 系数后，等价于只对预测噪声 $\epsilon_\theta$ 的切向分量做同等放大，所以也可以理解成"对预测噪声的几何感知再缩放"。之所以选切向放大而不是别的方向，是因为图 2 的可视化把两个分量画出来看：径向部分像无结构噪声，切向部分则是一张清晰的语义图——语义信息天然集中在切向，放大它就能选择性增强语义而不放大噪声。

**2. 一阶 Taylor 增益单调性证明：解释为什么放大切向不会破坏轨迹**

光有"切向看着像语义"的直觉还不够，作者要回答放大它为什么不会像盲目调大 CFG 那样毁掉样本。思路是把推理时引导写成一个带步长约束的局部对数似然最大化问题：在 $\|\bm{x}_k - \bm{x}_{k+1}\|_2 \le \delta_k$ 的约束下求 $\max (\bm{x}_k - \bm{x}_{k+1})^\top \nabla_{\bm{x}}\log p(\bm{x}\mid t_{k+1})|_{\bm{x}_{k+1}}$，并定义一阶 Taylor 增益 $G(\eta) := (\Delta^{\text{TAG}}_{k+1})^\top \nabla_{\bm{x}}\log p$。论文证明 $G(\eta)$ 关于 $\eta$ 单调递增（Theorem 4.1），也就是只要 $\eta>1$ 就严格提升局部似然、把样本往高密度区拉。关键之处在于径向分量始终保持不变，意味着 Tweedie 恒等式给出的"一步标定"被保住了——$\langle \hat{\bm{x}}_{k+1}, \Delta^{\text{TAG}}_{k+1}\rangle = \langle \hat{\bm{x}}_{k+1}, \Delta_{k+1}\rangle$（公式 22），噪声调度的时间表没被动过，所以放大切向是在似然上单调爬升而不会撑坏轨迹。

**3. CFG 上的切向对齐（$\text{TAG}_{\text{cfg}}$）：把条件/无条件分支拉回同一切向**

作者把同一套几何视角搬到 CFG，认为条件与无条件预测的失配主要发生在切向子空间（径向被噪声调度共同约束住了），所以与其用标量 $\omega$ 调大整条残差、连不该放大的不一致部分也一起放大，不如只在切向里重新对齐。具体做法是：记 CFG 残差 $\bm{g}_k = \bm{\varepsilon}_c - \bm{\varepsilon}_u$，取其切向投影 $\bm{g}_k^\perp = \bm{P}^\perp_{\mathcal{M}}(\bm{x}_k)\bm{g}_k$，再把条件预测 $\bm{\varepsilon}_c$ 投到这个切向子空间 $\mathrm{span}(\bm{g}_k^\perp)$ 上得到对齐向量

$$\bm{g}_k^{\text{align}} = \frac{\langle\bm{\varepsilon}_c, \bm{g}_k^\perp\rangle}{\|\bm{g}_k^\perp\|_2^2}\,\bm{g}_k^\perp,\qquad \tilde{\bm{\varepsilon}}_k = \bm{\varepsilon}_u + \omega\,\bm{g}_k + \eta\,\bm{g}_k^{\text{align}}.$$

这样两支预测在切向方向上"同向前进"，既保留了 CFG 原本的引导强度 $\omega$，又额外补一个只作用在切向的对齐项 $\eta$，避免了标量缩放无差别放大不一致带来的伪影。

### 损失函数 / 训练策略
完全无需训练或微调。算法只在采样循环里插入投影和向量重加权（Algorithm 1，11 行代码量级），唯一超参为 $\eta \ge 1$（无条件采样典型设置 $\eta \approx 1.x$；CFG 版本另有标量 $\omega, \eta$ 两个）。

## 实验关键数据

主要 backbone：SD v1.5 / v2.1 / XL / SD3，在 COCO 2014 上评测；另在 T2I-CompBench 评测组合性。

### 主实验

| 设置 | Backbone | FID ↓ | IS ↑ | AES ↑ | CMMD ↓ |
|------|----------|-------|------|-------|--------|
| Uncond. | SD v1.5 | 58.41 | 15.59 | 5.003 | 1.069 |
| Uncond. | SD v1.5 + **TAG** | **46.20** | **16.77** | **5.064** | **0.778** |
| Uncond. | SDXL | 119.14 | 9.08 | 5.645 | 2.474 |
| Uncond. | SDXL + **TAG** | **90.71** | 8.91 | 5.577 | **2.201** |
| Uncond. | SD3 | 84.26 | 11.53 | 5.261 | 1.671 |
| Uncond. | SD3 + **TAG** | **79.11** | **11.73** | **5.365** | **1.564** |

| 设置 | Backbone | FID ↓ | ImageReward ↑ | CLIP ↑ |
|------|----------|-------|---------------|--------|
| Cond. (T2I) | SD v1.5 | 33.49 | −0.342 | 25.00 |
| Cond. (T2I) | SD v1.5 + **TAG** | **26.61** | **−0.339** | **25.09** |
| Cond. (T2I) | SD v2.1 | 26.12 | 0.143 | 25.35 |
| Cond. (T2I) | SD v2.1 + **TAG** | **21.59** | **0.424** | **26.16** |
| Cond. (T2I) | SD3 | 29.02 | 1.030 | 26.39 |
| Cond. (T2I) | SD3 + **TAG** | **27.54** | **1.043** | **26.56** |

### 消融实验

| 配置（SD v1.5, Uncond.） | FID ↓ | IS ↑ | CMMD ↓ | 说明 |
|--------------------------|-------|------|--------|------|
| No guidance | 58.41 | 15.59 | 1.069 | 不加任何引导 |
| TAG | 46.20 | 16.77 | 0.778 | 只用 TAG |
| PAG | 53.72 | 21.13 | 0.723 | 现有 SOTA 引导 |
| **TAG + PAG** | **52.61** | **21.20** | **0.701** | 叠加后再提升 |
| SEG | 47.69 | 18.50 | 0.835 | 另一现有引导 |
| **TAG + SEG** | **42.71** | **19.45** | **0.746** | 叠加后再提升 |

另在 T2I-CompBench 的 Spatial / Complex 子集上，TAG 把 SDXL 的 2DSpatial 从 0.1857 → 0.1980、BLIP-VQA 从 0.4443 → 0.4650、ImageReward 从 0.2596 → 0.3978，说明对结构性幻觉（如三条腿）也有明显改善。

### 关键发现
- **径向放大反而毁掉样本**：图 5 显示如果把 normal（径向）分量也一起放大（"+TAG + Normal"），会发生严重过平滑——这从公式 21 的 $\kappa$ 倍径向收缩可解释，验证了"只放大切向"的必要性。
- **50 NFE 的 TAG 打过 250 NFE 的 baseline**：在同一 backbone 上，TAG 把 50 步的样本质量推到接近 5× 步数的原始采样，说明它确实是把"采样轨迹本身"修得更准，而非靠多算。
- **真正即插即用**：和 PAG / SEG / 自定义 CFG 都能叠加且单调更优，没有发现互斥情况；唯一新增超参 $\eta$ 很稳。

## 亮点与洞察
- **把"几何感知"从经验提到理论**：以前 PAG/SEG/CFG++ 多是经验上选择"哪些方向不要放大"，TAG 用 Tweedie + 高斯环面给出了"为什么切向放大就是单调提升一阶似然"的证明，是一种把扩散引导从启发式推向几何最优化的视角。
- **巧妙利用"样本自身做投影基"**：不需要外部分类器、不需要额外网络前向，仅用 $\hat{\bm{x}}_{k+1}$ 就把高维空间分成"调度规定的方向"和"模型自由发挥的方向"——这种"用自身归一化向量当几何标架"的 trick 可迁移到任何带球壳先验的迭代算法（语言模型 logits、流匹配、薛定谔桥等）。
- **CFG 的"切向不一致"假说**：把条件/无条件预测的失配归因于切向子空间这一断言很有 insight；如果成立，后续工作可以专门去训一支"切向校准网络"，而不再做整向量的残差缩放。

## 局限与展望
- 作者承认的局限：理论是基于 $\log p$ 局部 $C^2$ 光滑假设和一阶 Taylor 展开；当 $\eta$ 过大时高阶项不可忽略，仍会出现可见伪影；CFG 变体引入额外超参 $\eta$ 需调。
- 自己观察到的局限：投影基取 $\hat{\bm{x}}_{k+1}$ 即假设噪声分布严格各向同性、数据流形是同心球壳；对非欧几何潜空间（如球面流匹配、SE(3) 上的扩散）需要重新定义投影；表 1 中 SDXL 上 IS 略降（9.08→8.91）说明 FID/CMMD 的提升一定程度上以多样性细节为代价。
- 改进思路：把单点的径向投影换成估计的局部 tangent bundle（用 PCA / Jacobian 估流形切空间）；或把 $\eta$ 做成 $t_k$ 的函数，前期小后期大，与 Aithal 等观察到的"幻觉主要出现在中段时间步"对齐。

## 相关工作与启发
- **vs CFG / PAG / SEG**：传统引导直接对预测噪声 $\bm{\varepsilon}$ 做标量缩放或对自注意力扰动；TAG 在向量层面做几何分解，理论上证明"只放切向不放径向"是局部最优，且与上述方法正交可叠加。
- **vs Mode Interpolation (Aithal 2024)**：那篇工作指出幻觉源于轨迹穿过低密度模式之间的"谷"；TAG 是直接基于这一观察设计的：切向放大就是沿数据流形多走、避免落入谷里。
- **vs Tweedie-based score correction**：传统用 Tweedie 估计 $\mathbb{E}[\bm{x}_0\mid\bm{x}_k]$ 做后验校正，TAG 不估计 $\bm{x}_0$，只用 Tweedie 恒等式来论证径向分量已被噪声调度"吃掉"——同一恒等式被用出新角度。

## 评分
- 新颖性: ⭐⭐⭐⭐ 切向/径向分解 + 一阶单调性证明在扩散引导里是清晰新颖的角度。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 SD1.5/2.1/XL/SD3 + 无/有条件 + 与 PAG/SEG 叠加 + T2I-CompBench，规模合理但缺人评。
- 写作质量: ⭐⭐⭐⭐ 推理流畅，从几何直觉到定理顺承，公式偏多但配图很到位。
- 价值: ⭐⭐⭐⭐ 不改架构、不重训、~10 行代码即可插到任何扩散主干，工业落地友好。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Ground What You See: Hallucination-Resistant MLLMs via Caption Feedback, Diversity-Aware Sampling, and Conflict Regularization](../../AAAI2026/hallucination/ground_what_you_see_hallucination-resistant_mllms_via_caption_feedback_diversity.md)
- [\[ICML 2026\] Building Reliable Long-Form Generation via Hallucination Rejection Sampling](building_reliable_long-form_generation_via_hallucination_rejection_sampling.md)
- [\[ICML 2026\] Capturing Gaze Shifts for Guidance: Cross-Modal Fusion Enhancement for VLM Hallucination Mitigation](capturing_gaze_shifts_for_guidance_cross-modal_fusion_enhancement_for_vlm_halluc.md)
- [\[CVPR 2026\] Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance](../../CVPR2026/hallucination/residual_decoding_mitigating_hallucinations_in_large_vision-language_models_via_.md)
- [\[ICML 2026\] Automatic Layer Selection for Hallucination Detection](automatic_layer_selection_for_hallucination_detection.md)

</div>

<!-- RELATED:END -->
