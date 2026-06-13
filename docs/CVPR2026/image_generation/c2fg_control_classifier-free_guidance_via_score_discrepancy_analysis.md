---
title: >-
  [论文解读] C$^2$FG: Control Classifier-Free Guidance via Score Discrepancy Analysis
description: >-
  [CVPR2026][图像生成][Classifier-Free Guidance] 本文用严格的 score 差距上界证明"条件与无条件分布在前向扩散中以指数速率收敛"，据此把 CFG 里那个固定的引导权重 $\omega$ 换成一个指数衰减的时变控制函数 $\omega(t)$，无需训练、即插即用，在 DiT / SiT / Stable Diffusion / EDM2 等多种框架上把 FID/IS 都进一步刷到了 SOTA。
tags:
  - "CVPR2026"
  - "图像生成"
  - "Classifier-Free Guidance"
  - "扩散模型"
  - "score 差距"
  - "时变引导权重"
  - "training-free"
---

# C$^2$FG: Control Classifier-Free Guidance via Score Discrepancy Analysis

**会议**: CVPR2026  
**arXiv**: [2603.08155](https://arxiv.org/abs/2603.08155)  
**代码**: 待确认  
**领域**: 图像生成  
**关键词**: Classifier-Free Guidance, 扩散模型, score 差距, 时变引导权重, training-free  

## 一句话总结
本文用严格的 score 差距上界证明"条件与无条件分布在前向扩散中以指数速率收敛"，据此把 CFG 里那个固定的引导权重 $\omega$ 换成一个指数衰减的时变控制函数 $\omega(t)$，无需训练、即插即用，在 DiT / SiT / Stable Diffusion / EDM2 等多种框架上把 FID/IS 都进一步刷到了 SOTA。

## 研究背景与动机
**领域现状**：Classifier-Free Guidance（CFG）是现代条件扩散模型的基石。它通过 $\hat{\epsilon}=\epsilon_\theta(x_t,t,\varnothing)+\omega[\epsilon_\theta(x_t,t,y)-\epsilon_\theta(x_t,t,\varnothing)]$ 在条件与无条件预测之间做插值，用一个标量 $\omega$ 控制条件信息的注入强度，从而在保真度和多样性之间权衡。

**现有痛点**：原始 CFG 在整个采样过程中把 $\omega$ 固定为常数。后续工作（Interval Guidance、FDG、CFG++、$\beta$-CFG、RAAG 等）已经意识到固定权重不是最优，提出了各种动态调度，但它们几乎全是启发式的——靠经验观察和调参拍出来的曲线，缺乏理论依据，也说不清"为什么该这样变"。

**核心矛盾**：这些方法忽略了 CFG 设计里最根本的一点——条件分布 $p(x_t|y)$ 与无条件分布 $p(x_t)$ 之间的差异在扩散过程中本身就是动态变化的。如果不知道这个差异随时间怎么变，就无法原理性地决定每一步该给多大引导。

**本文目标**：① 从理论上刻画条件/无条件 score 差距随时间步的演化规律；② 据此设计一个与扩散动力学严格对齐的时变引导权重。

**切入角度**：把前向扩散写成 Ornstein–Uhlenbeck 过程（VP-SDE / VE-SDE），直接对两个由不同初值出发的分布的 score 差求上界。直觉是：前向加噪会让所有条件下的分布都趋向同一个高斯，所以条件信息必然随 $t$ 增大而流失——关键是流失得有多快、是不是均匀。

**核心 idea**：证明 score 差距随时间近似按 $e^{-t}$ 指数衰减，于是把固定 $\omega$ 替换成同样指数形状的 $\omega(t)$，让引导强度"哪里差距大就给多大"。

## 方法详解

### 整体框架
C$^2$FG 不改网络、不改训练，只改采样时每一步用的引导权重。逻辑链是三段：先在理论上对前向扩散中条件/无条件 score 的差距求出严格上界（Theorem 1/2），证明它随时间近似指数衰减；再用 Harnack 型不等式（Theorem 3/4）从概率密度角度补充说明"$t\to 0$ 的临界区差异最大、最需要强引导"；最后据此把标准 CFG 的固定 $\omega$ 换成指数衰减控制函数 $\omega(t)=\omega_0\exp(\lambda(1-t/t_{\max}))$，在反向采样每一步即插即用。整套方法只是一个权重函数的替换，因此天然 training-free，且与 autoguidance、interval guidance 等已有策略正交可叠加。

### 关键设计

**1. score 差距的严格上界：把"凭经验"变成"有定理"**

针对"现有动态 CFG 全是启发式"这个痛点，本文直接对 score 差距给出可证明的上界。设条件密度 $\tilde p(x_t,t)=p(x_t,t|y)$ 与无条件密度 $p(x_t,t)$ 由不同初值经同一前向 SDE 演化。在 VP-SDE 下（Theorem 1）有

$$\|\nabla\log p(x,t)-\nabla\log\tilde p(x,t)\|\le \frac{\alpha(t)}{\sigma^2(t)}\,C,$$

其中 $\alpha(t)=\exp(-\tfrac12\int_0^t\beta_s\,\mathrm{d}s)$，$\sigma(t)=\alpha(t)\sqrt{\int_0^t \beta_s/\alpha^2(s)\,\mathrm{d}s}$，$C$ 是常数。做时间重参数化 $t'=\tfrac12\int_0^t\beta_s\,\mathrm{d}s$ 后，上界化简为 $\frac{e^{-t'}}{1-e^{-2t'}}C\sim O(e^{-t'})$；VE-SDE 下（Theorem 2）上界为 $C/\sigma^2(t)$，同样随时间衰减。这条定理说明：前向加噪越久，条件信息流失越彻底，两个 score 越像。反过来在反向采样里就意味着——高 $t$（接近纯噪声）时两者几乎重合、低 $t$（接近数据）时差距最大。这正好解释了为什么固定 $\omega$ 必然次优：它对所有时间步一视同仁，高 $t$ 时过度引导破坏结构、低 $t$ 时引导又不够拉不回条件流形。论文还用反向采样中实测的 score MSE 与余弦相似度（Figure 1）验证了这条上界确实成立

**2. Harnack 型 PDF 不等式：补上 $t\to 0$ 临界区的视角**

score 差距上界在 $t\to 0$ 处会发散（分母含 $\sigma^2(t)\to 0$），既不能直接当权重用，而且近 $t=0$ 时 score 本身就难估计。为了说清这块"临界区"到底发生了什么，本文额外给出 PDF 自身的 Harnack 型不等式。VP-SDE 下（Theorem 3）对 $0<s_1<s_2$、任意 $\alpha>1$ 有

$$p(x_1,t(s_1))\le p(x_2,t(s_2))\Big(\frac{s_2}{s_1}\Big)^{\frac{m\alpha}{2}}\exp\!\Big(\frac{\alpha^2\|x_1-x_2\|^2}{4(s_2-s_1)}+\frac{\|x_2\|^2-\|x_1\|^2}{2}\Big),$$

VE-SDE 下（Theorem 4）形式类似。固定 $x_2,s_2$ 后可以看出：$s_1$ 越小（越靠近初始时刻）、$x_1$ 离 $x_2$ 越远，$p(x_1,t(s_1))$ 的上界越大、越难控制。这说明早期（$t\to 0$）PDF 的幅度和多样性都急剧放大，不同初始条件之间的差异被放大到最大。和上一个设计互补：score MSE 告诉你"差距随时间指数衰减"，Harnack 不等式告诉你"差距最大、最不可控的地方恰恰是 $t\to 0$"——因此那里必须给最强的引导信号才能精确收敛到目标条件分布

**3. 指数衰减控制函数：让引导强度对齐扩散动力学**

既然 score 差距在反向过程中近似按 $e^{-t}$ 增长（$t:T\to0$），理想的引导调度就该是同样形状。本文把固定 $\omega$ 替换为

$$\omega(t)=\omega_0\exp\!\Big(\lambda\Big(1-\frac{t}{t_{\max}}\Big)\Big),$$

其中 $t_{\max}$ 是前向过程的最大扩散时间，$\lambda>0$ 控制增长速率：权重从 $t=t_{\max}$ 时的 $\omega_0$ 增长到 $t=0$ 时的 $\omega_0 e^{\lambda}$，正好捕捉理论证明的指数增长趋势。采样时每一步把标准 CFG 更新换成 $\hat\epsilon_c^\omega(x_t)=\hat\epsilon_\varnothing(x_t)+\omega(t)[\hat\epsilon_c(x_t)-\hat\epsilon_\varnothing(x_t)]$ 即可。这个形式有几个好处：与理论/实测的指数衰减自洽；指数函数连续可微，比阶梯或分段调度数值更稳定；只引入 $\omega_0$（最大引导强度，等同标准 CFG）和 $\lambda$（衰减速率）两个可解释超参，$\lambda$ 直接控制保真↔多样的权衡。更妙的是，该框架还能反过来解释 Interval Guidance：早期 $t$ 高时条件/无条件 score 差距可忽略，只需用条件网络，于是 [25] 的"区间内 $\omega_0>1$、区间外退回 1"分段调度恰好是本框架的一个特例；两者还能叠加，把引导只施加在最有效的区间从而省下计算

### 损失函数 / 训练策略
无训练、无额外损失。C$^2$FG 是纯推理期的即插即用方法，只在采样循环里把常数 $\omega$ 换成 $\omega(t)$，调的是 $\omega_0$ 与 $\lambda$ 两个超参，对各类预训练扩散权重直接生效。

## 实验关键数据

### 主实验
ImageNet 256×256 类条件生成（50k 样本，250 步），与各 backbone 的固定权重 baseline 对比（↓越低越好 / ↑越高越好）：

| 模型 / 采样器 | FID↓ | IS↑ | sFID↓ | Prec↑ | Rec↑ |
|--------------|------|-----|-------|-------|-------|
| DiT-XL/2 ($\omega=1.5$, ODE) | 2.29 | 276.8 | 4.6 | 0.83 | 0.57 |
| DiT-XL/2 + Rectified Diffusion | 2.13 | / | / | 0.83 | 0.58 |
| **DiT-XL/2 + Ours** ($\omega_0{=}1,\lambda{=}\ln2$, ODE) | **2.07** | **291.5** | 4.6 | 0.83 | 0.59 |
| SiT-XL/2 (REPA) ($\omega=1.35$, SDE) | 1.80 | 284.0 | 4.5 | 0.81 | 0.61 |
| **SiT-XL/2 (REPA) + Ours** ($\omega_0{=}1,\lambda{=}1$, SDE) | **1.51** | **315.0** | 4.6 | 0.80 | 0.62 |
| SiT-XL/2 (REPA, Interval) ($\omega=1.8$, SDE) | 1.42 | 305.7 | 4.7 | 0.80 | 0.65 |
| **SiT (REPA, Interval) + Ours** ($\omega_0{=}1.8,\lambda{=}0.03$, SDE) | **1.41** | **308.0** | 4.7 | 0.80 | 0.65 |

ImageNet 512×512（10k 样本，100 步，SDE）：DiT-XL/2 从 FID 6.81 / IS 229.5 降到 **6.54 / 280.9**，sFID 20.0→19.7，验证高分辨率同样有效。

其他框架与数据集（Table 2）：

| 设置 | 指标 | Baseline | + Ours |
|------|------|----------|--------|
| MS-COCO, U-ViT ($\omega=2$) | FID↓ | 5.37 | **5.28** |
| MS-COCO, Stable Diffusion 1.5 ($\omega=5$) | CLIP↑ | 31.8 | **31.9** |
| ImageNet-64, EDM2-S + autoguidance ($\omega=1.7$) | FID↓ | 1.04 | **1.03** |

值得注意的是 EDM2-S+autoguidance 的 FID 1.04 已是像素空间扩散近饱和的极强 baseline，C$^2$FG 仍能再降一点到 1.03，说明它确实即插即用、与 autoguidance 正交。

### 消融实验
采样器鲁棒性（SiT-XL/2 REPA，ImageNet 256×256，更少步数，$\omega_0{=}1.7,\lambda{=}0.15$）：

| 配置 | FID↓ | sFID↓ | Prec↑ | Rec↑ |
|------|------|-------|-------|-------|
| 50 步 SDE baseline ($\omega=1.8$) | 3.36 | 4.5 | 0.86 | 0.54 |
| 50 步 SDE + Ours | **3.20** | 4.6 | 0.86 | 0.54 |
| 50 步 ODE baseline | 3.46 | 4.5 | 0.86 | 0.54 |
| 50 步 ODE + Ours | **3.25** | 4.4 | 0.86 | 0.55 |
| 20 步 ODE baseline | 3.29 | 4.6 | 0.85 | 0.54 |
| 20 步 ODE + Ours | **3.10** | 4.5 | 0.85 | 0.54 |

### 关键发现
- **越是强 baseline 越能看出框架的"原理性"价值**：在已经很难提升的 SiT-XL/2 (REPA) 上，SDE 采样 FID 从 1.80 直接降到 1.51（IS 284→315），是所有设置里增益最大的；而在已经动态化的 Interval Guidance 上叠加只微降（1.42→1.41），说明 C$^2$FG 抓的是和 interval 同源的规律。
- **步数越少增益越明显**：20 步 ODE 上 FID 3.29→3.10，比 50 步的增益更突出，说明把引导集中到"最该引导"的低 $t$ 区在算力紧张时更划算。
- **几乎不损其他指标**：FID/IS 普遍变好的同时，sFID、Precision、Recall 基本持平甚至略升，说明指数调度没有用牺牲多样性来换保真度。

## 亮点与洞察
- **把启发式调参变成可证明的定理**：最"啊哈"的地方在于先证明 score 差距按 $e^{-t}$ 衰减，再让权重 $\omega(t)$ 取一模一样的指数形状——曲线形状不是调出来的，而是从扩散动力学推出来的，这让方法的可解释性远超同类。
- **两套理论互补成闭环**：score MSE 上界负责"长时趋势"（差距随时间指数衰减），Harnack 不等式负责"短时临界区"（$t\to0$ 最不可控、最需强引导），两者各覆盖一段时间尺度，合起来才完整支撑指数权重的设计。
- **能反向解释并吸收已有方法**：把 Interval Guidance 解释成自己框架在"早期 score 差距可忽略"下的特例，这种"我能解释你"的姿态既是理论自信，也带来工程收益（叠加后省算力）。
- **可迁移性强**："先刻画某个量随时间/层数的演化规律、再让超参取同样形状"这个思路，可以迁移到任何带时变/步进调度的生成任务（噪声调度、温度调度、step-aware LoRA 等）。

## 局限与展望
- **理论与实现之间有一道工程缝**：score 差距上界在 $t\to0$ 处发散，作者直接"忽略这一段"并用一个非奇异的指数函数外推近似，定理给的是上界形状而非精确权重，最终 $\omega_0,\lambda$ 仍要按经验调（⚠️ 这点作者坦承）。
- **超参仍需按任务/框架重调**：不同 backbone 的最优 $(\omega_0,\lambda)$ 差异不小（如 SD15 用 $\lambda=0.2$、SiT-SDE 用 $\lambda=1$），方法把"调一条曲线"简化成"调两个数"，但并未完全免调。
- **在已动态化的强 baseline 上增益边际递减**：叠加到 Interval Guidance 上往往只降 0.01～0.02 FID，绝对收益有限。
- **评测集中在图像类条件生成**：虽宣称可推广到语音、3D 等条件扩散，但论文实验只覆盖 ImageNet/MS-COCO 图像，其他模态的有效性仍待验证。

## 相关工作与启发
- **vs 固定权重 CFG**：标准 CFG 全程用常数 $\omega$，对所有时间步一视同仁；本文证明这与扩散动力学相悖（高 $t$ 过度引导、低 $t$ 引导不足），用 $\omega(t)$ 对齐 score 差距，理论上更优。
- **vs Interval Guidance [25]**：它把引导限制在某个噪声区间内（区间内 $\omega_0$、区间外退回 1），是分段常数；本文证明它是自身指数框架的特例，二者可叠加，叠加后还能省算力。
- **vs $\beta$-CFG / RAAG 等动态调度**：同样让引导随时间变，但它们靠启发式/经验分布拍曲线，缺理论依据；本文从 SDE score 差距上界推出指数形状，是 principled 的。
- **vs autoguidance [23] / CFG++ [10] / FDG [40]**：这些从频率、数据流形等不同角度改进引导，与本文的时间维度正交；实验中 C$^2$FG 叠加在 autoguidance 之上仍能再降 FID，验证了正交性。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次对 CFG 的条件/无条件 score 差距给出严格上界，并据此原理性地设计时变权重，理论深度在同类动态 CFG 中突出。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 DiT/SiT/U-ViT/SD/EDM2 多框架、SDE/ODE 多采样器、多步数，但增益在强 baseline 上偏小、模态局限于图像。
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰、定理与方法衔接自然；但理论上界到工程权重之间的近似处理略显仓促。
- 价值: ⭐⭐⭐⭐⭐ training-free、即插即用、与现有策略正交，能直接给几乎所有条件扩散框架带来"白拿"的提升，落地价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance](cfg-ctrl_control-based_classifier-free_diffusion_guidance.md)
- [\[AAAI 2026\] DICE: Distilling Classifier-Free Guidance into Text Embeddings](../../AAAI2026/image_generation/dice_distilling_classifier-free_guidance_into_text_embedding.md)
- [\[AAAI 2026\] Studying Classifier(-Free) Guidance From A Classifier-Centric Perspective](../../AAAI2026/image_generation/studying_classifier-free_guidance_from_a_classifier-centric_perspective.md)
- [\[ICLR 2026\] PolyGraph Discrepancy: a classifier-based metric for graph generation](../../ICLR2026/image_generation/polygraph_discrepancy_a_classifier-based_metric_for_graph_generation.md)
- [\[CVPR 2026\] FG-Portrait: 3D Flow Guided Editable Portrait Animation](fg-portrait_3d_flow_guided_editable_portrait_animation.md)

</div>

<!-- RELATED:END -->
