---
title: >-
  [论文解读] Enhancing Membership Inference Attacks on Diffusion Models from a Frequency-Domain Perspective
description: >-
  [ICML 2026][图像生成][成员推断攻击] 本文从频域视角分析了扩散模型成员推断攻击（MIA）的失败模式，指出高频内容会同时放大 member 和 hold-out 样本得分的标准差从而稀释成员优势，提出一个无需训练、零额外推理代价的"高频滤波器"模块，只需在计算重建误差前对预测图与目标图做相同的 FFT 低通处理，就能把 Naive/SecMI/PIA 等主流 MIA 在 DDIM、Stable Diffusion 上的 ASR/AUC/TPR@1%FPR 普遍拉高 4–11 个百分点（个别场景 TPR@1%FPR 直接从 6% 跃到 41%）。
tags:
  - "ICML 2026"
  - "图像生成"
  - "成员推断攻击"
  - "扩散模型"
  - "频域分析"
  - "高频缺陷"
  - "即插即用滤波器"
---

# Enhancing Membership Inference Attacks on Diffusion Models from a Frequency-Domain Perspective

**会议**: ICML 2026  
**arXiv**: [2505.20955](https://arxiv.org/abs/2505.20955)  
**代码**: https://github.com/poetic2/FreMIA  
**领域**: AI安全 / 隐私攻击（成员推断 / 扩散模型）  
**关键词**: 成员推断攻击, 扩散模型, 频域分析, 高频缺陷, 即插即用滤波器

## 一句话总结
本文从频域视角分析了扩散模型成员推断攻击（MIA）的失败模式，指出高频内容会同时放大 member 和 hold-out 样本得分的标准差从而稀释成员优势，提出一个无需训练、零额外推理代价的"高频滤波器"模块，只需在计算重建误差前对预测图与目标图做相同的 FFT 低通处理，就能把 Naive/SecMI/PIA 等主流 MIA 在 DDIM、Stable Diffusion 上的 ASR/AUC/TPR@1%FPR 普遍拉高 4–11 个百分点（个别场景 TPR@1%FPR 直接从 6% 跃到 41%）。

## 研究背景与动机

**领域现状**：扩散模型在图像生成上效果惊艳，但训练集"记忆"风险也随之放大，因此评估其训练数据隐私的成员推断攻击（MIA）成了热门方向。针对扩散模型的主流 MIA（Naive loss、SecMI、PIA、PIAN 等）属于"重建误差派"：给一张待测图 $x_i$，让模型在某个时间步 $t$ 上预测一份 $x_{i,t}$ 与目标 $x_{i,t}^{target}$，以两者距离 $\|x_{i,t}-x_{i,t}^{target}\|_q$ 作为成员分数，再卡阈值判定。

**现有痛点**：实证发现，这些 attack 在某些"看起来不难"的样本上系统性失败——**含高频内容多的 member 图常被误判为非成员，含高频内容少的 hold-out 图反而被误判为成员**。作者在 MS-COCO/Flickr/CIFAR-100/TINY-IN 上画散点（图 1）和失败样本统计（表 1）都验证了这一规律。

**核心矛盾**：扩散模型本身有"频率层次性"——先恢复低频整体结构，再补高频细节，而高频部分的恢复天然伴随更大的方差与不确定性（Yang et al. 2023; Falck et al. 2025 已从频谱与 SNR 角度证明）。但现有 MIA 用的是像素级误差，把高频带来的"模型自身随机性"和"是否在训练集"两个信号搅在一起，导致 member 与 hold-out 的分数分布同时被高频"拉宽"。

**本文目标**：（1）刻画高频内容如何系统性损害现有 MIA；（2）给出一个不动 attack 主体、不引入训练、不增加推理时间的通用增强模块；（3）从理论上证明这种增强为何有效。

**切入角度**：既然高频是"共同噪声源"，那就把它从误差里减掉。利用 Yeom et al. (2018) 的成员优势公式 $Adv^M(\mathcal{A})\propto \sigma_H/\sigma_M$（hold-out 分数标准差 / member 分数标准差，越大越好攻击），观察到高频带来的额外方差 $\sigma^{high}$ 对 member 和 hold-out 几乎同等贡献，会拉低 $\sigma_H/\sigma_M$ 这个比例（因为分母变大不止）。

**核心 idea**：在送入距离函数前，把待测图 $x_{i,t}$ 与目标图 $x_{i,t}^{target}$ 一起 FFT，对半径大于 $r_t$ 的高频区乘以衰减因子 $s$（默认 0），再 IFFT 回空间域，**用低频版的重建误差替代原始重建误差**，这一步对任何"重建误差派"MIA 都即插即用，称为 FreMIA。

## 方法详解

### 整体框架
本文要解决的是"重建误差派 MIA 在高频丰富的样本上系统性失灵"这一问题，做法是不碰攻击主体、只在误差计算前插一道对称低通滤波。作者先把 Naive/SecMI/PIA 等攻击统一成一个通用范式（公式 6）：$\mathcal{A}(x_i,\theta)=\mathbb{1}[\|x_{i,t}-x_{i,t}^{target}\|_q \le \tau]$，它们的差别仅在于如何拿到模型预测图 $x_{i,t}$ 与目标图 $x_{i,t}^{target}$（Naive 用一步加噪去噪损失、SecMI 用 DDIM 多步逆向、PIA 用近端初始化得确定性噪声预测）。在这个统一接口上，FreMIA 对两张图施加同一个频域低通滤波 $\mathcal{F}(\cdot)$，把判别量升级为（公式 11）$\mathcal{A}'(x_i,\theta)=\mathbb{1}[\|\mathcal{F}(x_{i,t})-\mathcal{F}(x_{i,t}^{target})\|_q \le \tau]$，整条流程只在距离计算前多了一次 FFT/IFFT，没有任何可学习参数。

### 关键设计

**1. MIA 通用范式形式化：让一层滤波改造对所有攻击同时生效**

不同攻击表面上做法各异，若逐个打补丁既繁琐又难以统一分析。作者在附录 B 证明，Naive/SecMI/PIA 等的判别量都能改写成"模型预测图 vs 目标图"的 $\ell_q$ 距离 $\|x_{i,t}-x_{i,t}^{target}\|_q$，区别仅在 $x_{i,t}^{target}$ 的构造（如 SecMI 取 DDIM 逆过程中间结果、PIA 取确定性初始噪声）。把所有攻击收敛到这一个表达式后，"在距离前插一层滤波"就成了对全体方法同时生效的单点修改，既保证了模块的即插即用性，也让后续基于成员分数标准差的理论分析能一次性覆盖该范式下的所有攻击。

**2. 对称高频滤波器 $\mathcal{F}$：把模型在高频上的随机性从误差里减掉**

针对"高频是假信号主要来源"这一痛点，$\mathcal{F}$ 在送进距离函数前抹掉两张图中频率半径 $r>r_t$ 的高频分量。具体做法是先对 $x_{i,t}$ 做 DFT 得 $\mathbf{X}=FFT(x_{i,t})$，乘上掩码 $\beta_{i,t}(r)=s$（当 $r>r_t$）否则 $1$，再 IFFT，得到 $\mathcal{F}(x_{i,t})=IFFT(FFT(x_{i,t})\odot\beta_{i,t}(r))$，实验中 $s=0$ 即硬截止低通。关键在于对 $x_{i,t}$ 和 $x_{i,t}^{target}$ 用同一个 mask，所以两张图被减掉的高频信号同分布、在相减时相消，距离里只剩"低频部分对训练数据的拟合差异"——正好是模型最稳定、最能反映 memorization 的频段。这种"对称低通"有两重依据：频谱上扩散模型先学低频再学高频、高频天然更不稳定；统计上失败样本里 member 的高频含量显著高于 hold-out（表 1），说明高频正是把 member 误判成非成员的噪声来源。

**3. 成员优势的频域分解与理论保证（Proposition 4.2）：证明去高频必然涨**

为给"去高频为何更强"一个可证明依据，作者把 member/hold-out 的总分数标准差按频段分解为 $\sigma_M^2=\sigma_M^{low\,2}+\sigma_M^{high\,2}$、$\sigma_H^2=\sigma_H^{low\,2}+\sigma_H^{high\,2}$。设低频部分 $\sigma_H^{low}-\sigma_M^{low}=\Delta$（hold-out 比 member 更分散，这是攻击能赢的本钱），高频部分 $\sigma_M^{high}=k\cdot\sigma_H^{high}$。论文证明当 $k\ge 1$（高频部分 member 方差不小于 hold-out）时必有 $\sigma_H'/\sigma_M' > \sigma_H/\sigma_M$；按成员优势公式 $Adv^M(\mathcal{A})\propto \sigma_H/\sigma_M$（公式 8），优势严格上升。直觉是高频对 member 和 hold-out 加同一份噪声方差，但分母（member）被加上去的相对比例更高，减掉后比值变好。这把一个工程 trick 升级为"常见条件下必然提升"的结论，也解释了为何跨数据集、跨 baseline 都稳赢——附录 D.2 实测 $k\ge 1$ 在标准设置下几乎总成立。

### 损失函数 / 训练策略
**无任何训练**。该模块只在 inference 时给已有 MIA 的距离计算加一道 FFT→mask→IFFT，复杂度 $O(N\log N)$，相对扩散模型的多步采样几乎可忽略，作者强调"negligible additional time overhead"。唯一超参是高频半径 $r_t$（随分辨率而定：CIFAR-100/TINY-IN 用 $r_t=2$，MS-COCO/Flickr 用 $r_t=5$），衰减因子 $s$ 默认取 0。

## 实验关键数据

### 主实验

DDIM 上三个数据集 × 三个 baseline（Naive / SecMI / PIA）加 +F（高频滤波）后的对比（表 2 摘录）：

| 数据集 / 方法 | ASR (基线 → +F) | AUC (基线 → +F) | TPR@1%FPR (基线 → +F) |
|---|---|---|---|
| STL10-U / SecMI | 81.14 → 86.51 | 87.39 → 91.39 | 11.11 → 14.63 |
| CIFAR-100 / SecMI | 80.56 → 88.09 | 87.21 → 93.74 | 16.50 → 24.32 |
| Tiny-IN / PIA | 80.87 → 89.12 | 86.30 → 93.23 | 14.66 → 32.91 |
| 三个数据集平均增量 | +5.4~+7.3 | +4.5~+6.4 | +4.5~+11.8 |

Fine-tuned Stable Diffusion 上（表 3）效果更夸张，尤其在小数据集 Pokémon / 大分辨率自然图上：

| 数据集 / 方法 | ASR (基线 → +F) | AUC (基线 → +F) | TPR@1%FPR (基线 → +F) |
|---|---|---|---|
| Pokémon / Naive | 79.50 → 87.88 | 86.97 → 94.14 | 6.49 → 41.25 |
| MS-COCO / Naive | 80.29 → 93.60 | 87.85 → 98.32 | 4.80 → 41.99 |
| Flickr / Naive | 79.29 → 90.90 | 86.14 → 96.82 | 16.59 → 67.60 |

TPR@1%FPR 是 MIA 最严苛的低 FPR 指标，能从 4.80% 跳到 41.99% 说明滤波器在"高置信度判定"这种实际更危险的工作点上提升尤其明显。

### 消融实验

| 配置 / 现象 | 表现 / 说明 |
|---|---|
| Baseline（不滤波） | 失败样本里 member 的高频含量比 hold-out 高约 0.07–0.18（表 1） |
| +F（对称低通） | 消除上述偏差，所有 baseline 在所有数据集上 ASR/AUC 单调上升 |
| 改变高频半径 $r_t$ | 在合理区间内不敏感，过小（保留太多）退化为基线，过大（滤掉太多）开始损失 memorization 信号 |
| 是否依赖正态性假设 | 命题 4.2 在正态下证明，附录 D.2 实测分布偏离正态时仍然有提升，说明结论稳健 |
| 时间开销 | FFT/IFFT 一次的代价相对 DDIM 多步采样 < 1%，作者称"negligible" |

### 关键发现
- **高频是"假信号"的主要来源**：失败案例统计（表 1）+ 散点图（图 1）+ 像素级距离可视化（附录 D.1）三组证据收敛到同一结论——重建误差里随高频含量飙升的部分跟 memorization 几乎无关，反而是噪声
- **滤波后小 FPR 区间收益最大**：TPR@1%FPR 平均提升远大于 ASR/AUC 提升，原因是低频信号更"干净"、阈值更易卡在高置信度区间；这意味着在真实"是否需要发起隐私调查"这种高门槛场景里收益最显著
- **跨架构/跨数据集普适**：DDIM unconditional + Stable Diffusion fine-tuned 都涨，CIFAR-100 像 STL10 这种低分辨率自然图、Pokémon 这种小样本、MS-COCO/Flickr 这种自然大图都涨，说明"高频缺陷"是扩散模型的内禀属性，不是某个 attack 或数据集的偶然现象

## 亮点与洞察
- **理论 + 工程的完美闭环**：把一个看似简单的"加个低通滤波"trick，配上 Yeom et al. 的成员优势公式做了正面证明（$\sigma_H/\sigma_M$ 必涨），这种"小工程改造 + 可证明保证"的研究范式在 MIA 这种偏经验的领域很有借鉴价值
- **频域视角统一了攻击与防御**：以前讨论扩散模型频域多在生成质量上（先低频后高频），本文把它移植到"隐私可识别性"上，提示我们任何依赖重建误差的隐私/安全分析（不只是 MIA，还有 attribution、unlearning verification）都可以照这套思路在频域上重做一遍
- **零成本即插即用**：不需要重新训练、不需要 shadow model、不需要调 attack 内部、推理几乎不变慢，这种工程友好度让它有望被快速集成进所有现有 MIA benchmark
- **"对称扰动 + 误差分解"是可迁移的设计思想**：把扰动对称地作用于两张被比较的图，让噪声项相消、信号项保留——这一招在其它"基于距离的判别器"（如 OOD 检测、对抗样本检测）上也可能直接复用

## 局限与展望
- **只覆盖"重建误差派"攻击**：基于梯度的白盒 attack（如 GSA）、基于似然/token 概率的 attack（自回归生成模型常用）不在范式内，需要重新推导
- **超参 $r_t$ 是数据集相关的**：论文按经验给了 CIFAR/TINY-IN 用 2、COCO/Flickr 用 5，但缺乏自动选择策略；对未知分辨率/未知数据集需要小规模 sweep
- **理论假设较强**：命题 4.2 在正态分布 + $k\ge 1$ 下严格成立，作者用附录实测撑住了，但极端 outlier 数据集（如医学高对比度图）能否仍满足值得追加检验
- **防御方未触及**：本文是 attack 增强，自然引出反向问题：模型方能否通过"训练时主动平衡高低频学习"或"输出时在高频上做差分隐私加噪"来抵抗这类增强后的 MIA
- **多模态条件扩散与视频扩散未充分实验**：Stable Diffusion 只在三个数据集 fine-tune 验证，视频扩散、3D 扩散下高频规律可能不同

## 相关工作与启发
- **vs SecMI (Duan et al., 2023)**：SecMI 用 DDIM 多步逆向得到中间预测作为 $x_{i,t}^{target}$，本文把它作为基线之一并直接套用 $\mathcal{F}$，所有指标普涨 → 表明 SecMI 的失败模式确实来自高频不稳定，而不是逆向过程本身
- **vs PIA (Kong et al., 2023)**：PIA 用近端初始化拿到确定性噪声，强调"消去采样随机性"；本文展示即使消去采样随机性，**图像本身的高频依然带来 attack 不确定性**，两类不确定性是正交的，可以叠加缓解
- **vs LiRA (Carlini et al., 2022; 2023)**：LiRA 通过训练大量 shadow model 估计似然比，对扩散模型计算代价高到不实用；FreMIA 提供了"零额外训练成本"的替代路径，但当然不如 LiRA 在理论最优意义上紧
- **vs Frequency-aware Generation (Yang et al., 2023; Falck et al., 2025)**：这些工作从生成质量角度刻画频率层次，本文借用其结论来解释 attack 失败，**示范了"生成机制研究 → 隐私分析"这条技术迁移路径**，启示后续可以反过来用 MIA 改进结果反哺生成模型设计（如自适应频率重加权训练）

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次把频域视角引入扩散模型 MIA，框架统一性强；改造本身简单但视角真新
- 实验充分度: ⭐⭐⭐⭐ 3 baseline × 6 数据集 × 2 架构覆盖较广，附录补了正态性、$k$ 取值、可视化等论证；缺少视频/3D 扩散与防御方实验
- 写作质量: ⭐⭐⭐⭐ 范式形式化清晰、命题陈述和直觉解释配合好；个别符号（$h_M/h_H$）在正文未严格定义略影响阅读
- 价值: ⭐⭐⭐⭐⭐ 即插即用、零额外成本、跨方法全涨、TPR@1%FPR 在 SD 上有数量级提升，几乎肯定会成为后续扩散模型 MIA benchmark 的标配

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] FouriScale: A Frequency Perspective on Training-Free High-Resolution Image Synthesis](../../ECCV2024/image_generation/fouriscale_a_frequency_perspective_on_training-free_high-resolution_image_synthe.md)
- [\[AAAI 2026\] UNSEEN: Enhancing Dataset Pruning from a Generalization Perspective](../../AAAI2026/image_generation/unseen_enhancing_dataset_pruning_from_a_generalization_perspective.md)
- [\[ICML 2026\] Balancing Fidelity and Diversity in Diffusion Models via Symmetric Attention Decomposition: Hopfield Perspective](balancing_fidelity_and_diversity_in_diffusion_models_via_symmetric_attention_dec.md)
- [\[ICLR 2026\] Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective](../../ICLR2026/image_generation/continual_unlearning_for_text-to-image_diffusion_models_a_regularization_perspec.md)
- [\[ICML 2026\] Stable Velocity: A Variance Perspective on Flow Matching](stable_velocity_a_variance_perspective_on_flow_matching.md)

</div>

<!-- RELATED:END -->
