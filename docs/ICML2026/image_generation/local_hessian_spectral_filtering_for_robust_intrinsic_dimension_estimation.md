---
title: >-
  [论文解读] Local Hessian Spectral Filtering for Robust Intrinsic Dimension Estimation
description: >-
  [ICML 2026][图像生成][局部内禀维度] 本文提出 LHSD，把 score 模型的对数密度 Hessian 做一个 Hill 型谱滤波只保留近零特征值来数切空间维数，再用 Stochastic Lanczos Quadrature 把 $\mathcal{O}(D^3)$ 的代价压到 $\math…
tags:
  - "ICML 2026"
  - "图像生成"
  - "局部内禀维度"
  - "Hessian 谱滤波"
  - "扩散模型"
  - "Stochastic Lanczos Quadrature"
  - "记忆化检测"
---

# Local Hessian Spectral Filtering for Robust Intrinsic Dimension Estimation

**会议**: ICML 2026  
**arXiv**: [2605.01221](https://arxiv.org/abs/2605.01221)  
**代码**: 无  
**领域**: 扩散模型 / 流形学习 / 内禀维度估计  
**关键词**: 局部内禀维度, Hessian 谱滤波, 扩散模型, Stochastic Lanczos Quadrature, 记忆化检测

## 一句话总结
本文提出 LHSD，把 score 模型的对数密度 Hessian 做一个 Hill 型谱滤波只保留近零特征值来数切空间维数，再用 Stochastic Lanczos Quadrature 把 $\mathcal{O}(D^3)$ 的代价压到 $\mathcal{O}(D)$，从而在 3072 维图像空间稳定估计局部内禀维度，并用于诊断扩散模型的训练样本记忆化。

## 研究背景与动机

**领域现状**：流形假设认为高维数据沿着低维流形分布，局部内禀维度（LID）描述了样本邻域的真实自由度，是分析泛化、检测异常/对抗样本以及最近用于检测生成模型“原样复现训练数据”这类记忆化现象的关键指标。早期的 LID 估计依赖 kNN（MLE / TwoNN / LPCA），近年则转向基于扩散模型 score 函数的几何方法，如 LIDL、FLIPD 和 NB。

**现有痛点**：kNN 类方法在高维空间被维度灾难压垮，邻域距离集中化导致估计严重偏差；FLIPD 这类基于 score divergence 的方法把 Hessian 所有特征值不加区分地求和，在高 co-dimension 的图像数据上，法向方向的曲率以 $1/\sigma(t)^2$ 量级发散，把真正的切向信号淹没；NB 通过 score 矩阵 SVD 估秩，但 $\mathcal{O}(D^3)$ 的复杂度让 3072 维图像几乎不可算。

**核心矛盾**：LID 本质上是切空间维数，要可靠估计就必须把切向与法向贡献**显式分离**；但 score / Hessian 这些量天然把两者混在一起，且任何显式构造 $D\times D$ Hessian 都会触发 $\mathcal{O}(D^3)$ 代价。

**本文目标**：(1) 设计一个对法向噪声鲁棒的 LID 估计器；(2) 把它做成线性复杂度可在 $D{>}1000$ 上跑；(3) 提供可视化诊断让超参选择有据可依而不是靠玄学。

**切入角度**：在小噪声 $\sigma(t)$ 下，log-density Hessian $H(\mathbf{x}) = -\nabla^2\log p_t(\mathbf{x})$ 的谱呈现“两堆”结构——切向 $\mathcal{O}(1)$ 的小特征值与法向 $\mathcal{O}(1/\sigma(t)^2)$ 的大特征值，中间天然存在 spectral gap。这意味着只要在 gap 处放一道“截止线”就能区分切向与法向。

**核心 idea**：用 Hill 型平滑滤波器 $f(\lambda)$ 把 Hessian 特征值压缩到 $[0,1]$，使切向 $\approx 1$、法向 $\approx 0$，于是 $\mathrm{tr}(f(H))$ 就是 LID；进一步用 SLQ 估迹避免构造 Hessian，把复杂度降到 $\mathcal{O}(D)$。

## 方法详解

### 整体框架
LHSD 要解决的是：在 3000+ 维图像空间里稳定估计一个样本的局部内禀维度（LID），而 LID 本质上就是流形切空间的维数。本文把这件事转化为"数 log-density Hessian $H(\mathbf{x})=-\nabla^2\log p_t(\mathbf{x})$ 里近零特征值的个数"——切向对应小特征值、法向对应大特征值，只要用一个谱滤波器把切向特征值压到 $\approx 1$、法向压到 $\approx 0$，再求迹就是 LID。整个估计在推断时完成，底下复用一个训好的 score 模型，并用 Stochastic Lanczos Quadrature（SLQ）在不显式构造 $D\times D$ Hessian 的前提下把迹算出来。

### 关键设计

**1. 切–法谱分离的 Hessian 滤波估计器：把"数维数"变成"过滤后求迹"**

直接数 Hessian 的零空间维数在数值上不可行，本文的突破口是 Hessian 谱本身的"两堆"结构。在流形附近用 tangent–normal 坐标对 $\log p_t$ 展开可得 $H(\mathbf{x}) = \Pi_\text{nor}(\mathbf{x})/\sigma(t)^2 + \mathcal{O}(1)$，即在小噪声 $\sigma(t)$ 下 Hessian 基本上就是法向投影矩阵除以 $\sigma(t)^2$。于是法向特征值 $\approx 1/\sigma(t)^2$、切向特征值 $\approx \mathcal{O}(1)$，二者中间天然出现一个明显的 spectral gap。

利用这个 gap，本文定义 $\text{LHSD}(\mathbf{x}) := \sum_i f(\lambda_i) = \mathrm{tr}(f(H(\mathbf{x})))$，滤波器取 Hill 型 $f(\lambda;\sigma(t)) = 1/(1+(|\lambda|/\kappa(t))^p)$，截止 $\kappa(t) := c/\sigma(t)^2$ 直接吸收掉法向曲率随噪声尺度的 $\sigma(t)^{-2}$ 缩放。相比 FLIPD 把所有特征值不加区分地累加（$\nabla\cdot \mathbf{s}_\theta$），这个滤波器在切向响应 $\approx 1$、在法向响应 $\approx 0$，等于把"magnitude 求和"改成"个数计数"，从根本上消除了法向 magnitude 发散对估计的污染。选 Hill 型而非 sigmoid，是因为它有更平坦的 passband，更适配后面 SLQ 的低阶多项式逼近精度。

**2. 基于 transition mass 的可验证超参选择：让"截止落在 gap 里"看得见**

LHSD 能不能成立，全看截止 $\kappa(t)$ 是否恰好落进 spectral gap，以往这类 cutoff 选择往往只能靠合成数据试错。本文把"截止是否在 gap 中"这一几何条件量化成一条一维曲线：固定 $c, p$、扫描 $t$，定义 transition mass $M(t) := \frac{1}{D}\sum_i \mathbb{I}(\lambda_i(t) \in [\kappa(t) - \delta, \kappa(t) + \delta])$，计数落在截止边界附近的特征值比例。

当 $M(t) \approx 0$ 且位置处于两个特征值"峰"中间时，截止线正好穿过 gap，被记为 safe zone；若落进峰内或越过两个峰之外，$M(t)$ 都会把它揭穿。论文图 3 显示 safe zone 表现为 $M(t)$ 的低谷，让超参选择从"盲选"变成"看图选"，这是把玄学操作落到实证依据上的关键一步。

**3. SLQ 加速：把 $\mathcal{O}(D^3)$ 压到 $\mathcal{O}(D)$**

要在 3072 维上算 $\mathrm{tr}(f(H))$，绝不能显式构造完整的 $D\times D$ Hessian——传统 NB 用 SVD 估秩走 $\mathcal{O}(D^3)$ 对图像直接放弃。本文走 SLQ 路径：先用 Hutchinson 估计 $\mathrm{tr}(f(H)) \approx \mathbb{E}_\mathbf{v}[\mathbf{v}^\top f(H) \mathbf{v}]$，用 $K$ 个 Rademacher 探针向量做蒙特卡洛平均；每个二次型 $\mathbf{v}^\top f(H)\mathbf{v}$ 又通过 $m$ 步 Lanczos 把 $H$ 在 Krylov 子空间上三对角化成 $T$，再用 $T$ 的特征对 $(\tilde\lambda_j, \tau_j)$ 做高斯求积近似 $\mathbf{v}^\top f(H)\mathbf{v} \approx \|\mathbf{v}\|^2 \sum_{j=1}^m \tau_j^2 f(\tilde\lambda_j)$。

整条链路里唯一接触 Hessian 的是 Hessian-向量乘，借自动微分实现 $H\mathbf{v} = -\nabla(\mathbf{s}_\theta(\mathbf{x})^\top \mathbf{v})$，每次只需一次反传。因为 Krylov 子空间只看一个低秩近似，论文实验显示 $m=5$ 步就够，最终复杂度线性于 $D$，把高维图像上的 LID 估计从"理论上可做"变成"真能跑"。

### 损失函数 / 训练策略
LHSD 是**纯推断时**算法，不引入任何训练参数。它假设底下的 score 模型 $\mathbf{s}_\theta$ 已用标准 denoising score matching 训练好；超参 $c$（截止位置）、$p$（滤波器陡度，论文用 $p=4$）、$\delta$（transition mass 边距，$\delta = 0.2$）、$K$（Rademacher 探针数）、$m$（Lanczos 步数）通过 transition mass 曲线诊断设定。

## 实验关键数据

### 主实验
合成流形数据（线性子空间 $\mathcal{L}$ 与 Funnel $\mathcal{F}$）下的 MAE（越低越好）：

| 维度 $D$ | 数据集 | FLIPD | NB | LHSD ($m{=}5$) |
|---|---|---|---|---|
| 1024 | $\mathcal{L}^{10+80+200}$ | 86.03 | 528.95 | **3.47** |
| 1024 | $\mathcal{F}^{10+80+200}$ | 373.80 | 937.82 | **6.90** |
| 3072 | $\mathcal{L}^{900}$ | 7.78 | 2171.00 | **11.53** |
| 3072 | $\mathcal{F}^{900}$ | 782.50 | 2171.00 | **18.79** |
| 3072 | $\mathcal{L}^{10+80+200}$ | 256.40 | 2949.10 | **4.70** |
| — | 平均（9 个设定） | 307.4 | 1319.9 | **6.6** |

差距非常显著：在 $D=3072$ 的 Funnel 流形上，FLIPD 的 MAE 是 782，NB 是 2171，LHSD 是 18.79，整整两个数量级。

### 消融实验

| 配置 | 平均 MAE | 说明 |
|---|---|---|
| LHSD ($m{=}2$) | 20.7 | Lanczos 步数太少，谱近似粗糙 |
| LHSD ($m{=}5$) | **6.6** | 默认配置，已经足够 |
| FLIPD（$\nabla\cdot\mathbf{s}_\theta + \|\mathbf{s}_\theta\|^2$，无滤波） | 307.4 | 不分切/法直接相加，法向噪声爆炸 |
| NB（SVD 估秩） | 1319.9 | 不仅慢且高维下完全崩 |

### 关键发现
- 高维设置（$D \geq 1024$）下，所有基线方法（kNN 类 / FLIPD / NB）在至少一个数据集上误差量级达到 $10^2 \sim 10^3$；LHSD 在每个设定都稳定在个位到几十的量级，说明“显式过滤法向”不是锦上添花而是必需。
- Lanczos 只要 $m=5$ 步就足够，这对应一个很小的三对角矩阵，验证 SLQ 的高效性。
- transition mass 诊断曲线（图 3）揭示 $t$ 选错时（如 $t=0.22$）截止线会跑过两个特征值峰之外，单看 $M(t)\approx 0$ 还不够，必须结合峰位置共同判断，这是论文很务实的实操点。

## 亮点与洞察
- **把 LID 从 magnitude 求和改成 indicator 计数**：FLIPD 之所以爆，本质是用一个发散的量去逼近一个有界的整数（维数）；LHSD 通过把谱压缩到 $[0,1]$ 再求和，把估计目标恢复成“有界、与噪声尺度解耦”的形式，这种“先归一化再聚合”的思路可以迁移到任何依赖 Hessian 谱的下游任务（如 sharpness、模型几何）。
- **截止 $\kappa(t) := c/\sigma(t)^2$ 是一个漂亮的“自适应归一化”**：它把法向曲率的噪声依赖性 $\sigma(t)^{-2}$ 直接吸收进截止线，让滤波器在不同 $t$ 下行为一致——这种“让常数自适应”的设计在多尺度噪声场景里很有借鉴价值。
- **SLQ + Hill 滤波的工程联姻**：Hill 滤波光滑、低多项式阶就能拟合，恰好契合 SLQ 用低阶高斯求积的要求；如果换成阶跃式 hard cutoff，SLQ 的多项式逼近就会失效，这个细节解释了为什么必须用平滑滤波器。

## 局限与展望
- LHSD 假设 spectral gap 真的存在：当 $\sigma(t)$ 偏大时 Hessian 谱会塌陷成各向同性（论文附录 E 讨论），此时方法失效。但作者承认还没给出对“gap 是否存在”的自动检测，只能借 transition mass 间接判断。
- 估计精度依赖底下 score 模型 $\mathbf{s}_\theta$ 的质量，未训练充分的扩散模型 Hessian 谱会被网络噪声污染。这个隐藏前提在实际应用（如评估别人的扩散 checkpoint）中可能成为坑。
- 实验主要在 UNet 类小到中等扩散模型上做，对超大模型（如 SD/DALL·E 级）的 LID 计算开销尚未充分验证，尽管复杂度是线性的。
- 后续可探索把 Hill 滤波换成可学习的谱滤波器，端到端用记忆化检测/异常检测 loss 监督，让 LID 估计直接为下游任务对齐。

## 相关工作与启发
- **vs FLIPD**：FLIPD 用 $\sigma(t)^2(\nabla\cdot \mathbf{s}_\theta + \|\mathbf{s}_\theta\|^2)$，把所有 Hessian 特征值不加区分相加；LHSD 用滤波器只数切向特征值，在高维上能差两个数量级。
- **vs NB（Normal Bundle）**：NB 通过堆叠多个 noisy score 的 SVD 来估法空间秩，$\mathcal{O}(D^3)$ 算不动；LHSD 走二阶 Hessian + SLQ 路径，复杂度线性，能上 3072 维。
- **vs kNN 估计器（MLE / TwoNN / LPCA / ESS）**：传统估计器依赖样本邻域距离，受维度灾难拖累；扩散模型 score 取代了邻域搜索，但只有 LHSD 把切–法分离贯彻到底。
- **启发**：Hessian 谱分析在深度学习里早已有 bulk + outliers 这种刻板印象，本文把它用在生成模型几何上是个聪明迁移；可以推广到把 score 模型当作 implicit 几何探针的更多任务（如曲率、reach、流形拓扑特征数）。

## 评分
- 新颖性: ⭐⭐⭐⭐ 切–法谱分离 + Hill 滤波 + SLQ 三件套的组合自洽且新颖
- 实验充分度: ⭐⭐⭐⭐ 合成 + 真实数据，覆盖低/中/高维与多种流形，但缺少在实际 SOTA 扩散模型上的大规模检测案例
- 写作质量: ⭐⭐⭐⭐ 几何动机清晰，公式推导扎实，transition mass 的可视化诊断尤其加分
- 价值: ⭐⭐⭐⭐ 让高维 LID 估计第一次在 3000+ 维真正可算，对扩散模型记忆化诊断、OOD 检测有直接落地价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] A Connection Between Score Matching and Local Intrinsic Dimension](../../NeurIPS2025/image_generation/a_connection_between_score_matching_and_local_intrinsic_dimension.md)
- [\[ICML 2026\] Order within Chaos: Capturing Intrinsic Energy Anomalies for AI-Manipulated Image Forgery Localization](order_within_chaos_capturing_intrinsic_energy_anomalies_for_ai-manipulated_image.md)
- [\[ICML 2026\] Caracal: Causal Architecture via Spectral Mixing](caracal_causal_architecture_via_spectral_mixing.md)
- [\[ICML 2026\] Spectral Guidance for Flexible and Efficient Control of Diffusion Models](spectral_guidance_for_flexible_and_efficient_control_of_diffusion_models.md)
- [\[ICML 2026\] DiScoFormer: Plug-In Density and Score Estimation with Transformers](discoformer_plug-in_density_and_score_estimation_with_transformers.md)

</div>

<!-- RELATED:END -->
