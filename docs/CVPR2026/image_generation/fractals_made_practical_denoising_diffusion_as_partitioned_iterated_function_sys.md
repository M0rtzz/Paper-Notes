---
title: >-
  [论文解读] Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems
description: >-
  [CVPR 2026][图像生成][扩散模型] 证明了DDIM确定性反向链本质上是一个分区迭代函数系统(PIFS)，并从该框架推导出三个无需模型评估的可计算几何量，从第一性原理统一解释了扩散模型的双阶段去噪动力学、自注意力的有效性…
tags:
  - "CVPR 2026"
  - "图像生成"
  - "扩散模型"
  - "分区迭代函数系统(PIFS)"
  - "分形几何"
  - "Jacobian分析"
  - "Kaplan-Yorke维度"
---

# Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems

**会议**: CVPR 2026  
**arXiv**: [2603.13069](https://arxiv.org/abs/2603.13069)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 扩散模型, 分区迭代函数系统(PIFS), 分形几何, Jacobian分析, Kaplan-Yorke维度

## 一句话总结

证明了DDIM确定性反向链本质上是一个分区迭代函数系统(PIFS)，并从该框架推导出三个无需模型评估的可计算几何量，从第一性原理统一解释了扩散模型的双阶段去噪动力学、自注意力的有效性，以及四种经验设计选择（cosine schedule offset、分辨率相关logSNR偏移、Min-SNR损失加权、Align Your Steps采样）。

## 研究背景与动机

### 1. 领域现状
扩散模型通过顺序去噪过程生成高质量图像，理论基础是连续时间SDE或概率流ODE，已有全局 $\mathcal{W}_2$ 分布收敛保证。但连续视角将学习到的score网络视为黑盒。

### 2. 痛点
现有理论无法结构性地解释两个关键现象：(a) 为什么早期步骤组装全局空间上下文、后期步骤合成局部细节？(b) 为什么自注意力作为生成原语如此有效？诸多经验设计（cosine offset、Min-SNR加权等）缺乏统一的几何解释。

### 3. 核心矛盾
连续SDE/ODE框架提供了优雅的分布收敛保证，但无法揭示离散采样链如何在每一步中组装图像结构——理论优雅性与结构可解释性之间存在张力。

### 4. 要解决什么
- 为DDIM反向链的双阶段动力学提供结构性证明
- 解释自注意力在扩散模型中的几何角色
- 从统一框架推导出实用设计准则，解释现有经验技巧

### 5. 切入角度
将DDIM确定性反向链 $\Phi = \Phi_1 \circ \cdots \circ \Phi_T$ 视为分区迭代函数系统(PIFS)——这是分形图像压缩中的经典数学结构，天然适配处理局部自相似性。

### 6. 核心idea
每一步DDIM算子 $\Phi_t$ 的Jacobian可分解为对角块（patch内）和交叉块（patch间），其收缩/膨胀行为由三个仅依赖噪声调度和patch协方差的闭式常量完全刻画，无需运行模型即可分析去噪动力学。

## 方法详解

### 整体框架

这篇论文不训练新模型，而是给已经训练好的扩散模型套一层新的数学透镜：把DDIM的确定性反向链 $\Phi = \Phi_1 \circ \cdots \circ \Phi_T$ 整体看成一个分区迭代函数系统(PIFS)——分形图像压缩里用来刻画局部自相似性的经典结构。一旦接受这个视角，每一步去噪算子 $\Phi_t$ 的Jacobian就能被拆成patch内的对角块和patch间的交叉块，而这两块的收缩还是膨胀，完全由三个只依赖噪声调度和数据协方差的闭式常量决定，根本不用跑一次模型。

沿着这条主线，分析分四层展开：先建立单步算子的收缩条件，回答"什么时候图像结构会被压实、什么时候会被展开"(§3)；再用真实数据统计和网络的抑制行为，把整条采样链切成早期组装全局上下文、后期合成局部细节的两个阶段(§4)；接着用Lyapunov谱算出PIFS吸引子（即生成流形）的分形维度(§5)；最后反过来把这套几何语言变成实用准则，统一解释cosine offset、Min-SNR加权等四个原本各自为政的经验技巧(§6)。

### 关键设计

**1. 双重收缩条件 (EC) 与 (PC)：用一个阈值判断每步到底压不压得动结构。**

要分析采样链先得回答最基础的问题——单步算子 $\Phi_t(x) = \frac{\sqrt{\bar\alpha_{t-1}}}{\sqrt{\bar\alpha_t}} x + b_t \hat\varepsilon_\theta(x,t)$ 到底是把点拉近（收缩、组装结构）还是推远（膨胀、注入细节）。看它的Jacobian $J_x\Phi_t = \frac{\sqrt{\bar\alpha_{t-1}}}{\sqrt{\bar\alpha_t}} I + b_t J_x\hat\varepsilon_\theta$ 就清楚了：第一项是尺度 $>1$ 的恒等缩放，天然膨胀；第二项因 $b_t<0$ 是score修正带来的收缩，谁占上风取决于score Jacobian的谱。论文据此给出两套判据。全局的欧氏收缩(EC)定义了一个只依赖噪声调度的收缩阈值 $L_t^* = \frac{\sqrt{\bar\alpha_{t-1}/\bar\alpha_t} - 1}{|b_t|}$，$L_t^*$ 越小这一步越容易收缩。但自然图像的自相似性是局部的而非全局的，所以真正关键的是patch级的块-最大范数收缩(PC)：把Jacobian按patch分成对角块 $\kappa_t^{\mathrm{diag}}$ 和交叉块 $\delta_t^{\mathrm{cross}}$，只要 $\kappa_t^{\mathrm{diag}} + \delta_t^{\mathrm{cross}} < 1$ 就保证patch层面的收缩。这正是经典PIFS比全局IFS强的地方——它允许各patch按自己的节奏收缩，而这恰好对应后面观察到的"patch逐个解锁"现象。

**2. 方向性抑制场与分层释放：解释为什么早期对角块卡在 $\approx 1$、后期才逐个膨胀。**

纯高斯基线给出的预测是"灾难性"的：对角块谱范数 $f_t(\lambda_k)$ 在所有CIFAR-10 patch上都 $>1$，意味着每步都该立刻膨胀、细节早早炸开，这和实测的双阶段行为对不上。论文的解释是训练把网络偏离了高斯——它学到一个方向性抑制场 $S_{k,t}(x) = |b_t| \langle v_k^{(1)}, [\nabla_x \Delta_t(x)]_{kk} v_k^{(1)} \rangle$，其中 $\Delta_t$ 是score网络相对高斯score的非高斯修正、$v_k^{(1)}$ 是该patch的主方向。这个抑制场把有效Rayleigh商硬压到1以下，于是对角块在早期被钉在 $\approx 1$ 而非膨胀。更妙的是抑制不是同时撤掉的：分层释放定理（Stratified Crossover, Thm 22）证明在边际单调性条件(MM)下，低方差patch会先释放抑制、高方差patch后释放，形成严格按方差排序的解锁次序。这就给出了Regime II里patch逐个"解锁"、细节由粗到细合成的结构性来源。

**3. 吸引子的Kaplan-Yorke维度公式：不跑模型就能预测生成流形的分形维度。**

既然每步的收缩/膨胀都有闭式刻画，整条链的吸引子几何也就能算出来。在高斯数据加块对角协方差的假设下，Lyapunov谱完全由各步对角膨胀函数 $f_t(\lambda)$ 决定；把所有步的贡献乘起来得到离散Moran方程 $\prod_t f_t(\lambda^{**}) = 1$，其解 $\lambda^{**}$ 就是区分膨胀与收缩方向的全局阈值——协方差特征值 $\lambda_k > \lambda^{**}$ 的方向才膨胀。由此得到Kaplan-Yorke维度的闭式公式：

$$d_{\mathrm{KY}} = N^+ + \frac{\sum_{k:\lambda_k > \lambda^{**}} n_k \Lambda(\lambda_k)}{|\Lambda_{k^*}^-|}$$

其中 $N^+$ 数膨胀方向、分式补上分维的小数部分。把上一条的抑制场代进去得到的修正版 $d_{\mathrm{KY}}^{\mathrm{eff}} \leq d_{\mathrm{KY}}$，说明训练学到的抑制只会让吸引子变"瘦"、不会变胖。这条公式把抽象的噪声调度选择直接翻译成了生成流形的几何尺寸，是后面推导设计准则的支点。

### 损失函数 / 训练策略

这套PIFS视角同样能落到训练目标上。论文证明去噪score matching(DSM)目标本质就是PIFS的collage误差最小化（Collage类比, Thm 12，差一个SNR加权），即训练在做的事情等价于把每个patch往它的自相似映射上对齐；进一步地 $L^2$–$\mathcal{W}_1$ Bridge（Thm 14）说明训练损失直接控制了采样分布到PIFS不动点的Wasserstein-1距离。顺着这条线还能反推出一个显式正则项 $\mathcal{L}_{\mathrm{PIFS}}(\theta) = \mathcal{L}(\theta) + \mu_{\mathrm{reg}} \sum_{t,k,j\neq k} \|[J_x\hat\varepsilon_\theta]_{kj}\|_F^2$（Thm 15），它惩罚交叉块范数、直接强制(PC)条件，且能用JVP/VJP高效估计，不必显式构造Jacobian。

## 实验关键数据

### 主实验：Block-Jacobian分解验证双阶段结构

在预训练DDPM CIFAR-10模型上，使用8×8 patch（$M=16$, $n_k=192$）和50步DDIM采样器验证理论预测。

| 训练步 $t$ | $\hat\kappa_t^{\mathrm{diag}}$ | $\hat\delta_t^{\mathrm{cross}}$ | 全局 $\hat\kappa_t$ | 阶段 |
|---|---|---|---|---|
| 980 | 1.0004 | 0.0007 | 1.0011 | 高噪声 |
| 800 | 1.0002 | 0.0008 | 1.0010 | 高噪声 |
| 600 | 1.0000 | 0.0853 | 1.0853 | Regime I |
| 400 | 1.0026 | 0.1273 | 1.1300 | Regime I |
| 220 | 1.0325 | 0.0768 | 1.1092 | Regime II |
| 20 | 1.2111 | 0.1858 | 1.3969 | 细节 |

**关键发现**：Regime I中全局膨胀完全由交叉patch耦合驱动（对角块 $\approx 1$）；Regime II中对角块开始膨胀，注意力局部化。

### 注意力熵与交叉patch耦合

| $t$ | 注意力熵 $H(A_t)$ (nats) | $\hat\delta_t^{\mathrm{cross}}$ | 阶段 |
|---|---|---|---|
| 980 | 4.963 | 0.00946 | 高噪声 |
| 560 | 4.662 | 0.09463 | Regime I |
| 160 | 4.541 | 0.42899 | Regime II |
| 20 | 4.063 | 2.06175 | 细节 |

$\hat\delta_t^{\mathrm{cross}}$ 从 $t=980$ 到 $t=20$ 增长218倍。Spearman $\rho(H, \hat\delta^{\mathrm{cross}}) = -1.000$（$p < 0.001$），耦合与熵完美反相关。

### 消融实验

**(PC)条件crossover验证**

| $t$ | 阶段 | 平均margin slack | 违反比例 |
|---|---|---|---|
| 700 | Regime I | $-0.003942$ | 16/16 |
| 200 | I/II过渡 | $-0.000304$ | 14/16 |
| 160 | Regime II | $+0.001382$ | 0/16 |
| 40 | Regime II | $+0.006412$ | 0/16 |

Crossover发生在 $t \in [160, 200]$，约40步窗口。Regime I全面违反(PC)，Regime II全面满足。

**分层释放的Spearman相关**

在crossover区间（$t=240,260$），$\rho(\hat\lambda_k, \hat\kappa_t^{\mathrm{diag}})$ 为负且显著（$p \leq 0.047$），确认低方差patch先释放。深度Regime II（$t=40$）回到正 $\rho = 0.771$（$p=0.001$），高斯排序恢复。

**抑制修正KY维度（CelebA-HQ实验）**

在google/ddpm-celebahq-256模型上，$\lambda_k \in [38.7, 231.7]$，高斯基线预测 $d_{\mathrm{KY}} = 12288$（全维膨胀），但抑制修正Moran阈值 $\lambda^{***} = 500 \gg \lambda_{\max}$，预测 $d_{\mathrm{KY}}^{\mathrm{eff}} = 0$。全部16个patch的预测Lyapunov指数与实测指数符号一致（100%）。

### 关键发现

1. **Score偏差缩放**：高噪声区域 $\|\Delta_t\|_2 = O(\sqrt{\bar\alpha_t})$，OLS拟合斜率 $0.95$（95% CI $[0.88, 1.02]$），与理论预测一致
2. **信息增益-KY维度正比性**：$\rho(\mathrm{IG}_t, |\Delta d_t|) \geq 0.9999$，比值的CV仅3.4%，几乎完美满足Cauchy-Schwarz等号条件
3. **噪声调度对比**：线性调度 $L_t^*$ 均衡性最好（CV 0.341），cosine调度信息增益均衡性更好（CV 0.867 vs 1.107）

## 亮点与洞察

1. **深刻的数学联系**：将扩散模型与分形图像压缩通过PIFS框架桥接，揭示了score matching = collage误差最小化这一精妙对应
2. **三个无模型常量**：$L_t^*$（收缩阈值）、$f_t(\lambda)$（对角膨胀函数）、$\lambda^{**}$（全局膨胀阈值）完全由调度和数据统计决定，构成通用设计语言
3. **解释四种经验设计的统一理论**：cosine offset提升最弱环节 $L_1^*$（4倍），Min-SNR实现KY维度增长均衡化，分辨率shift保持Moran比值不变，AYS将步骤集中在 $L_t^*$ 最小处
4. **自注意力的几何角色**：query token = range block，key/value token = domain block，$A_{kj}$ = 软域-范围配对；hard attention极限精确恢复经典PIFS结构，交叉耦合 $\delta_t^{\mathrm{cross}}$ 由注意力权重界定

## 局限与展望

1. **高斯patch假设**：核心分析依赖块对角高斯协方差假设，非高斯情况（如纹理丰富的数据）需更精细的抑制场建模
2. **PIFS正则化器未实验验证**：提出了 $\mathcal{L}_{\mathrm{PIFS}}$ 正则项但未进行端到端训练实验，实际训练效果待验证
3. **仅限DDIM确定性采样**：分析主要针对DDIM（概率流ODE），对DDPM随机采样的适用性未详细讨论
4. **注意力梯度的regime I界松散**：Thm 23中 $\|\nabla_x A_{k\ell}\|_{\mathrm{op}}$ 在Regime I是松的，更精确的query/key温度刻画留作未来工作
5. **skip connection的影响**：UNet架构的encoder skip connection超出注意力界的范围，$\delta_t^{\mathrm{cross,skip}}$ 在Regime II可能主导但未精确量化

## 相关工作与启发

- **分形图像压缩**（Jacquin 1992, Barnsley 1988）：PIFS和Collage定理是本文的数学基础，作者将经典编码理论重新解释为生成模型的分析工具
- **双阶段行为**（Raya & Ambrogioni 2023）：先前经验观察的两阶段现象，本文给出了结构性证明（收缩 vs 膨胀）
- **信息常数调度**（Kingma et al. 2021, Chen et al. 2023）：本文证明IG均衡等价于KY维度增长均衡（Thm 32），赋予信息理论准则以几何意义
- **Align Your Steps**（Sabour et al. 2024）：从KL散度上界优化步骤分配，本文从收缩margin给出互补推导，两者因 $\sqrt{v_t}$ 共同控制而一致

## 评分

⭐⭐⭐⭐⭐ 理论深度极高的工作，将分形几何与扩散模型完美对接，从第一性原理统一解释了多个孤立的经验技巧，数学推导严谨且实验验证全面（CIFAR-10 + CelebA-HQ），对扩散模型的理解和设计具有范式级启发价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Smoothing the Score Function for Generalization in Diffusion Models: An Optimization-based Explanation Framework](smoothing_the_score_function_for_generalization_in_diffusion_models.md)
- [\[CVPR 2026\] Reviving ConvNeXt for Efficient Convolutional Diffusion Models](reviving_convnext_for_efficient_convolutional_diffusion_models.md)
- [\[CVPR 2026\] Diffusion Mental Averages](diffusion_mental_averages.md)
- [\[CVPR 2026\] Learnability-Guided Diffusion for Dataset Distillation](learnability-guided_diffusion_for_dataset_distillation.md)
- [\[CVPR 2026\] Exploring Conditions for Diffusion Models in Robotic Control](exploring_conditions_for_diffusion_models_in_robotic_control.md)

</div>

<!-- RELATED:END -->
