---
title: >-
  [论文解读] Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems
description: >-
  [CVPR 2026][图像生成][扩散模型] 证明了DDIM确定性反向链本质上是一个分区迭代函数系统(PIFS)，并从该框架推导出三个无需模型评估的可计算几何量，从第一性原理统一解释了扩散模型的双阶段去噪动力学、自注意力的有效性，以及四种经验设计选择（cosine schedule offset、分辨率相关logSNR偏移、Min-SNR损失加权、Align Your Steps采样）。
tags:
  - CVPR 2026
  - 图像生成
  - 扩散模型
  - 分区迭代函数系统(PIFS)
  - 分形几何
  - Jacobian分析
  - Kaplan-Yorke维度
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

论文建立了DDIM → PIFS的数学映射，核心链路为：

1. **收缩结构分析**(§3)：推导单步DDIM算子的收缩条件
2. **双阶段结构分析**(§4)：从数据统计和架构属性计算常量，解释两个去噪阶段
3. **吸引子几何**(§5)：通过Lyapunov谱计算PIFS吸引子的分形维度
4. **实用设计准则**(§6)：从PIFS框架推导三个优化准则，统一解释四种经验设计

### 关键设计

#### 设计一：双重收缩条件 (EC) 和 (PC)

**功能**：为DDIM单步算子 $\Phi_t(x) = \frac{\sqrt{\bar\alpha_{t-1}}}{\sqrt{\bar\alpha_t}} x + b_t \hat\varepsilon_\theta(x,t)$ 建立两种收缩性条件。

**核心思路**：Jacobian $J_x\Phi_t = \frac{\sqrt{\bar\alpha_{t-1}}}{\sqrt{\bar\alpha_t}} I + b_t J_x\hat\varepsilon_\theta$ 包含膨胀项（尺度 $>1$ 的恒等缩放）和收缩项（$b_t < 0$ 的score修正），收缩性取决于score Jacobian的代数性质。

- **(EC) 欧氏收缩**：全局条件，定义收缩阈值 $L_t^* = \frac{\sqrt{\bar\alpha_{t-1}/\bar\alpha_t} - 1}{|b_t|}$，仅依赖噪声调度
- **(PC) 块-最大范数收缩**：patch级条件，将Jacobian分解为对角块 $\kappa_t^{\mathrm{diag}}$ 和交叉块 $\delta_t^{\mathrm{cross}}$，要求 $\kappa_t^{\mathrm{diag}} + \delta_t^{\mathrm{cross}} < 1$

**设计动机**：自然图像具有局部自相似性（非全局），需要patch级收缩保证而非全局收缩。这正是经典PIFS优于IFS的地方。

#### 设计二：方向性抑制场与分层释放

**功能**：引入方向性抑制场 $S_{k,t}(x) = |b_t| \langle v_k^{(1)}, [\nabla_x \Delta_t(x)]_{kk} v_k^{(1)} \rangle$，量化训练score网络对每个patch的非高斯修正。

**核心思路**：高斯基线下，对角块谱范数 $f_t(\lambda_k)$ 在所有CIFAR-10 patch上都 $>1$（膨胀），但训练后网络学习到抑制场 $S_{k,t} > 0$，将有效Rayleigh商压到1以下。关键定理（Stratified Crossover, Thm 22）证明：在Margin Monotonicity条件(MM)下，低方差patch先释放抑制、高方差patch后释放，产生严格的方差顺序化释放。

**设计动机**：解释Regime I中对角块保持 $\approx 1$ 而非按高斯预测膨胀的现象，以及Regime II中patch逐个"解锁"细节合成的机制。

#### 设计三：吸引子的Kaplan-Yorke维度公式

**功能**：推导PIFS吸引子的分形维度，建立离散Moran方程 $\prod_t f_t(\lambda^{**}) = 1$ 求解全局膨胀阈值 $\lambda^{**}$。

**核心思路**：在高斯数据 + 块对角协方差假设下，Lyapunov谱完全由每步对角膨胀函数 $f_t(\lambda)$ 确定。对角方向 $\lambda_k > \lambda^{**}$ 的为膨胀方向，KY维度公式为：

$$d_{\mathrm{KY}} = N^+ + \frac{\sum_{k:\lambda_k > \lambda^{**}} n_k \Lambda(\lambda_k)}{|\Lambda_{k^*}^-|}$$

对非高斯数据，抑制修正版本 $d_{\mathrm{KY}}^{\mathrm{eff}} \leq d_{\mathrm{KY}}$，抑制只会缩小吸引子维度。

**设计动机**：提供无需模型评估的吸引子维度预测，连接噪声调度设计与生成流形的几何性质。

### 损失函数/训练策略

- **Collage类比**（Thm 12）：DSM训练目标等价于PIFS的collage误差最小化（up to SNR加权）
- **$L^2$–$\mathcal{W}_1$ Bridge**（Thm 14）：训练损失控制到PIFS不动点的Wasserstein-1距离
- **PIFS正则化器**（Thm 15）：$\mathcal{L}_{\mathrm{PIFS}}(\theta) = \mathcal{L}(\theta) + \mu_{\mathrm{reg}} \sum_{t,k,j\neq k} \|[J_x\hat\varepsilon_\theta]_{kj}\|_F^2$，直接强制(PC)条件，可通过JVP/VJP高效计算

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

#### (PC)条件crossover验证

| $t$ | 阶段 | 平均margin slack | 违反比例 |
|---|---|---|---|
| 700 | Regime I | $-0.003942$ | 16/16 |
| 200 | I/II过渡 | $-0.000304$ | 14/16 |
| 160 | Regime II | $+0.001382$ | 0/16 |
| 40 | Regime II | $+0.006412$ | 0/16 |

Crossover发生在 $t \in [160, 200]$，约40步窗口。Regime I全面违反(PC)，Regime II全面满足。

#### 分层释放的Spearman相关

在crossover区间（$t=240,260$），$\rho(\hat\lambda_k, \hat\kappa_t^{\mathrm{diag}})$ 为负且显著（$p \leq 0.047$），确认低方差patch先释放。深度Regime II（$t=40$）回到正 $\rho = 0.771$（$p=0.001$），高斯排序恢复。

#### 抑制修正KY维度（CelebA-HQ实验）

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
