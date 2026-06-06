---
title: >-
  [论文解读] The Lie We Tell: Correcting the Euclidean Fallacy in Vision-Language-Action Policies via Score Matching on Tangent Space
description: >-
  [ICML 2026][机器人][SE(3) 流形] Lie Diffuser Actor (LDA) 把扩散过程从把 SE(3) 位姿展平成 $\mathbb{R}^{12}$ 的"欧氏谎言"修正回流形原生：通过左不变 SDE 在李代数 $\mathfrak{se}(3)$ 中注入噪声、用指数映射回拉到流形…
tags:
  - "ICML 2026"
  - "机器人"
  - "SE(3) 流形"
  - "Lie 群扩散"
  - "左不变 SDE"
  - "切空间 score matching"
  - "CALVIN"
---

# The Lie We Tell: Correcting the Euclidean Fallacy in Vision-Language-Action Policies via Score Matching on Tangent Space

**会议**: ICML 2026  
**arXiv**: [2606.01847](https://arxiv.org/abs/2606.01847)  
**代码**: 论文未声明（仅 NSTC/NVIDIA 致谢，未给仓库链接）  
**领域**: 机器人 / VLA 策略 / 扩散模型  
**关键词**: SE(3) 流形, Lie 群扩散, 左不变 SDE, 切空间 score matching, CALVIN

## 一句话总结
Lie Diffuser Actor (LDA) 把扩散过程从把 SE(3) 位姿展平成 $\mathbb{R}^{12}$ 的"欧氏谎言"修正回流形原生：通过左不变 SDE 在李代数 $\mathfrak{se}(3)$ 中注入噪声、用指数映射回拉到流形、tangent-space 预测 score，理论上同时获得流形闭合、坐标系等变与测地线最优性，在 CALVIN ABC→D 上把平均任务长度从 3.27 推到 3.51。

## 研究背景与动机
**领域现状**：扩散类 VLA 策略（3D Diffuser Actor、Diffusion Policy、Octo 系）已成为机器人操作的主流方法，能捕捉多模态行为且擅长长程一致性。共同套路是把 SE(3) 位姿序列 $\mathbf{g} = (g^1, \dots, g^H)$ 平摊成 $\mathbb{R}^{12 \times H}$ 向量（9 维旋转矩阵 + 3 维平移），在欧氏空间里加 Gaussian 噪声、训练去噪网络，再用 SVD 或四元数归一化把输出投回 SO(3)。

**现有痛点**：作者把这类做法统一命名为 "Euclidean Fallacy"，并指出三个具体毛病——(1) 流形漂移：高斯噪声 + 旋转矩阵几乎必然破坏 $R^\top R = I$，迫使网络浪费容量学 SVD 后处理；(2) 等变性破坏：欧氏噪声分布在工作空间发生整体刚体变换时不会协变，score 函数被绑死在坐标系上；(3) 非测地轨迹：欧氏插值跨越物理不可实现的中间位姿，丢掉 Chasles 定理意义下的螺旋运动结构，轨迹角加加加速度（jerk）偏大。文中图 2 给出实测：3D Diffuser Actor 在 148K 次预测的正交性误差曲线在百分位上明显高于流形原生方法，且轨迹步间方差大，源自 SVD 投影对接近退化矩阵的放大效应。

**核心矛盾**：score-based 扩散需要在每步加 Gaussian 噪声，但 Gaussian 是为平直向量空间设计的；SE(3) 是 6 维曲率非零的 Riemannian 流形，加性噪声与流形几何根本不兼容。靠后处理投影看似补救，实际带来训练-推理不匹配、SVD 对近退化矩阵敏感、推理时的不可微投影改写了反向 SDE 三重隐患。

**本文目标**：构造一个 SE(3) 上的扩散框架，使得 (i) 任意时刻的中间样本本身就 $\in$ SE(3)；(ii) score 函数在工作空间整体刚体变换下协变；(iii) 反向过程的确定性极限趋于流形上的测地线。

**切入角度**：李代数 $\mathfrak{se}(3)$ 是 SE(3) 在单位元处的切空间，是平直 6 维向量空间，Gaussian 噪声在那里是良定义的；指数映射 $\exp: \mathfrak{se}(3) \to SE(3)$ 是满射，把任意 twist 映回合法的刚体变换。于是把"在流形上加噪"重新定义成"在切空间加噪后用 $\exp$ 把扰动结合到位姿上"，几何结构就闭环了。

**核心 idea**：把前向扩散写成左不变 SDE $g_t = g_0 \cdot \exp(\sigma_t \boldsymbol{\xi})$，score 网络输出 $\boldsymbol{\xi} \in \mathfrak{se}(3)$ 而非 $\mathbb{R}^{12}$ 噪声，反向更新用 $\exp$ 回拉，所有几何不变性靠群结构而非后处理保证。

## 方法详解

### 整体框架
LDA 由几何上下文编码器、迭代去噪 Transformer、切空间预测头与切空间 score matching 训练目标四部分组成。输入是 $K$ 路 RGB-D 观测和语言指令 $\mathcal{L}$，输出是 horizon $H$ 的位姿轨迹 $\mathbf{g} = (g^1, \dots, g^H) \in SE(3)^H$ 加 gripper 二值序列。点云被 back-project 后过 GAT 图注意力 Transformer 得几何特征 $\mathbf{F}_{\text{geo}}$，CLIP 文本编码器产语言特征 $\mathbf{F}_{\text{lang}}$，cross-attention 融合成 context $\mathcal{C}$；去噪 Transformer 在每个 diffusion step $t$ 接收 noisy 轨迹 $\mathbf{g}_t$ 与时间嵌入 $\tau(t)$，self-attention 建轨迹时间依赖、cross-attention 注入 $\mathcal{C}$；最后切空间预测头对每个 waypoint 输出 6 维 twist $\boldsymbol{\xi}^h = (\boldsymbol{\omega}^h, \mathbf{v}^h)$，去噪更新走 $g_{t-1}^h = g_t^h \cdot \exp(-\beta_t \boldsymbol{\xi}^h)$，所有中间态都自动 $\in$ SE(3)。

### 关键设计

1. **左不变 SDE 前向扩散 + 指数映射回拉**:

    - 功能：把"在 SE(3) 上加噪"变成切空间加 Gaussian、流形端做群乘，从而消除流形漂移并赋予等变性。
    - 核心思路：前向过程取 Stratonovich 形式 $\mathrm{d}g_t = g_t \cdot (\sigma_t \sum_{i=1}^6 E_i \circ \mathrm{d}W_t^i)$，其中 $\{E_i\}$ 是 $\mathfrak{se}(3)$ 的正交基、$W_t^i$ 是独立 Wiener；离散化即 $g_t = g_0 \cdot \exp(\sigma_t \boldsymbol{\xi})$，$\boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_6)$。由于 SE(3) 在群乘下闭合、$\exp$ 把任意 twist 映成合法刚体变换，命题 4.1 给出 $g_t \in SE(3)$ a.s. 全程成立。配合时间反演理论得到反向 SDE $\mathrm{d}g_t = g_t \cdot (\sigma_t^2 s_\theta(g_t, t) \mathrm{d}t + \sigma_t \mathrm{d}\bar{\mathbf{B}}_t)$，离散实现是 $g_{t-\Delta t} = g_t \cdot \exp(\sigma_t^2 s_\theta(g_t, t) \Delta t + \sigma_t \sqrt{\Delta t} \boldsymbol{\zeta})$。
    - 设计动机：欧氏加噪后 SVD 投影既改写了反向 SDE，又对近退化矩阵敏感（小预测误差被放大成大旋转误差），还破坏训练-推理一致性。把所有几何约束推到群结构里，网络再也不用学"我先输出一个糟糕矩阵、再让 SVD 救场"，可以专注操作语义。

2. **切空间 score matching + Adjoint 等变性**:

    - 功能：让 score 网络输出 $\mathfrak{se}(3)$ 中的 6 维 twist 而非 $\mathbb{R}^{12}$ 噪声，并自动获得坐标系等变。
    - 核心思路：预测头分两路 MLP 分别回归角速度 $\boldsymbol{\omega}^h \in \mathbb{R}^3$ 与线速度 $\mathbf{v}^h \in \mathbb{R}^3$，对应 SE(3) 半直积结构；输出 twist 立即喂给 $\exp$ 做去噪更新。训练目标是切空间 denoising score matching：从专家轨迹 $\mathbf{g}_0$ 采 $t \sim \mathcal{U}(0, T)$、$\boldsymbol{\xi}^h \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_6)$，构造 $g_t^h = g_0^h \cdot \exp(\sigma_t \boldsymbol{\xi}^h)$，最小化 $\|s_\theta(g_t^h, t) - \boldsymbol{\xi}^h\|^2$。理论 4.2 证明最优 score 满足 $s_\theta(h \cdot g, t) = \mathrm{Ad}_h(s_\theta(g, t))$，其中 adjoint $\mathrm{Ad}_{(R, \mathbf{p})}(\boldsymbol{\omega}, \mathbf{v}) = (R\boldsymbol{\omega}, R\mathbf{v} + [\mathbf{p}]_\times R\boldsymbol{\omega})$，即工作空间整体刚体变换时输出会按 adjoint 协变。
    - 设计动机：等变性是机器人部署的硬需求——同一任务换个相机外参或换个桌面位置不该让策略重学。把 score 定义在 body-fixed 切空间，网络学到的是"任务本身的几何"而不是"实验室的坐标系"，CALVIN ABC→D 的零样本迁移正好考这个。

3. **几何确定性 ODE → 测地线偏置**:

    - 功能：让反向过程的确定性极限对应 SE(3) 上 bi-invariant 度量下的测地线，使生成轨迹接近螺旋运动而非欧氏直线。
    - 核心思路：反向 SDE 对应的概率流 ODE 是 $\mathrm{d}g_t/\mathrm{d}t = g_t \cdot \sigma_t^2 s_\theta(g_t, t)$。命题 4.3 给出：若 score 沿轨迹近似常向量 $\boldsymbol{\xi}^*$，ODE 解就是流形上的测地线，即角速度与线速度都恒定的螺旋运动（Chasles 定理意义下的 screw motion）。即使实际 score 随时间变化，"intrinsic 形式"也把生成轨迹偏置到接近测地的方向。
    - 设计动机：欧氏插值跨越非正交矩阵造成的角加加加速度（angular jerk）会直接转化为执行器的抖动。论文里"look-ahead consistency"实验测了相邻 diffusion step 上 $\hat{x}_0^{(t)}$ 与 $\hat{x}_0^{(t-1)}$ 的 geodesic jitter，欧氏基线比 LDA 高近一个数量级，对真机部署是直接的稳态控制收益。

### 损失函数 / 训练策略
总损失 $\mathcal{L} = \lambda_s \mathbb{E}_{t, \boldsymbol{\xi}}\left[\sum_h \|s_\theta(g_t^h, t) - \boldsymbol{\xi}^h\|^2\right] + \lambda_p \mathcal{L}_{\text{pos}} + \lambda_g \mathcal{L}_{\text{grip}}$。score matching 项是主目标，位置 MSE 与 gripper 二元交叉熵作辅助；CALVIN 上训 300K～600K 步与基线 600K～800K 步对齐对比。

## 实验关键数据

### 主实验
CALVIN ABC→D（A/B/C 训、D 零样本测）与 ABCD→D 设置上的成功率与平均链长：

| Setting | 方法 | SR1 | SR2 | SR3 | SR4 | SR5 | Avg Len |
|---------|------|-----|-----|-----|-----|-----|---------|
| ABC→D | 3D Diffuser Actor (600K) | 92.2 | 78.7 | 63.9 | 51.2 | 41.2 | 3.27 |
| ABC→D | LDA (600K) w/o GAT | 89.6 | 78.0 | 66.6 | 55.7 | 46.9 | 3.368 |
| ABC→D | LDA (300K) w/o Lie | 90.2 | 80.3 | 69.6 | 58.5 | 48.8 | 3.474 |
| ABC→D | **LDA (300K) full** | **93.7** | **83.4** | **70.3** | 57.6 | 46.2 | **3.512** |
| ABCD→D | 3D Diffuser Actor (800K) | 90.3 | 77.3 | 65.8 | 53.8 | 41.6 | 3.288 |
| ABCD→D | LDA (300K) w/o GAT | 90.8 | 77.3 | 66.4 | 57.6 | 48.3 | 3.404 |
| ABCD→D | LDA (400K) w/o Lie | 91.0 | 76.1 | 63.4 | 51.6 | 41.8 | 3.239 |
| ABCD→D | **LDA (300K) full** | 90.6 | 80.4 | **71.1** | **62.6** | **53.7** | **3.584** |

可见两个模块独立有效、组合后超越 baseline 训练预算 1/2 仍领先。OpenVLA-OFT 上把 score matching 切到 SE(3) 后，LIBERO Long 成功率从 92.20 升到 94.13，说明收益来自 Lie 形式本身而非具体编码器。

### 真机消融
真机 20 trials/task，4 个任务覆盖 6-DOF 转运到精细装配：

| 任务 | Baseline | LDA | 任务对几何的核心要求 |
|------|----------|-----|----------------------|
| Move Doll Platform | 90% | **100%** | 整段转运保持稳定朝向 |
| Put Block in Box | 80% | 75% | 紧公差插入（基线略优） |
| Sort Blocks | 55% | **75%** | 精确平移放置 |
| Stack Cups | 55% | **60%** | 亚厘米对齐的倾斜插入 |

LDA 在需要旋转-平移耦合保持精度的任务上拉开差距；紧公差插入这种主要靠 translation 的任务上欧氏探索略有优势。

### 关键发现
- 流形约束违反实测（Fig. 5）：欧氏 baseline 在反向扩散过程中正交性误差达 $\mathcal{O}(10^0)$、行列式误差与四元数范数偏离 1 都很显著（0.5–2.0 之间），LDA 全程保持在 $\sim 10^{-7}$ 浮点精度内；四元数轨迹投影到 $\mathbb{S}^3$ 上，欧氏轨迹从球面内部穿过，LDA 始终贴着球面走测地线。
- Look-ahead 一致性（Fig. 6）：相邻去噪步预测出的"最终干净动作" $\hat{x}_0^{(t)}$ 与 $\hat{x}_0^{(t-1)}$ 的 geodesic jitter 欧氏基线高出近一个数量级，特别在早期 noisy 步段表现为大幅跳变。这意味着 LDA 的目标估计在去噪过程中单调收敛，对实时控制信号的可预测性是直接收益。
- 等变性的实操好处：CALVIN ABC→D 的零样本迁移要求模型在没见过的环境布局上工作，LDA 的 adjoint 等变保证 score 函数对工作空间整体平移/旋转协变，因此布局变化不会让策略失效；ABCD→D 上 baseline 反而被环境多样性拖累出训练不稳，LDA 仍随训练数据扩展稳步提升。

## 亮点与洞察
- "Euclidean Fallacy" 这个命名很有传播性，把过去一堆 diffusion policy 论文的隐含工程债（SVD 后处理、四元数归一化）打包成一个可测量的几何错误。它给出了一个明确的攻击面：旋转表示选择 vs 噪声分布的协变性。
- 把"等变性"从架构层（GNN/equivariant CNN）推到"生成过程"层是一个干净的代际跃迁。过去等变工作只能在 encoder 端保证，生成端一加噪声就破功；LDA 用左不变 SDE 把等变性从编码器一路保留到采样末端，给出了 Riemannian 生成模型在机器人上落地的样板。
- 兼容性强：LDA 是即插即用的"几何头部"——把 3D Diffuser Actor 的 12 维欧氏头换成 6 维 twist 头 + $\exp$ 回拉就行，不需要重设计 encoder 或 RL 训练循环。OpenVLA-OFT 上的实验侧面证明几何修正与底层视觉骨干解耦。

## 局限与展望
- 论文未公开代码仓库，可复现性需要等作者放出实现细节（特别是 left Jacobian $V(\boldsymbol{\omega})$ 数值实现与小角度近似的处理）。
- 真机 Put Block in Box 上 LDA 略落后 baseline（75% vs 80%），暗示在主要靠 translation、对 rotation 几乎无要求的任务上，欧氏空间的无约束探索可能反而更宽松。论文给的解释偏向描述性，缺少消融。
- 与大规模 VLA 基础模型（如 $\pi_0$、OpenVLA 全量）的对比仅限于 OpenVLA-OFT 的子任务，没有展示在更大数据规模或更多语言指令下 Lie 形式是否仍是主导收益来源；和 Flow Matching on Riemannian manifolds 的直接基线对比也缺失。
- 命题 4.3 的测地线性质要求 score 沿轨迹近似常向量，实际训练得到的 score 显然不是常向量，论文只说"偏置到接近测地"，缺少定量度量。

## 相关工作与启发
- **vs 3D Diffuser Actor (Ke et al., 2024)**: 共享点云 + Transformer + diffusion 框架，区别在欧氏 vs 流形：3DDA 用 12 维 + SVD 后处理，LDA 把噪声、score、更新全搬到 SE(3) 内。受控消融显示几何头与 GAT 编码器独立有效，两者叠加得最优。
- **vs 等变策略学习（Yang et al., Ryu et al., Tie et al., Zhu et al.）**: 这一线把等变性塞进 encoder，生成端仍是欧氏 diffusion，产生"前等变后破缺"的不对称。LDA 让生成全过程都左不变，等变是端到端的。
- **vs Riemannian 生成模型（De Bortoli et al., Lou et al., Chen & Lipman 2024）**: 数学基础来自这一线，但 Riemannian flow matching 偏确定性传输，作者认为对操作中的多模态行为表达力不足；LDA 保留 SDE 噪声以维持多模态采样能力，同时拿到流形闭合与等变。
- **vs 旧式四元数 / 6D 旋转表示**: 那些工作主要解决"输出层怎么表示旋转"的问题，但仍在欧氏空间加噪；LDA 解决的是"噪声本身在哪里"，是更上游的修正，因此能同时治好流形漂移、等变性、几何 jitter 三个症状。

## 评分
- 新颖性: ⭐⭐⭐⭐ 旋转表示与 Lie 群扩散单点不算新，但 "Euclidean Fallacy" 的系统化命名 + 命题 4.1/4.2/4.3 三连证明 + 真机覆盖把 Riemannian diffusion 推到了 VLA 实战层。
- 实验充分度: ⭐⭐⭐ CALVIN 双设置 + 真机 4 任务 + OpenVLA-OFT 跨架构验证 + 几何违反/jitter 量化均到位，但缺与最新大 VLA 基线的横评、缺 RLBench 等异源 benchmark、缺训练曲线稳定性证据。
- 写作质量: ⭐⭐⭐⭐ 命题与几何动机讲得清楚，Fig. 1/5/6 直观；定理陈述与实证现象一一对应，便于复现工程师对照实现。
- 价值: ⭐⭐⭐⭐ 给所有 diffusion-based 操作策略提供了一个低改动量、高几何收益的升级路径；"换掉欧氏头部"几乎是任何继任工作可以直接复用的模块。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies](discrete_diffusion_vla_bringing_discrete_diffusion_to_action_decoding_in_vision-.md)
- [\[ICML 2026\] Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action](spatial_memory_for_out-of-vision_manipulation_in_vision-language-action.md)
- [\[ICML 2026\] LangForce: Bayesian Decomposition of Vision-Language-Action Models via Latent Action Queries](langforce_bayesian_decomposition_of_vision_language_action_models_via_latent_act.md)
- [\[ICML 2026\] Contrastive Representation Regularization for Vision-Language-Action Models](contrastive_representation_regularization_for_vision-language-action_models.md)
- [\[ICML 2026\] Neural Implicit Action Fields: From Discrete Waypoints to Continuous Functions for Vision-Language-Action Models](neural_implicit_action_fields_from_discrete_waypoints_to_continuous_functions_fo.md)

</div>

<!-- RELATED:END -->
