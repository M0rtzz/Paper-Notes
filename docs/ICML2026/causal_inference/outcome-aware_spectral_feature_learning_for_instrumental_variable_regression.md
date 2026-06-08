---
title: >-
  [论文解读] Outcome-Aware Spectral Feature Learning for Instrumental Variable Regression
description: >-
  [ICML 2026][因果推理][工具变量回归] 针对非参数工具变量（NPIV）回归中 SpecIV 学到的谱特征"只看 X-Z 关系、不看结果 Y"的盲点，本文提出 Augmented Spectral Feature Learning：在 SpecIV 的对比损失里加上一项 Y 投影到 Z 特征上的回归…
tags:
  - "ICML 2026"
  - "因果推理"
  - "工具变量回归"
  - "谱特征学习"
  - "增广算子"
  - "对比损失"
  - "因果效应估计"
---

# Outcome-Aware Spectral Feature Learning for Instrumental Variable Regression

**会议**: ICML 2026  
**arXiv**: [2512.00919](https://arxiv.org/abs/2512.00919)  
**代码**: 无  
**领域**: 因果推断 / 工具变量回归 / 非参数 IV / 谱方法  
**关键词**: 工具变量回归, 谱特征学习, 增广算子, 对比损失, 因果效应估计

## 一句话总结
针对非参数工具变量（NPIV）回归中 SpecIV 学到的谱特征"只看 X-Z 关系、不看结果 Y"的盲点，本文提出 Augmented Spectral Feature Learning：在 SpecIV 的对比损失里加上一项 Y 投影到 Z 特征上的回归损失，等价于对一个把 Y 信息拼进去的"增广算子" $\mathcal{T}_\delta = [\mathcal{T} \mid \delta r_0]$ 做截断 SVD，从而在结构函数 $h_0$ 与 $\mathcal{T}$ 顶端奇异函数对齐很差的"坏"情形下也能用极低秩特征恢复因果效应。

## 研究背景与动机

**领域现状**：NPIV 是隐变量存在下做因果效应估计的核心工具，形式为 $Y=h_0(X)+U,\;\mathbb{E}[U|Z]=0$，等价于解线性反问题 $\mathcal{T}h_0=r_0$，其中 $\mathcal{T}$ 是从 $L_2(X)$ 到 $L_2(Z)$ 的条件期望算子。两阶段最小二乘（2SLS）是经典做法，近年研究把"特征 $\varphi(X)$、$\psi(Z)$"换成神经网络自适应学习，代表方法包括 DFIV、SpecIV、minimax saddle-point 类方法等。SpecIV（Sun et al., 2025）通过最小化对比损失，让学到的特征逼近 $\mathcal{T}$ 的顶端 $d$ 维奇异子空间，相当于做了一个低秩近似 $\mathcal{T}_d$。

**现有痛点**：Meunier et al. (2025) 已经证明，SpecIV 只有在 $h_0$ 主要展开在 $\mathcal{T}$ 的前 $d$ 个右奇异函数 $v_1,\ldots,v_d$ 上时才最优；一旦 $h_0$ 主要落在某个"小奇异值 $\lambda_k$ 对应的 $v_k$"上（spectral misalignment），SpecIV 要么需要把秩 $d$ 拉到 $k$ 以上（带 $k-1$ 个无信息维度）才能覆盖到 $v_k$，要么直接失败。

**核心矛盾**：SpecIV 学特征的目标里**根本没有 Y**，所以它只能挑"$X,Z$ 之间关系最强的方向"，而不是"对预测 Y 最有用的方向"。当这两者重合时无所谓，当它们不重合时（比如 $h_0$ 在 $\mathcal{T}$ 的尾部分量上），算法把所有容量都浪费在 Y 无关的方向上。

**本文目标**：让谱特征学习"看见" Y——既保留 SpecIV"特征要能解释 $\mathcal{T}$"的好性质，又让特征额外向"对 Y 有预测力"的方向倾斜，从而在 spectral misalignment 场景下也能用更小的特征维度 $d$ 恢复 $h_0$。

**切入角度**：把"对 Y 有预测力"形式化为"用 $\psi(Z)$ 线性预测 Y 的 MSE 要小"，并把这一项加进 SpecIV 的对比损失里。作者随即发现：加这一项后整个优化目标恰好等价于对一个"增广算子" $\mathcal{T}_\delta = [\mathcal{T} \mid \delta r_0]$（把 $r_0=\mathbb{E}[Y|Z]$ 作为额外一列）做截断 SVD。这给了方法一个干净的算子视角和后续理论的入口。

**核心 idea**：用"$\mathcal{T} \to \mathcal{T}_\delta = [\mathcal{T}\mid\delta r_0]$"替代裸算子，让谱分解结果天然地把 $r_0$（进而 $h_0$ 的"信号方向"）拉到主奇异子空间里——一个超参 $\delta$ 控制偏向 Y 的强度，$\delta=0$ 退化为 SpecIV。

## 方法详解

### 整体框架

方法要解决的是 NPIV 反问题 $\mathcal{T}h_0=r_0$（$\mathcal{T}:L_2(X)\to L_2(Z)$ 是 $X|Z$ 的条件期望算子，$r_0=\mathbb{E}[Y|Z]$），核心痛点是 SpecIV 学特征时只逼近 $\mathcal{T}$ 的顶端奇异子空间、看不到 $Y$，于是把"对 $Y$ 有预测力"写成一项正则塞进对比损失里。整套流程仍是两阶段：先在数据集 $\tilde{\mathcal{D}}_m$ 上学一对神经网络特征 $\varphi_\theta:\mathcal{X}\to\mathbb{R}^d$ 与 $\psi_\theta:\mathcal{Z}\to\mathbb{R}^d$（外加辅助向量 $\omega\in\mathbb{R}^d$），最小化增广对比损失 $\mathcal{L}_\delta^{(d)}(\theta,\omega)$；再在独立数据集 $\mathcal{D}_n$ 上把学到的特征代入闭式 2SLS 估计 $\widehat{h}_\theta(x)=\varphi_\theta(x)^\top \widehat{C}_{\psi\varphi}^{-1}\widehat{\mathbb{E}}_n[Y\psi_\theta(Z)]$（$\widehat{C}_{\psi\varphi}$ 是 $\psi(Z)\varphi(X)^\top$ 的样本均值）。相比 SpecIV，只有"学特征"这一步的目标函数变了，下游 2SLS 原样复用。

### 关键设计

**1. 增广算子 $\mathcal{T}_\delta$ 与正则项 $\mathcal{R}_\delta$：把 $Y$ 信息注进谱分解**

SpecIV 学特征的目标里根本没有 $Y$，所以 $h_0$ 一旦落在 $\mathcal{T}$ 的小奇异方向上就会被错过。本文的修法是在 SpecIV 对比损失 $\mathcal{L}_0^{(d)}$ 上加一项 $\mathcal{R}_\delta^{(d)}(\theta) = -\delta^2 \mathbb{E}[Y\psi_\theta(Z)]^\top C_{\psi_\theta}^{-1}\mathbb{E}[Y\psi_\theta(Z)]$，得到 $\mathcal{L}_\delta^{(d)}=\mathcal{L}_0^{(d)}+\mathcal{R}_\delta^{(d)}$。直观上 $-\delta^{-2}\mathcal{R}_\delta^{(d)}$ 恰是"用 $\psi_\theta(Z)$ 线性回归预测 $Y$ 的 MSE"（差一个无关常数），因此这一项强迫 $\psi$ 特征张成"能解释 $Y$"的方向。

真正干净的地方是它的算子解释。定义增广算子 $\mathcal{T}_\delta:L_2(X)\times\mathbb{R}\to L_2(Z)$，$\mathcal{T}_\delta(h,a)=\mathcal{T}h+a\cdot\delta\cdot r_0$，相当于把 $\delta r_0$ 当作一额外列拼到 $\mathcal{T}$ 上，记作 $\mathcal{T}_\delta=[\mathcal{T}\mid\delta r_0]$。Proposition 4.1 证明 $\mathcal{L}_\delta^{(d)}\ge -\|\mathcal{T}_\delta^{(d)}\|_{HS}^2$，且下界恰好在学到的算子等于 $\mathcal{T}_\delta$ 的最佳秩-$d$ 截断 $\mathcal{T}_\delta^{(d)}$ 时达成——也就是说，最小化这个增广损失等价于对 $\mathcal{T}_\delta$ 做截断 SVD。之所以有效，是因为把 $\delta r_0$ 拼进去，会把 $h_0$ 里"原本被 $\mathcal{T}$ 压在小奇异值上"的信号分量放大到顶端奇异子空间，从根本上化解了 spectral misalignment；超参 $\delta$ 控制偏向 $Y$ 的强度，$\delta=0$ 退化为 SpecIV。

**2. 引入辅助变量 $\omega$ 把目标改写成可微的联合优化**

正则项 $\mathcal{R}_\delta$ 里含 $C_{\psi_\theta}^{-1}$，直接对网络反传这个矩阵求逆数值上很不稳。为此把 $\mathcal{R}_\delta$ 里隐含的"内层最小化"显式化：引入辅助向量 $\omega\in\mathbb{R}^d$，对 $(\theta,\omega)$ 联合最小化 $\mathcal{L}_\delta^{(d)}(\theta,\omega) = \mathcal{L}_0^{(d)}(\theta) - 2\delta\mathbb{E}[Y\psi_\theta(Z)]^\top \omega + \omega^\top C_{\psi_\theta}\omega$。固定 $\theta$ 时它关于 $\omega$ 是凸二次的，最优 $\omega_\theta^* = \delta C_{\psi_\theta}^{-1}\mathbb{E}[Y\psi_\theta(Z)]$，回代正好恢复原始 $\mathcal{L}_\delta^{(d)}$。这样网络反传只剩 $C_{\psi_\theta}$ 的常规矩阵乘法、不再求逆。更巧的是 $\omega$ 在理论上恰好对应 $\mathcal{T}_\delta$ 的 SVD 中"额外一行"的坐标（Eq. 6 的 $\omega_{*,i}$），训练得到的 $\hat\omega$ 还能直接拿去选 $\delta$（见下一点）。

**3. 用两条启发式给无监督特征学习挑出"恰到好处"的 $\delta$**

特征学习没有验证集，$\delta$ 不能靠交叉验证，需要可观察的停止信号。第一条是**对齐估计**：Proposition 6.1 给出 $h_0$ 在学到子空间上的投影长度 $\|\Pi_{\varphi_\star}h_0\|^2 = \alpha^\top(I_d-\omega_\star\omega_\star^\top)^{-1}\alpha$（$\alpha_i=\mathbb{E}[Y\psi_{\star,i}(Z)]\sigma_{\star,i}^{-1}$），全部能用学到的 $\hat\sigma,\hat\psi,\hat\omega$ 插值估出，于是逐步增大 $\delta$、只要估计对齐显著上升就接受。第二条是**损失平衡**：把 $\mathcal{R}_\delta$ 看作对 $\mathcal{L}_0$ 的正则，逐步增 $\delta$ 直到 $\mathcal{R}_\delta$ 大幅下降而 $\mathcal{L}_0$ 明显上升就收手——因为一旦 $\mathcal{R}_\delta$ 占主导，特征会过拟合 $r_0$ 的张成方向、丢掉对 $\mathcal{T}$ 的整体逼近。两条判据这么设计，是因为实验里"小正 $\delta$"几乎总改进，但过大 $\delta$ 会让 $\omega_*$ 范数发散、$(I-\omega\omega^\top)^{-1}$ 不稳，所以必须有个"在哪收手"的可算判据；两者分别从"对齐改善"和"主损失退化"给信号，交叉验证后比较稳健。

### 损失函数 / 训练策略

最终训练目标（经验形式，用 $\tilde{\mathcal{D}}_m$）：

$\widehat{\mathcal{L}}_\delta^{(d)}(\theta,\omega) = \widehat{\mathbb{E}}_X\widehat{\mathbb{E}}_Z[(\varphi_\theta(X)^\top\psi_\theta(Z))^2] - 2\widehat{\mathbb{E}}[\varphi_\theta(X)^\top\psi_\theta(Z)] - 2\delta \widehat{\mathbb{E}}[Y\psi_\theta(Z)]^\top\omega + \omega^\top \widehat{C}_{\psi_\theta}\omega$。

第一项里 $\widehat{\mathbb{E}}_X\widehat{\mathbb{E}}_Z$ 是边缘乘积分布上的期望，按 SpecIV 标准做法用 in-batch 边缘采样（把同一 batch 的 X 和 Z 重新打乱配对）近似。优化器用 Adam，$(\theta,\omega)$ 一起更新；特征学完后冻结，在独立数据集 $\mathcal{D}_n$ 上做闭式 2SLS。两数据集划分确保 2SLS 阶段误差与特征学习阶段误差解耦，对应理论 Theorem 5.1 中的 $\sqrt{d/n}$ 速率。

## 实验关键数据

### 主实验

**合成数据**：构造可控算子 $\mathcal{T}=\mathbf{1}_Z\otimes\mathbf{1}_X+\sum_{i=1}^{d-1}\sigma_i u_i\otimes v_i$，$\sigma_i$ 从 $\sigma_1$ 线性衰减到 $\sigma_{d-1}=c_\sigma\sigma_1$；结构函数 $h_0=\sum\alpha_i v_i$，$\alpha_i$ 线性变化，$c_\alpha=\alpha_{d-1}/\alpha_1$ 控制 $h_0$ 与 $\mathcal{T}$ 谱的对齐程度。报告归一化 IV 损失 $\|\widehat{h}_\theta-h_0\|^2$（以 $\delta=0$ 时的均值为 1）。

| 场景 ($c_\sigma$, $c_\alpha$) | $\delta=0$ (SpecIV) | $\delta=0.5$ | $\delta=1.0$ | $\delta=3.0$ | $\delta=5.0$ |
|---|---|---|---|---|---|
| $c_\sigma=0.2,\;c_\alpha=5.0$（严重 misalignment） | 1.00 | 显著下降 | 进一步下降 | 最优区 | 略回升 |
| $c_\sigma=0.8,\;c_\alpha=0.2$（高度对齐） | 1.00 | 仍有改进 | 改进 | 持平 | 略差 |

定性结论：在"$h_0$ 与 $\mathcal{T}$ 谱对齐很差 + 奇异值衰减很快"的最难场景下，增广方法把 IV 损失打到 SpecIV 的若干分之一；在"对齐良好"场景下，小正 $\delta$ 仍稳定带来改进。

**dSprites IV 基准**：除原版 $h_{\text{old}}$（用心形 sprite、$h_0$ 落在 $\mathcal{T}$ 顶端奇异函数上的"好"场景）外，作者新提出 $h_{\text{new}}$（用椭圆 sprite，$h_0$ 近似椭圆朝向，预期需要小奇异值方向的特征）。对比 DFIV 与 KIV：

| 设定 | SpecIV ($\delta=0$) | AugSpecIV（小正 $\delta$） | DFIV |
|---|---|---|---|
| $h_{\text{old}}$（"好"） | 基线 | 较 SpecIV 平均 **~20% 改进** | 强基线 |
| $h_{\text{new}}$（"坏"） | **严重不及** DFIV | 持平或略超 DFIV | 当前最强基线 |

关键发现：在 $h_{\text{new}}$ 上裸 SpecIV 的失败正好验证理论——顶端奇异空间不是 $h_0$ 的最佳基；调大 $\delta$ 让特征向 $h_{\text{new}}$ 投影显著增大，损失大幅下降并赶上 DFIV。

**Off-Policy Evaluation（OPE）**：在 BSuite Cartpole、Mountain Car、Catch 三个环境上，把 OPE 转写成一个迭代式 NPIV 问题（因为目标 $Y_k=-\gamma^{-1}(R-Q_k(s,a))$ 每步会变，是典型动态 spectral misalignment）。结果显示 AugSpecIV 在 Cartpole 上显著超过 SpecIV 和 DFIV，在 Catch 上与 SpecIV 都强、在 Mountain Car 上不如 DFIV——但没有任何方法一致最优（与 Chen et al., 2022 的结论一致）。三个环境自动调出的 $\delta$ 分别为 $1,\,10^{-3},\,10^{-2}$，说明方法能适应不同程度的对齐结构。

### 消融实验

| 配置 | 表现 | 说明 |
|---|---|---|
| Full ($\mathcal{L}_0+\mathcal{R}_\delta$, joint $(\theta,\omega)$) | 最优 | 完整方法 |
| $\delta=0$（去掉 $\mathcal{R}_\delta$） | 退化为 SpecIV | misalignment 下崩盘 |
| $\delta\to\infty$ 区（$\mathcal{R}_\delta$ 主导） | IV 损失反弹 | 特征过度只学 $r_0$ 张成方向 |
| 仅 $\mathcal{R}_\delta$（去掉 $\mathcal{L}_0$） | 不能保证学到 $\mathcal{T}$ 的整体结构 | 论文不推荐 |

### 关键发现

- **小正 $\delta$ 是几乎免费的午餐**：合成与 dSprites 都观察到只要 $\delta>0$ 就有改进，且对具体取值不敏感，这降低了实践中调参负担。
- **$\delta$ 越大越好不成立**：当 $\mathcal{R}_\delta$ 大到压过 $\mathcal{L}_0$，特征会"逃跑"到只解释 $r_0$ 的方向，丢失对 $\mathcal{T}$ 的逼近能力，IV 损失反而上升；论文给出的两条启发式正是为此服务。
- **OPE 上 $\delta$ 自动选出的量级跨 3 个数量级**（$1\sim 10^{-3}$），证明"动态 misalignment 程度"会因任务不同差异极大，强证据支持"outcome-aware"在实际系统里必要。

## 亮点与洞察

- **把"加正则项"等价转化成"算子增广"**：正则 $\mathcal{R}_\delta$ 看上去只是"让 $\psi(Z)$ 多解释一点 Y"的工程改动，但作者证明它精确等价于对 $[\mathcal{T}\mid\delta r_0]$ 做截断 SVD。这一等价让整套方法继承了 SpecIV 的算子-理论框架（Wedin sin-Θ、Weyl 不等式），从而推出可控的高概率 2SLS 误差界，而不是只能给"经验上更好"的解释。
- **辅助变量 $\omega$ 一举两用**：它既是规避矩阵求逆的优化技巧，又是 $\mathcal{T}_\delta$ SVD 里"额外一行"的真实坐标，训练完直接拿来估计 $\|\Pi_{\varphi_\star}h_0\|^2$ 选 $\delta$。把"训练技巧"和"模型选择信号"复用是非常优雅的设计。
- **新 dSprites 基准 $h_{\text{new}}$**：旧基准事后被发现属于"好"场景，掩盖了 SpecIV 的弱点；作者刻意用椭圆朝向构造一个落在小奇异方向的 $h_{\text{new}}$，把 SpecIV 的失败暴露出来。这种"先诊断 benchmark 的盲点再补救方法"的研究路径很值得借鉴到其他领域。
- **可迁移到的任务**：任何"用谱方法做条件算子分解"的场景——比如 OPE 的价值函数估计、分子动力学 / 气候里的演化算子学习——只要存在一个"任务相关的额外信号"（不必是 $Y$，可以是 $\mathbb{E}[Y^k|Z]$ 等高阶矩，论文 Appendix F 提示），同样的"增广操作子"思路都可以套用。

## 局限与展望

- 当前理论只处理**秩-1 增广**（$\delta r_0$ 一列），论文承认更一般的秩-$K$ 增广（同时增广 $r_0,\mathbb{E}[Y^2|Z],\ldots$）需要更复杂的扰动分析，留作未来工作；这意味着多输出 $Y$ 或多任务 IV 场景下方法还不能直接调用现成保证。
- **下游 2SLS 误差上界依赖"成功的表征学习"**——即假设网络能把经验增广损失驱到接近最优，但论文坦言"为 DNN 训练这一目标证明收敛"超出本文范围，与谱对比损失的优化/泛化耦合在一起目前还是 open problem。
- **$\delta$ 选择仍偏经验**：两条启发式都建立在"看到曲线趋势"上，没有像交叉验证那样的封闭式自动选法；理论上选 $\delta$ 的最优率（依赖 $\|q_d\|$、$\lambda_k$）需要不可观测量，自然给不出。
- **OPE 实验里"没有方法一致最优"**：AugSpecIV 在 Mountain Car 上仍不如 DFIV，说明 spectral misalignment 不是 OPE 失败的唯一原因，方法对"非紧致算子+迭代目标"的根本困难只是局部缓解。
- 自己看到的改进方向：把 $\delta$ 在训练过程中按梯度信号或对齐估计**自适应退火**（而不是固定常数），可能进一步把"全程偏向 Y"和"末段精修 $\mathcal{T}$"两个阶段分离开，理论上对应"先用大 $\delta$ 找信号方向，再用 $\delta\to 0$ 收敛到真实算子"。

## 相关工作与启发

- **vs SpecIV (Sun et al., 2025)**：本文是 SpecIV 的直接超集，$\delta=0$ 完全退化为 SpecIV。SpecIV 的对比损失只看 $X,Z$ 关系；本文证明它在 misalignment 时崩盘，并通过加 $\mathcal{R}_\delta$ 一项以低代价修复。优势是无监督特征也能对齐 Y，劣势是多了一个 $\delta$ 要调。
- **vs DFIV (Xu et al., 2020)**：DFIV 用神经网络直接学第一阶段 $\mathbb{E}[\varphi(X)|Z]$ 的条件期望，不显式做低秩分解。两者目标对偶——DFIV 灵活但缺谱视角下的可控逼近误差；本文有理论保证、计算更轻，但需要假设 $\mathcal{T}$ 紧致。在 $h_{\text{new}}$ 上两者打成平手，验证了"显式谱 + Y-aware"路径可以追上更黑盒的端到端方法。
- **vs Bruns-Smith (2024) 的 outcome-aware NPIV**：他先学一组"对 Y 有预测力的 Z-特征"，再在大函数类里找投影预测 Y 的元素；与本文都属于"让特征显式见 Y"的思路，但他的方法对 Z-特征是否真的对 IV 问题有用是事后验证，强范数收敛还要更严的 ill-posedness 条件。本文的算子等价性把"对 Y 有用"和"对 $\mathcal{T}$ 有用"一次性绑定，理论条件更弱。
- **vs Minimax / saddle-point 方法 (Dikkala et al., 2020; Liao et al., 2020; Bennett et al., 2023)**：saddle-point 走的是 GMM-对抗框架，绕开条件期望估计；本文相反，紧抓条件期望算子的 SVD 视角。saddle-point 在高维更通用但难调参；本文方法结构更简、计算更便宜（一次闭式 2SLS）。
- **启发**：把"任务相关额外信号"作为额外一列拼到原算子里做谱分解，是一个高度可迁移的范式——可推广到任何"算子学习 + 下游任务"组合（off-policy evaluation、conditional density 估计、operator learning for dynamical systems）。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把"加一项 Y-MSE 正则"和"算子增广 + 截断 SVD"建立精确等价，是干净且新的视角；但具体方法相对 SpecIV 是直接扩展，不是颠覆性创新。
- 实验充分度: ⭐⭐⭐⭐ 合成 + dSprites（含新构造的 $h_{\text{new}}$） + OPE 三类任务覆盖完整，并明确展示失败-修复的对照；缺真实经济学/医疗 IV 数据稍可惜。
- 写作质量: ⭐⭐⭐⭐⭐ 动机—算子视角—理论保证—实验—$\delta$ 选择层层递进，记号严谨，附录给完整证明，是 ICML 顶刊水准。
- 价值: ⭐⭐⭐⭐ 直接修复 SpecIV 已知盲点、对 OPE 等下游任务有立竿见影的改进，且方法实现成本极低，工程上有立刻采用的价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Demystifying Spectral Feature Learning for Instrumental Variable Regression](../../NeurIPS2025/causal_inference/demystifying_spectral_feature_learning_for_instrumental_variable_regression.md)
- [\[ICML 2026\] ECSEL: Explainable Classification via Signomial Equation Learning](ecsel_explainable_classification_via_signomial_equation_learning.md)
- [\[ICML 2026\] Evaluating Bivariate Causal Statements Based on Mutual Compatibility](evaluating_bivariate_causal_statements_based_on_mutual_compatibility.md)
- [\[ICML 2026\] Rank-Learner: Orthogonal Ranking of Treatment Effects](rank-learner_orthogonal_ranking_of_treatment_effects.md)
- [\[ICML 2026\] Unveiling the Structure of Do-Calculus Reasoning via Derivation Graphs](unveiling_the_structure_of_do-calculus_reasoning_via_derivation_graphs.md)

</div>

<!-- RELATED:END -->
