---
title: >-
  [论文解读] Learning General Causal Structures with Hidden Dynamic Process for Climate Analysis
description: >-
  [ICML 2026][图像生成][因果发现] 本文提出 CaDRe，用一个带结构约束的时序 VAE 把"观测变量之间的因果图"与"驱动观测的隐动力过程"放在同一个非参数框架下联合识别，并给出了从时序数据同时恢复二者的可识别性定理…
tags:
  - "ICML 2026"
  - "图像生成"
  - "因果发现"
  - "因果表征学习"
  - "隐变量动力系统"
  - "非线性ICA"
  - "VAE"
---

# Learning General Causal Structures with Hidden Dynamic Process for Climate Analysis

**会议**: ICML 2026  
**arXiv**: [2501.12500](https://arxiv.org/abs/2501.12500)  
**代码**: https://github.com/MinghaoFu/CaDRe (有)  
**领域**: 因果推断 / 因果表征学习 / 时序生成模型 / 气候建模  
**关键词**: 因果发现, 因果表征学习, 隐变量动力系统, 非线性ICA, VAE

## 一句话总结
本文提出 CaDRe，用一个带结构约束的时序 VAE 把"观测变量之间的因果图"与"驱动观测的隐动力过程"放在同一个非参数框架下联合识别，并给出了从时序数据同时恢复二者的可识别性定理，在合成数据上验证理论、在 CESM2 气候数据上得到与领域专家一致的因果图与有竞争力的温度预测精度。

## 研究背景与动机
**领域现状**：理解气候系统的因果结构既是科学推理的基础，也是可靠预测的前提。目前主流路线分两支：一支是因果表征学习（CRL），通常基于非线性 ICA，用辅助变量、稀疏性或受限生成函数从观测中恢复"潜在驱动因子" $\mathbf{z}_t$；另一支是经典的因果发现（CD），如 LiNGAM、加性噪声模型、PC/FCI 等，在观测变量之间构造 DAG。

**现有痛点**：两条路线各有死角。CRL 一般假设观测变量之间没有直接因果联系，且 $\mathbf{z}_t \to \mathbf{x}_t$ 的生成是无噪可逆的——气候数据完全不满足（如相邻区域的温度通过热传导互相影响，且观测带强随机扰动）。CD 这一支虽然能处理观测间的图结构，但要么参数化（LiNGAM 假设线性非高斯），要么在存在隐混杂的非参数设定下只能给出 PAG 等价类（如 FCI），既识别不出潜在变量本身，也识别不出潜在变量之间的因果关系。

**核心矛盾**：气候系统里"看不到的大气/辐射动力"和"看得到的区域间空间耦合"是耦合在一起的——隐过程驱动观测，观测之间又有自己的因果图，二者无法各自单独被现有方法处理。

**本文目标**：在同一个时序非参数框架下，同时识别（i）观测变量 $\mathbf{x}_t$ 之间的因果图 $\mathbf{J}_g(\mathbf{x}_t)$；（ii）潜在变量 $\mathbf{z}_t$ 及其瞬时与时滞依赖；并把这两者放进一个可估计的生成模型里。

**切入角度**：作者两个关键观察。第一，DAG 结构保证从噪声源 $\mathbf{s}_t$ 到观测 $\mathbf{x}_t$ 的分布变换是单射的——观测间的因果传播不会破坏潜在空间的可恢复性。第二，原始 SEM 与一个带隐变量的非线性 ICA 在数据生成意义上完全等价，因此可以借 ICA 的可识别性工具去做 CD。

**核心 idea**：用三段连续观测 $\{\mathbf{x}_{t-1},\mathbf{x}_t,\mathbf{x}_{t+1}\}$ 的上下文信息把潜在空间恢复到值级 $\hat{\mathbf{z}}_t=h_z(\mathbf{z}_t)$，再借 SEM↔ICA 等价把观测因果图表示成 $\mathbf{J}_g(\mathbf{x}_t)=\mathbf{I}-\mathbf{D}_m(\mathbf{s}_t)\mathbf{J}_m^{-1}(\mathbf{s}_t)$，最后在 VAE 上加稀疏 + DAG 约束统一估计。

## 方法详解

### 整体框架
数据生成过程被写成一个含隐变量的时序 SEM：观测 $x_{t,i}=g_i(\mathbf{pa}_O(x_{t,i}),\mathbf{pa}_L(x_{t,i}),s_{t,i})$ 同时受其他观测分量 $\mathbf{x}_{t,\setminus i}$、当前与历史隐变量 $\mathbf{z}_t,\mathbf{z}_{t-1}$、以及依赖 $\mathbf{z}_t$ 的内生噪声 $s_{t,i}=g_{s_i}(\mathbf{z}_t,\epsilon^x_{t,i})$ 影响；隐变量 $z_{t,j}=f_j(\mathbf{pa}_L(z_{t,j}),\epsilon^z_{t,j})$ 既有瞬时也有时滞依赖。

整个 CaDRe 是一个状态空间 VAE：输入是一段时序 $\mathbf{x}_{1:T}$，z-encoder $\phi$ 从观测推 $\hat{\mathbf{z}}_t$，s-encoder $\eta$ 推非平稳噪声 $\hat{\mathbf{s}}_t$，decoder $\psi$ 再从 $(\hat{\mathbf{z}}_t,\hat{\mathbf{s}}_t)$ 重构 $\hat{\mathbf{x}}_t$；两个 prior 网络（normalizing flow）分别学 $\hat{\mathbf{z}}_t$ 的时序先验和 $\hat{\mathbf{s}}_t$ 的条件先验，并在它们的 Jacobian 上读出因果图。损失是 ELBO + 稀疏惩罚 $\mathcal{L}_s$ + DAG 惩罚 $\mathcal{L}_d$。

### 关键设计

1. **潜在空间的非参数值级可识别性（Theorem 3.2）**:

    - 功能：在不假设 $\mathbf{z}_t \to \mathbf{x}_t$ 无噪可逆的前提下，把潜在变量从带观测因果图 + 随机噪声的混合中恢复到 $\hat{\mathbf{z}}_t=h_z(\mathbf{z}_t)$（$h_z$ 可逆可微）。
    - 核心思路：先用 Lemma 3.1 证 $L_{\mathbf{x}_t\mid \mathbf{s}_t}$ 单射——DAG 保证信息沿因果通路单向流动，观测间的图不会阻断分布级恢复。再通过三段上下文 $\{\mathbf{x}_{t-1},\mathbf{x}_t,\mathbf{x}_{t+1}\}$ 在算子 $L_{\mathbf{x}_{t+1}\mid \mathbf{z}_t}$、$L_{\mathbf{x}_{t-1}\mid \mathbf{x}_{t+1}}$ 上的单射条件（A2），加上 latent drift（A3，不同 $\mathbf{z}_t$ 给出不同条件分布 $p(\mathbf{x}_t\mid\mathbf{z}_t)$）和可微性（A4），先证后验集合 $\{p(\mathbf{x}_t\mid\hat{\mathbf{z}}_t)\}$ 唯一，再把值级映射拎出来。
    - 设计动机：先前基于 eigendecomposition 的非参数识别（Hu & Schennach 类）只能恢复到 $p_{\hat{\mathbf{z}}_t}=p_{\mathbf{z}_t}$ 的分布级，无法支撑下游 component-wise CD；同时它们需要部分已知生成函数，对气候这种黑箱不友好。本文用"差一个可微编码器即可"的 A4 把结论抬到值级，是后续 Theorem 3.5/3.6 能用的前提。

2. **SEM ↔ 非线性 ICA 等价与观测因果图的闭式表达（Lemma 3.3 + Theorem 3.5）**:

    - 功能：把"观测之间的非参数因果发现"转化为"带隐变量的非线性 ICA 的混合结构识别"，从而绕开 FCI/CDNOD 等只能给等价类的非参数 CD 困境。
    - 核心思路：证明对每个 $i$ 存在 $m_i$ 使 $x_{t,i}=g_i(\mathbf{pa}_O,\mathbf{pa}_L,s_{t,i})=m_i(\mathbf{z}_t,\mathbf{s}_t)$ 描述同一数据生成过程——SEM 里 $x_{t,2}\to x_{t,1}$ 的边在 ICA 视角下等价于"间接路径" $s_{t,2}\to x_{t,2}\to x_{t,1}$。定义雅可比 $[\mathbf{J}_g(\mathbf{x}_t)]_{i,j}=\partial x_{t,i}/\partial x_{t,j}$、$[\mathbf{J}_m(\mathbf{s}_t)]_{i,j}=\partial x_{t,i}/\partial s_{t,j}$ 与对角块 $\mathbf{D}_m(\mathbf{s}_t)$，可推出 $\mathbf{J}_g(\mathbf{x}_t)\mathbf{J}_m(\mathbf{s}_t)=\mathbf{J}_m(\mathbf{s}_t)-\mathbf{D}_m(\mathbf{s}_t)$，进而 $\mathbf{J}_g(\mathbf{x}_t)=\mathbf{I}-\mathbf{D}_m(\mathbf{s}_t)\mathbf{J}_m^{-1}(\mathbf{s}_t)$。在 A5（generation variability，要求 $\mathbf{V}(t,k),\mathbf{U}(t,k)$ 这 $2d_x$ 个雅可比向量线性独立）下，进一步给出 Theorem 3.6：观测因果图的支持集 $\text{supp}(\mathbf{J}_g(\mathbf{x}_t))$ 可识别。
    - 设计动机：先前的 ICA-based CD（Monti 2020、Reizinger 2023）要求无隐混杂；本文把隐过程 $\mathbf{z}_t$ 作为"连续条件先验"塞进 ICA，DAG 结构同时帮 ICA 去掉了可逆性假设——这是能在存在隐动力的气候系统里得到 Table 2 中"既能拿潜在因果图、又能拿观测因果图、又是非参数"的关键。

3. **基于 normalizing flow 的双 prior + 雅可比读图的估计架构**:

    - 功能：把上面三条定理落地成可训练的 VAE，并直接从两个 prior 网络的雅可比里读出"潜在瞬时图 $\mathbf{J}_r(\hat{\mathbf{z}}_t)$、潜在时滞图 $\mathbf{J}_r(\hat{\mathbf{z}}_{t-1})$、观测因果图 $\mathbf{J}_{\hat g}(\hat{\mathbf{x}}_t)$"三类结构。
    - 核心思路：z-prior 用条件 normalizing flow 学逆转换 $\hat{\epsilon}^z_{t,i}=r_i(\hat{\mathbf{z}}_{t-1},\hat{\mathbf{z}}_t)$，由变换 $\kappa$ 的分块 Jacobian 直接读出潜在的瞬时与时滞结构；s-prior 同理学 $\hat{\epsilon}^x_{t,i}=w_i(\hat{\mathbf{z}}_t,\hat{\mathbf{s}}_t)$。观测因果图按 Corollary 2 用 decoder 的 $\mathbf{J}_{\hat m}(\hat{\mathbf{s}}_t)$ 直接算出 $\mathbf{J}_{\hat g}(\hat{\mathbf{x}}_t)=\mathbf{I}-\mathbf{D}_{\hat m}\mathbf{J}_{\hat m}^{-1}$，因此 DAG 是随 $\hat{\mathbf{z}}_t$ 时变的。为避免冗余边和环，加稀疏惩罚 $\mathcal{L}_s=\|\mathcal{M}(\mathbf{J}_r(\hat{\mathbf{z}}_t))\|_1+\|\mathcal{M}(\mathbf{J}_r(\hat{\mathbf{z}}_{t-1}))\|_1+\|\mathbf{J}_{\hat g}(\hat{\mathbf{x}}_t)\|_1$，其中 $\mathcal{M}(\mathbf{J})=(\mathbf{I}+\mathbf{J})^\top(\mathbf{I}+\mathbf{J})-\mathbf{I}$ 是 Markov 网络结构；DAG 约束 $\mathcal{L}_d=\mathcal{D}_g(\mathbf{J}_{\hat g}(\hat{\mathbf{x}}_t))+\mathcal{D}_g(\mathbf{J}_r(\hat{\mathbf{z}}_t))$，$\mathcal{D}_g(A)=\mathrm{tr}[(\mathbf{I}+\tfrac1d A\circ A)^d]-d$（沿用 Yu et al. 2019）。
    - 设计动机：把 ICA 的 invertibility 通过 DAG 拿掉之后，混合结构 $\mathbf{J}_m$ 仍是可逆方阵（Corollary 1），所以用解析闭式而不是再学一个 graph network 就能拿到 $\mathbf{J}_g$，推理快（CESM2 上仅 1.1 ms）且与理论同源；用 flow 而不是高斯 prior 是为了让"独立成分"假设真的成立，让 KL 项有效驱动 $\hat{\bm{\epsilon}}^z, \hat{\bm{\epsilon}}^x \sim \mathcal{N}(\mathbf{0},\mathbf{I})$，从而满足 Theorem 3.6 中关于条件独立的前提。

### 损失函数 / 训练策略
总损失 $\mathcal{L}_{ALL}=\mathcal{L}_{ELBO}+\alpha\mathcal{L}_s+\beta\mathcal{L}_d$。ELBO 的两个 KL 系数 $\lambda_1=4\times10^{-3}$、$\lambda_2=1.0\times10^{-2}$（控 s 与 z 的先验匹配强度），结构损失系数 $\alpha=1.0\times10^{-4}$、$\beta=5.0\times10^{-5}$。z/s encoder、decoder、prior 网络全部用 MLP，prior 端用 normalizing flow 把后验逼近成 $\mathcal{N}(\mathbf{0},\mathbf{I})$。一阶 Markov 假设下用三步窗口作为上下文，二阶以上的推广在附录 E.3。

## 实验关键数据

### 主实验
合成数据上做"潜在表征恢复"和"观测因果图恢复"的双指标对比，气候数据上做"因果图质量"和"温度预测"的双任务对比。

| 数据集 | 指标 | CaDRe | 之前最优 | 提升 / 差异 |
|--------|------|-------|----------|------|
| Synthetic Independent ($d_z=3,d_x=10$) | MCC | **0.9811** | TDRL 0.9106 | +0.07，潜在因子几乎完全恢复 |
| Synthetic Sparse | MCC | **0.9306** | G-CaRL 0.7701 | +0.16，最贴近实际气候稀疏结构的设定 |
| Synthetic Dense | MCC | **0.6750** | G-CaRL 0.6714 | 持平，密集潜图下所有方法都退化 |
| CESM2 (CD) | WSHD ↓ | **0.012** | LPCMCI 0.019 | 比所有 PC/FCI/CDNOD/PCMCI/LPCMCI/TCDF/TDRL/IDOL 都低 |
| CESM2 (CD) | WTPR ↑ | **0.532** | TCDF 0.327 | 真边召回比次优高 60% |
| CESM2 (CD) | Latency (ms) ↓ | **1.095** | TDRL 0.974 | 与最快 baseline 一档，比 PCMCI/LPCMCI 快 ~3000× |
| CESM2 Forecast (H=96) | MSE | **0.410** | CARD 0.409 | 与 SOTA 持平，但同时给出可解释因果图 |
| CESM2 Forecast (H=192) | MSE | **0.412** | CARD 0.422 | 长程预测略优 |

### 消融实验
论文用三档合成结构（Independent / Sparse / Dense）天然构成了一组结构层面的消融，下表抽出代表性对比：

| 配置 | MCC (Sparse) | $R^2$ (Sparse) | 说明 |
|------|-------------|---------------|------|
| CaDRe（完整） | 0.9306 | 0.9102 | z/s 双 encoder + flow prior + 稀疏 + DAG |
| iCITRIS（无 $\mathbf{s}_t$ 显式建模） | 0.4531 | 0.6326 | 没把内生噪声从潜在因子里剥离 → 因子缠绕 |
| TDRL / LEAP（无观测 DAG 约束） | 0.6628 / 0.6453 | 0.6953 / 0.4637 | 把观测间因果当成独立噪声 → 稀疏场景下严重掉点 |
| G-CaRL（无 flow + 弱条件先验） | 0.7701 | 0.5443 | prior 表达力不够，潜在成分独立性失效 |

### 关键发现
- 稀疏场景（最贴合气候真实结构）下，CaDRe 比次优方法 MCC 高 0.16、$R^2$ 高 0.22——说明"显式建模 $\mathbf{s}_t$ + DAG 约束观测图"两个设计在结构稀疏时贡献最大；Dense 设定下所有方法都退化，反向印证了稀疏性假设（附录 A.3）是 component-wise 可识别性的实质前提。
- 在 CESM2 上推理延迟比 PCMCI/LPCMCI 这类经典 CD 快约 3 个数量级（1.1 ms vs 3000+ ms），原因是 $\mathbf{J}_g$ 由 decoder 的 $\mathbf{J}_m$ 一步闭式得到，而非每次都跑条件独立检验。
- 在温度预测任务上，CaDRe 与 CARD、iTransformer 等纯预测模型几乎打平，但额外免费交付一张与领域专家认知一致的因果图——说明"加因果结构"在这一类任务上没有付出明显预测精度代价。

## 亮点与洞察
- **DAG 救了 ICA 的可逆性**：传统非线性 ICA 必须假设 $\mathbf{z}\to\mathbf{x}$ 可逆，本文用 SEM 的 DAG 直接证出 $\mathbf{J}_m$ 可逆（Corollary 1）。这意味着只要数据生成符合无环假设，就可以放心地把噪声维数和观测维数对齐，让 ICA 在"有隐混杂 + 有观测耦合"的更现实场景里跑起来。
- **把"观测因果图"写成 decoder 雅可比的解析函数**：$\mathbf{J}_g=\mathbf{I}-\mathbf{D}_m\mathbf{J}_m^{-1}$ 是这套理论里最实用的一条——一旦 VAE 训好，因果图就免费生成、随时间变化、毫秒级推理，可以直接迁移到任何 "想在生成模型基础上画一张时变因果图" 的任务（例如脑科学 fMRI、宏观经济时序）。
- **把"上下文"用在分布级 ICA 上**：用 $\{\mathbf{x}_{t-1},\mathbf{x}_t,\mathbf{x}_{t+1}\}$ 三窗口构造算子单射条件（A2）这一招，本质上是把传统辅助变量 ICA 的"外生标签"换成了"时间邻域"，对所有没有显式 domain label 的连续时序系统都适用。

## 局限与展望
- 一阶 Markov + 稀疏潜在结构是默认假设；高阶 Markov 虽然在附录 E.3 给了推广，但实际气候系统的多尺度耦合（日尺度、季节尺度、年尺度）能否在同一套 flow 里被一次性识别还没实证。
- Assumption A5（generation variability）要求 $2d_x$ 个雅可比向量线性独立，是较强的非退化条件；当观测维数 $d_x$ 增大、变量高度共线时，该假设在实际气候网格上的可验证性需更仔细的讨论。
- 实验主要在 CESM2 单一气候数据集上跑，没在更多真实物理场景（如海洋温盐环流、生态系统）上系统验证；同时 forecasting 部分只对比了 96/192 这两个相对短的 horizon。
- decoder $\mathbf{J}_m^{-1}$ 在高维时数值稳定性可能成为瓶颈，论文未给出大规模观测维数下的稳定性分析。

## 相关工作与启发
- **vs iCITRIS / TDRL / LEAP / G-CaRL（CRL 系）**：这些方法只识别潜在因子和潜在因果，把观测之间的依赖当噪声；本文同时识别"观测图 + 潜图 + 跨层效应"，在稀疏设定下 MCC 高 0.16–0.5。本文优势在于把内生噪声 $\mathbf{s}_t$ 显式分离出来；代价是模型更复杂、需要两个 prior 网络。
- **vs FCI / LPCMCI（CD 系）**：FCI/LPCMCI 在隐混杂下只能给 PAG 等价类，且非参数 CI 检验在高维时序上很慢。本文把 CD 转成 ICA + 雅可比读图，既能识别到"具体的有向图"，又把推理代价从秒级压到毫秒级。
- **vs CDSD（brouillard2024causal）**：CDSD 也尝试统一 CRL 与 CD，但它不识别观测之间的因果图；本文是 Table 2 中唯一在五项属性（非参数 / 潜变量 / 潜在因果图 / 观测因果图 / 无等价类）上全部勾选的方法。
- **启发**：这种"用 DAG 让 ICA 抛掉可逆性 + 用 flow prior 让独立性真正满足 + 用 Jacobian 读图"的三件套，可能是把"任何带 VAE 风格生成器的时序模型"升级成"自带因果解释"的通用范式。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个在非参数 + 隐动力 + 观测因果图三者同时存在时给出 component-wise 可识别性的统一框架
- 实验充分度: ⭐⭐⭐⭐ 合成数据三档结构 + CESM2 因果图 + 温度预测三个任务齐备，但真实气候数据集仅 CESM2 一个
- 写作质量: ⭐⭐⭐⭐ 理论推进与方法落地衔接清晰，三个定理之间的依赖明示得很好；个别符号（如 $\mathbf{J}_d$ vs $\mathbf{J}_r$）在公式中略有混用
- 价值: ⭐⭐⭐⭐⭐ 同时解决 CRL 与 CD 两条线长期各自为政的痛点，且产出可在气候/脑科学/经济等领域复用的"VAE + 因果图"通用模板

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Recovering Hidden Reward in Diffusion-Based Policies](recovering_hidden_reward_in_diffusion-based_policies.md)
- [\[ICML 2026\] Caracal: Causal Architecture via Spectral Mixing](caracal_causal_architecture_via_spectral_mixing.md)
- [\[ICLR 2026\] Embracing Discrete Search: A Reasonable Approach to Causal Structure Learning](../../ICLR2026/image_generation/embracing_discrete_search_a_reasonable_approach_to_causal_structure_learning.md)
- [\[ICML 2026\] Image Restoration via Diffusion Models with Dynamic Resolution](image_restoration_via_diffusion_models_with_dynamic_resolution.md)
- [\[CVPR 2026\] DynaVid: Learning to Generate Highly Dynamic Videos using Synthetic Motion Data](../../CVPR2026/image_generation/dynavid_learning_to_generate_highly_dynamic_videos_using_synthetic_motion_data.md)

</div>

<!-- RELATED:END -->
