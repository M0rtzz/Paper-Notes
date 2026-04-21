---
title: >-
  [论文解读] Information-Theoretic Discrete Diffusion
description: >-
  [NeurIPS 2025][图像生成][离散扩散模型] 将连续扩散中经典的 I-MMSE 恒等式推广到离散域，建立 I-MDSE 和 I-MDCE 关系——证明 DSE/DCE 损失不仅是变分上界而是对数似然的**精确分解**，并由此推导出 time-free 公式、条件似然估计和耦合似然比估计器，在 LLaDA 等大模型上验证了低方差和 OOD 检测能力。
tags:
  - NeurIPS 2025
  - 图像生成
  - 离散扩散模型
  - 信息论
  - 似然估计
  - 分数匹配
  - 扩散模型
  - I-MMSE
---

# Information-Theoretic Discrete Diffusion

**会议**: NeurIPS 2025  
**arXiv**: [2510.24088](https://arxiv.org/abs/2510.24088)  
**代码**: [github.com/Dongjae0324/infodis](https://github.com/Dongjae0324/infodis)  
**领域**: 生成模型理论/离散扩散  
**关键词**: 离散扩散模型, 信息论, 似然估计, 分数匹配, masked diffusion, I-MMSE

## 一句话总结
将连续扩散中经典的 I-MMSE 恒等式推广到离散域，建立 I-MDSE 和 I-MDCE 关系——证明 DSE/DCE 损失不仅是变分上界而是对数似然的**精确分解**，并由此推导出 time-free 公式、条件似然估计和耦合似然比估计器，在 LLaDA 等大模型上验证了低方差和 OOD 检测能力。

## 背景与动机

连续扩散模型（如高斯扩散）有一套成熟的信息论基础：Guo et al. (2005) 的 **I-MMSE 恒等式**将互信息的变化率与最小均方误差联系起来，Kong et al. (2023) 进一步将其推广为对数似然的 pointwise 分解。然而，离散扩散模型（处理文本、DNA 等类别数据）虽然发展迅速（D3PM、SEDD、LLaDA 等），却缺乏与之对应的信息论框架。

现有的离散扩散训练损失（DSE, DCE）通常被视为负对数似然的**变分上界**，这留下两个未解问题：

1. 这些损失是否只是上界，还是能精确估计似然？
2. 能否像连续情况一样，用一阶 score 函数就获得精确的似然分解？

## 核心问题

**目标**：为离散扩散模型建立严格的信息论框架，证明训练损失与对数似然之间的精确（等式而非不等式）关系，并由此衍生出实用的似然估计方法。

## 方法详解

### 1. I-MDSE 关系（一般离散扩散）

考虑由连续时间马尔可夫链（CTMC）驱动的离散扩散前向过程 $\frac{dp_t}{dt} = Q_t p_t$。定义最小去噪 score 熵（MDSE）：

$$\mathrm{mdse}(x_0, t) := \mathbb{E}_{p_{t|0}(x_t|x_0)}[\ell_{\mathrm{DSE}}(x_0, x_t, t, s_t^\star)]$$

其中 $s_t^\star$ 是使 DSE 损失最小的最优 score 函数。

**定理 3.1（I-MDSE 恒等式）**：

$$\frac{d}{dt} D_{\mathrm{KL}}(p_{t|0}(\cdot|x_0) \| p_t) = -\mathrm{mdse}(x_0, t)$$

取期望得到边际形式：$\frac{d}{dt} I(x_0; x_t) = -\mathrm{mdse}(t)$

**核心洞察**：互信息的衰减速率恰好等于最小 DSE 损失的负值。这是连续域 I-MMSE 恒等式的离散对应。

**定理 3.2（NLL 分解）**：对时间积分可得

$$-\log p_0(x_0) = \int_0^\infty \mathrm{mdse}(x_0, t) \, dt$$

→ DSE 损失不是变分上界，而是对数似然的**精确估计器**。

### 2. I-MDCE 关系（masked 扩散）

对于实际更常用的 absorbing（掩码）扩散模型，引入去噪交叉熵（DCE）损失：

$$\ell_{\mathrm{DCE}}(\mathbf{x}_0, \mathbf{x}, c) := \sum_{i=1}^L \mathbb{1}[x^i = [\mathrm{M}]] \log \frac{1}{c(\mathbf{x})_{i, x_0^i}}$$

**引理 3.3（逐点等价）**：通过时间重参数化 $\lambda = 1 - e^{-\bar{\sigma}(t)}$，DSE 和 DCE 在逐点层面精确等价：

$$\ell_{\mathrm{DSE}}(\mathbf{x}_0, \mathbf{x}, t, s_t) = \frac{\bar{\sigma}(t)(1-\lambda)}{\lambda} \ell_{\mathrm{DCE}}(\mathbf{x}_0, \mathbf{x}, c)$$

**定理 3.4（训练损失等价）**：$\mathcal{L}_{\mathrm{DSE}}^T(\mathbf{x}_0) = \mathcal{L}_{\mathrm{DCE}}^\Lambda(\mathbf{x}_0)$，对任意有限 $T$，不仅仅是 $T \to \infty$ 时的渐近等价。

**推论 3.7（I-MDCE NLL 分解）**：

$$-\log p_0(\mathbf{x}_0) = \int_0^1 \frac{1}{\lambda} \mathrm{mdce}(\mathbf{x}_0, \lambda) \, d\lambda$$

### 3. Time-Free 似然估计

时间积分形式在实际中需要采样扩散时间。本文推导出等价的**无时间公式**：

**定理 4.1**：

$$-\log p_0(\mathbf{x}_0) = H_L \, \mathbb{E}_{p(I)} \left[ \sum_{i \notin I} \log \frac{1}{p_0(x_0^i | \mathbf{x}_0^I)} \right]$$

其中 $I$ 是按 Beta 分布 $p(I) = B(L-|I|, |I|+1)/H_L$ 采样的 unmasked 索引集，$H_L$ 是第 $L$ 个调和数。直接随机 mask 子集做 Monte Carlo 估计，无需显式时间积分。

### 4. 条件似然估计

**定理 4.2**：对不相交索引集 $I_1$（target）和 $I_2$（context），有

$$-\log p_0(\mathbf{x}_0^{I_1} | \mathbf{x}_0^{I_2}) = \int_0^1 \frac{1}{\lambda} \mathbb{E}[\cdots] \, d\lambda$$

同样可转化为 time-free 形式（推论 4.3），适用于 prompt-response 建模场景。

### 5. 耦合似然比估计

利用**共享 mask**（相同的随机 unmasked 集 $I$ 同时用于两个序列），得到耦合估计器：

$$\log \frac{p_0(\mathbf{y})}{p_0(\mathbf{x})} = H_L \, \mathbb{E}_{p(I)} \left[ \sum_{i \notin I} \log \frac{p_0(y^i | \mathbf{y}^I)}{p_0(x^i | \mathbf{x}^I)} \right]$$

共享随机性使正相关项相互抵消，显著降低方差，对 alignment（如 DPO）等下游任务有用。

## 训练与推理

### 训练

模型训练采用标准的 masked diffusion 框架：对序列 $\mathbf{x}_0$ 按 masking rate $\lambda$ 随机掩码，训练网络 $c^\theta$ 预测被掩码 token 的条件分布。损失函数即为 DCE loss：

$$\mathcal{L}_{\mathrm{DCE}}(\mathbf{x}_0) = \int_0^1 \frac{1}{\lambda} \mathbb{E}_{p_{\lambda|0}} \left[ \sum_i \mathbb{1}[x_\lambda^i = [\mathrm{M}]] \log \frac{1}{c^\theta(\mathbf{x}_\lambda)_{i, x_0^i}} \right] d\lambda$$

本文的核心理论贡献在于：证明该损失在最优 $c^\theta = c^\star$ 时**精确等于** $-\log p_0(\mathbf{x}_0)$，而非仅为其上界。因此现有 masked diffusion LM（如 LLaDA）的训练目标已是理论最优的似然最大化，无需额外修正项。

实际训练中，时间 $\lambda$ 通过均匀采样 Monte Carlo 近似积分，网络架构为标准 Transformer（与 LLaDA 一致），采用 time-independent 参数化。

### 推理/似然估计

本文的主要推理任务不是生成（采样），而是**似然估计**。提供三种实用方式：

1. **Time-integral 估计**：直接在多个 $\lambda$ 上采样，数值积分 DCE loss，与训练损失形式一致
2. **Time-free 估计**（推荐）：随机采样 unmasked 子集 $I$，按 Beta 分布加权，无需显式时间积分，方差降低 5-7 倍
3. **耦合似然比**：两个序列共享同一组 mask 索引 $I$，对 $\log$-ratio 做 Monte Carlo 估计，方差降低约 7 倍

Time-free 估计的计算成本约为 $K$ 次前向推理（$K$ 为 MC 样本数），每次只需对不同 mask 模式做一次模型推理，实现简单。

## 实验

### 实验设置

- **合成数据**：128 个长度为 8 的 DNA 序列（字母表 {A,T,G,C}），已知精确分布；4th-order Markov chain 生成的 5M 长 DNA 序列（用于条件似然验证）
- **真实数据**：LLaDA 8B 模型（Nie et al., 2025），text8 语料 + RADD 模型
- **评估基准**：HellaSwag、ARC-hard、PIQA（条件似然方差）；BeaverTails（似然比方差）

### 关键结果

| 实验 | 关键结果 |
|------|---------|
| **合成 DNA 似然恢复** | time-free 估计值与真实 NLL 高度吻合（图 1a 无条件 / 图 1b 条件） |
| **方差对比（HellaSwag）** | 128 MC 样本：time-integral 方差 70.97 → time-free 方差 **11.57**（降低 6×） |
| **方差对比（ARC-hard）** | 128 MC 样本：23.18 → **5.73** |
| **方差对比（PIQA）** | 128 MC 样本：19.77 → **4.93** |
| **似然比方差（BeaverTails）** | 5 MC 样本：decoupled 62469 → coupled **8897**（降低 7×） |
| **OOD 检测（text8 + RADD）** | NLL 直方图清晰分离 in-distribution vs GPT-4 生成文本 |
| **LLaDA 模型审计** | LLaMA 3.1 生成文本比 WikiText 获得更高似然 → 暗示 LLaDA 训练数据可能受 LLaMA 3.1 影响 |

### 结果分析

**方差优势随样本增多而保持**：从 128 到 512 MC 样本，time-free 在 HellaSwag 上方差从 11.57 降至 2.92（vs time-integral 从 70.97 降至 13.38），相对优势始终约 4-6 倍。这说明方差降低不是偶然的，而是 time-free 公式消除了时间采样噪声的结构性优势。

**耦合估计器的优势更显著**：在 BeaverTails 上仅用 5 个 MC 样本，耦合方差就比解耦低 7 倍，且随样本增加差距保持稳定。这对需要高效似然比估计的 alignment 任务（如 DPO）具有直接实用价值。

**模型审计发现**：LLaDA 对 LLaMA 3.1 生成文本赋予更高似然这一发现暗示了训练数据中可能包含合成数据（data contamination）。这一应用场景展示了精确似然估计在 AI safety 和 model provenance 方面的潜力。

## 亮点

1. **从不等式到等式**：证明 DSE/DCE 损失不是变分上界而是对数似然的精确分解——信息论层面的根本性突破
2. **统一框架**：I-MDSE（一般离散扩散）和 I-MDCE（masked 扩散）构成完整的信息论工具包
3. **实用性强**：time-free 公式消除了时间积分的 Monte Carlo 噪声，方差降低 5-7 倍
4. **耦合似然比**：共享 mask 的技巧简单优雅，方差降低约 7 倍，直接适用于 alignment
5. **模型审计**：用条件似然检测 OOD 和推断训练数据来源，展示了理论成果的实际价值
6. **推论深远**：证明一阶 score 足以精确重建似然，无需高阶修正

## 局限与展望

1. **实验规模有限**：真实数据实验主要用 LLaDA 8B，未在更大模型（如几十B参数）上验证
2. **仅限离散域**：框架专为离散扩散设计，对连续-离散混合扩散模型（如图文联合生成）的推广尚未探讨
3. **OOD 检测仅定性**：模型审计实验以可视化为主，缺乏 AUROC 等定量评估指标
4. **条件似然估计精度**：当 context 和 target 长度极不平衡时，估计器的行为未深入分析
5. **与 ELBO 的关系**：虽然证明了等式关系，但与 VAE 框架下的 ELBO 之间的深层联系未充分讨论

## 与相关工作的对比

| 方法 | 定位 | 本文优势 |
|------|------|---------|
| **SEDD** (Lou et al., 2024) | DSE 损失作为变分上界训练 | 本文证明 DSE 是精确似然估计器 |
| **RADD** (Ou et al., 2025) | DSE-DCE 渐近等价（$T \to \infty$） | 本文证明对任意有限 $T$ 精确等价 |
| **LLaDA** (Nie et al., 2025) | 大规模 masked diffusion LM | 本文为其训练损失提供信息论正当性 |
| **Kong et al. (2023/2024)** | 连续扩散 I-MMSE/条件似然 | 本文是其离散域推广 |
| **Zhu et al. (2025)** | masked diffusion alignment | 本文为似然比估计提供理论基础 |

## 个人评注 (My Notes)

### 启发与关联

- **对 masked diffusion LM 训练的理论保证**：本文直接证明了 LLaDA 等模型的 DCE 训练目标就是在最小化真实 NLL，而不是某个松弛的上界——这使得 masked diffusion 在理论上与 autoregressive 模型的 MLE 具有同等地位
- **Alignment 应用**：耦合似然比估计器天然适配 DPO 等偏好学习方法，可以更稳定地估计 $\log \frac{p_\theta(y_w)}{p_\theta(y_l)}$
- **模型溯源**：条件似然估计 + OOD 检测的组合可用于检测数据污染和训练数据来源分析（membership inference 的信息论版本）
- **连续↔离散的桥梁**：I-MMSE → I-MDSE/I-MDCE 的推广路径暗示，更多连续扩散的信息论工具（如 capacity、编码定理）或许也能推广到离散设定

### 论文意义

这篇工作的核心贡献不在于提出新模型或新算法，而在于为已有方法提供了更深刻的理论理解。它将 DSE/DCE 从"好用但理论保证不足的训练目标"提升为"信息论层面精确正当的似然估计器"。这种理论深化对领域的影响是长期的：它让研究者可以放心地基于 DCE 做后续理论推导（如收敛性分析、模型比较），而不必担心变分间隙带来的理论漏洞。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 首次建立离散扩散的完整信息论框架，将变分上界提升为精确等式
- 实验充分度: ⭐⭐⭐⭐ — 合成验证 + 方差分析 + 模型审计覆盖全面，但缺少大规模定量评估
- 写作质量: ⭐⭐⭐⭐⭐ — 数学严谨、逻辑清晰，理论与实验衔接自然
- 价值: ⭐⭐⭐⭐⭐ — 为离散扩散模型提供根本性理论基础，对训练、评估和下游应用均有深远影响

<!-- RELATED:START -->

## 相关论文

- [ItDPDM: Information-Theoretic Discrete Poisson Diffusion Model](itdpdm_information-theoretic_discrete_poisson_diffusion_model.md)
- [Information Theoretic Learning for Diffusion Models with Warm Start](information_theoretic_learning_for_diffusion_models_with_warm_start.md)
- [Constrained Discrete Diffusion](constrained_discrete_diffusion.md)
- [MMG: Mutual Information Estimation via the MMSE Gap in Diffusion](mmg_mutual_information_estimation_via_the_mmse_gap_in_diffusion.md)
- [Learnable Sampler Distillation for Discrete Diffusion Models](learnable_sampler_distillation_for_discrete_diffusion_models.md)

<!-- RELATED:END -->
