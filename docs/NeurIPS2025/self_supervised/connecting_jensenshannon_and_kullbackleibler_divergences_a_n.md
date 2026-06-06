---
title: >-
  [论文解读] Connecting Jensen-Shannon and Kullback-Leibler Divergences: A New Bound for Representation Learning
description: >-
  [NeurIPS 2025][自监督学习][mutual information] 推导了一般情况下KL散度关于JS散度的最优紧致下界$\Xi(D_{\text{JS}}) \leq D_{\text{KL}}$，证明训练判别器最小化交叉熵损失等价于最大化互信息的一个保证下界…
tags:
  - "NeurIPS 2025"
  - "自监督学习"
  - "mutual information"
  - "Jensen-Shannon divergence"
  - "KL divergence"
  - "variational bound"
  - "representation learning"
---

# Connecting Jensen-Shannon and Kullback-Leibler Divergences: A New Bound for Representation Learning

**会议**: NeurIPS 2025  
**arXiv**: [2510.20644](https://arxiv.org/abs/2510.20644)  
**代码**: [https://github.com/ReubenDo/JSDlowerbound](https://github.com/ReubenDo/JSDlowerbound)  
**领域**: 自监督学习 / 信息论  
**关键词**: mutual information, Jensen-Shannon divergence, KL divergence, variational bound, representation learning

## 一句话总结
推导了一般情况下KL散度关于JS散度的最优紧致下界$\Xi(D_{\text{JS}}) \leq D_{\text{KL}}$，证明训练判别器最小化交叉熵损失等价于最大化互信息的一个保证下界，为JSD基于的判别式表示学习方法提供了缺失的理论基础，并在MI估计和Information Bottleneck框架中验证了紧致性与实用性。

## 研究背景与动机

**互信息在表示学习中的核心地位**：互信息（MI）作为统计依赖性的标准度量，是众多表示学习框架的理论基石——从信息瓶颈（Information Bottleneck）到对比学习（CPC/SimCLR），其目标都可以表述为最大化或约束MI。MI的定义是联合分布与边际乘积之间的KL散度：$I[U;V] = D_{\text{KL}}[p_{UV} \| p_U \otimes p_V]$。

**直接优化MI的困境**：然而直接优化MI在实践中通常不可计算。现有的变分下界（VLB）方法如MINE（基于Donsker-Varadhan表示）和NWJ虽然提供了MI的可优化下界，但存在高方差、不稳定、需要对抗训练等问题。InfoNCE通过对比学习给出了稳定的下界，但受batch size $b$限制——估计值上界为$\log b$，无法捕获高MI值。

**JSD替代优化的成功与理论缺失**：在实践中，许多成功的方法（如Deep InfoMax）绕过了KLD的直接优化，转而最大化联合分布与边际乘积之间的Jensen-Shannon散度（JSD），即$I_{\text{JS}}[U;V] = D_{\text{JS}}[p_{UV} \| p_U \otimes p_V]$。JSD因其对称性和有界性（上界为$\log 2$），优化更稳定且不需要大batch size。Deep InfoMax的实验表明JSD与真实MI良好相关。但一个根本性的理论问题未被回答：**最大化JSD是否真的在最大化MI？两者之间的定量关系是什么？** 已知的不等式——JSD ≤ MI（trivial, JSD是MI的弱下界）和Pinsker不等式$D_{\text{KL}} \geq 2 D_{\text{JS}}$——要么太松要么仅在低MI时紧致，无法给出一般性的紧致定量联系。

**本文的核心目标**：填补这一理论空白——推导KLD关于JSD的最优（即最紧致的不可改进）下界，从而严格证明最大化JSD确实在增加MI的一个保证下界，并将这一理论联系与判别器训练（交叉熵最小化）完整连通。

## 方法详解

### 整体框架
本文建立了一条从判别器训练到MI最大化的完整理论链路：训练判别器区分联合样本和边际样本（最小化交叉熵$\mathcal{L}_{\text{CE}}$）→ 增加JSD的变分下界（$I_{\text{JS}} \geq \log 2 - \mathcal{L}_{\text{CE}}$）→ 增加MI的保证下界（$I[U;V] \geq \Xi(I_{\text{JS}})$）。最终得到端到端的MI下界：$\Xi(\log 2 - \mathcal{L}_{\text{CE}}) \leq \Xi(I_{\text{JS}}) \leq I[U;V]$。

### 关键设计

1. **KLD关于JSD的最优下界（Theorem 4.1）**:

    - 功能：推导对任意两个分布$p, q$成立的、最紧致的KLD-JSD不等式$\Xi(D_{\text{JS}}[p \| q]) \leq D_{\text{KL}}[p \| q]$
    - 核心思路：利用$f$-散度的联合值域（joint range）理论——给定两个$f$-散度$D_f$和$D_g$，它们在所有分布对$(p,q)$上的联合取值集合$\mathcal{R}_{f,g}$是凸集，其下包络即为最优下界。关键工具是Harremoës-Vajda定理（Theorem 3.1）：完整联合值域$\mathcal{R}_{f,g}$可仅通过Bernoulli分布对完全刻画，即$\mathcal{R}_{f,g} = \text{co}(\mathcal{R}_{2;f,g})$。因此只需分析映射$\phi: (\mu, \nu) \mapsto (D_{\text{JS}}[B(\mu) \| B(\nu)], D_{\text{KL}}[B(\mu) \| B(\nu)])$在单位正方形上的像即可。分析三条边界上的行为后，发现下界恰好由$\mu=1$这条边（即$D_{\text{KL}}[B(1) \| B(\nu)]$）对应的曲线给出。$\Xi$是严格递增函数，其逆函数有解析形式$\Xi^{-1}(y) = D_{\text{JS}}[B(1) \| B(e^{-y})]$
    - 设计动机：此前的Pinsker不等式$D_{\text{KL}} \geq 2 D_{\text{JS}}$仅在散度趋近零时紧致，$f$-散度文献中的联合值域理论虽然在经典信息论中早有研究，但从未被应用于表示学习和MI估计的context中。将这一经典工具引入现代ML场景是本文的核心理论贡献

2. **JSD的交叉熵变分下界**:

    - 功能：证明训练一个判别器最小化交叉熵损失等价于最大化JSD的$f$-散度变分下界
    - 核心思路：设置混合模型：$(U,V) | Z=1 \sim p_{UV}$（联合样本），$(U,V) | Z=0 \sim p_U \otimes p_V$（独立样本），$Z \sim B(1/2)$。通过$f$-散度的变分表示，将JSD重写为$I_{\text{JS}} = \frac{1}{2}\max_t [\mathbb{E}_{p_{UV}}[t] - \mathbb{E}_{p_U \otimes p_V}[-\log(2-e^t)]]$。令$t(u,v) = \log(2 q_\theta(z=1|u,v))$（判别器输出的重参数化），代入后得到$I_{\text{JS}} \geq \log 2 - \min_\theta \mathcal{L}_{\text{CE}}(\theta)$，其中$\mathcal{L}_{\text{CE}}$是判别器的二分类交叉熵损失。该界在非参数极限下紧致（判别器capacity无限且数据无限时取等）
    - 设计动机：GAN文献中Goodfellow等人已注意到最优判别器与JSD的对应关系，但未从MI估计的角度建立完整链路。本文的新贡献在于量化了非最优判别器带来的gap，并将其与MI下界串联

3. **MI估计器的构建（两步法）**:

    - 功能：基于上述理论，构建从数据到MI估计值的完整估计流程
    - 核心思路：利用联合模型的后验可得$I[U;V] = \mathbb{E}_{p_{UV}}[\mathbb{L}(\tilde{p}(z=1|u,v))]$，其中$\mathbb{L}(\cdot) = \log \frac{x}{1-x}$是Logit函数。两步估计：(1) 训练判别器$q_\theta$区分联合样本和边际样本（最小化CE loss）；(2) 将$q_\theta$代入上式得到MI估计。同时，$\Xi$的平滑可微近似为$\Xi(x) \approx 1.15 \cdot \mathbb{L}(0.5(x / \log 2 + 1))$，方便端到端优化
    - 设计动机：两步法将优化（训练判别器）和估计（计算MI）解耦，避免了VLB方法中优化目标与估计目标耦合带来的不稳定性。这一策略等价于GAN-DIME方法，本文给出了其理论基础

### 损失函数 / 训练策略
判别器使用标准二分类交叉熵训练：正样本为联合分布$(u,v) \sim p_{UV}$采样的配对，负样本为打乱配对$(u, v') \sim p_U \otimes p_V$获得的独立样本。网络结构为全连接网络（输入维度$2d$，两层256 ReLU隐藏层，标量输出）。训练4000步，Adam优化器，batch size 64。由于$\Xi$严格递增，最大化$\Xi(\log 2 - \mathcal{L}_{\text{CE}})$等价于最小化$\mathcal{L}_{\text{CE}}$，因此$\Xi$的近似不影响优化过程，仅在插入式MI估计时使用。

## 实验关键数据

### 主实验
**高斯分布MI估计（$d=5$, batch size 64, staircase设定）**：

| 估计器 | 低MI区 | 中MI区 | 高MI区(20 nats) | 方差 | 是否超出真实MI |
|--------|--------|--------|-----------------|------|--------------|
| MINE | 准确 | 偏高 | 严重高估 | 极高 | 经常超出 |
| NWJ | 准确 | 偏低 | 严重偏低 | 高 | 较少 |
| CPC (InfoNCE) | 准确 | 受限于$\log b=4.16$ | 受限 | 低 | 不超（有限） |
| **JSD-LB (ours)** | **准确** | **紧致下界** | **紧致下界** | **低** | **从不超出** |
| GAN-DIME (两步MI估计) | 准确 | 准确 | 最准确 | 低 | 偶尔略超 |

**Information Bottleneck（MNIST分类）**：

| 方法 | 测试准确率(%) | 对抗鲁棒性($\epsilon=0.1$) | 对抗鲁棒性($\epsilon=0.3$) | OOD检测AUROC(%) |
|------|-------------|--------------------------|--------------------------|---------------|
| VIB | 97.6 | 73.4 | 4.2 | 94.6 |
| NIB | 97.2 | 75.2 | 3.4 | 94.2 |
| DisenIB | 98.2 | 90.2 | 67.8 | - |
| **JSD-LB (ours)** | **98.8** | **94.6** | **86.1** | **最高** |

### 消融实验
**离散分布紧致性验证**：

| 类别数$k$ | 依赖强度$\alpha$ | $\Xi(I_{\text{JS}})$ | 真实MI | Gap |
|----------|----------------|---------------------|--------|-----|
| 2 | 0→1 | 紧贴下界 | 紧贴下界 | 极小 |
| 50 | 0→1 | 紧贴下界 | 略偏离 | 小 |
| 500 | 0→1 | 紧贴下界 | 略偏离 | 小 |

**非高斯分布泛化性（asinh/half-cube/Student/uniform）**：

| 分布 | JSD-LB | MINE | NWJ | CPC |
|------|--------|------|-----|-----|
| Cubic | 紧致低方差 | 高方差 | 严重偏低 | 受限 |
| Asinh | 紧致低方差 | 高方差 | 偏低 | 受限 |
| Student | 紧致低方差 | 方差大 | 偏低 | 受限 |

### 关键发现
- JSD-LB在所有MI范围（低/中/高）均提供紧致且稳定的下界，不像MINE/NWJ有高方差和overestimation风险
- 离散分布上，本文的一般性下界（对任意$p,q$）在特化到联合-边际对后仍接近紧致——存在无穷族离散分布对恰好落在下界曲线上
- Information Bottleneck中使用JSD-LB替代传统VLB，获得了更好的分类准确率（98.8%）、对抗鲁棒性（$\epsilon=0.3$时86.1% vs 67.8%）和OOD检测性能
- CPC/InfoNCE受batch size约束（$\leq \log 64 \approx 4.16$ nats），在高MI区完全失效；JSD-LB无此限制

## 亮点与洞察
- **经典信息论工具的现代应用**：$f$-散度联合值域这一经典概念（Harremoës-Vajda 2011）此前从未在表示学习领域使用过。作者将其引入MI估计问题，得到了最优下界——这种"跨领域嫁接经典结果"的研究范式值得学习
- **完整的理论链路**：从CE loss → JSD VLB → KLD下界 → MI保证，每一步都有严格的数学推导和明确的gap刻画。这为整个JSD基于的表示学习方法族（Deep InfoMax、SMILE等）提供了一直缺失的理论根基
- **Logit-like的$\Xi$函数形态**：下界函数$\Xi$类似于Logit函数，在JSD接近0时近线性（$\Xi(x) \approx 2x$，退化为Pinsker），在JSD接近$\log 2$时快速增长趋向无穷，完美捕获了高MI区JSD饱和但KLD仍在增长的行为

## 局限与展望
- 理论贡献为主，缺乏在大规模视觉SSL任务（如SimCLR/BYOL在ImageNet上的训练）中的直接应用验证
- $\Xi$函数虽有平滑近似但无闭式表达，需要数值求解或近似
- 下界毕竟是下界——在某些场景下估计值可能显著低于真实MI（尤其当判别器次优时gap叠加）
- JSD估计器本身的质量决定了下界的紧致程度，需要足够好的判别器

## 相关工作与启发
- **vs MINE**: MINE直接估计KLD的VLB，方差大且可能overestimate（超出真实MI），优化不稳定。JSD-LB通过JSD间接下界KLD，方差更低且保证不超出
- **vs InfoNCE/CPC**: InfoNCE给出$I[U;V] \geq \log b - \mathcal{L}_{\text{InfoNCE}}$的下界，受batch size $b$约束。JSD-LB无此限制，在高MI区优势明显
- **vs Deep InfoMax (Hjelm et al.)**: Deep InfoMax实证观察到JSD与MI良好相关并以此为依据使用JSD目标。本文给出了严格证明：$\Xi(I_{\text{JS}}) \leq I[U;V]$，最大化$I_{\text{JS}}$确实在提升MI的一个保证下界
- **vs GAN框架**: Goodfellow 2014指出最优判别器下GAN目标是JSD的仿射变换。本文从MI估计角度看待同一判别器，揭示了GAN损失与MI最大化的深层联系

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次建立一般性KLD-JSD最优紧致下界，将经典信息论工具引入现代表示学习，理论贡献突出
- 实验充分度: ⭐⭐⭐⭐ MI估计benchmark全面（高斯/非高斯/离散），IB应用有说服力，但缺乏大规模视觉SSL实验
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨清晰，Figure 1/2的可视化直观展示了联合值域和下界紧致性
- 价值: ⭐⭐⭐⭐ 为JSD基于的判别式SSL方法提供了缺失的理论基础，IB中的实际效果（86.1%对抗鲁棒性）令人印象深刻

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Soft Task-Aware Routing of Experts for Equivariant Representation Learning](soft_task-aware_routing_of_experts_for_equivariant_representation_learning.md)
- [\[ICML 2026\] Learning to Extrapolate to New Tasks: A Relational Approach to Task Extrapolation](../../ICML2026/self_supervised/learning_to_extrapolate_to_new_tasks_a_relational_approach_to_task_extrapolation.md)
- [\[NeurIPS 2025\] Adv-SSL: Adversarial Self-Supervised Representation Learning with Theoretical Guarantees](adv-ssl_adversarial_self-supervised_representation_learning_with_theoretical_gua.md)
- [\[NeurIPS 2025\] TRIDENT: Tri-Modal Molecular Representation Learning with Taxonomic Annotations and Structural Relationships](trident_tri-modal_molecular_representation_learning_with_taxonomic_annotations_a.md)
- [\[NeurIPS 2025\] STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking](starformer_semi-supervised_task-informed_representation_learning_via_dynamic_att.md)

</div>

<!-- RELATED:END -->
