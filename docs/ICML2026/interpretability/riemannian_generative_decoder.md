---
title: >-
  [论文解读] Riemannian Generative Decoder
description: >-
  [ICML 2026][可解释性][黎曼流形] 本文针对 Riemannian VAE 必须为每种流形手工设计复杂概率密度的痛点，提出 Riemannian Generative Decoder (RGD)——彻底丢掉 encoder…
tags:
  - "ICML 2026"
  - "可解释性"
  - "黎曼流形"
  - "生成式解码器"
  - "几何正则化"
  - "MAP 估计"
  - "计算生物学"
---

# Riemannian Generative Decoder

**会议**: ICML 2026  
**arXiv**: [2506.19133](https://arxiv.org/abs/2506.19133)  
**代码**: https://github.com/yhsure/riemannian-generative-decoder (有)  
**领域**: 表示学习 / 几何深度学习 / 可解释生成模型  
**关键词**: 黎曼流形、生成式解码器、几何正则化、MAP 估计、计算生物学

## 一句话总结
本文针对 Riemannian VAE 必须为每种流形手工设计复杂概率密度的痛点，提出 Riemannian Generative Decoder (RGD)——彻底丢掉 encoder，把每个样本的 latent 当作自由参数用黎曼优化器 (RiemannianAdam) 直接训，同时引入"按局部度量逆缩放的输入噪声"作为几何正则，在合成分支扩散树、人类线粒体 DNA、细胞周期 scRNA-seq 三个真实生物数据上恢复出更忠实的几何，且在高维下数值稳定胜过 VAE 基线。

## 研究背景与动机

**领域现状**：真实数据（进化树、社交网络、周期信号）常具有非欧几何结构，但主流表示学习几乎都假设 $\mathbb{R}^d$ 隐空间，导致几何信息被强行压扁。Riemannian VAE 系列（$\mathcal{S}$-VAE 用 von Mises–Fisher、$\mathcal{P}$-VAE 用 Poincaré、$\Delta$VAE 用 Brownian motion 采样）尝试把 VAE 搬到球面、双曲、torus 等流形上，是当前 geometry-aware representation learning 的主流。

**现有痛点**：每种流形都需要专门设计概率密度，难处包括 (i) von Mises–Fisher 归一化常数含贝塞尔函数；(ii) Poincaré 上 Riemannian 正态分布的归一化常数 + volume correction 都得 Monte Carlo；(iii) wrapped normal 需要截断求和；(iv) $\Delta$VAE 走 Brownian motion 模拟。所有这些近似既数值不稳定（高维下尤其爆炸），又把流形选择牢牢绑在"是否能写出 tractable density"上，导致无法直接处理 ProductManifold 等异构组合几何。

**核心矛盾**：VAE 框架要求 encoder $q_\phi(z|x)$ 与 prior $p(z)$ 都在流形上有可算的密度，但绝大多数 Riemannian 流形上根本没有 closed-form density。强行近似的代价是 bias、unstable optimizer 与 limited manifold coverage——三者直接互锁。

**本文目标**：(i) 让表示学习能在**任何** geoopt 支持的黎曼流形上做（包括 Product 组合）；(ii) 摆脱对 manifold density 的依赖；(iii) 给出一个**几何感知**的正则项，使 decoder 平滑度与流形局部度量对齐而不是固定欧式同向。

**切入角度**：Deep Generative Decoder (DGD) 思路——不要 encoder，把 latent $z_i$ 当 free parameter 用 MAP 直接优化。把这一思路升级为 Riemannian DGD：用黎曼优化器替代欧式 Adam，每个 latent 始终留在流形上，从而绕过 density 近似。

**核心 idea**：抛掉 encoder + 用 RiemannianAdam 把 latent 当流形上的自由参数 + 在训练中加入协方差为 $\sigma^2 G^{-1}(z)$ 的几何噪声作为正则，让 decoder 在度量大的方向自然变光滑。

## 方法详解

### 整体框架
RGD 要解决的是 Riemannian VAE 必须为每种流形手写概率密度这一痛点，做法是把整个 encoder 删掉。给定数据 $X=\{x_i\}_{i=1}^N\in\mathbb{R}^D$ 和一个选定的 $d$ 维黎曼流形 $(\mathcal{M},g)$，它把每个样本的隐变量 $z_i$ 当成一个可学习的自由参数，和 decoder $f_\theta:\mathcal{M}\to\mathcal{X}$ 一起做 MAP 估计 $\mathcal{L}(\theta,Z)=\sum_i(-\log p_\theta(x_i|z_i)-\log p(z_i))-\log p(\theta)$。训练时 $\theta$ 用普通 Adam 走欧式步、$Z$ 用 RiemannianAdam 走黎曼步，每步靠 retraction $R_z(\cdot)$ 把切向量映回流形，保证 $z_i$ 始终留在曲面上；同时给 latent 注入按局部度量逆缩放的几何噪声作正则，让 decoder 的平滑方向与流形几何对齐。于是任何 density 近似都不再需要，借助 geoopt 库实现极其简洁。

### 关键设计

**1. Encoder-less MAP + RiemannianAdam：把 latent 当流形上的自由参数直接优化**

VAE 的麻烦根源在 encoder：$q_\phi(z|x)$ 必须在曲面上是一个可算密度的概率分布，而绝大多数黎曼流形上根本没有 closed-form density。RGD 干脆放弃 amortized inference，对每条样本的隐变量直接用黎曼梯度 $\nabla_z^{\mathcal{R}}\mathcal{L}=G(z)^{-1}\nabla_z^E\mathcal{L}$ 配合 retraction（通常取 exponential map）更新 $z^{(t+1)}=R_{z^{(t)}}(-\eta\,\nabla_{z^{(t)}}^{\mathcal{R}}\mathcal{L})$；RiemannianAdam 在切空间里维护自适应方向，收敛速度与 Adam 相当却每步严格停在流形上。prior 也随之大幅简化——compact 流形直接取常数 $1/\text{Vol}(\mathcal{M})$（不影响梯度），non-compact 才退而用 wrapped 或 Riemannian normal。Goldberg 的 DGD 已经证明这种丢掉 encoder 的 MAP 范式在欧式空间可行，RGD 把它原样 lift 到流形上就绕过了整条 density 近似链；一个顺带的红利是 ProductManifold 这类异构积流形也立刻可用，因为框架不再需要为每个流形单独写 prior。

**2. 几何感知的输入噪声正则：用度量逆缩放让 decoder 平滑度对齐流形几何**

光选对流形只是给了空间，要让 decoder 真正把"几何上该相似的点"解到相似输出，还得有一个与度量对齐的正则。RGD 在训练时给 latent 注入协方差按流形度量逆缩放的噪声 $\epsilon\sim\mathcal{N}(0,\sigma^2 G^{-1}(z))$——噪声在度量大的方向更弱、小的方向更强——再用 $z'=\text{Exp}_z(\epsilon)\approx z+\epsilon$ 映回流形。沿 Bishop (1995) 的二阶 Taylor 展开可证，这等价于在损失上加一项被度量加权的 Jacobian 范数惩罚 $\mathbb{E}_\epsilon[L(z')]\approx L(z)+\sigma^2\,\text{Tr}\big(J(z)^\top G^{-1}(z)J(z)\big)$，其中 $J(z)=\partial_z f$。这一步之所以有效，是因为普通欧式高斯噪声在曲面上会过度惩罚度量大的方向、欠惩罚小的方向；换成 $G^{-1}(z)$ 缩放后，正则在球面这种 homogeneous 流形上退化为近似各向同性、在曲率非均匀的双曲流形上则按位置自适应。相比 Lee & Park (2023) 涉及 Hessian-vector product、公式占整页的二阶曲率正则，RGD 只需一次 Jacobian 计算，可扩展性强一个量级。

**3. 统一框架支持任意黎曼流形与 Product 组合：把流形从算法假设降为配置项**

以往每引入一种新流形都要重推 prior 与近似，研究者根本没法快速比较"哪种几何更适合我的数据"。RGD 由于不依赖 manifold density，可以直接复用 geoopt 已实现的全部流形（Euclidean、Sphere、Stereographic、PoincareBall、Lorentz、SPD、Stiefel、UpperHalf、ProductManifold…），只要该流形给出 exponential map / retraction / 度量即可。异构需求通过积流形 $\mathcal{M}=\mathcal{M}_1\times\cdots\times\mathcal{M}_K$ 表达，度量取各分量的 direct sum，从而自然覆盖"一部分维度球面 + 一部分维度双曲"这种组合。这样一来流形选择不再是写死的算法假设，而变成一个可一键切换的配置项，hypothesis-based exploration 真正可行——例如细胞周期数据上 torus、sphere、Euclidean 都能即时切换对比。

### 损失函数 / 训练策略
目标为公式 (10) 的负后验 $\mathcal{L}=\sum_i(-\log p_\theta(x_i|z_i)-\log p(z_i))-\log p(\theta)$，$\theta$ 用 Adam、$Z$ 用 RiemannianAdam 交替更新。几何正则通过给 latent 加 $\mathcal{N}(0,\sigma^2 G^{-1})$ 噪声、一次 retraction 映回流形实现；重构 likelihood 按数据性质选（连续 → Gaussian/MSE，离散 → categorical）。全程无 KL 项、无 ELBO，也不需要 Monte Carlo 估计归一化常数。

## 实验关键数据

### 主实验
三个真实/合成生物数据集：(a) 合成分支扩散树（7 层、$d=50$、6350 样本）→ 双曲流形天然适配；(b) 人类线粒体 DNA 67k 条序列 + haplogroup 标签 → 双曲适配 phylogeny；(c) 细胞周期 scRNA-seq 5367 细胞 × 189 基因 → 周期性，torus 适配。

| 数据集 | 任务 | 最佳几何 | 关键指标 | 备注 |
|--------|------|----------|----------|------|
| Cell cycle scRNA-seq | 相位距离 vs latent 距离相关 | Sphere $\mathbb{S}^2$ | Train Pearson 0.58、Test 0.60，重构 MAE 0.31 | 超过 $\mathcal{S}$-VAE / $\Delta$VAE |
| Branching diffusion | 树距离 vs latent 测地距离相关 | Lorentz $\mathbb{H}^2$（$\sigma=1.0$） | Train Pearson 0.81、Test 0.80 | $\mathcal{P}$-VAE 仅 0.68 |
| hmtDNA haplogroup 分类 | 下游 24/128 类 logistic 回归准确率 | Hyperbolic $\mathbb{H}^2_{\sigma=0.5}$ | 24-way LR 0.70 / XGB 0.85；128-way LR 0.43 | 全面胜 Euclidean 和 $\mathcal{P}$-VAE |

### 消融实验

| 配置 | 关键发现 |
|------|----------|
| 几何噪声 $\sigma$ 扫 0→2.6（双曲，分支扩散） | $\sigma\approx 0.9$ 前相关性快速上升，之后噪声过大 decoder 无法保持局部精度 → 局部-全局 trade-off |
| 双曲 $\mathbb{H}^2$ vs Euclidean / Sphere（分支扩散） | $\mathbb{H}^2$ 显著恢复树拓扑（Pearson 0.81 vs 0.53/0.56） |
| UMAP（分支扩散） | 完全看不出树拓扑 → 验证 RGD 在解释性上的明显优势 |
| 与 $\mathcal{P}$-VAE/$\mathcal{S}$-VAE/$\Delta$VAE 比 | 高维下 VAE baseline 数值崩溃，RGD 仍可用 |
| 生成判别（XGBClassifier 区分真伪重构） | RGD Sphere 0.58 = $\mathcal{S}$-VAE Sphere 0.58 < $\Delta$VAE 0.62（越接近 0.5 越逼真） |

### 关键发现
- **encoder-less 在高维数据下更稳**：在 cell cycle 全基因（高维）+ ProductManifold 等设置下，$\mathcal{S}$/$\Delta$/$\mathcal{P}$-VAE 都因为 normalizing constant 或采样 Monte Carlo 数值崩溃，RGD 仍可训练并给合理重构。
- **几何噪声是真正学到几何的关键**：移除它后双曲模型的距离-相关性接近 Euclidean baseline，说明流形选择本身只给了空间，真正把 decoder 推到与几何对齐还得靠正则。
- **流形选择 = 假设测试**：分支扩散数据用双曲 latent 一秒还原树结构、用 UMAP 完全看不出，说明合理的几何先验比通用降维更可解释。
- **生成质量与 VAE 持平**：discrimination test 上 RGD 的合成样本可分性与 $\mathcal{S}$-VAE 相同（0.58），说明丢掉 encoder 没损失生成保真度。

## 亮点与洞察
- **思路足够干净**：把 DGD 从欧式空间一步推到任意黎曼流形，绕过了 Riemannian VAE 的整条 density approximation 链——这是少有的"减法贡献"，但带来的可扩展性极大。
- **几何噪声推导简单且 actionable**：用 Bishop 噪声-正则等价的经典结论加一层度量逆缩放，就能在任何流形上得到 metric-aware Jacobian 惩罚；公式只有一行，工程实现也只是替换协方差。
- **使流形成为 hypothesis 而非算法负担**：让科学家可以快速对同一份数据试 torus、sphere、双曲、SPD，pick 最符合先验的几何，对 computational biology / phylogenetics 有直接生产力提升。
- **可迁移到 LLM/VLM 隐表示**：把 LLM 的 routing/expert 表示放到 Stiefel 或 Sphere、把 vision-language alignment 放到 ProductManifold，都能套用这套框架。

## 局限与展望
- 没有 amortized encoder 意味着新样本必须**重新优化 latent**（per-sample MAP 步）才能 encode，online inference 比 VAE 慢；可加 amortized warm-start。
- 几何正则中 $\sigma$ 与流形曲率耦合，需要手工 sweep；自动调度是显然的下一步。
- prior 在 non-compact 流形上仍用 wrapped/Riemannian normal，理论上引入轻微 bias，但作者明确把生成质量当作 sanity check。
- 实验全部在生物/合成数据上，未在 NLP/图像等大规模任务上验证（虽然原理可直接套用）。
- ProductManifold 的维度分配方案需要人工设定，对复杂数据可能不易选；自动几何选择是未来方向。

## 相关工作与启发
- **vs $\mathcal{S}$-VAE / $\mathcal{P}$-VAE / $\Delta$VAE**：他们必须为每流形手写概率密度，RGD 全部绕过，且高维稳定性显著更好。
- **vs DGD (Schuster & Krogh 2023)**：DGD 是欧式 encoder-less；RGD 是其黎曼扩展，把 free-parameter MAP 推广到任意流形。
- **vs Lee & Park (2023) 的曲率正则**：他们用二阶曲率（含 Hessian-vector product，公式整页）实现"全局拉平"；RGD 用一阶 Jacobian 正则 + 度量逆噪声，计算成本低一个量级。
- **vs UMAP / Isomap**：经典非线性降维只可视化，不学生成模型也不显式指定流形；RGD 既可视化又生成且 hypothesis-testing。
- **跨任务启发**：molecular 表示（用 SE(3) 流形）、机器人 pose 表示（SO(3)、SE(3)）、强化学习的 belief 空间表示，都可以套用 RGD 的"流形 + free latent + Riemannian Adam + 度量噪声"组合拳。

## 评分
- 新颖性: ⭐⭐⭐⭐ "丢掉 encoder 把 DGD 升级到任意黎曼流形" + "metric-逆噪声" 都是干净独立的贡献，框架级简化效果明显。
- 实验充分度: ⭐⭐⭐⭐ 三个数据集覆盖周期/树/层次三种几何，多基线对比 + 噪声扫描 + downstream + 生成判别俱全；缺大规模 NLP/视觉验证。
- 写作质量: ⭐⭐⭐⭐⭐ 公式推导清晰，附录详尽，对前置工作介绍极完整，理论与工程的过渡非常顺。
- 价值: ⭐⭐⭐⭐ 在 geometric DL 中把 manifold 从"算法门槛"降为"配置项"，对 representation learning、computational biology 生产力提升显著。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Position: We Need An Algorithmic Understanding of Generative AI](../../ICML2025/interpretability/position_we_need_an_algorithmic_understanding_of_generative_ai.md)
- [\[NeurIPS 2025\] Uncovering Graph Reasoning in Decoder-only Transformers with Circuit Tracing](../../NeurIPS2025/interpretability/uncovering_graph_reasoning_in_decoder-only_transformers_with_circuit_tracing.md)
- [\[ACL 2025\] Towards Explainable Temporal Reasoning in Large Language Models: A Structure-Aware Generative Framework](../../ACL2025/interpretability/towards_explainable_temporal_reasoning_in_large_language_models_a_structure-awar.md)
- [\[ICML 2026\] BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking](block-em_preventing_emergent_misalignment_via_latent_blocking.md)
- [\[ICML 2026\] Interpretable Self-Supervised Learning via Representer Landmarks and Nyström Approximation](interpretable_self-supervised_learning_via_representer_landmarks_and_nyström_app.md)

</div>

<!-- RELATED:END -->
