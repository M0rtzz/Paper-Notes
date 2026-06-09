---
title: >-
  [论文解读] Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models
description: >-
  [ICML 2026][计算生物][单细胞 RNA-seq] scLDM 用统一的多头交叉注意力块 (MCAB) 把可交换的基因表达数据编成固定长度、置换不变的潜变量集合，再用 DiT + 流匹配 + 联合多属性 classifier-free guidance 替代 Gaussian 先验…
tags:
  - "ICML 2026"
  - "计算生物"
  - "单细胞 RNA-seq"
  - "可交换性 (exchangeability)"
  - "多头交叉注意力"
  - "潜在扩散"
  - "流匹配"
  - "多条件 CFG"
---

# Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models

**会议**: ICML 2026  
**arXiv**: [2511.02986](https://arxiv.org/abs/2511.02986)  
**代码**: https://github.com/czi-ai/scldm (有)  
**领域**: 计算生物学 / 单细胞转录组 / 潜在扩散模型 / Transformer VAE  
**关键词**: 单细胞 RNA-seq, 可交换性 (exchangeability), 多头交叉注意力, 潜在扩散, 流匹配, 多条件 CFG

## 一句话总结
scLDM 用统一的多头交叉注意力块 (MCAB) 把可交换的基因表达数据编成固定长度、置换不变的潜变量集合，再用 DiT + 流匹配 + 联合多属性 classifier-free guidance 替代 Gaussian 先验，在多个 scRNA-seq 数据集上的重构、(有/无条件) 生成、扰动响应预测全面超过 scVI / scDiffusion / CFGen。

## 研究背景与动机

**领域现状**：单细胞 RNA-seq 让我们可以一次性测量上百万个细胞、每个细胞数万个基因的表达，进而研究细胞分化、疾病进展、药物扰动。主流生成式建模方案有三类：(i) VAE 路线 (scVI / scVAEDer)，(ii) GAN 路线 (scGAN)，(iii) 扩散路线 (scDiffusion) 以及最新的 latent diffusion + flow matching 的 CFGen。

**现有痛点**：

- 几乎所有方法都把基因表达当成"固定顺序的向量"——把第 i 维硬绑到基因 $g_i$，输入维度等于基因词表大小。这要求训练前先选一份"高变基因"子集 (HVG)，跨组织、跨物种就得重训或者动手术式置换权重。
- 这种"按位编码"还和生物学事实直接冲突：基因表达是**可交换集合 (exchangeable set)**，谁先谁后没有意义。
- GAN 路线天生有训练不稳定、模式塌缩问题；纯 MLP-based autoencoder 容量有限，scale 上去后边际收益迅速递减。
- scRNA-seq 数据 70%+ 是 0 (dropout)，把所有零位也喂进 transformer 既浪费算力又稀释信号。

**核心矛盾**：要同时拿到 (a) 真正可交换的概率模型 (b) 可扩展到大词表 / 大上下文的 transformer 架构 (c) 对计数数据 (NB 分布) 的精确建模 (d) 支持多属性可控生成。已有方法每条都只能选一两个。

**本文目标**：

1. 设计一个**置换不变 (编码) + 置换等变 (解码) 的 transformer-based VAE**，潜变量数量固定，与输入基因数解耦。
2. 用 **DiT + 线性插值 + flow matching** 在潜空间训练一个 LDM 替换 Gaussian 先验，支持多属性联合 CFG。
3. 在重构、无条件生成、条件生成、扰动响应预测、下游分类五个任务上全面验证。

**切入角度**：作者注意到 SetTransformer / Perceiver IO 已经提供了"用 learnable pseudo-inputs 把变长输入池化成固定长度 token 集"的工具。把 pseudo-inputs 换成基因 embedding，就同时拿到 (i) 编码侧的置换不变池化和 (ii) 解码侧的置换等变 unpooling——**同一个 block 两用**，省掉 SetTransformer (PMA + ISAB) 那套分离的 pool/unpool 架构。

**核心 idea**：用一个**统一的 MCAB (Multi-head Cross-Attention Block)** 同时担任 encoder 的"置换不变池化"与 decoder 的"置换等变解池化"，并在 fixed-size 潜空间上跑 DiT-based latent diffusion，于是模型对基因顺序天然不变、对词表大小可扩展、对计数分布精确建模、对多条件可控生成。

## 方法详解

### 整体框架

scLDM 要解决的核心问题是：基因表达本质是一个"谁先谁后无所谓"的可交换集合，但主流模型却把它当成固定顺序的向量，既绑死了基因词表又违背生物学事实。它的破局思路是把每个细胞表示成"基因 ID 集合 + 对应计数"$(\mathbf{x}_{\mathcal{I}}, \mathcal{I})$，先用一个置换不变的 transformer-VAE 把它压成固定长度、与基因数解耦的潜变量集合，再在这个干净的潜空间上跑 DiT 扩散来做可控生成。训练分两阶段：Stage 1 学 VAE（编码 NB 计数似然），Stage 2 冻结 VAE、在潜空间上训一个流匹配扩散模型替换简陋的高斯先验；采样时先由扩散模型采潜变量，再过 VAE 解码采计数。

### 关键设计

**1. 统一的 MCAB：一个 block 同时拿到置换不变与置换等变**

可交换集合建模的难点在于：编码时希望"打乱基因顺序、潜变量不变"（置换不变），解码时希望"打乱查询基因、输出跟着同样打乱"（置换等变）。传统方案 SetTransformer 要分别造 PMA（池化）和 ISAB（解池化）两套 block，归纳偏置不一致、参数也不共享。scLDM 发现这两种对称性可以用同一个多头交叉注意力块表达，区别只在于 query 是什么。它定义 $\mathrm{MCAB}_{\mathbf{S}}(\mathbf{X}) = F(\mathbf{X},\mathbf{S}) + \mathrm{MLP}(\mathrm{LN}(F(\mathbf{X},\mathbf{S})))$，其中 $F$ 以 $\mathbf{S}$ 为 query、输入集合 $\mathbf{X}$ 为 key/value 做 cross-attention。

在 encoder 端，$\mathbf{S}$ 是 $m$ 个**与基因 ID 无关**的可学习 pseudo-inputs：对 $\mathbf{X}$ 任意置换时 $\mathbf{S}$ 不动、attention 聚合结果不变，于是潜变量 $\mathbf{Z}$（固定 $m \times D$）天然置换不变；在 decoder 端，把 $\mathbf{S}$ 换成查询基因的 embedding $\mathbf{E}_{\mathcal{I}}$，对 $\mathcal{I}$ 的置换就等价于对 $\mathbf{S}$ 做行置换，输出随之同样置换，得到置换等变。这一设计的额外红利是潜空间大小 $m$ 和基因词表彻底解耦——后者只通过 $\mathbf{E}$ 这一个 embedding 矩阵进入模型，跨组织、跨物种迁移只需扩展 $\mathbf{E}$ 而不必改动主体网络。

**2. 稀疏感知的输入处理：只把"真正有信号"的基因喂进 transformer**

scRNA-seq 有 70%+ 的位置是 dropout 零，若把数万维全部喂给 transformer，既浪费 $O(D^2)$ 算力又把有效信号稀释掉。scLDM 在 encoder 入口做了一次稀疏裁剪 $G(\mathbf{x},\mathcal{I})$：先选出表达基因集合 $\mathcal{J} = \{i : x_i > 0\}$，若不足目标长度 $d$ 就用 PAD token（计数 0、索引 PAD）补齐，即 $\mathrm{Out} = \{(x_i, i)\}_{i \in \mathcal{J}} \cup \{(0, \mathrm{PAD})\}^{d - |\mathcal{J}|}$；再把计数和基因 embedding 拼接进 token，$\mathrm{Emb}(\bar{\mathbf{x}}_{\mathcal{J}}, \mathcal{J}) = \mathrm{Linear}(\mathrm{repeat}_d(\bar{\mathbf{x}}_{\mathcal{J}}) \,\Vert\, \mathbf{E}_{\mathcal{J}})$。

关键在于这只是**编码端的上下文裁剪**，并不削弱模型表达结构性零的能力：decoder 仍然对完整的 $\mathcal{I}$ 输出 NB 分布参数，而 NB 在 0 处本来就有大量概率质量，所以那些被裁掉的零依然能被解码器恢复（Table 15、17 的 $R^2$ Zeros 实验证实），算力却全部集中在真正携带信号的基因上——是一次"省算力还不掉点"的免费午餐。

**3. DiT 潜空间扩散 + 联合多属性 CFG：用强先验替换高斯先验并解锁组合生成**

VAE 训完后，aggregated posterior 的真实形状远比标准高斯 $\mathcal{N}(0, I)$ 复杂，直接用高斯先验采样会因"先验-后验 mismatch"导致生成质量塌陷（Tomczak 2024）。scLDM 把 $m$ 个潜 token 当作 DiT 的输入序列，用线性插值 + 流匹配训一个速度场 $v_{t,\epsilon}(\mathbf{Z}; y)$ 来逼近这个复杂的潜分布，相当于把简陋的高斯先验换成一个学出来的强先验。

多属性可控生成上，它把属性向量 $\mathbf{y} \in \{0,1\}^J$（cell type / perturbation / batch 等）**整体**当成一个条件做联合 CFG：$\tilde{v}_{t,\epsilon}(\mathbf{Z}, y) = v_{t,\epsilon}(\mathbf{Z}; \mathrm{Null}) + \omega [v_{t,\epsilon}(\mathbf{Z}; y) - v_{t,\epsilon}(\mathbf{Z}; \mathrm{Null})]$，而不是 CFGen 那种 $\sum_j \omega_j [v(\mathbf{Z}; y_j) - v(\mathbf{Z}; \mathrm{Null})]$ 的加性分解。加性 CFG 隐含属性 one-hot 互斥假设 $\sum_j y_j = 1$，无法表达"perturbation A + cell type B"这类组合；联合 CFG 直接把组合编码进同一个条件 embedding，因此能捕捉属性间的交互，对扰动 benchmark 上的多属性可控生成尤其关键。

### 损失函数 / 训练策略

- **Stage 1**：$\beta$-VAE 形式的 ELBO：$\mathcal{L} = \mathbb{E}_q[\ln p(\mathbf{x}_{\mathcal{I}} | \eta(\mathbf{Z}, \mathcal{I}))] - \beta \cdot \mathrm{KL}(q(\mathbf{Z}|\mathbf{x}_{\mathcal{I}}) \,\Vert\, p(\mathbf{Z}))$。计数似然用 NB；极端情形 $\beta = 0$ 退化成 deterministic autoencoder（Stable Diffusion 同款做法）。
- **Stage 2**：冻结 VAE，DiT 用 flow matching 损失 $\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{t, \mathbf{Z}_0, \mathbf{Z}_1, \mathbf{y}} \| v_{t,\epsilon}(\mathbf{Z}_t; \mathbf{y}) - (\mathbf{Z}_1 - \mathbf{Z}_0) \|^2$ 拟合线性插值 $\mathbf{Z}_t = (1-t)\mathbf{Z}_0 + t \mathbf{Z}_1$。CFG drop-out 概率 $\rho$ 决定 mini-batch 中以多大概率把条件置 Null。采样用 SiT 库 (Scalable Interpolant Transformers)。

## 实验关键数据

### 主实验

**Table 1：细胞重构（NB 似然 + Pearson + MSE）**

| 数据集 | 模型 | RE ↓ | PCC ↑ | MSE ↓ |
|--------|------|------|-------|-------|
| Dentate Gyrus | scVI | 5193.2 | 0.058 | 0.378 |
| Dentate Gyrus | CFGen | 5468.8 | 0.076 | 0.253 |
| Dentate Gyrus | **scLDM (NB)** | **4571.6** | **0.273** | **0.206** |
| Tabula Muris | scVI | 5588.2 | 0.221 | 0.132 |
| Tabula Muris | CFGen | 5547.6 | 0.136 | 0.127 |
| Tabula Muris | **scLDM (NB)** | **4993.6** | **0.376** | **0.106** |
| HLCA | scVI | 5659.2 | 0.125 | 0.238 |
| HLCA | CFGen | 5428.7 | 0.146 | 0.117 |
| HLCA | **scLDM (NB)** | **4898.9** | **0.310** | **0.095** |

PCC 在 Tabula Muris 上 0.376 vs CFGen 0.136，**几乎是 3 倍**——transformer-VAE 对复杂细胞群体的重构能力显著好于 MLP-based VAE。

**Table 2：(无)条件生成 (HVG, Wasserstein-2 / MMD / 1-NN accuracy → 0.5 / Precision / Recall)**

| 数据集 | 设定 | 模型 | W2 ↓ | MMD² RBF ↓ | 1-NN →0.5 | Prec ↑ | Rec ↑ |
|--------|------|------|------|-----------|----------|--------|-------|
| Dentate Gyrus | Uncond | CFGen | 12.617 | 0.022 | 0.856 | 0.278 | **0.385** |
| Dentate Gyrus | Uncond | **scLDM (NB)** | **10.710** | **0.017** | **0.709** | **0.664** | 0.291 |
| Tabula Muris | Uncond | CFGen | 11.658 | 0.008 | 0.773 | 0.255 | 0.591 |
| Tabula Muris | Uncond | **scLDM (NB)** | **7.267** | **0.002** | **0.596** | **0.539** | **0.608** |
| HLCA | Uncond | CFGen | 12.433 | 0.007 | 0.760 | 0.272 | 0.583 |
| HLCA | Uncond | **scLDM (NB)** | **9.272** | **0.004** | **0.605** | — | — |

W2 在 Tabula Muris 上几乎砍半 (7.27 vs 11.66)，1-NN classifier 准确率从 0.77 降到 0.60（越接近 0.5 表示生成样本与真实分布越难区分），说明 scLDM 的生成分布更贴近真实。条件设定 (cell type 控制) 也是全面领先。

### 消融实验

| 配置 | 关键发现 | 说明 |
|------|---------|------|
| scLDM (NB) | W2 = 10.71 / 7.27 / 9.27 (DG/TM/HLCA, uncond) | 完整模型 |
| scLDM (Gauss) | W2 = 17.68 / 14.67 / — | 把 NB 换成 Gaussian 似然 → 性能崩盘，说明计数分布建模不可省 |
| w/o LDM (Gaussian 先验) | 生成质量大跌 (见 Appendix K) | 印证"aggregated posterior ≠ 标准高斯"是 VAE 生成质量瓶颈 |
| 加性 CFG (CFGen 同款) vs 联合 CFG | 联合 CFG 在扰动 benchmark 上更优 (Appendix K.4) | 联合编码多属性组合优于加性独立处理 |
| 输入过滤 vs 全 context | 过滤后重构指标持平/更好 (Table 15) | 稀疏过滤是"算力优化"而非"模型表达力损失" |
| MCAB vs SetTransformer pooling 等聚合算子 | MCAB 更优 (Appendix K.1) | 验证统一 block 设计的有效性 |

### 关键发现

- **NB 似然不可换成 Gaussian**：scLDM (Gauss) 在所有指标上塌陷到 scDiffusion 的水平甚至更差，说明对计数数据**必须用 NB 这种支持零膨胀的离散分布**，连续化建模 (log-normalize 后 Gaussian) 会丢掉关键的离散结构。
- **重构 PCC 提升远大于 W2 提升**：从 0.14 到 0.38 的 PCC 跃升说明 MCAB transformer 在"保留细胞间相对差异"这件事上质变；这直接转化为下游分类任务的优势 (Appendix K)。
- **联合 CFG > 加性 CFG**：当条件是"多属性组合"而不是 one-hot 互斥时，联合编码能捕捉属性交互，对 perturbation prediction 任务尤其重要。
- **稀疏过滤是免费午餐**：把 70% 的零位剔除后重构反而**更好**，因为 decoder 仍然能通过 NB 在 0 处的概率质量恢复 structural zeros，而 encoder 算力都集中在真正有信号的基因上。

## 亮点与洞察

- **MCAB 一个 block 两用**——同一份注意力机制，靠"pseudo-inputs 是否随输入置换"切换 invariant / equivariant 两种语义。这种"用同一参数空间表达两种对称性"的设计很优雅，原则上可以迁移到任何 set-to-set 任务（点云 → 点云、原子坐标 → 力场等）。
- **把基因从"位置维度"解放到"embedding 维度"**——这是该工作和 scVI / CFGen / scDiffusion 的本质差别。基因 ID 通过 $\mathbf{E}$ 进入模型，新增基因 / 跨物种迁移只需扩展 embedding，不需要改模型主体。这呼应了 NLP 里 "tokenizer 解耦 vocabulary 与架构" 的成熟范式。
- **Stable Diffusion paradigm 在生物数据上的成功复现**——VAE 压维 → DiT 在潜空间扩散 → CFG 控制，整套范式在 scRNA-seq 上原样跑通，且击败领域内专门设计的 baseline。这暗示**潜在扩散 + transformer 几乎是所有"高维稀疏科学数据生成"问题的通用配方**。
- **联合 CFG 的工程价值**：作者敏锐地发现 CFGen 的加性 CFG 隐含 $\sum_j y_j = 1$ 假设，对多扰动组合无效；改成联合 CFG 就解锁了组合生成。这个细节在以后做"多属性可控生成"时值得直接借鉴。

## 局限性 / 可改进方向

- **两阶段训练成本高**：VAE 和 LDM 分开训，DiT 在潜空间上又要单独调优；端到端 (e.g. ELBO + flow matching 联合) 是否可行没有讨论。
- **MCAB 中 $m$ (潜 token 数) 的选择没有自适应机制**：固定数值意味着同一个 $m$ 既要承载"几百维 HVG"的 Dentate Gyrus 也要承载"几万维全基因"的 HLCA，对极端 case 可能不够紧凑或不够表达。
- **未在跨物种迁移上做实验**：架构上虽然天然支持扩展 $\mathbf{E}$，但论文没有展示"在小鼠数据预训练 → 直接迁移到人类基因"这种 killer experiment。
- **PAD token 的语义偏置**：所有"未表达"基因被映射到同一个 PAD token，丢失了"这个基因在这个细胞里到底是 dropout 还是真零"的区分；引入显式 dropout mask 可能进一步提升。
- **Wall-clock 与算力开销未充分对比**：transformer 显然比 scVI 的 MLP 慢得多，论文没给出"在同等训练 GPU-小时下" 的公平比较。

## 相关工作与启发

- **vs CFGen** (Palma 2025a)：同样是 scVI + 潜空间 flow matching，但 CFGen 的 encoder/decoder 仍是 MLP，对基因维度有固定排序；CFG 也是加性的。scLDM 把两者都升级 (transformer + 联合 CFG)，重构和生成两端都明显胜出。
- **vs scDiffusion** (Luo 2024)：直接在原始 (log-normalize 后) 表达空间跑扩散，没有潜空间，计算昂贵且对计数离散性建模欠佳。scLDM 用 latent diffusion + NB 直接两面碾压。
- **vs SetTransformer / SetVAE** (Lee 2019, Kim 2021)：思想接近——都希望对集合做置换不变 / 等变建模——但 SetTransformer 需要分开的 PMA (pool) 和 ISAB (unpool) 两套 block，SetVAE 还引入双潜变量和分层结构；MCAB 用单个 block + 共享 embedding 表达两种对称性，更简洁且参数共享。
- **vs Perceiver IO** (Jaegle 2022)：encoder 侧 MCAB 与 Perceiver IO block 几乎相同，但 scLDM 的 decoder **复用同一 block**，只是把 query 换成基因 embedding，这一点是新意所在。
- **vs Stable Diffusion** (Rombach 2022) / **DiT** (Peebles 2023)：架构 paradigm 完全继承，可以视作"Stable Diffusion in scRNA-seq"。这种"工业级生成模型 → 科学计算"的迁移路径很值得关注：分子 (La-proteina)、材料 (all-atom DiT)、动力学 (LaM-SLidE) 都在走同一条路。
- **启发**：(i) 任何"高维稀疏 + 可交换 + 计数"的科学数据生成 (e.g. metagenomics, ATAC-seq, spatial transcriptomics) 都值得套用 MCAB + latent DiT 模板；(ii) 联合 CFG 应该成为多属性可控生成的默认选择，而不是加性 CFG；(iii) "可交换性" 这种数据天然对称性如果被架构正确编码，会比"靠 data augmentation 灌输"得到的模型显著更强——这是几何深度学习的核心信条在 omics 数据上的又一次胜利。

## 评分
- 新颖性: ⭐⭐⭐⭐ MCAB 双用 + 联合 CFG 是真正新的架构贡献；整体范式 (latent diffusion + transformer) 是已知配方在新领域的落地。
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集 × 重构 + (无)条件生成 + 扰动 + 下游分类 + 多个消融，覆盖非常完整。
- 写作质量: ⭐⭐⭐⭐ 数学符号严谨，可交换性 properties 写得清楚；但 MCAB 公式密集、初读门槛高，缺一张直观的 attention pattern 图解。
- 价值: ⭐⭐⭐⭐⭐ 直接给 scRNA-seq 社区一个开源、SOTA、可扩展的生成模型 + 编码器；同时为"set-structured 科学数据 + latent diffusion"提供了一个干净的参考实现。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Towards Universal Gene Regulatory Network Inference: Unlocking Generalizable Regulatory Knowledge in Single-cell Foundation Models](towards_universal_gene_regulatory_network_inference_unlocking_generalizable_regu.md)
- [\[CVPR 2026\] HINGE: Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene Expression Generation from Histology Images](../../CVPR2026/computational_biology/adapting_a_pre-trained_single-cell_foundation_model_to_spatial_gene_expression_g.md)
- [\[AAAI 2026\] Gene Incremental Learning for Single-Cell Transcriptomics](../../AAAI2026/computational_biology/gene_incremental_learning_for_single-cell_transcriptomics.md)
- [\[CVPR 2026\] Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images](../../CVPR2026/computational_biology/cell-type_prototype-informed_neural_network_for_gene_expression_estimation_from_.md)
- [\[ICML 2026\] What Makes a Representation Good for Single-Cell Perturbation Prediction?](what_makes_a_representation_good_for_single-cell_perturbation_prediction.md)

</div>

<!-- RELATED:END -->
