---
title: >-
  [论文解读] Neural Collapse by Design: Learning Class Prototypes on the Hypersphere
description: >-
  [ICML2026][可解释性][神经坍缩] 把"分类器学习 (CE) "和"监督对比学习 (SCL) "统一成超球面上的原型对比，并通过两个新损失 NTCE/NONL（修 CE 侧）和固定原型分类器 FP（修 SCL 侧）让神经坍缩 NC 真正"按设计达成"，同时在精度、迁移、长尾、鲁棒性上全面占优。
tags:
  - "ICML2026"
  - "可解释性"
  - "神经坍缩"
  - "超球面"
  - "原型对比"
  - "归一化 Softmax"
  - "监督对比学习"
---

# Neural Collapse by Design: Learning Class Prototypes on the Hypersphere

**会议**: ICML2026  
**arXiv**: [2605.20302](https://arxiv.org/abs/2605.20302)  
**代码**: https://github.com/pakoromilas/nc_by_design  
**领域**: interpretability  
**关键词**: 神经坍缩, 超球面, 原型对比, 归一化 Softmax, 监督对比学习

## 一句话总结
把"分类器学习 (CE) "和"监督对比学习 (SCL) "统一成超球面上的原型对比，并通过两个新损失 NTCE/NONL（修 CE 侧）和固定原型分类器 FP（修 SCL 侧）让神经坍缩 NC 真正"按设计达成"，同时在精度、迁移、长尾、鲁棒性上全面占优。

## 研究背景与动机

**领域现状**：监督学习有一个理论最优——神经坍缩 (Neural Collapse, NC)：同类特征塌缩到类均值 (NC1)，类均值构成 simplex ETF (NC2)，分类器权重与类均值对齐 (NC3)，偏置塌缩 (NC4)。两大主流范式是 CE（交叉熵）和 SCL（监督对比学习 + 线性探测）。

**现有痛点**：理论说 NC 是全局最优，但实际上两条路都达不到。CE 因为特征/权重可以联合放缩而保持预测不变（径向自由度无约束），损失能被无止境地"通过放大 logits"驱向 0，几何结构永远收不紧；SCL 在预训练阶段确实能逼近 NC 几何，但随后扔掉投影头、在未归一化的编码器特征上重新训一个带偏置的线性分类器（linear probing, LP），把刚学到的 ETF 几何又破坏掉。

**核心矛盾**：CE 从一开始就没有约束径向；SCL 学到了几何却又主动丢掉。两者的失败本质都是"径向自由度 + 偏置"破坏了超球面上的角度结构。同时，归一化 Softmax (NormFace) 虽然把分类压回了超球面，但负样本只有 $K$ 个（类原型数），远少于对比学习需要的负样本数，并且对齐项 (alignment) 与均匀项 (uniformity) 被共享归一化耦合在一起。

**本文目标**：(i) 用统一视角把 CE 和 SCL 同时刻画为"超球面上的原型对比"；(ii) 在 CE 侧修复"负样本太少 + 对齐-均匀耦合"两个缺陷；(iii) 在 SCL 侧证明类均值已经是最优分类器，干掉 LP。

**切入角度**：作者注意到归一化 Softmax 中分类器权重 $\mathbf{w}_c$ 就是一个可学习的"类原型"，而 SCL 的类均值 $\hat{\bm{\mu}}_c$ 是"涌现出来的类原型"。两者只是显式 vs 隐式之差，本质上都是 $\mathbb{S}^{d-1}$ 上的原型 ↔ 实例对比。

**核心 idea**：在超球面上做原型对比时，CE 应该把锚点从"实例对比 $K$ 个原型"翻转为"原型对比 $M$ 个 batch 实例"（NTCE），并把同类正样本从归一化项里剔除以解耦 alignment-uniformity（NONL）；同时 SCL 不需要再训分类器，直接用类均值作权重（FP）即可。

## 方法详解

### 整体框架
全文给出三个改动，分别落在两条路径上：

- 输入：标注图像 $\mathbf{x}_i$，编码器 $f_{\bm{\theta}}$，可选投影头 $g_{\bm{\phi}}: \mathcal{Z}\to\mathbb{S}^{d-1}$。
- CL 路径（NTCE / NONL）：把特征 $\bm{u}_i=\mathbf{z}_i/\|\mathbf{z}_i\|$ 和权重 $\hat{\bm{w}}_c$ 都归一化到 $\mathbb{S}^{d-1}$，并把损失换成把"类原型 $\hat{\bm{w}}_{y_i}$ 当锚点、batch 内 $M$ 个实例当对比对象"的对比式。
- SCL 路径（FP）：照常用 SCL 训完编码器+投影头，**不**再训 LP，直接令 $\bm{w}_c=\hat{\bm{\mu}}_c$（batch 内类均值）作为分类器权重。
- 理论：定理 4.1 证明 NormFace/NTCE/NONL 的全局最优都满足 NC1–NC3；定理 4.2 证明 SCL 与"原型 Softmax"损失共享同一组全局最优，所以类均值本就是 SCL 在整条优化轨迹上想要的分类器。

### 关键设计

1. **NTCE：把锚点从实例翻转到类原型，扩负样本集**：

    - 功能：在 NormFace 的归一化 Softmax 基础上，把"每个实例 vs $K$ 个类权重"换成"每个类权重 vs batch 内 $M$ 个实例"，让对比的有效负样本数从 $K$ 跳到 $M$。
    - 核心思路：损失形如 $L_{\text{NTCE}}=\frac{1}{M}\sum_i -\log\frac{e^{\hat{\bm{w}}_{y_i}^\top \bm{u}_i/\tau}}{\sum_{j=1}^{M} e^{\hat{\bm{w}}_{y_i}^\top \bm{u}_j/\tau}}$。锚点是类原型 $\hat{\bm{w}}_{y_i}$，分母遍历整个 batch 而不是 $K$ 个类，从而把 NormFace 退化成"$K$ 路对比"的瓶颈打掉，恢复了对比学习"负样本越多估计越准"的优势。
    - 设计动机：He 等人早就指出对比目标需要大量负样本才能逼近真实期望；NormFace 形式上是对比损失，但其实只用了 $K$ 个负样本，导致类间分离收敛慢、对小 batch 鲁棒。NTCE 让 CL 真正享受到对比优化的好处，从经验上把 ETF 几何更快推到位（在 Table 3 上 inter-class effective rank 直接到达理论上限 $K-1$）。

2. **NONL：把同类正样本从归一化项里剔除，解耦 alignment-uniformity**：

    - 功能：在 NTCE 基础上进一步去掉分母里所有与锚点同类的样本，消除"正样本被当作负样本"导致的梯度对冲。
    - 核心思路：定义 $L_{\text{NONL}}=\frac{1}{M}\sum_i -\log\frac{e^{\hat{\bm{w}}_{y_i}^\top \bm{u}_i/\tau}}{\sum_{j\notin\mathcal{C}(i)} e^{\hat{\bm{w}}_{y_i}^\top \bm{u}_j/\tau}}$，其中 $\mathcal{C}(i)$ 是 batch 内与 $i$ 同类的索引集。NTCE 的分母里同类实例 $j$ 会通过 $e^{\hat{\bm{w}}_{y_i}^\top \bm{u}_j/\tau}$ 产生"把同类实例从类原型推开"的梯度，恰好与分子的对齐项相反——这就是已知的 alignment-uniformity 耦合问题；NONL 通过显式排除同类样本把这个内耗去掉。
    - 设计动机：均匀项最优是当同类实例间距最大，这与"同类塌缩"的对齐目标天然矛盾。NONL 把这种内耗一刀切断，使 NC1（intra-class collapse）显著更好——Table 3 中 intra-class effective rank 从 NTCE 的 9.0/12.6 降到 4.0/11.4，几个 NC 指标在 CL 家族里整体最佳。

3. **FP（Fixed Prototypes）：用类均值当 SCL 的分类器，干掉线性探测**：

    - 功能：SCL 训完后不再做 LP，直接把投影头输出空间里每类的样本均值 $\hat{\bm{\mu}}_c$ 当作分类器权重，零偏置，零额外训练。
    - 核心思路：作者证明（定理 4.2）：在单位范数特征 + 平衡标签下，SCL 损失 $L_{\text{SCL}}$ 与"原型 Softmax"损失 $L_{\text{proto}}$（分子是 $e^{\bm{a}_i^\top \hat{\bm{\mu}}_{y_i}/\tau}$，分母按 batch 频次 $n_c$ 加权所有类原型）拥有同一组全局最优。这意味着 SCL 在整条优化轨迹上都隐式优化"以类均值为权重的原型分类器"，类均值不是事后凑出来的，而是 SCL 一直想要的分类器。FP 推理时只需对 batch 计算类均值，做 nearest-prototype 决策。
    - 设计动机：标准 LP 在未归一化特征上加可学权重+偏置，重新引入径向自由度和偏置，把 SCL 好不容易学到的 ETF 几何破坏掉，且要多花 $T$ 个 epoch。FP 用 $N$ 次前向取代 $T\times N$ 的训练，几何上也强制 NC3 严格成立，从理论和算力两头同时占便宜。

### 损失函数 / 训练策略
温度 $\tau$ 与对比学习同款（CIFAR/ImageNet 默认值）；CL 侧只需对 $\mathbf{z}$ 和 $\mathbf{w}$ 各做一次 L2 归一化，偏置置零；SCL 侧沿用 Khosla 等的标准训练配方，唯一改动是把 LP 阶段换成"计算类均值 → 直接当权重"。NTCE/NONL 在小 batch 下会因负样本不足退化（对比损失通病），ImageNet-1K 建议大 batch。

## 实验关键数据

### 主实验
四个 benchmark：CIFAR-10、CIFAR-100、ImageNet-100、ImageNet-1K；CIFAR 用 ResNet18，ImageNet 用 ResNet50。

| 数据集 | 指标 | NONL (ours) | NTCE (ours) | NormFace | CE |
|---|---|---|---|---|---|
| CIFAR-10 | Top-1 | **94.9** | 94.7 | 94.8 | 94.6 |
| CIFAR-100 | Top-1 | **73.6** | 72.9 | 72.4 | 72.1 |
| ImageNet-100 | Top-1 | **84.9** | 84.7 | 84.4 | 84.4 |
| ImageNet-1K | Top-1 | 76.5 | **76.7** | 76.4 | 75.4 |

SCL 侧：FP 在 ImageNet-100 上相对 LP **+2.0%**（86.8 vs 84.8），其余三个数据集与 LP 持平，但只需 $N$ 次前向（LP 需要 $T\times N$）。

### NC 几何与收敛速度（CIFAR-10 / CIFAR-100 training）

| 方法 | Intra erank ↓ | Inter erank ↑ | Weight align ↓ | Instance align ↓ |
|---|---|---|---|---|
| CE | 22.5 / 96.4 | 8.6 / 57.1 | 0.59 / 0.83 | 0.69 / 1.05 |
| NormFace | 10.5 / 13.6 | 9.0 / 96.2 | 0.12 / 0.01 | 0.14 / 0.06 |
| NTCE | 9.0 / 12.6 | **9.0 / 99.0** | 0.08 / 0.01 | 0.10 / 0.05 |
| **NONL** | **4.0 / 11.4** | **9.0 / 99.0** | 0.11 / 0.01 | 0.16 / 0.06 |
| SCL + LP | 4.5 / 7.5 | 9.0 / 66.7 | 0.99 / 1.03 (NC3 破坏) | 0.10 / 0.34 |

NTCE/NONL 的 NC 指标达到理论值的 $\geq 95\%$，并且匹配 CE 的最终 NC 指标只需 CE 训练迭代数的 $\leq 7.5\%$（附录 Table 7）。

### 下游迁移与鲁棒性

- **迁移学习（8 数据集均值，Table 4）**：NONL 70.7% vs CE 67.0%，平均相对提升 **+5.5%**；在 Cars 上相对 CE **+47.1%**（38.1 vs 25.9）。
- **长尾分类（CIFAR-10/100-LT, Table 6）**：极端不平衡 $\tau=0.01$ 下，NONL 在 CIFAR-10-LT 把 CE 的 70.2 推到 76.3，CIFAR-100-LT 上 NTCE 从 37.4 推到 39.0；最大相对提升 **+8.7%**。
- **鲁棒性（ImageNet-C, Table 5）**：NTCE mCE = 77.6（CE 80.1），所有 corruption 类型 (Noise/Blur/Weather/Digital) 都更稳。

### 关键发现
- NONL 在 CL 家族里 NC 几何最干净，主要得益于把 alignment 和 uniformity 解耦，intra-class effective rank 比 NTCE 又降一半。
- "SCL + LP" 在 NC1（intra erank 4.5/7.5）上其实比所有 CL 都更紧，但 NC3 完全垮掉（w-inst 0.99/1.03），印证了"LP 把几何破坏掉"的核心论断。
- 信息论指标 MIR/HDR 上 CE 略胜，但作者指出那只是反映"总熵高"而非"NC 结构好"——下游迁移/长尾/鲁棒性三战全负的事实更能说明问题。

## 亮点与洞察
- **一个统一视角，两个失败诊断同时被治好**：把 CE 和 SCL 都看成超球面上的原型对比，让原本看似无关的两条修复路径（NTCE/NONL on CL；FP on SCL）变成同一个图景里的两半，理论上由两条定理闭环 ($\bm{w}_c=\hat{\bm{\mu}}_c$ at 全局最优)。
- **锚点翻转 + 同类剔除是简单到几乎一行代码的改动**：NTCE 只是把 Softmax 的"维度"从 $K$ 个类换成 $M$ 个 batch 实例；NONL 只是从分母里去掉同类——但带来的 NC 改善和下游收益巨大，非常值得迁移到其他归一化对比损失（如 InfoNCE 的有监督变体）。
- **干掉 LP 是 SCL 工程实践上的实打实优化**：在大规模训练里 LP 通常要再花数小时，FP 直接把这段省掉而精度持平甚至更好，对工业部署友好。
- **NC 不只是"漂亮的几何"，也是下游性能的关键**：作者用迁移/长尾/鲁棒三组实验把"几何更紧 → 下游更好"这条链条串起来，回应了 NC 文献里长期被质疑的"NC 是否真有实用价值"。

## 局限与展望
- 对比式目标天然依赖大 batch，ImageNet-1K 上小 batch 训练 NTCE/NONL 会显著退化（附录 G.4），低资源场景部署需注意。
- 主要验证在 ResNet18/50 + 标准图像分类；Transformer backbone、检测/分割等下游任务、多标签场景下的 NC 行为没有覆盖。
- 长尾分析虽给出实测增益，但理论上 NC2/ETF 在不平衡数据下会发生 minority collapse（Thrampoulidis 2022），NONL/NTCE 究竟是"接近 ETF"还是"接近某种带权 ETF"还需要更细的几何分析。
- FP 在 ImageNet-1K 上只与 LP 持平而不是反超，可能与投影头维度 / 类均值估计噪声有关，值得进一步研究 EMA 类均值或多 batch 累积估计。

## 相关工作与启发
- **vs NormFace (Wang et al. 2017)**：NormFace 是 NTCE/NONL 的直系前身，作者把它解释为"$K$ 路对比 + 锚点是实例"，NTCE 翻转锚点并扩负样本；NONL 进一步解耦 alignment-uniformity，是 NormFace 的"对比化升级版"。
- **vs ETF + DR (Yang et al. 2022)**：ETF + DR 直接固定分类器为预设 ETF，只训 backbone；本文反过来让分类器自由学习但通过归一化和正确的对比形式让它"自然收敛到 ETF"。Table 1/4 上 ETF + DR 迁移崩盘 (mean 54.6 vs NONL 70.7)，说明强制固定几何会丢掉表征的迁移能力，"自然涌现"更稳。
- **vs Hyperspherical / EBV 原型 (Mettes 2019; Shen 2023)**：这些方法事先构造好原型；本文证明 SCL 的类均值原型在整条优化轨迹上就是最优分类器，提供了一个理论根基让"emergent prototype"取代"prescribed prototype"。
- **vs Graf et al. 2021**：Graf 等只证明 SCL 在全局最优处出现 ETF；本文的定理 4.2 把结论加强到"SCL 全局最优集合 = 原型 Softmax 全局最优集合"，从而合法化 FP 在非最优实际训练点上也好用。
- **vs Kini et al. 2024**：Kini 等在特定架构假设下分析 SCL 最优；本文在 UFM/LPM 设定下给出更松的等价性结论，并把 CL 与 SCL 统一进来。

## 评分
- 新颖性: ⭐⭐⭐⭐ "把 CE 和 SCL 统一成超球面原型对比"的视角清晰，NTCE/NONL/FP 的具体改动都不复杂但贡献明确。
- 实验充分度: ⭐⭐⭐⭐⭐ 四数据集 + NC 几何 + 收敛速度 + 迁移 + 长尾 + 鲁棒，证据链很完整。
- 写作质量: ⭐⭐⭐⭐ 行文连贯，理论定理与经验现象互相呼应；少数段落公式密度偏高。
- 价值: ⭐⭐⭐⭐ 理论 + 工程双重价值：CL 侧涨点、SCL 侧省训，且方法简单可迁移到现有 pipeline。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Prototype Transformer: Towards Language Model Architectures Interpretable by Design](prototype_transformer_towards_language_model_architectures_interpretable_by_desi.md)
- [\[ICML 2026\] ShaplEIG: Bayesian Experimental Design for Shapley Value Estimation](shapleig_bayesian_experimental_design_for_shapley_value_estimation.md)
- [\[ACL 2026\] NOSE: Neural Olfactory-Semantic Embedding with Tri-Modal Orthogonal Contrastive Learning](../../ACL2026/interpretability/nose_neural_olfactory-semantic_embedding_with_tri-modal_orthogonal_contrastive_l.md)
- [\[ICML 2025\] Sum-of-Parts: Self-Attributing Neural Networks with End-to-End Learning of Feature Groups](../../ICML2025/interpretability/sum-of-parts_self-attributing_neural_networks_with_end-to-end_learning_of_featur.md)
- [\[ICML 2026\] Breaking the Simplification Bottleneck in Amortized Neural Symbolic Regression](breaking_the_simplification_bottleneck_in_amortized_neural_symbolic_regression.md)

</div>

<!-- RELATED:END -->
