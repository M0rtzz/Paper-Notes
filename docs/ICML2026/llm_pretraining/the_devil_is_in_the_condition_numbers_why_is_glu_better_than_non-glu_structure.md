---
title: >-
  [论文解读] The Devil is in the Condition Numbers: Why is GLU Better than non-GLU Structure?
description: >-
  [ICML 2026][预训练][GLU] 在 NTK 视角下证明 GLU 把两层网络的核矩阵改写成"原 NTK 与数据 Gram 阵的 Hadamard 积"，从而显著压缩条件数、加速收敛，同时实证显示 GLU 并不改善泛化间隔，其全部红利都来自更好的优化。
tags:
  - "ICML 2026"
  - "预训练"
  - "GLU"
  - "SwiGLU"
  - "神经正切核"
  - "条件数"
  - "训练动力学"
---

# The Devil is in the Condition Numbers: Why is GLU Better than non-GLU Structure?

**会议**: ICML 2026  
**arXiv**: [2605.20749](https://arxiv.org/abs/2605.20749)  
**代码**: https://github.com/Zemdalk/GLU-NTK (有)  
**领域**: 优化理论 / 神经正切核 / 大模型架构  
**关键词**: GLU, SwiGLU, 神经正切核, 条件数, 训练动力学  

## 一句话总结
在 NTK 视角下证明 GLU 把两层网络的核矩阵改写成"原 NTK 与数据 Gram 阵的 Hadamard 积"，从而显著压缩条件数、加速收敛，同时实证显示 GLU 并不改善泛化间隔，其全部红利都来自更好的优化。

## 研究背景与动机

**领域现状**：从 LLaMA、Qwen、DeepSeek 到几乎所有现代开源大模型，FFN 层都默认换成了 SwiGLU/GEGLU 等 GLU 变体；公式上 $\mathrm{GLU}_\phi(\mathbf{x}) = (\mathbf{P}\mathbf{x}) \odot \phi(\mathbf{W}\mathbf{x})$ 就是在原非门控前馈块上多乘了一支线性 gate，但论文与工程经验都一致地报告 GLU 比纯 ReLU/GELU 收敛更快、效果更好。

**现有痛点**：GLU 的优势完全停留在实证层面，几乎没有可解释的理论。已有解释多停留在"门控提供二阶非线性、增强表达力"这种泛泛之词，既不能告诉你 GLU 为什么从两层 MLP 这种小模型起就好用，也不能解释训练曲线中一个反直觉的现象：早期 ReLU 反而比 ReGLU 收敛更快，后期才被反超（loss-crossing）。

**核心矛盾**：把"训练误差快"和"泛化间隔小"两件事混在一起谈，会得到错误的归因。一个干净的拆分是 $\mathcal{L}_\mathcal{D}(f_\theta) = \mathcal{L}_S(f_\theta) + (\mathcal{L}_\mathcal{D}(f_\theta) - \mathcal{L}_S(f_\theta))$，前者是优化，后者是泛化，必须分开讨论才能定位 GLU 的真实贡献。

**本文目标**：(1) 用一个理论可分析的框架把 GLU 的核矩阵写出来，刻画其谱性质相对于非门控对应物的变化；(2) 把谱差异翻译成可观察的训练曲线现象（包括 loss-crossing）；(3) 实证地比较 GLU 和非 GLU 模型的泛化间隔，回答 GLU 究竟改善了优化还是泛化。

**切入角度**：作者选择 NTK 框架——一方面 NTK 把训练动力学完全编码到一个核矩阵的谱里，另一方面已有结论显示梯度下降到 $\epsilon$ 误差所需步数是 $\mathcal{O}(\kappa\log(1/\epsilon))$，其中 $\kappa = \lambda_{\max}/\lambda_{\min}$ 是 NTK 的条件数。只要能算出 GLU 和非 GLU 模型 NTK 的极端特征值，就能直接对比收敛速度。

**核心 idea**：在 LeCun 初始化下，两层 ReGLU 模型的 NTK 近似满足 $\tilde{\mathbf{K}} \approx \mathbf{K} \odot (\mathbf{X}\mathbf{X}^\top/d)$，即"原 ReLU NTK 与数据 Gram 阵的 Hadamard 积"。这个 Hadamard 重权使 NTK 谱被显著压缩（$\lambda_{\max}$ 降一阶、$\lambda_{\min}$ 不降反升），条件数从 $\mathcal{O}(n/d)$ 改进到 $\mathcal{O}(n/d^2)$，因此 GLU 的全部优势可以归约为"更好条件的 NTK 矩阵"。

## 方法详解

### 整体框架

本文不提新方法，而是用 NTK（神经正切核）框架把"GLU 为什么好"做成一个机制级证明。整条论证链是：先在 LeCun 初始化下写出门控/非门控两层网络 NTK 的解析形式，发现两者差一个 Hadamard 积；再用随机矩阵理论估极端特征值，把这个乘法结构翻译成"条件数低一阶"的定量界；最后把谱差异接回训练动力学，既解释了收敛更快、也解释了 loss-crossing 这条反直觉曲线，并用实证把"GLU 改善泛化"的解释排除掉。

具体设定是：输入 $\mathbf{x}\in\mathbb{R}^d$、隐藏宽度 $m$，非门控模型 $z(\mathbf{x}) = \mathbf{V}\phi(\mathbf{W}\mathbf{x})$，门控模型 $z(\mathbf{x}) = \mathbf{V}[(\mathbf{P}\mathbf{x}) \odot \phi(\mathbf{W}\mathbf{x})]$；权重独立 Gauss 初始化 $W_{ij}\sim\mathcal{N}(0,\sigma_w^2)$、$P_{ij}\sim\mathcal{N}(0,\sigma_p^2)$、$V_{ij}\sim\mathcal{N}(0,\sigma_v^2)$，取 LeCun 设置 $\sigma_w^2 = \sigma_p^2 = 1/d$、$\sigma_v^2 = 1/m$。所有谱估计都建立在这套初始化和 Marchenko–Pastur 分布、El Karoui 核矩阵展开、Weyl 不等式三件工具上。

### 关键设计

**1. GLU 的 NTK 等于"原 NTK ⊙ 数据 Gram 阵"：把架构差异压成一个乘法项**

GLU 的优势过去只有"门控增强表达力"这种泛泛之词，根子在于没人写得出门控模型 NTK 和非门控 NTK 的可比关系。作者的做法是对参数取期望、代入 LeCun 初始化，并利用大 $m$ 下 $\sigma_v^2 + \sigma_p^2 \approx \sigma_p^2$，得到逐元素关系 $\tilde{K}_{ij} \approx K_{ij}\cdot(\mathbf{x}_i^\top\mathbf{x}_j/d)$，矩阵化即 $\tilde{\mathbf{K}} \approx \mathbf{K}\odot(\mathbf{X}\mathbf{X}^\top/d)$——门控等价于用"数据 Gram 矩阵的归一版本"对原 NTK 做逐元素重加权。这一步之所以关键，是因为它把一个看似纯架构的设计（多乘一支线性 gate）和一个纯统计对象（数据 Gram 阵）画上等号：此后所有差异都集中在 $\mathbf{X}\mathbf{X}^\top/d$ 这一项，谱分析只需研究 Wishart 矩阵及其自 Hadamard 乘积，而不必对门控模型从头做渐近分析。

**2. 条件数尺度下降一阶（核心定理 3.1）：把"GLU 更好"变成"差一个 $d$ 倍"的可验证陈述**

有了 Hadamard 结构还不够，得把它翻译成能算的收敛量。作者先用 arc-cosine 核公式把 ReLU NTK Taylor 展开成三块 $\mathbf{K} = \alpha\mathbf{X}\mathbf{X}^\top + \beta\mathbf{rr}^\top + \gamma\mathbf{D}$（数据 Gram、$\mathbf{r}_i = \|\mathbf{x}_i\|$ 的秩 1 更新、对角修正），门控版本经 Hadamard 后变成

$$\tilde{\mathbf{K}} = \frac{\alpha}{d}(\mathbf{X}\mathbf{X}^\top)\odot(\mathbf{X}\mathbf{X}^\top) + \frac{\beta}{d}(\mathbf{rr}^\top)\odot(\mathbf{X}\mathbf{X}^\top) + \frac{\gamma}{d}\mathbf{D}^2.$$

对每块用 Marchenko–Pastur 与 El Karoui 展开估 $\lambda_{\max}$、$\lambda_{\min}$，再用 Weyl 不等式叠合，得到 $\lambda_{\max}(\mathbf{K}) = \mathcal{O}(mn/d)$ 而 $\lambda_{\max}(\tilde{\mathbf{K}}) = \mathcal{O}(mn/d^2)$，两者最小特征值同阶为 $\mathcal{O}(m)$ 但 $\lambda_{\min}(\tilde{\mathbf{K}}) \geq \lambda_{\min}(\mathbf{K})$，于是条件数从 $\kappa(\mathbf{K}) = \mathcal{O}(n/d)$ 降到 $\kappa(\tilde{\mathbf{K}}) = \mathcal{O}(n/d^2)$——整整低一个 $d$ 倍。由于梯度下降到 $\epsilon$ 误差需 $\mathcal{O}(\kappa\log(1/\epsilon))$ 步，这就直接把"GLU 收敛更快"坐实成定理，同时给出 GLU NTK 更"对角占优"的几何图像：对角项被放大、非对角项被抑制。

**3. 用谱分解解释 loss-crossing：早期 ReLU 更快、后期 ReGLU 反超不是噪声**

训练曲线里有个反直觉现象——早期 ReLU 反而比 ReGLU 收敛快，后期才被反超，容易被当成随机扰动。作者用同一张谱图就解释清楚：NTK 区里 MSE 沿各特征方向独立衰减，第 $i$ 个方向的误差按 $(1 - \eta\lambda_i)^t$ 收缩，早期由 $\lambda_{\max}$ 主导、后期由 $\lambda_{\min}$ 主导。ReLU 的 $\lambda_{\max}$ 更大，所以前期沿主方向衰减更猛；但 ReGLU 的 $\lambda_{\min}$ 更大，剩余分量收得更快，于是越过某个步数后实现反超。这一图像被形式化成命题 4.1 的无限宽损失闭式 $\mathbb{E}_\theta[L_k] \propto \mathrm{Tr}[(\mathbf{I}-\eta\mathbf{K})^{2k}\mathbf{K}] + \mathbf{Y}^\top(\mathbf{I}-\eta\mathbf{K})^{2k}\mathbf{Y}$ 和推论 4.2（在 $d\geq 5,n\geq 300$ 等条件下存在一个时间分界点把两阶段分开），从而把第 2 点的条件数定理延伸成完整的"谱压缩 → 训练动力学"叙事，也给出一个可单调验证的判据：用 loss-crossing 诊断新激活/新结构是否真的改善了优化。

### 损失函数 / 训练策略

本文是理论分析，不引入新损失或训练策略；实验沿用 MSE / 交叉熵和 SGD/AdamW 默认设置，在两层 MLP、MLP-Mixer、ViT、GPT-2 上对照 ReLU/ReGLU、GELU/GEGLU、SiLU/SwiGLU 三组激活对。

## 实验关键数据

### 主实验

| 验证对象 | 主要现象 | 与理论的吻合 |
|--------|------|------|
| 数值合成 NTK 矩阵（变 $d$） | ReGLU 的 $\lambda_{\max}$ 显著小于 ReLU、$\lambda_{\min}$ 略大，条件数低一阶 | 与命题 B.6/B.9 的解析估计在不同输入维度上几乎重合 |
| ViT FFN 替换 GLU 变体（CIFAR/ImageNet 设定） | 条件数随激活类型的变化趋势：非门控 > 门控 | 与定理 3.1 一致：GLU 在真实架构上仍压缩谱 |
| GPT-2 FFN 替换 SwiGLU/GEGLU 等 | 训练前后 NTK 条件数均小于 SiLU/GELU 对照 | 表明 LLM 上同样存在条件数改善 |

### 消融 / 分析实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 两层 MLP，ReLU vs ReGLU，Gaussian 输入，lr=0.005 | 早期 ReLU loss 低 → 后期 ReGLU 反超 | 复现 loss-crossing；与推论 4.2 的双阶段判据一致 |
| GELU/GEGLU、SiLU/SwiGLU 同样对比 | 同样观察到 crossing | 现象不依赖具体激活，验证机制级解释 |
| lr 提高（如 0.008） | crossing 被压缩、早期优势消失 | 大 lr 一刀切地加速所有特征方向，与谱解释一致 |
| MLP-Mixer/CIFAR-10、ViT/Tiny-ImageNet、GPT-2/FineWeb-Edu 的 $(L_S, L_\mathcal{D}-L_S)$ 散点 | GLU 与非 GLU 分布几乎重叠；能量距离置换检验 $p\geq 0.05$ | 证伪"GLU 减小泛化间隔"的假设；优化器（SGD→AdamW）反而显著左下移 |

### 关键发现

- 在两层网络的 NTK 视角下，GLU 的优势可以被一个干净的 Hadamard 积公式完全捕获，与具体激活函数关系不大，这解释了为什么 ReGLU/GEGLU/SwiGLU 几乎都好用。
- "GLU 学得更好"在实验上完全等价于"GLU 在相同训练 loss 下的优化效率更高"，**不是因为它泛化更强**——这是过去年内最容易被混淆的归因，本文用能量距离统计检验给出反例。
- Loss-crossing 不是噪声，而是 $\lambda_{\max}/\lambda_{\min}$ 谱差异的副产品；这个判据可以用于诊断新激活/新结构是否真的改善了优化。

## 亮点与洞察

- Hadamard-积结构是这篇论文最优雅的结果：它把一个看起来纯架构的设计（多乘一支 gate）与一个纯统计对象（数据 Gram 矩阵）联系起来，机制清晰、便于扩展到 attention 等其他场景。
- "把训练误差与泛化间隔显式拆开"的论文哲学很值得借鉴：现代大模型很多"涨点"现象都是两者混淆的产物，按 $\mathcal{L}_\mathcal{D} = \mathcal{L}_S + \text{gap}$ 分别画散点是非常便宜但有效的诊断工具。
- 论文给出的"对角占优 + 模型梯度角度增大"几何图像（$\cos\tilde{\phi}_{ij} = \cos\phi_{ij}\cdot\cos\alpha_{ij}$）说明门控等价于把样本在梯度特征空间里更好地分开，与 Liu et al. 2025 的梯度角度理论自然对接，可以反过来作为设计新激活的指南。

## 局限与展望

- 全部理论建立在两层网络的 NTK 区，对真实 LLM 的解释主要靠数值条件数和 loss-crossing 的实证延伸，深层 + 注意力机制下的严格谱分析仍是开问题。
- 推论 4.2 需要 $d\geq 5$、$n\geq 300$ 等较强条件，对小数据/低维场景不一定适用。
- 论文只回答了"GLU 为什么快"，没有回答"该把多少计算预算从其他模块挪给 gate"——这一资源分配问题在 LLM 工程中其实更迫切；一个自然的延伸是把条件数作为架构搜索的代理目标。
- 泛化结论建立在能量距离检验上，对超参数和数据规模的依赖还需更多场景验证。

## 相关工作与启发

- **vs De Ryck et al. 2024 / Liu et al. 2025（NTK 收敛理论）**：本文复用了 $\mathcal{O}(\kappa\log(1/\epsilon))$ 这条主干，但首次把它落到 GLU 这一具体架构上，并用 Hadamard 积给出条件数的显式改进幅度。
- **vs Shazeer 2020（GLU Variants 经验研究）**：Shazeer 给出经验排序，本文给出第一性原理的解释；两者互为佐证。
- **vs El Karoui 2010（核随机矩阵理论）**：本文的关键技术工具，被用来处理 Wishart 矩阵的自 Hadamard 乘积，是把架构问题转化为随机矩阵问题的桥梁。
- **vs Wang 2025 等"在 attention 上加 gate"工作**：本文的 Hadamard 解释可直接迁移到 GLU-attention，预测同样的条件数压缩效应——这是一个值得用本文框架去验证的方向。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Splintering Nonconcatenative Languages for Better Tokenization](../../ACL2025/llm_pretraining/splintering_nonconcatenative_languages_for_better_tokenization.md)
- [\[AAAI 2026\] ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences](../../AAAI2026/llm_pretraining/elspr_evaluator_llm_training_data_self-purification_on_non-transitive_preference.md)
- [\[NeurIPS 2025\] LCDB 1.1: A Database Illustrating Learning Curves Are More Ill-Behaved Than Previously Thought](../../NeurIPS2025/llm_pretraining/lcdb_11_a_database_illustrating_learning_curves_are_more_ill-behaved_than_previo.md)
- [\[ACL 2025\] Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning](../../ACL2025/llm_pretraining/making_llms_better_many-to-many_speech-to-text_translators_with_curriculum_learn.md)
- [\[CVPR 2025\] Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics](../../CVPR2025/llm_pretraining/influence_malleability_in_linearized_attention_dual_implications_of_non-converge.md)

</div>

<!-- RELATED:END -->
