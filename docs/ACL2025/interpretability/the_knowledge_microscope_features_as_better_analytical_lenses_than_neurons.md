---
title: >-
  [论文解读] The Knowledge Microscope: Features as Better Analytical Lenses than Neurons
description: >-
  [ACL 2025][可解释性][稀疏自编码器] 本文通过系统实验验证了 SAE（稀疏自编码器）分解出的特征（features）在知识表达影响力、可解释性、单义性（monosemanticity）三个维度上全面优于传统神经元（neurons）作为分析单元，并提出首个基于 feature 的模型编辑方法 FeatureEdit，在隐私知识擦除任务上大幅超越神经元方法。
tags:
  - "ACL 2025"
  - "可解释性"
  - "稀疏自编码器"
  - "知识神经元"
  - "特征分析"
  - "单义性"
  - "隐私知识擦除"
---

# The Knowledge Microscope: Features as Better Analytical Lenses than Neurons

**会议**: ACL 2025  
**arXiv**: [2502.12483](https://arxiv.org/abs/2502.12483)  
**代码**: 无（论文中提到 "Code and dataset will be available"）  
**领域**: 其他  
**关键词**: 稀疏自编码器, 知识神经元, 特征分析, 单义性, 隐私知识擦除

## 一句话总结

本文通过系统实验验证了 SAE（稀疏自编码器）分解出的特征（features）在知识表达影响力、可解释性、单义性（monosemanticity）三个维度上全面优于传统神经元（neurons）作为分析单元，并提出首个基于 feature 的模型编辑方法 FeatureEdit，在隐私知识擦除任务上大幅超越神经元方法。

## 研究背景与动机

1. **领域现状**: 理解 LLM 中事实知识的存储和表达机制是 mechanistic interpretability 的核心问题。主流方法以 MLP 神经元为分析单元，产生了"知识神经元"理论——某些特定神经元负责存储特定知识。

2. **现有痛点**: 神经元存在严重的**多义性（polysemanticity）**问题——一个神经元对多种不相关的事实都会响应，导致：(1) **有限的知识表达能力**：知识分散存储在大量神经元中，单个神经元贡献微弱；(2) **差的可解释性**：一个神经元与多种不相关事实耦合，难以准确描述其功能。例如 Gemma-2 2B 只有约 230k 个神经元，但训练数据约 2 万亿 token，单个神经元必然关联多种事实。

3. **核心矛盾**: 用多义的神经元来分析单一知识的机制，是一个根本性的粒度不匹配。

4. **本文目标**: (1) 能否用 SAE 将神经元分解为更细粒度的特征来作为分析单元？(2) 特征是否能解决神经元的有限知识表达和差可解释性问题？(3) 特征在下游任务中是否优于神经元？

5. **切入角度**: SAE 将低维神经元空间映射到高维特征空间（类似升维使原本不可分的变成可分的），使不同事实对应不同特征，实现"知识显微镜"的效果。

6. **核心 idea**: 用稀疏自编码器将神经元分解为特征，特征在知识表达、可解释性和单义性上全面优于神经元，并能在隐私保护任务中发挥更好的效果。

## 方法详解

### 整体框架

研究分四部分。(1) 初步实验验证 SAE 是最佳的神经元→特征分解方法（vs PCA、ICA、随机方向）；(2) 比较特征和神经元的知识表达影响力（通过消融后预测概率变化 $\Delta Prob$）；(3) 比较可解释性（通过 LLM 预测激活值的相关性 $IS$）和单义性（通过激活分布分离度）；(4) 提出 FeatureEdit 方法在隐私知识擦除任务上验证实用价值。

### 关键设计

1. **SAE 特征提取与评估方法论**:

    - 功能：将 MLP 激活分解为可解释的高维特征
    - 核心思路：使用 Gemma Scope 提供的预训练 SAE，对 MLP 激活 $\mathbf{h}$ 通过编码器函数 $\mathbf{f}(\mathbf{h}) = \sigma(\mathbf{W}_{enc}\mathbf{h} + \mathbf{b}_{enc})$ 得到特征激活（$\sigma$ 为 JumpReLU）。通过阈值 $\tau_1$ 选择高激活特征：$\mathbf{F_a} = \{(l,p) | f_{l,p}(\mathbf{h}) > \tau_1 \cdot \max f \}$。对特征进行消融时（清零），通过 SAE 解码器重构激活向量来替换原始激活。两个核心评估指标：$\Delta Prob$（消融后答案概率下降比例）和 $IS$（LLM 预测激活与真实激活的相关性即可解释性分数）
    - 设计动机：SAE 的高维稀疏表示天然适合"一个特征对应一类知识"的理想分析范式，解决了神经元"一对多"的根本问题

2. **多组件特征对比分析**:

    - 功能：找出最有效的特征提取位置
    - 核心思路：比较三个 Transformer 组件的特征：post-attention residual、MLP activation、post-MLP residual。发现 post-MLP residual 特征在 $\Delta Prob$（约 0.85，消融单个特征即可达 0.6）和 $IS$（约 0.6，是神经元的 4 倍）上均最优
    - 设计动机：全面比较不同位置的特征有助于理解知识在 Transformer 中的流动路径——注意力处理后经 MLP 加工的残差连接包含最丰富的事实知识信息

3. **FeatureEdit 知识擦除方法**:

    - 功能：首个基于 feature 的模型编辑方法，用于擦除隐私敏感知识
    - 核心思路：对于要擦除的知识对应的 feature $f_l^i$，构造 one-hot 探测向量 $\mathbf{p}^i$，通过 SAE 解码器重构得到该 feature 在 MLP 权重空间中的贡献模式 $\mathbf{h}^i = \mathbf{W}_e^T \mathbf{p}^i$。在 $\mathbf{h}^i$ 中找到绝对值超过阈值 $\tau_2$ 的位置，在 MLP 第二层权重 $\mathbf{W}_l^{(2)}$ 中将这些位置清零。与神经元方法（直接清零整列）不同，FeatureEdit 在权重矩阵中选择性清零特定位置，粒度更细
    - 设计动机：特征的单义性使得擦除操作更精准——只影响目标知识，不波及无关事实。神经元方法清零整列影响面太大

### 损失函数 / 训练策略

本文不涉及新模型训练，使用 Gemma Scope 的预训练 SAE。FeatureEdit 是无需训练的推理时编辑方法。隐私数据集 PrivacyParaRel 通过增量微调注入模型后再进行擦除实验。

## 实验关键数据

### 初步实验：SAE vs 其他分解方法

Gemma-2 9B 上:

| 方法 | $\Delta Prob$ | IS (可解释性分数) |
|------|-----|-----|
| SAE | ~0.78 | ~0.64 |
| ICA | ~0.60 (×1.3) | ~0.32 (×2.0) |
| PCA | 较低 | 较低 |
| 随机方向 | 最低 | 最低 |

### 主实验：Features vs Neurons

| 分析单元 | $\Delta Prob$ (知识表达) | IS (可解释性) |
|---------|---------|---------|
| Post-MLP Features | **~0.85** | **~0.6** |
| MLP Features | ~0.75 | ~0.6 |
| Post-Attention Features | ~0.55 | ~0.5 |
| Knowledge Neurons | ~0.45 (1.9×差距) | ~0.15 (4×差距) |

精细消融（逐步消除 features/neurons）:

| 消除数量 | Post-MLP Feature $\Delta Prob$ | Neuron $\Delta Prob$ |
|---------|---------|---------|
| 1 | ~0.6 | ~0.2 (3×差距) |
| 5 | ~0.8 | ~0.35 |
| 10 | ~0.85 | ~0.4 |

### 隐私知识擦除 (FeatureEdit vs Neuron-based)

| 指标 | FeatureEdit | 神经元方法 |
|------|--------|--------|
| Reliability (擦除成功率) | **~0.8** | ~0.65 |
| Generalization (跨表述泛化) | **~0.7** | ~0.25 |
| Locality (无关知识保留) | **~0.7** | ~0.2 |
| $\Delta$PPL (困惑度变化↓) | **~0.1** | ~0.3 |

### 关键发现

- **消除单个 post-MLP feature 的影响相当于消除约3个知识神经元**，说明 feature 是更精准的知识定位单元
- Feature 的 IS 可解释性评分是神经元的 4 倍（~0.6 vs ~0.15），验证了"多义性→差可解释性"的因果链
- 单义性实验中，feature 的激活分布随相关事实比例增加呈现清晰的分离波形，而神经元的分布高度重叠，在 0% 相关事实时也有显著激活（p<0.001，Cohen's d>0.8）
- FeatureEdit 在 Generalization 上优势最大（0.7 vs 0.25），说明 feature 方法有效防止了"换个问法就泄露"的 jailbreak 问题
- 特征分布模式在不同特征数量 $N$（1×到8×模型维度）下保持一致，可以固定 $N=4\times d_{model}$ 而无需调参

## 亮点与洞察

- **"知识显微镜"的类比非常贴切**：SAE 就像把粗颗粒的神经元用更高分辨率的显微镜观察，看到了之前看不到的细粒度知识结构。这个直觉可以启发更多用 SAE 分析模型内部机制的工作
- **FeatureEdit 的粒度优势**：神经元方法清零权重矩阵的整列（影响该神经元的所有下游连接），而 FeatureEdit 通过 SAE 解码器追踪 feature 在权重空间中的分布，仅清零相关位置，实现"精准手术"而非"粗暴截肢"
- **单义性的直接验证方法很优雅**：通过控制输入中相关事实的比例（0%→100%），观察 feature/neuron 的激活分布变化，直观展示了 feature 的"该亮则亮、该灭则灭"vs neuron 的"总是半亮不灭"
- **Post-MLP residual 是最佳分析位置**：这一发现对后续使用 SAE 分析 LLM 的工作有重要参考价值

## 局限与展望

- 实验仅在 Gemma-2 (2B/9B) 上进行，需要在更多架构和更大规模模型上验证
- SAE 本身的训练质量影响分析结论——依赖 Gemma Scope 的预训练 SAE，换其他 SAE 实现可能结果不同
- FeatureEdit 的阈值 $\tau_2$ 需要手动设定，缺少自适应确定方法
- PrivacyParaRel 使用合成的隐私数据，与真实模型中自然习得的隐私信息可能分布不同
- 未探索 feature 在更复杂知识操作（如知识更新、知识冲突解决）中的应用

## 相关工作与启发

- **vs Knowledge Neurons (Dai et al. 2022; Chen et al. 2024a)**: 知识神经元理论假设特定神经元存储特定知识，但本文证明这种对应关系不够精确，feature 是更合适的分析粒度
- **vs SAE 可解释性 (Bricken et al. 2023)**: Bricken 等首次展示 SAE 可将神经元分解为可解释 feature，但主要在一般文本处理领域。本文专注于事实知识领域，提供了系统的定量比较
- **vs ROME/MEMIT (Meng et al. 2022, 2023)**: 这些方法基于知识神经元理论进行知识编辑，FeatureEdit 提供了 feature 层面的替代方案，理论上更精准
- SAE 特征用于模型编辑的思路可以扩展到更多场景——知识冲突检测、事实验证、安全对齐中的有害知识定位等

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 SAE 特征系统性地与神经元对比用于事实知识分析是新的视角，FeatureEdit 是首个 feature-based 编辑方法
- 实验充分度: ⭐⭐⭐⭐⭐ 从多个角度（知识表达、可解释性、单义性、下游任务）系统验证，统计检验充分，可视化直观
- 写作质量: ⭐⭐⭐⭐ 研究问题层层递进（Q1→Q2→Q3）的结构清晰，但公式较密集
- 价值: ⭐⭐⭐⭐⭐ 为 LLM 可解释性研究提供了更好的分析工具和方法论，FeatureEdit 有实际应用价值（隐私保护）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Byte Latent Transformer: Patches Scale Better Than Tokens](byte_latent_transformer.md)
- [\[ACL 2025\] Better Embeddings with Coupled Adam](better_embeddings_with_coupled_adam.md)
- [\[ICLR 2026\] FastLSQ: Solving PDEs in One Shot via Fourier Features with Exact Analytical Derivatives](../../ICLR2026/others/fastlsq_solving_pdes_in_one_shot_via_fourier_features_with_exact_analytical_deri.md)
- [\[ACL 2025\] Hierarchical Attention Generates Better Proofs](hierarchical_attention_generates_better_proofs.md)
- [\[ACL 2025\] Towards Better Evaluation for Generated Patent Claims](patclaimeval_patent_evaluation.md)

</div>

<!-- RELATED:END -->
