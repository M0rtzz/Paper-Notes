---
title: >-
  [论文解读] SHAP Values via Sparse Fourier Representation
description: >-
  [NeurIPS 2025][SHAP值] 提出 FourierShap 算法，先将黑盒预测器近似为稀疏 Fourier 表示，再利用 Fourier 基函数的 SHAP 值闭式公式高效计算特征归因，实现相比 KernelShap 10-10000 倍的加速，同时支持精度-效率的可调权衡。
tags:
  - NeurIPS 2025
  - SHAP值
  - Fourier表示
  - 稀疏Walsh-Hadamard变换
  - 特征归因
  - 加速计算
---

# SHAP Values via Sparse Fourier Representation

**会议**: NeurIPS 2025  
**arXiv**: [2410.06300](https://arxiv.org/abs/2410.06300)  
**代码**: 暂无  
**领域**: 可解释AI / Shapley值计算  
**关键词**: SHAP值, Fourier表示, 稀疏Walsh-Hadamard变换, 特征归因, 加速计算

## 一句话总结

提出 FourierShap 算法，先将黑盒预测器近似为稀疏 Fourier 表示，再利用 Fourier 基函数的 SHAP 值闭式公式高效计算特征归因，实现相比 KernelShap 10-10000 倍的加速，同时支持精度-效率的可调权衡。

## 研究背景与动机

SHAP（SHapley Additive exPlanations）是可解释 AI 中最广泛使用的局部特征归因方法。其核心是 Shapley 值公式：

$$\phi_i(v) = \frac{1}{n} \sum_{S \subseteq [n] \setminus \{i\}} \frac{v(S \cup \{i\}) - v(S)}{\binom{n-1}{|S|}}$$

该公式涉及对所有可能子集的指数级求和（$2^n$ 项），计算代价极高。在工业级场景中（如广告定向、订阅倾向模型），需要对数百万样本进行解释，SHAP 计算成为严重瓶颈。

现有方法的局限：
- **KernelShap / LinRegShap**：通过随机采样子集近似指数和，每个待解释样本都需要求解一个代价高昂的最小二乘优化问题
- **TreeShap**：仅适用于树模型的白盒设置
- **FastShap**：训练 MLP 直接输出 SHAP 值，但精度依赖 MLP 的函数逼近能力，且更换背景数据集需重新训练

关键洞察是：许多实际模型具有**频谱偏置**——神经网络倾向于学习低阶函数，决策树集成本身就是稀疏 Fourier 函数。这意味着可以用紧凑的 Fourier 表示来近似这些模型，然后利用 Fourier 基的结构性质来高效计算 SHAP 值。

## 方法详解

### 整体框架

FourierShap 是一个两阶段算法：
1. **第一阶段**：将黑盒预测器近似为稀疏 Fourier 表示（对树模型可精确计算，对黑盒模型通过稀疏 Walsh-Hadamard 变换近似）
2. **第二阶段**：利用推导出的闭式公式，直接从 Fourier 表示计算 SHAP 值（无需指数级求和）

第一阶段只需执行一次，后续对每个新样本的 SHAP 计算均摊到第二阶段的高效计算中——这是**摊销化**SHAP 计算的核心优势。

### 关键设计

1. **Fourier 表示与稀疏性**：对于二值输入 $\{0,1\}^n$ 上的函数 $h$，其 Fourier 展开为 $h(x) = \frac{1}{\sqrt{2^n}} \sum_{f \in \{0,1\}^n} \hat{h}(f)(-1)^{\langle f, x \rangle}$。若只有 $k$ 个非零 Fourier 系数，则称 $h$ 为 $k$-稀疏的。关键结论是：

    - 深度为 $d$ 的决策树集成（$T$ 棵树）是 $O(T \cdot 4^d)$-稀疏的
    - 神经网络由于谱偏置倾向于学习低阶（因而稀疏）函数
    - 度为 $d$ 的函数是 $O(n^d)$-稀疏的（Proposition 2）

2. **SHAP 值的闭式公式（核心贡献）**：对单个 Fourier 基函数 $\Psi_f(x) = (-1)^{\langle f, x \rangle}$，通过组合论证推导出 SHAP 值的闭式解（Lemma 1）：
    $\phi_i^{\Psi_f} = -\frac{2f_i}{|\mathcal{D}|} \sum_{(x,y) \in \mathcal{D}} \mathbb{1}_{x_i \neq x_i^*} (-1)^{\langle f, x \rangle} \frac{(|A|+1) \bmod 2}{|A|+1}$
   其中 $A = \{j \in [n] | x_j \neq x_j^*, j \neq i, f_j = 1\}$。关键组合洞察是：指数和中大量正负项相互抵消，不需要实际计算就能统计抵消的数量。

3. **总体复杂度（Theorem 2）**：利用 SHAP 值关于被解释函数的线性性，最终公式为：
    $\phi_i^h = -\frac{2}{|\mathcal{D}|} \sum_{f \in \text{supp}(h)} \hat{h}(f) \cdot f_i \cdot \sum_{(x,y) \in \mathcal{D}} \mathbb{1}_{x_i \neq x_i^*} (-1)^{\langle f, x \rangle} \frac{(|A|+1) \bmod 2}{|A|+1}$
   计算复杂度为 $\Theta(n \cdot |\mathcal{D}| \cdot k)$，完全消除了指数级求和。且两个求和都可以用 JAX 的 vmap 在 GPU 上高效并行化。

### 树模型的精确设置

对决策树集成，可通过递归公式精确计算其 Fourier 表示：$t(x) = \frac{1+(-1)^{\langle e_i, x \rangle}}{2} t_{\text{left}}(x) + \frac{1-(-1)^{\langle e_i, x \rangle}}{2} t_{\text{right}}(x)$。由于 Fourier 变换的线性性，集成的 Fourier 变换等于各树 Fourier 变换的平均。在此设置下，FourierShap 计算的 SHAP 值与 TreeShap 完全一致。

## 实验关键数据

### 主实验（黑盒设置）

在四个真实数据集上评估（Entacmaea n=13, SGEMM n=40, GB1 n=80, avGFP n=236），对比 SHAP 值精度（$R^2$）与计算速度（相对 KernelShap 的加速倍数）。

| 方法 | 类型 | Entacmaea 加速 | SGEMM 加速 | 精度控制 |
|------|------|------|------|------|
| KernelShap | 黑盒 | 1× (基准) | 1× (基准) | 需收敛检查 |
| LinRegShap | 黑盒 | ~10× | ~10× | 需收敛检查 |
| DeepLift | 白盒(NN) | ~100× | ~10× | 固定 |
| FastShap | 黑盒 | ~1000× | ~100× | 不可靠 |
| **FourierShap** | **黑盒** | **~10000×** | **~1000×** | **精细可调(k)** |

FourierShap 比 KernelShap 快 10-10000×，且在 3/4 数据集上比 FastShap 精度更高。

### 树模型设置

| 方法 | Entacmaea 加速 | SGEMM 加速 | GB1 加速 | avGFP 加速 |
|------|------|------|------|------|
| TreeShap | 1× | 1× | 1× | 1× |
| FastTreeShap | ~2-5× | ~2-5× | ~2-5× | ~2-5× |
| GPU TreeShap | ~5-10× | ~5× | ~3× | ~3× |
| PLTreeShap | ~3× | ~3× | ~2× | ~2× |
| **FourierShap** | **~100×** | **~50×** | **~10-30×** | **~5-20×** |

在树模型上，FourierShap 同样实现了数量级的加速，计算结果与 TreeShap 完全一致（精确）。优势在低维和浅树时最显著，随深度和特征数增加而减弱。

### 关键发现

- 控制稀疏度参数 $k$ 可以**精细、可靠地**权衡精度与速度：更大的 $k$ 意味着更精确的函数近似但更慢的计算
- 函数近似（第一阶段）只需做一次，之后每个新样本的 SHAP 计算比 KernelShap 快数个数量级
- 与 FastShap 相比，FourierShap 精度-速度曲线更平滑可控，且更换背景数据集无需重新训练
- 实验验证了神经网络确实具有频谱偏置——Fourier 函数近似的 $R^2$ 在相对较少的系数下就达到 0.95+

## 亮点与洞察

- 将频谱分析与博弈论解释性巧妙结合，通过消除指数和实现了本质性的计算加速
- Lemma 1 的组合论证是核心创新：不是近似指数和，而是解析地计算指数和（对 Fourier 基函数）
- 摊销化设计使其特别适合需要解释大量样本的工业场景
- 在黑盒设置下超越了需要模型白盒访问的 DeepLift，令人印象深刻

## 局限与展望

- 假设输入为二值特征——连续特征需先离散化为分位数，可能丢失信息
- 稀疏 WHT 的第一阶段计算本身也有开销，当模型不够稀疏时近似质量下降
- 随着特征数和树深度增加，Fourier 系数数量快速增长（$O(T \cdot 4^d)$），优势减弱
- 仅计算 Interventional SHAP 值，不支持 Conditional SHAP

## 相关工作与启发

- 频谱偏置理论（Yang & Salman 2019）为本方法提供了关键的理论基础
- 稀疏 Fourier 变换算法（Amrollahi et al. 2019）是实现的关键工具
- 启发：对于有结构的函数类，利用其数学结构（如稀疏性）可以实现质的计算加速
- 更广泛的教训：可解释性方法的计算瓶颈可以通过换一种函数表示来突破

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 将 Fourier 分析引入 SHAP 计算是全新思路，闭式公式是重要理论贡献
- **实验充分度**: ⭐⭐⭐⭐ 黑盒和白盒两种设置，四个数据集，多个基线对比
- **写作质量**: ⭐⭐⭐⭐ 数学推导清晰，但符号较多需仔细阅读
- **价值**: ⭐⭐⭐⭐⭐ 解决了 SHAP 计算的核心瓶颈，对工业应用有直接价值

<!-- RELATED:START -->

## 相关论文

- [Deep Value Benchmark: Measuring Whether Models Generalize Deep Values or Shallow Preferences](deep_value_benchmark_measuring_whether_models_generalize_deep_values_or_shallow_.md)
- [Why Is Attention Sparse in Particle Transformer?](why_is_attention_sparse_in_particle_transformer.md)
- [Do Different Prompting Methods Yield a Common Task Representation?](do_different_prompting_methods_yield_a_common_task_representation_in_language_mo.md)
- [A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](a_is_for_absorption_studying_feature_splitting_and_absorption_in_sparse_autoenco.md)
- [From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit](from_flat_to_hierarchical_extracting_sparse_representations_with_matching_pursui.md)

<!-- RELATED:END -->
