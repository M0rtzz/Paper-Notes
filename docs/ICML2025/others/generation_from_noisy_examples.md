---
title: >-
  [论文解读] Generation from Noisy Examples
description: >-
  [ICML2025][生成理论] 将 Kleinberg & Mullainathan (2024) 的"极限语言生成"理论框架扩展至噪声样本流场景，提出 Noisy Closure 维度，完整刻画了均匀噪声依赖可生成性的充要条件，并证明所有可数假设类在有限噪声下仍可非均匀生成。
tags:
  - "ICML2025"
  - "生成理论"
  - "噪声鲁棒性"
  - "组合维度"
  - "学习理论"
  - "极限生成"
---

# Generation from Noisy Examples

**会议**: ICML2025  
**arXiv**: [2501.04179](https://arxiv.org/abs/2501.04179)  
**代码**: 无（纯理论工作）  
**领域**: 其他  
**关键词**: 生成理论, 噪声鲁棒性, 组合维度, 学习理论, 极限生成

## 一句话总结

将 Kleinberg & Mullainathan (2024) 的"极限语言生成"理论框架扩展至噪声样本流场景，提出 Noisy Closure 维度，完整刻画了均匀噪声依赖可生成性的充要条件，并证明所有可数假设类在有限噪声下仍可非均匀生成。

## 研究背景与动机

Kleinberg & Mullainathan (2024) 提出了"极限语言生成"模型：给定可数语言族 $C$，对手选择一个语言 $K \in C$ 并逐一展示其中的字符串，生成器需要输出 $K$ 中新的、未见过的字符串。他们证明了一个惊人的正面结果——**所有可数语言族都可极限生成**，这与 Gold (1967) 语言辨识中"大多数语言族不可辨识"形成鲜明对比。

Li et al. (2024) 进一步将该框架推广到二元假设类 $\mathcal{H} \subseteq \{0,1\}^{\mathcal{X}}$，定义了均匀 (uniform) 和非均匀 (non-uniform) 可生成性，并用 Closure 维度给出了完整刻画。

然而，上述工作都假设样本流是**无噪声的**——每个样本都是正样本。现实中这一假设不成立：
- LLM 常用其他 LLM 的幻觉输出训练 (Burns et al., 2023)
- 训练数据可能遭受数据投毒攻击 (Zhang et al., 2023)

本文提出的核心问题是：**当对手可在正样本流中插入有限个负样本时，生成能力会如何变化？**

## 方法详解

### 问题形式化

设 $\mathcal{X}$ 为可数样本空间，$\mathcal{H} \subseteq \{0,1\}^{\mathcal{X}}$ 为满足均匀无界支撑 (UUS) 性质的二元假设类。在噪声模型中，对手选定 $h \in \mathcal{H}$ 及其正样本序列后，可插入至多 $n^{\star}$ 个负样本，生成器不知道哪些是负样本。

### 四种噪声可生成性定义

论文按"最终完美生成"所需样本数的依赖关系，定义了由强到弱的四种噪声可生成性：

| 定义 | $d^{\star}$ 依赖于 | 强度 |
|------|-------------------|------|
| 均匀噪声无关 (Uniform Noise-Independent) | 仅类 $\mathcal{H}$ | 最强 |
| 均匀噪声依赖 (Uniform Noise-Dependent) | $\mathcal{H}$ + 噪声级 $n^{\star}$ | 较强 |
| 非均匀噪声依赖 (Non-uniform Noise-Dependent) | $\mathcal{H}$ + $n^{\star}$ + 假设 $h$ | 较弱 |
| 噪声极限可生成 (Noisy in the Limit) | $\mathcal{H}$ + $n^{\star}$ + $h$ + 流 | 最弱 |

### 核心工具：Noisy Closure 维度

定义噪声级 $n$ 下的闭包算子：

$$\langle x_{1:d} \rangle_{\mathcal{H},n} := \bigcap_{h \in \mathcal{H}(x_{1:d};n)} \operatorname{supp}(h)$$

其中 $\mathcal{H}(x_{1:d};n) = \{h \in \mathcal{H} : |\{x_{1:d}\} \cap \operatorname{supp}(h)| \geq d - n\}$ 是与序列至多不一致 $n$ 个样本的假设集。

**$n$-Noisy Closure 维度** $\mathrm{NC}_n(\mathcal{H})$ 定义为满足 $\langle x_{1:d} \rangle_{\mathcal{H},n} \neq \bot$ 且 $|\langle x_{1:d} \rangle_{\mathcal{H},n}| < \infty$ 的最大 $d$。

与无噪声 Closure 维度的关键区别：Noisy Closure 维度是**尺度敏感**的——每个噪声级 $n$ 对应一个维度值。

### 构造性生成器

对于均匀噪声依赖可生成的类，论文构造了一个不需要事先知道噪声级的生成器：在观测到 $\mathrm{NC}_n(\mathcal{H}) + 1$ 个不同样本后即可完美生成（当噪声级为 $n$ 时）。对于噪声极限可生成性，通过 Algorithm 1 将无噪声非均匀生成器 $\mathcal{Q}$ 包装为噪声鲁棒生成器 $\mathcal{G}$：每轮取流的后半段（确保无噪声）调用 $\mathcal{Q}$ 反复生成候选，过滤掉已见样本。

## 理论结果

| 结果 | 内容 | 意义 |
|------|------|------|
| **Thm 3.1** | 均匀噪声无关可生成 $\Leftrightarrow$ $\|\bigcap_{h\in\mathcal{H}} \operatorname{supp}(h)\| = \infty$ | 极端困难：仅"平凡"类可行 |
| **Thm 3.3** | 均匀噪声依赖可生成 $\Leftrightarrow$ $\forall n, \mathrm{NC}_n(\mathcal{H}) < \infty$ | 完整刻画，样本复杂度 $\Theta(\mathrm{NC}_n(\mathcal{H}))$ |
| **Cor 3.4** | 所有有限类均匀噪声依赖可生成 | 有限类噪声鲁棒 |
| **Lem 3.5** | 存在可数类无噪声均匀可生成但 $\mathrm{NC}_1 = \infty$ | 噪声严格增加难度 |
| **Lem 3.6** | 可分解为 $\bigcup \mathcal{H}_i$（$\mathrm{NC}_i(\mathcal{H}_i) < \infty$）则非均匀噪声依赖可生成 | 充分条件 |
| **Cor 3.7** | 所有可数类非均匀噪声依赖可生成 | 可数类噪声"免费" |
| **Thm 3.9** | 无噪声非均匀可生成 $\Rightarrow$ 噪声极限可生成 | 无噪声能力蕴含噪声鲁棒 |
| **Thm 3.10** | 有限个均匀噪声无关可生成类之并 → 噪声极限可生成 | 类的可分解充分条件 |

## 亮点与洞察

- **噪声对生成的影响是温和的**：虽然噪声使均匀生成严格变难（Lem 3.5），但对可数类的非均匀/极限可生成性完全无影响（Cor 3.7）
- **Noisy Closure 维度的尺度敏感性**与 PAC 回归中的 fat-shattering 维度类似，是该领域首次引入此类结构
- **构造性证明**：所有正面结果都给出了显式生成器算法，而非纯存在性证明
- **生成 vs 辨识的分离持续成立**：在噪声下生成仍比辨识容易得多
- **巧妙的反例**（Lem 3.5）：用素数幂负集构造的假设类，无噪声 Closure 维度为 0 但 $\mathrm{NC}_1 = \infty$

## 关键证明思路

**Thm 3.3 必要性**：若 $\exists n$ 使 $\mathrm{NC}_n(\mathcal{H}) = \infty$，则对任意生成器 $\mathcal{G}$ 和任意样本数 $d$，都能构造一个假设 $h$ 和噪声流使 $\mathcal{G}$ 在观测 $d$ 个不同样本后犯错。利用无穷 Noisy Closure 维度保证了总能找到使闭包有限的序列来"欺骗"生成器。

**Thm 3.3 充分性**：显式构造生成器：观测 $\mathrm{NC}_n(\mathcal{H}) + 1$ 个不同样本后，对所有可能噪声级别，闭包 $\langle x_{1:d} \rangle_{\mathcal{H},n}$ 必为无穷集（或 $\bot$），从而总能从中选取新的正样本。核心技巧是生成器不需要知道真实噪声级——它同时"对冲"所有可能的 $n$。

**Cor 3.4**：有限类 $|\mathcal{H}| = q$ 时，$\mathrm{NC}_n(\mathcal{H}) < nq + d + 1 < \infty$，通过鸽巢原理和子集支撑交集的有限性获得上界。

## 局限与展望

- **非均匀噪声依赖可生成的完整刻画仍是开放问题**：充分和必要条件不匹配
- **噪声极限可生成的完整刻画未解决**（继承自 Li et al. 2024 的开放问题）
- **仅考虑插入式噪声**：对手只能插入负样本、不能替换正样本，这是较弱的噪声模型
- **纯信息论视角**：未讨论计算可行性，给定 membership oracle 的可计算噪声生成算法仍是开放问题
- **噪声有限**：要求负样本总数有限，未覆盖无限噪声场景（Jain 1994 的密度噪声模型）
- 无实验验证——纯理论贡献

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将噪声鲁棒性引入极限生成理论，Noisy Closure 维度是自然而非平凡的新组合维度
- 理论深度: ⭐⭐⭐⭐⭐ — 均匀噪声依赖可生成性给出完整刻画（充要条件+样本复杂度），证明技术扎实
- 写作质量: ⭐⭐⭐⭐ — 层次清晰，定义-定理-证明结构严谨，但符号较重
- 价值: ⭐⭐⭐⭐ — 推进了生成理论基础，但开放问题的存在表明工作尚未完全闭环

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Joint Asymmetric Loss for Learning with Noisy Labels](../../ICCV2025/others/joint_asymmetric_loss_for_learning_with_noisy_labels.md)
- [\[ICLR 2026\] ANO: Faster is Better in Noisy Landscapes](../../ICLR2026/others/ano_faster_is_better_in_noisy_landscape.md)
- [\[ACL 2025\] Predicting Through Generation: Why Generation Is Better for Prediction](../../ACL2025/others/predicting_through_generation_why_generation_is_better_for_prediction.md)
- [\[ECCV 2024\] Foster Adaptivity and Balance in Learning with Noisy Labels](../../ECCV2024/others/foster_adaptivity_and_balance_in_learning_with_noisy_labels.md)
- [\[ACL 2025\] The Noisy Path from Source to Citation: Measuring How Scholars Engage with Past Research](../../ACL2025/others/the_noisy_path_from_source_to_citation_measuring_how_scholars_engage_with_past_r.md)

</div>

<!-- RELATED:END -->
