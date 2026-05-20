---
title: >-
  [论文解读] Disentangling Direction and Magnitude in Transformer Representations: A Double Dissociation Through L2-Matched Perturbation Analysis
description: >-
  [ICML 2026][可解释性][线性表征假设] 本文用 L2 匹配扰动协议，证明 Pythia 系列里方向（角度）扰动对语言建模 loss 的破坏力是同等位移幅值扰动的 42.9 倍，而幅值扰动对句法（主谓一致）的破坏远高于角度——这是一对认知神经科学意义上的 "双重分离"…
tags:
  - "ICML 2026"
  - "可解释性"
  - "线性表征假设"
  - "方向 vs 幅值"
  - "L2 匹配扰动"
  - "LayerNorm"
  - "注意力路径"
---

# Disentangling Direction and Magnitude in Transformer Representations: A Double Dissociation Through L2-Matched Perturbation Analysis

**会议**: ICML 2026  
**arXiv**: [2602.11169](https://arxiv.org/abs/2602.11169)  
**代码**: 未公开  
**领域**: 可解释性 / 表征几何 / 因果干预  
**关键词**: 线性表征假设, 方向 vs 幅值, L2 匹配扰动, LayerNorm, 注意力路径

## 一句话总结
本文用 L2 匹配扰动协议，证明 Pythia 系列里方向（角度）扰动对语言建模 loss 的破坏力是同等位移幅值扰动的 42.9 倍，而幅值扰动对句法（主谓一致）的破坏远高于角度——这是一对认知神经科学意义上的 "双重分离"，对应方向走 attention 路径、幅值走 LayerNorm 路径。

## 研究背景与动机
**领域现状**：线性表征假设（LRH）是当前可解释性研究的基石——把概念编码成激活空间里的方向，用线性 probe 抽取语义特征。activation patching、TunedLens、representation engineering 都建立在 "方向重要" 这一假定上。

**现有痛点**：LRH 对幅值（norm）默认沉默——但 norm 在 transformer 里并非常数：Kobayashi 等发现它跨 token、跨 layer 显著变化；LayerNorm 显式操纵 norm；representation engineering 通过缩放 vector 修改行为。没人系统比较过方向和幅值的因果重要性。

**核心矛盾**：朴素比较 = 把方向扰一个小角度、把幅值缩一个小因子，但二者在表征空间的实际位移大小完全不同。如果角度扰动破坏更大，到底是因为方向更重要，还是因为这种扰动 "更暴力"？没有控制位移，所有比较都是无效的。

**本文目标**：(1) 构造 L2 匹配的扰动协议消除大小混淆；(2) 在 Pythia 上系统测量方向与幅值对不同下游任务的因果重要性；(3) 通过 pathway 修复实验定位影响传播的机制路径。

**切入角度**：借用认知神经科学的 "double dissociation" 工具——如果 A 操作主要损害任务 X 而不损害 Y，B 操作主要损害 Y 而不损害 X，那就说明 X 和 Y 由可分离的子系统支持。

**核心 idea**：用 $\delta$ 参数化 "扰动强度"，强制角度扰动和幅值扰动在 intervention layer 处的 Euclidean 位移完全等于 $\delta$，然后比较二者对 loss / 句法准确率的影响。

## 方法详解

### 整体框架
对 Pythia-410M 的中层（layer 8-15），在每个 token 位置的隐状态 $\mathbf{h}$ 上应用两种扰动之一：

1. **磁度扰动**：$\mathbf{h}'_{\text{mag}} = \alpha \mathbf{h}$，方向不变、长度变。
2. **角度扰动**：$\mathbf{h}'_{\text{ang}} = \|\mathbf{h}\| \cdot \hat{\mathbf{h}}'$，长度不变、方向旋转 $\theta$。

通过解析公式让两种扰动满足 $\|\mathbf{h} - \mathbf{h}'_{\text{mag}}\| = \|\mathbf{h} - \mathbf{h}'_{\text{ang}}\| = \delta$，然后测下游 (a) WikiText cross-entropy loss；(b) BLiMP 主谓一致准确率；(c) attention / LayerNorm pathway 修复后的恢复率。

### 关键设计

1. **L2 匹配扰动公式**:

    - 功能：消除扰动大小对方向 / 幅值比较的混淆。
    - 核心思路：磁度侧从 $|1-\alpha| \cdot \|\mathbf{h}\| = \delta$ 反解出 $\alpha = 1 \pm \delta / \|\mathbf{h}\|$，符号随机抽（缩大或缩小各占一半），需要 $\delta < \|\mathbf{h}\|$。角度侧采样一个正交单位向量 $\mathbf{v} \perp \mathbf{h}$，写 $\mathbf{h}'_{\text{ang}} = \|\mathbf{h}\|(\cos\theta \cdot \hat{\mathbf{h}} + \sin\theta \cdot \hat{\mathbf{v}})$，由 $\|\mathbf{h} - \mathbf{h}'_{\text{ang}}\| = \delta$ 推出 $\theta = \arccos(1 - \delta^2 / 2\|\mathbf{h}\|^2)$。每次实验经验性验证扰动后位移误差 < 0.01。
    - 设计动机：把 "方向 vs 幅值" 从两个不可比的轴投影到统一的 $\delta$ 维度，让因果效应差异完全归因于扰动 "类型"，这是整篇文章成立的方法论前提。

2. **Cross-over dissociation 测量**:

    - 功能：分别在 "宏观损失" 和 "细粒度句法" 两个维度测两种扰动的影响。
    - 核心思路：宏观侧用 WikiText-103 上 281 句话的 next-token cross-entropy，覆盖 10-64 token 长度；细粒度侧用 BLiMP 的 irregular / regular plural subject-verb agreement 200 对最小对（如 "The dogs run" vs "The dogs runs"），看模型是否仍给合法句更高概率。同样的 $\delta \in \{1, 2, 5, 10, 15, 20\}$ 6 档强度。5 个随机 seed，pair t-test + Bonferroni 校正。
    - 设计动机：选这两个任务是因为它们在 "信息密度" 和 "敏感的几何属性" 上互补——next-token 是高熵全局预测，对方向极敏感；主谓一致是低维离散决策，对 norm 这种数值幅度更敏感。

3. **Attention / LayerNorm pathway 修复**:

    - 功能：定位扰动影响通过哪条计算路径传播。
    - 核心思路：对扰动后的状态 $\mathbf{h}'$，单独把某条路径上的中间产物（attention pattern 或 LayerNorm 输出）替换为干净版本，看下游 loss 恢复多少。如果某条路径修复后恢复率高 → 该路径承载了主要扰动效应。具体 attention 修复 = 用未扰动前向的 attention weights 重放；LayerNorm 修复 = 用未扰动的 LN 输出替换扰动后版本。
    - 设计动机：相关性观察只能告诉你 "方向重要"，但要建立 "方向通过 attention 影响 loss" 这种机制陈述，必须做因果干预实验，靠修复某路径后效应是否消失来推断因果路径。

### 损失函数 / 训练策略
本文无训练，是纯推理时干预实验。Pythia-410M / 1.4B float32 精度跑前向，每个 $\delta$ 跨 5 个 seed 独立采样正交方向，统计置信度。

## 实验关键数据

### 主实验
Loss 损害（Table 1，baseline loss = 4.107）：

| $\delta$ | 幅值 $\Delta$loss | 角度 $\Delta$loss | 角/幅 比 | p |
|----------|-------------------|-------------------|----------|-----|
| 1.0 | 0.009 | 0.368 | **42.9×** | <0.001 |
| 2.0 | 0.042 | 0.983 | 23.2× | <0.001 |
| 5.0 | 0.700 | 3.757 | 5.4× | <0.001 |
| 10.0 | 3.262 | 7.061 | 2.2× | <0.001 |
| 20.0 | 5.433 | 7.750 | 1.4× | <0.001 |

句法准确率（Table 2，baseline 89.5%）：

| $\delta$ | 幅值后准确率 | 角度后准确率 | 幅值掉点 | 角度掉点 |
|----------|--------------|--------------|----------|----------|
| 5.0 | 69.1% | 87.9% | **20.4%** | 1.6% |
| 10.0 | 56.0% | 77.1% | 33.5% | 12.4% |
| 15.0 | 53.5% | 67.4% | 36.0% | 22.1% |

在 $\delta = 5$，loss 差 5.4 倍方向占优、句法差 12.8 倍幅值占优——两个矛盾方向的优势构成了双重分离。

### 消融实验
Pathway 修复（恢复占总损害的比例）：

| 修复路径 | 角度扰动恢复 | 幅值扰动恢复 | 偏向 |
|----------|--------------|--------------|------|
| Attention | **28.4%** | 15.2% | 角度→attention |
| LayerNorm | 13.7% | **29.9%** | 幅值→LayerNorm |

这模式在 Pythia-1.4B 上复现（角度/幅值比从 410M 的 23.2× 涨到 56.8×）。RMSNorm 架构（无 affine LN）上分离消失，说明这个现象与 LayerNorm 的 norm 操作机制紧密耦合。

层间传播（Table 4，$\delta = 5$）：

| Layer | 角度位移 L2 | 幅值位移 L2 | 比例 |
|-------|-------------|-------------|------|
| 8（干预起点）| 5.00 | 5.00 | 1.00× |
| 15（干预终点）| 35.9 | 12.7 | 2.82× |
| 23（最末）| 123.8 | 38.9 | 3.18× |

角度扰动放大 24.8 倍，幅值仅 7.8 倍——LayerNorm 对幅值有自然抑制，对方向放任自流。

### 关键发现
- **方向通过 attention 通道作用**：因为注意力本质是 $\text{softmax}(QK^T / \sqrt{d})$，依赖余弦相似度，方向扰动会直接改变 routing；LayerNorm 把 norm 重新归一化所以幅值变化部分被吸收。
- **句法是 norm 敏感任务**：主谓一致这类需要精细数值比较的决策，更依赖 norm 调控 "处理强度"，而非 attention routing 哪个 token 上。
- **小 $\delta$ 极端不对称、大 $\delta$ 趋于饱和**：低 $\delta$ 区角度优势 6.80×，高 $\delta$ 区降到 1.69×，因为模型预测有 "下限"——再扰动也只能差到 random 水平。
- **架构依赖**：把同样实验放到 RMSNorm 架构上分离不再出现，说明这是 LayerNorm 特有的几何分工，不是 transformer 通解。

## 亮点与洞察
- **L2 匹配 = 干净的实验设计**：能在概念清晰的同时保持数学简洁，是这种几何因果研究的范式级贡献，应该会被后续工作大量沿用。
- **借鉴 cognitive neuroscience 的术语**：把 "double dissociation" 这种成熟的因果推断框架引入可解释性，使结论更有论证强度，远胜单方向的 ablation。
- **机制定位 + 架构反例**：先建立现象（双重分离），再定位机制（attention / LN 路径），再用 RMSNorm 验证依赖性——这种 "现象 → 机制 → 边界条件" 的论证链非常严密。
- **对 representation engineering 的隐含警告**：方向编辑 (steering vectors) 和幅值缩放 (activation scaling) 不可互换，二者对应不同子能力。

## 局限与展望
- **pathway 只解释 ~30%**：attention / LN 修复加起来不到一半的损害，剩余 70% 走了什么路径还是黑箱，论文自认 "mechanistic picture 仍待完善"。
- **仅 5 个 seed**：作者承认统计 power 有限，效应虽大但样本少。
- **只测了主谓一致这一种句法**：BLiMP 还有大量其他句法现象（NPI 许可、岛限制等），是否同样幅值敏感未知。
- **干预层固定 8-15**：早层 / 末层是否同样分离没系统扫描，可能依赖处理阶段。
- **正交扰动方向随机**：但 representation space 各向异性（Ethayarajh 2019），"随机正交" 不一定等于 "语义中性"，部分扰动可能正好打到关键子空间。
- **未来推广**：建议测 RMSNorm + sandwich norm + 不同位置编码组合，看分离现象与具体 norm 形式的精细对应。

## 相关工作与启发
- **vs Park et al. 2023 (LRH 形式化)**: 那篇定义并实证 LRH 的方向编码假设；本文是对 LRH 的精细化扩展——加入幅值这个被忽略的维度。
- **vs Kobayashi et al. 2020 (注意力中的 norm)**: 他们发现 norm 调制注意力权重；本文用因果干预证明 norm 对句法功能特别重要。
- **vs Meng et al. 2022 (activation patching ROME)**: 同属因果干预家族，但 ROME patch 整个激活，本文 decompose 成方向 / 幅值再 patch，颗粒度更细。
- **启发**：模型编辑实践层面应区分用方向 steering（影响 attention routing 行为）和用幅值 scaling（影响处理强度），不应混用；安全研究可以测试 "jailbreak prompts 主要扰动方向还是幅值" 这种新维度。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ L2 匹配扰动协议 + double dissociation 跨学科借用，是可解释性领域少见的方法论级贡献。
- 实验充分度: ⭐⭐⭐⭐ 两个 Pythia 模型 + 双任务 + 双 pathway 修复 + RMSNorm 反例验证；扣分在 seed 数仅 5、模型规模较小。
- 写作质量: ⭐⭐⭐⭐⭐ 论证链 "现象 → 机制 → 边界" 工整，公式推导清晰，反例和置信区间充分讨论，可读性极高。
- 价值: ⭐⭐⭐⭐ 对 LRH 的精细化扩展和对 representation engineering 的实践指导都很重要，但实用门槛较高（需要因果干预设施）。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Similarity-Distance-Magnitude Activations](../../ACL2026/interpretability/similarity-distance-magnitude_activations.md)
- [\[AAAI 2026\] Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models](../../AAAI2026/interpretability/probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)
- [\[ACL 2026\] Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining](../../ACL2026/interpretability/crosscoding_through_time_tracking_emergence_consolidation_of_linguistic_represen.md)
- [\[ICML 2026\] Understanding LoRA as Knowledge Memory: An Empirical Analysis](understanding_lora_as_knowledge_memory_an_empirical_analysis.md)
- [\[CVPR 2026\] Beyond Semantics: Disentangling Information Scope in Sparse Autoencoders for CLIP](../../CVPR2026/interpretability/beyond_semantics_disentangling_information_scope_in_sparse_autoencoders_for_clip.md)

</div>

<!-- RELATED:END -->
