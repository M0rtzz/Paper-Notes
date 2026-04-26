---
title: >-
  [论文解读] MeshAnything V2: Artist-Created Mesh Generation with Adjacent Mesh Tokenization
description: >-
  [ICCV 2025][3D视觉][网格生成] MeshAnything V2 提出 Adjacent Mesh Tokenization (AMT)，通过用单个顶点（而非传统三个顶点）表示相邻面，将网格的 token 序列长度平均缩短一半，从而在不增加计算成本的前提下将最大生成面数从 800 提升到 1600，显著提高了自回归网格生成的效率和质量。
tags:
  - ICCV 2025
  - 3D视觉
  - 网格生成
  - 序列化
  - tokenization
  - 自回归
  - Artist-Created Mesh
---

# MeshAnything V2: Artist-Created Mesh Generation with Adjacent Mesh Tokenization

**会议**: ICCV 2025  
**arXiv**: [2408.02555](https://arxiv.org/abs/2408.02555)  
**代码**: https://buaacyw.github.io/meshanything-v2/  
**领域**: 3D视觉 / 网格生成  
**关键词**: 网格生成, 序列化, tokenization, 自回归, Artist-Created Mesh

## 一句话总结
MeshAnything V2 提出 Adjacent Mesh Tokenization (AMT)，通过用单个顶点（而非传统三个顶点）表示相邻面，将网格的 token 序列长度平均缩短一半，从而在不增加计算成本的前提下将最大生成面数从 800 提升到 1600，显著提高了自回归网格生成的效率和质量。

## 研究背景与动机

**领域现状**：自回归网格生成是近期热门方向，将三维网格视为面序列，用类似 LLM 的 Transformer 逐顶点生成。代表方法包括 PolyGen、MeshGPT、MeshXL、MeshAnything 等。这些方法从人类艺术家创建的网格分布中学习，生成高效、美观、可直接用于工业的 Artist-Created Mesh (AM)。

**现有痛点**：现有方法无法生成大量面数的复杂网格。根本原因在于 **tokenization 效率低下**：每个面用三个顶点表示，每个顶点需要三个 token（x/y/z 坐标），导致 token 序列长度为面数的 9 倍。这带来巨大的计算和内存开销，同时序列中存在大量冗余信息，损害序列学习效果。

**核心矛盾**：网格是图结构数据，存在无数种序列化方式。好的 tokenization 需要同时平衡两个目标：(a) **紧凑性**——更短的序列意味着更低的计算复杂度；(b) **规律性**——有规律的序列更易于 Transformer 学习。

**本文目标** 设计更高效的网格 tokenization 方法，在缩短序列的同时保持序列的规律性和学习友好性。

**切入角度**：NLP 领域已充分证明 tokenization 对序列学习的重要性（如 BPE vs WordPiece）。对于网格这种图结构数据，tokenization 的影响更为深远。关键观察是：当前方法的冗余主要来自每个面重复表示已访问过的顶点——如果相邻面共享边，只需一个新顶点即可表示新面。

**核心 idea**：相邻面共享两个顶点，因此只需一个新顶点即可表示下一个相邻面，当找不到相邻面时插入特殊 token "&" 重启。

## 方法详解

### 整体框架
输入为点云形状条件 $\mathcal{S}$（8192 个点），经预训练点云编码器编码为 token 前缀 $\mathcal{T}_S$。网格 $\mathcal{M}$ 通过 AMT 编码为紧凑 token 序列 $\mathcal{T}_M$。两者拼接后用 OPT-350M decoder-only Transformer 以 cross-entropy 损失学习条件分布 $p(\mathcal{M}|\mathcal{S})$。推理时给定 $\mathcal{T}_S$ 自回归生成 $\mathcal{T}_M$，再反向解码为网格。

### 关键设计

1. **Adjacent Mesh Tokenization (AMT)**:

    - 功能：将网格编码为更紧凑的 token 序列
    - 核心思路：传统方法将每个面表示为三个有序顶点 $f_i = (v_{i1}, v_{i2}, v_{i3})$，序列长度为 $3N$（$N$ 为面数）。AMT 的策略是：
        - 第一个面仍用三个顶点 $(v_1, v_2, v_3)$ 表示
        - 后续面如果与前一个面共享一条边（相邻面），则只需附加一个新顶点
        - 如果找不到相邻面，插入特殊 token "&" 标记中断，然后从未编码面列表中取出下一个面重新用三个顶点开始
    - 理想情况下（"&" 很少使用），序列长度可缩短至 $N+2$（约为传统方法的 1/3）。Objaverse 实测平均缩短约 50%
    - 设计动机：消除面序列中的冗余顶点重复，同时利用网格的拓扑邻接关系使序列空间上连续

2. **Vertices Swap（顶点交换）**:

    - 功能：扩大可探索的相邻面集合
    - 核心思路：考虑 $f_1 = (v_1, v_2, v_3)$ 和 $f_2 = (v_1, v_3, v_4)$，它们通过边 $(v_1, v_3)$ 相邻。但 AMT 默认用"最后两个顶点"查找相邻面，即 $(v_2, v_3)$，找不到 $f_2$。引入特殊 token "\$" 表示交换：序列 $(v_1, v_2, v_3, \$, v_4)$ 表示下一个面由第一个和最后一个顶点（而非最后两个）组成
    - 设计动机：减少因边不匹配导致的中断次数，进一步压缩序列长度

3. **Face Count Condition（面数控制）**:

    - 功能：允许用户指定生成网格的目标面数
    - 核心思路：初始化面数嵌入书（大小=最大面数），根据目标面数检索对应嵌入放在点云前缀之后。训练时加随机扰动防止过拟合到精确面数，10% 概率随机 drop 该条件增强鲁棒性
    - 设计动机：不同应用对面数有不同需求（游戏需要低面数，影视需要高面数），此前方法无法控制

4. **Masking Invalid Predictions（无效预测遮蔽）**:

    - 功能：推理时约束生成只输出合法 token
    - 核心思路：在推理阶段遮蔽不合法的 logits，例如不允许 "&" 紧接在 "&" 之后（新 strip 至少需要三个顶点），不允许违反坐标排序顺序的顶点
    - 设计动机：Transformer 可能生成不符合序列结构的输出，显式约束确保解码结果是合法网格

5. **AMT 位置编码**:

    - 为不同类型的 token 添加不同的位置嵌入：三顶点新面的三个位置各有专属嵌入，单顶点相邻面有不同嵌入，"&" token 有独立嵌入
    - 帮助 Transformer 区分当前 token 在 AMT 序列中的角色

### 训练策略
使用 Objaverse 数据集，面数上限从 800 提升到 1600。采样 8192 点（vs V1 的 4096）以适应更复杂网格。整体 batch size 256（32 块 A800 × 8），训练 4 天。与 V1 不同，V2 在训练中更新点云编码器权重以提高精度。

## 实验关键数据

### AMT 消融（MeshAnything V2 vs 无 AMT 变体）

| 方法 | CD↓ | ECD↓ | NC↑ | #V | #F | V_Ratio | F_Ratio | S_Ratio↓ |
|------|-----|------|-----|----|----|---------|---------|---------|
| V2 w/o AMT | 0.895 | 4.832 | 0.924 | 302.4 | 556.7 | 1.105 | 1.062 | 1.000 |
| **V2 (AMT)** | **0.874** | **4.721** | **0.933** | 308.6 | 571.8 | 1.127 | 1.097 | **0.497** |

AMT 将序列长度压缩至 49.7%（约一半），同时 CD 降低 2.3%、NC 提升。无 AMT 变体训练消耗的 GPU 小时接近 V2 的两倍。

### Tokenization 方法对比（OPT-125M，≤400 面）

| 方法 | CD↓ | ECD↓ | NC↑ | S_Ratio↓ | Perplexity↓ |
|------|-----|------|-----|---------|-------------|
| Baseline (3顶点/面) | 2.478 | 18.21 | 0.893 | 1.000 | 1.150 |
| Unsort (无排序) | 8.151 | 31.86 | 0.794 | 1.000 | 1.234 |
| PolyGen(AMT) | 3.226 | 22.97 | 0.872 | **0.372** | 1.589 |
| **AMT** | **2.348** | **19.33** | **0.904** | 0.492 | **1.363** |
| AMT(Swap) | 2.517 | 19.86 | 0.913 | 0.455 | 1.416 |

### 关键发现
- **排序至关重要**：Unsort 的 CD 是 Baseline 的 3.3 倍（8.151 vs 2.478），证明规律性的序列对学习至关重要
- **PolyGen 压缩率最高但质量差**：S_Ratio=0.372 是最短的，但 Perplexity 最高（1.589），说明其序列结构不适合自回归学习。原因是模型生成面时需要引用之前的顶点索引，这增加了序列学习难度
- **AMT 是最佳平衡**：序列缩短 50% 的同时 CD 还优于 Baseline（2.348 vs 2.478），Perplexity 也更低（1.363 vs 1.150），证明 AMT 生成的序列更紧凑且更易学习
- **Swap 略有帮助**：进一步改善压缩率（0.455 vs 0.492）且 NC 最高（0.913），但额外 special token 使 Perplexity 略升
- **V2 实现了面数翻倍**：最大面数从 800 提升到 1600，这完全归功于 AMT 将序列长度减半

## 亮点与洞察
- **将 NLP tokenization 思想迁移到 3D 网格**是核心洞察：BPE 通过合并频繁子词缩短序列，AMT 通过利用拓扑邻接缩短面表示。这一跨领域类比非常有启发性
- **紧凑性 vs 规律性的权衡分析**非常有价值：PolyGen 式的索引方法虽然更短，但破坏了序列的可预测性模式。AMT 的优雅之处在于它同时改善了两者——更短且更规律
- **"&" 特殊 token 的设计**简单而高效：当拓扑不连续时优雅降级，而不是强行扭曲序列结构
- **Masking Invalid Predictions**是自回归生成结构化数据的通用技巧，可迁移到任何需要满足语法约束的生成任务

## 局限性
- 仅处理三角形网格，虽然论文提到可扩展到多边形但未验证
- 对于拓扑不连续的网格（如大量独立面片），AMT 可能退化（过多 "&" 中断）
- 面数上限 1600 对于高精度工业应用仍然不够（常需要数千甚至数万面）
- 点云编码器的表征能力可能限制了复杂形状的重建质量
- 未与最新的扩散式网格生成方法（如 PolyDiff）进行对比

## 相关工作与启发
- **vs MeshAnything V1**: V2 通过 AMT 将面数上限翻倍且质量更优，核心区别就是 tokenization 方法。证明了在模型架构不变的情况下，改进数据表示本身就能带来显著提升
- **vs MeshGPT**: MeshGPT 使用 VQ-VAE 学习网格词表，AMT 直接用坐标离散化。AMT 可以与 VQ-VAE 正交组合使用
- **vs PolyGen**: PolyGen 先生成顶点再通过索引生成面，压缩率更高但序列学习更困难。AMT 选择了更利于学习的设计
- **关键启发**：在自回归生成领域，数据的序列化方式（tokenization）可能与模型架构同等重要甚至更重要

## 评分
- 新颖性: ⭐⭐⭐⭐ AMT 思路直觉但有效，Swap 设计和 Masking 是有价值的补充
- 实验充分度: ⭐⭐⭐⭐ 系统对比了多种 tokenization 方法，但缺少与其他网格生成方法的直接对比
- 写作质量: ⭐⭐⭐⭐ 算法描述清晰，但 ablation 设置（OPT-125M/400面 vs OPT-350M/1600面）不一致导致可比性受限
- 价值: ⭐⭐⭐⭐⭐ AMT 是网格生成领域的基础性贡献，几乎所有自回归网格方法都可受益

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] VertexRegen: Mesh Generation with Continuous Level of Detail](vertexregen_mesh_generation_with_continuous_level_of_detail.md)
- [\[CVPR 2025\] Scaling Mesh Generation via Compressive Tokenization](../../CVPR2025/3d_vision/scaling_mesh_generation_via_compressive_tokenization.md)
- [\[ICCV 2025\] MeshPad: Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation and Editing](meshpad_interactive_sketch-conditioned_artist-reminiscent_mesh_generation_and_ed.md)
- [\[ICCV 2025\] DeepMesh: Auto-Regressive Artist-Mesh Creation with Reinforcement Learning](deepmesh_auto-regressive_artist-mesh_creation_with_reinforcement_learning.md)
- [\[ICCV 2025\] Nautilus: Locality-aware Autoencoder for Scalable Mesh Generation](nautilus_locality-aware_autoencoder_for_scalable_mesh_generation.md)

<!-- RELATED:END -->
