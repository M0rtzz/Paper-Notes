---
title: >-
  [论文解读] VEDA: Scalable Video Diffusion via Distilled Sparse Attention
description: >-
  [ICML 2026][视频生成][稀疏注意力] VEDA 把视频 DiT 的稀疏注意力问题重新表述为"对全注意力结构的显式蒸馏"——通过统计感知的瓦片评分 + 头感知分组搜索 + 硬件高效内核，在 90-95% 极端稀疏度下保持生成质量…
tags:
  - "ICML 2026"
  - "视频生成"
  - "稀疏注意力"
  - "Transformer"
  - "蒸馏学习"
  - "硬件优化"
---

# VEDA: Scalable Video Diffusion via Distilled Sparse Attention

**会议**: ICML 2026  
**arXiv**: [2605.30325](https://arxiv.org/abs/2605.30325)  
**代码**: 待确认  
**领域**: 视频生成 / 扩散模型 / 模型加速  
**关键词**: 稀疏注意力, 视频扩散 Transformer, 蒸馏学习, 硬件优化

## 一句话总结
VEDA 把视频 DiT 的稀疏注意力问题重新表述为"对全注意力结构的显式蒸馏"——通过统计感知的瓦片评分 + 头感知分组搜索 + 硬件高效内核，在 90-95% 极端稀疏度下保持生成质量，给 Waver-12B 720P 10 秒视频带来 5.1× 端到端加速、10.5× 注意力加速。

## 研究背景与动机

**领域现状**：视频扩散 Transformer（DiT）已是高保真视频合成主流，但自注意力 $O(N^2)$ 计算瓶颈在高分辨率长时序生成时极严重。

**现有痛点**：现有稀疏注意力方法在高度剪枝（≥ 90%）下有两个根本问题：
- **静态方法**（SVG、STA）依赖预定义时空掩膜，缺对头部特异性注意力几何的自适应性。
- **动态方法**（VSA、VMOBA）通过隐式学习，缺显式监督；使用均值池化等粗糙统计量会忽略关键的信号峰值。

**核心矛盾**：高度稀疏剪枝导致"水纹畸变 / 空间翘曲 / 时间闪烁"等结构性伪影。但实验发现这**不是稀疏比例本身造成的**，而是稀疏掩膜与全注意力的瓦片级结构对齐度不足导致。

**本文目标**：在保持生成质量前提下实现视频 DiT 的激进稀疏化与实际加速。

**切入角度**：关键观察——"神谕级"掩膜（从全注意力 Top-k 得到）即便在 90% 稀疏度下也能保持高质量。这启发显式监督瓦片选择目标，而非依赖扩散目标的隐式学习。

**核心 idea**：把稀疏瓦片选择重新表述为对全注意力结构的显式蒸馏，加上头感知分组应对头部异质性，结合硬件高效内核实现真实加速。

## 方法详解

### 整体框架
VEDA 三个核心模块：
- **蒸馏瓦片评分**：用轻量级估计器学重建全注意力的瓦片级评分，把 token 级密集注意力映射为稀疏瓦片掩膜。
- **头感知分组搜索**：为每个注意力头搜索最优瓦片分组 $(p_t, p_h, p_w)$。
- **硬件高效内核**：通过 ThunderKittens DSL 和 NVIDIA Hopper TMA 实现瓦片跳过注意力内核，达 FlashAttention-3 80% 运算效率。

### 关键设计

1. **统计感知的瓦片评分估计器（TripPool）**:

    - 功能：通过压缩的瓦片表示重建全注意力的瓦片级评分，用以生成稀疏掩膜。
    - 核心思路：对每个查询 / 键瓦片构造 **TripPool 描述子**——均值 / 最大值 / 最小值的连接 $\text{TripPool}[\cdot] = \text{Avg}[\cdot] \oplus \text{Max}[\cdot] \oplus \text{Min}[\cdot]$。再通过头特异性 MLP 投影 $\phi_q, \phi_k$ 映射至共享潜在空间，计算预测评分 $S_{ij}^{\text{pred}} = \frac{\phi_q(\text{TripPool}[\tilde{Q}_i]) \cdot \phi_k(\text{TripPool}[\tilde{K}_j])^\top}{\sqrt{d'}}$。最后用 KL 散度损失 $\mathcal{L}_{\text{distill}} = \mathcal{D}_{KL}(A^{\text{tgt}} \| A^{\text{pred}})$ 对齐预测与全注意力。
    - 设计动机：相比平均池化忽略信号峰值，TripPool 的最大 / 最小统计保留关键依赖；显式蒸馏目标避免隐式学习的漂移；**关键的停梯度操作**解耦掩膜学习与特征学习，防止扰动预训练生成流形。

2. **头感知分组搜索**:

    - 功能：为每一层每个注意力头离线搜索最优时空瓦片分组配置。
    - 核心思路：把瓦片配置限制在硬件瓦片大小 $B$ 的因子分解 $\Omega = \{(p_t, p_h, p_w) \in \mathbb{N}^3 \mid p_t p_h p_w = B\}$。对每个候选 $\pi$，在校准集上最小化全注意力输出的稀疏近似误差 $\pi^*_{l, h} = \arg\min_{\pi \in \Omega} \mathbb{E}_{x \sim \mathcal{D}_{\text{cal}}} \|O^{\text{fu}}_{l, h}(x) - O^{\text{sp}}_{l, h}(x; \pi)\|_F^2$。
    - 设计动机：注意力头在空间 / 时间依赖上存在显著异质性；统一分组在高稀疏度下导致瓦片回忆率下降；针对性配置能保留不同头的关键信息。

3. **瓦片跳过硬件内核 + 两阶段训练**:

    - 功能：把稀疏掩膜高效执行在 GPU 上；通过稳定的两阶段训练避免收敛问题。
    - 核心思路：第一阶段冻结骨干只训投影器 1000 步对齐稀疏预测；第二阶段解冻所有参数在目标稀疏度下微调。利用异步 TMA + Warp 特化：生产者 warp 从全局内存非连续抓取选定的键 / 值瓦片到共享内存，消费者 warp 同时执行张量核心运算，达约 80% FlashAttention-3 效率。
    - 设计动机：两阶段解耦避免梯度反传破坏预训练流形；硬件优化确保算法稀疏性转化为实际端到端加速而非内核开销。

### 训练目标
$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diff}} + \lambda \mathcal{L}_{\text{distill}}$，KL 散度行级蒸馏 + 标准扩散去噪。**停梯度**确保骨干特征不接收掩膜估计器的梯度反传——实验证明允许梯度反传会导致显著生成质量下降。

## 实验关键数据

### 主实验（Waver-1B 与 Wan2.1-1.3B 上对比全注意力与 VSA）

| 模型 | 方法 | 稀疏度 | 主体一致性 | 背景一致性 | 运动平滑 | 美学质量 | 端到端时间 |
|------|------|--------|---------|---------|--------|--------|----------|
| Waver-1B | 全注意力 | 0% | 0.938 | 0.955 | 0.979 | 0.693 | 69.3s |
| Waver-1B | VSA | 87.5% | 0.933 | 0.949 | 0.978 | 0.692 | 34.3s |
| Waver-1B | **VEDA** | **90%** | **0.940** | **0.954** | **0.980** | **0.699** | **31.9s** |
| Waver-1B | **VEDA** | **95%** | 0.934 | 0.951 | 0.978 | 0.698 | **30.6s** |
| Wan2.1-1.3B | 全注意力 | 0% | 0.940 | 0.969 | 0.977 | 0.670 | 58.5s |
| Wan2.1-1.3B | **VEDA** | **90%** | 0.887 | 0.941 | 0.972 | 0.663 | **37.6s** |

### 消融实验

| 组件 | 配置 | 指标 ↓ | 说明 |
|------|------|------|------|
| 瓦片统计 | 平均池化 | 0.965 | 忽略峰值 |
| 瓦片统计 | 最大 / 最小 | 0.982 | 遗漏中等重要性 |
| 瓦片统计 | **TripPool** | **0.912** | 保留关键依赖 |
| 分组策略 | 静态 [8, 8, 2] | +3.2% 运动质量损失 | 偏空间 |
| 分组策略 | 静态 [4, 4, 8] | 基准 | 均衡配置 |
| 分组策略 | **头感知动态** | +7.2% 运动 / +9.6% 总体 | 适应头部异质性 |

### 关键发现
- **掩膜精度主导性能**：90% 固定稀疏度下"神谕"掩膜的生成质量远优于平均池化掩膜——问题根源不在稀疏比例而在对齐质量。
- **头部异质性显著**：不同层不同头的空间 / 时间依赖模式差异大，统一分组在高稀疏度下不行。
- **可扩展性**：Waver-12B 720P 10 秒视频生成实现 5.1× 端到端加速 + 10.5× 注意力加速，注意力开销从 92% 降到 50%；序列越长 VEDA 加速越大。

## 亮点与洞察
- **实验性的根本观察**："神谕掩膜"实验精准定位真正瓶颈是结构对齐度而非稀疏比例，推翻既往假设并奠定方法设计基础。
- **显式监督的范式转变**：相比让扩散目标隐式形塑稀疏结构，显式蒸馏直接监督瓦片评分，避免隐式学习的漂移；停梯度操作的设计巧妙保护预训练生成流形。
- **头感知分组的精细化设计**：识别头部异质性并针对性搜索时空分组配置，比同期 VSA 等静态 / 全局动态方法更细粒度，可迁移到其他多头 Transformer 加速任务。
- **硬件-算法协设计**：从 TMA 异步传输到 Warp 特化的完整内核实现，把 FLOPs 理论减少转化为真实端到端加速，工程闭环完整。

## 局限与展望
- 两阶段训练虽稳定但需手工设计学习率 / 步数，通用性待提升。
- 95%+ 稀疏度下仍需更多 kernel 融合以提升 MFU。
- 头感知分组依赖离线校准集，不同数据分布下可能需重新搜索。
- TripPool 对异常分布的鲁棒性未充分讨论（最大 / 最小值易被离群值影响）。

## 相关工作与启发
- **vs SVG / STA**（静态稀疏）：依赖预定义模式缺自适应性；本文通过显式蒸馏实现内容与头部敏感的动态选择。
- **vs VSA / VMOBA**（动态稀疏）：依赖隐式扩散目标 + 粗糙池化；本文显式蒸馏 + 精细统计量更准确捕捉全注意力结构。
- **vs 其他加速**（缓存复用 PAB / TeaCache、蒸馏 CausVid）：VEDA 与它们正交，可叠加使用。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐  在视频 DiT 稀疏化上首次系统引入显式监督 + 头感知分组；"掩膜精度主导" 的实验性发现改变了对稀疏注意力瓶颈的理解。
- 实验充分度: ⭐⭐⭐⭐⭐  多模型规模（1B / 12B）、多分辨率（480P / 720P）、长序列（34K-245K）、人类评估 + VBench、消融细致。
- 写作质量: ⭐⭐⭐⭐⭐  逻辑清晰层层递进，实验驱动的发现说服力强，方法各模块独立贡献明确。
- 价值: ⭐⭐⭐⭐⭐  5.1× 加速对工业应用意义重大；稀疏注意力设计思路对 LLM 加速也有参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention](light_forcing_accelerating_autoregressive_video_diffusion_via_sparse_attention.md)
- [\[ICML 2026\] DFSAttn: Dynamic Fine-Grained Sparse Attention for Efficient Video Generation](dfsattn_dynamic_fine-grained_sparse_attention_for_efficient_video_generation.md)
- [\[NeurIPS 2025\] VSA: Faster Video Diffusion with Trainable Sparse Attention](../../NeurIPS2025/video_generation/vsa_faster_video_diffusion_with_trainable_sparse_attention.md)
- [\[ICML 2026\] Attention Sparsity is Input-Stable: Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering](attention_sparsity_is_input-stable_training-free_sparse_attention_for_video_gene.md)
- [\[ICML 2026\] Lightning Unified Video Editing via In-Context Sparse Attention](lightning_unified_video_editing_via_in-context_sparse_attention.md)

</div>

<!-- RELATED:END -->
