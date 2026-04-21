---
title: >-
  [论文解读] SAGE: Spatial-visual Adaptive Graph Exploration for Efficient Visual Place Recognition
description: >-
  [ICLR 2026][目标检测][Visual Place Recognition] 提出 SAGE，一个统一的 VPR 训练框架：引入轻量 Soft Probing 模块增强局部特征判别力，每个 epoch 在线重建融合地理距离与视觉相似度的亲和图，再通过贪心加权团扩展聚焦最难样本，冻结 DINOv2 骨干仅训练 1.96M 参数即在 8 个基准上全面 SOTA。
tags:
  - ICLR 2026
  - 目标检测
  - Visual Place Recognition
  - DINOv2
  - Graph-based Sampling
  - Hard Sample Mining
  - parameter-efficient fine-tuning
---

# SAGE: Spatial-visual Adaptive Graph Exploration for Efficient Visual Place Recognition

**会议**: ICLR 2026  
**arXiv**: [2509.25723](https://arxiv.org/abs/2509.25723)  
**代码**: https://github.com/chenshunpeng/SAGE  
**领域**: 目标检测/视觉定位  
**关键词**: Visual Place Recognition, DINOv2, Graph-based Sampling, Hard Sample Mining, parameter-efficient fine-tuning

## 一句话总结

提出 SAGE，一个统一的 VPR 训练框架：引入轻量 Soft Probing 模块增强局部特征判别力，每个 epoch 在线重建融合地理距离与视觉相似度的亲和图，再通过贪心加权团扩展聚焦最难样本，冻结 DINOv2 骨干仅训练 1.96M 参数即在 8 个基准上全面 SOTA。

## 背景与动机

1. **视觉地点识别（VPR）核心挑战**：将查询图像匹配到大规模地理标注数据库中的正确位置，需在极端视角变化、光照变化、天气/季节漂移、动态遮挡等条件下保持鲁棒检索。
2. **静态采样的瓶颈**：现有方法（SALAD-CM、Cliquemining 等）使用"一次计算、全程使用"的离线策略——基于初始特征预聚类，训练全程不更新。随着嵌入空间演化，旧的"难样本"变成简单样本，新的决策边界难样本未被挖掘，导致学习效率下降。
3. **地理与视觉信息割裂**：多数方法独立使用地理邻近或视觉相似来构造训练批次，忽略了两者的动态交互——真正困难的样本由"地理近但视觉差异大"的耦合关系决定。
4. **局部特征均匀处理**：CFP（Centroid-Free Probing）等聚合方法将所有 patch 特征等权融合为全局描述符，无法突出微妙但具判别力的局部线索。
5. **参数效率需求**：全量微调 VFM 骨干参数量大、计算成本高，实际部署需要参数高效的适配方案。

## 方法详解

### 整体架构

SAGE 包含四个阶段：(1) 冻结 DINOv2 骨干提取 patch token；(2) Soft Probing 自适应增强局部特征后聚合为全局描述符；(3) InteractHead 建模跨图像关联并在线重建地理-视觉亲和图；(4) 贪心加权团扩展采样构造训练批次。

### 特征提取与 PEFT

使用预训练 DINOv2（ViT-B 或 ViT-L）作为冻结骨干，在最后 $N$ 个编码器块中插入可学习的 Dynamic Power Normalization (DPN) 层实现参数高效微调。输出一个 class token 和 $L$ 个 patch token，拼接为 $\mathbf{f} \in \mathbb{R}^{(L+1) \times M}$。

### Soft Probing（SoftP）

**动机**：CFP 对所有 patch 描述符等权聚合，埋没了微妙但关键的判别线索。SoftP 通过数据驱动的残差加权放大显著 patch。

对每个 patch 描述符 $X_i$：

1. 计算 $\ell_2$ 响应 $s_i = \|X_i\|_2 + \varepsilon$
2. 经两层 MLP $\phi$ 预测标量，再 sigmoid 压缩并缩放：$\beta_i = \alpha \cdot \sigma(\phi(s_i))$，范围 $[0, \alpha]$
3. 残差调制：$\widetilde{X}_i = (1 + \beta_i) X_i$

这种残差形式保留了原始特征的通道结构，同时选择性放大高响应位置的方差贡献。调制后的描述符集合 $\{\widetilde{X}_i\}$ 经 Feature Compression 和 Feature Probing 两个 MLP 分支降维至 $D=128$、$K=64$，最终聚合为全局描述符。SoftP 仅增加极少参数。

### InteractHead

将每个图像描述符 $\mathbf{f}_i \in \mathbb{R}^{D \times K}$ 确定性地切分为 $S$ 个等长片段，跨 batch 重排后送入两层 Transformer 编码器（768 维，16 头注意力，FFN 1024 维），在每个片段类型上跨图像做注意力，捕获视图间一致性关联，增强描述符鲁棒性。

### 在线地理-视觉图构建（Online Graph Creation）

每个 epoch 执行以下流程：

1. **采样代表特征**：对每个城市，按唯一聚类标签分组，随机采样一张图像通过模型获取聚类级描述符
2. **构建候选集**：按聚类数按比例采样城市，选一个 place，根据描述符余弦距离概率采样 $P=15$ 个相似 place
3. **地理图过滤**：计算所有节点对的欧式地理距离 $d_{\text{geo}}(i,j)$，低于阈值 $\tau_1=25$m 的连边
4. **提取团**：从地理图中提取完全子图（clique），取首个满足 $|V_C| \geq 10$ 的团
5. **计算乘性亲和度**：$W_{ij} = -(d_{\text{geo}}(i,j) \cdot d_{\text{vis}}(i,j))$，其中 $d_{\text{vis}} = \|\mathbf{F}_i - \mathbf{F}_j\|_2$
6. **构建稀疏亲和图**：仅保留 $W_{ij}$ 超过阈值 $\tau_2$ 的边

这种乘性距离设计使得"地理近 & 视觉相似"的节点对获得最高亲和度，精确捕获最易混淆的地点。由于每 epoch 重建，图始终与当前嵌入空间对齐。

### 贪心加权团扩展（Greedy Weighted Sampling）

1. **选锚点**：计算每个节点的种子得分 $S(i) = \frac{1}{N-1}\sum_{j \neq i} W_{ij}$，选得分最高者为初始团成员
2. **贪心扩展**：迭代选择与当前团成员平均亲和度最高的节点加入，直至团大小达到 $k=4$
3. **效果**：自动钻入亲和图中最稠密的子图区域——即模型最难做出细粒度区分的互相混淆样本簇

## 实验结果

### 主实验：8 个基准全面对比（Table 2 & 3）

| 方法 | 维度 | SPED R@1 | Pitts30k R@1 | MSLS-val R@1 | Nordland R@1 | AmsterTime R@1 | Tokyo24/7 R@1 |
|------|------|----------|-------------|-------------|-------------|---------------|-------------|
| CosPlace | 512 | 75.5 | 88.4 | 82.8 | 58.5 | - | - |
| MixVPR | 4096 | 84.7 | 91.5 | 88.0 | 76.2 | - | - |
| BoQ | 12288 | 92.5 | 93.7 | 93.8 | 90.6 | - | - |
| EMVP | 8448 | 94.6 | 94.0 | 93.9 | 88.7 | 65.6 | 96.8 |
| FoL | 8448 | 92.1 | 93.9 | 93.1 | 87.8 | 64.6 | 96.2 |
| **SAGE** | **8448** | **98.9** | **95.8** | **94.5** | **96.0** | **83.5** | **97.5** |

SAGE (8448-d) 在 SPED 上 R@10 达 100%，R@1 98.9%（超 EMVP +4.3pt）；Nordland 上 R@1 96.0%（超 EMVP +7.3pt）；AmsterTime 上 R@1 83.5%（超 EMVP +17.9pt），在跨时代历史图像检索场景优势巨大。

### 紧凑描述符也强劲

即使 PCA 降至 4096 维，SAGE 仍在 SPED 上达 97.7% R@1、R@10 100%；在 Pitts250k 上 R@1 98.2%，超越多数 8448 维方法。

### 参数效率（Table 4）

| 方法 | 总参数 | 可训练参数 | 需要 Adapter |
|------|--------|-----------|-------------|
| SALAD | 88.0M | 29.8M | 否 |
| SelaVPR | 102.8M | 16.2M | 是(14.2M) |
| CricaVPR | 95.7M | 9.15M | 是(9.2M) |
| EMVP | 88.5M | **1.96M** | 否 |
| **SAGE** | 88.5M(+7.88M) | **1.96M**(+7.88M) | 否 |

SAGE 骨干可训练参数与 EMVP 相同（1.96M），仅额外增加 7.88M 的 InteractHead（仅训练时需要，推理时可选用）。相比 SALAD 的 29.8M 或 SelaVPR 的 16.2M 可训练参数，效率优势明显。

### 消融实验（Table 5）

| 配置 | SPED R@1 | Pitts30k R@1 | MSLS-val R@1 | Nordland R@1 |
|------|----------|-------------|-------------|-------------|
| EMVP-B (CFP, 基线) | 91.8 | 93.1 | 93.2 | 80.8 |
| +SoftP+OGC | 96.8 | 94.6 | 93.6 | 95.2 |
| +SoftP+GWS（无 OGC） | 96.5 | 93.8 | 92.5 | 94.2 |
| +CFP+OGC+GWS | 97.5 | 94.9 | 93.9 | 95.4 |
| **+SoftP+OGC+GWS（完整 SAGE）** | **98.0** | **95.4** | **94.3** | **95.8** |

- SoftP vs CFP：相同图采样下 SoftP 提升 SPED R@1 约 0.5pt，Pitts30k 约 0.5pt
- OGC 贡献最大：在 Nordland 从 80.8% 跃升至 95.2%（+14.4pt），说明动态图重建对季节变化场景至关重要
- GWS 需与 OGC 配合：单独 GWS（无 OGC）性能不稳定，但 OGC+GWS 组合产生协同效果

### 在线 vs 离线图构建（Table 6）

| 策略 | 每 epoch 挖掘耗时 | SPED R@1 | MSLS-val R@1 |
|------|-----------------|----------|-------------|
| 离线 SAGE | 30.9 min（一次性） | 98.5 | 94.2 |
| **在线 SAGE** | **6.2 min** | **98.9** | **94.5** |

在线策略每 epoch 重建仅增 17.7% 训练时间，但 SPED R@1 提升 0.4pt、MSLS-val 提升 0.3pt，证明动态适应嵌入演化是值得的。推理阶段无额外开销。

### 收敛性分析

在 MSLS-val 上第 4 个 epoch，SAGE 即达到 93.4% R@1 vs Cliquemining 的 92.7%，早期训练阶段优势持续扩大，说明动态采样加速了有效学习。

## 亮点

- **"慢思考"范式**：跳出"一次挖掘、全程使用"的静态框架，每 epoch 重建图，让难样本定义随模型演化而动态调整
- **地理×视觉乘性耦合**：$W_{ij} = -(d_{\text{geo}} \cdot d_{\text{vis}})$ 精准捕获"地理近但视觉易混"的真正难样本
- **SoftP 残差加权**：极轻量（两层 MLP），通过 $\ell_2$ 响应驱动的残差缩放显著优于等权 CFP
- **参数极致高效**：冻结 DINOv2，骨干可训练参数仅 1.96M，在 8 个基准上超越需要 16-30M 可训练参数的方法
- **AmsterTime 惊人提升**：历史灰度 vs 当代彩色图像检索 R@1 从 65.6%→83.5%（+17.9pt），说明动态采样+特征增强对跨时代场景尤为有效

## 局限性

- InteractHead 训练时增加 7.88M 参数和跨图像注意力计算，大规模训练时可能成为瓶颈
- 在线图重建虽每 epoch 仅 6 分钟，但依赖地理标注信息，对无 GPS 数据的场景不适用
- 贪心团扩展的团大小 $k=4$ 是手动设定的超参数，不同数据集是否需要不同设置未充分讨论
- 仅使用 GSV-Cities + MSLS 训练，未探索更多训练数据的增益
- 推理阶段与标准单阶段方法相同，InteractHead 的跨图像注意力在推理时未使用，这部分训练开销的性价比有待分析

## 相关工作

- **全局描述符聚合**：NetVLAD（可学习 VLAD）→ MixVPR（特征混合）→ CFP/EMVP（无中心探测+二阶统计量）→ SoftP（残差加权探测）
- **训练采样策略**：静态难样本挖掘 → Cliquemining（离线图采样）→ SALAD-CM（离线聚类）→ SAGE（在线动态图+贪心团扩展）
- **参数高效微调**：Adapter（SelaVPR）→ 部分编码器微调（SALAD）→ DPN（EMVP/SAGE，冻结骨干+轻量归一化层）
- **跨图像关联**：CricaVPR / EMVP 的跨图像注意力 → SAGE InteractHead（确定性分段+Transformer 编码）

## 评分

- ⭐⭐⭐⭐ 新颖性：动态地理-视觉图+贪心团扩展是新颖的采样范式，SoftP 简洁有效
- ⭐⭐⭐⭐⭐ 实验充分度：8 个基准全面 SOTA，消融覆盖每个模块，含在线/离线对比和收敛分析
- ⭐⭐⭐⭐ 实用性：冻结骨干+极少可训练参数，推理无额外开销，代码开源
- ⭐⭐⭐⭐ 写作清晰度：图示直观，动机论述清楚，数学推导简洁

<!-- RELATED:START -->

## 相关论文

- [Evaluating Few-Shot Pill Recognition Under Visual Domain Shift](../../CVPR2026/object_detection/evaluating_fewshot_pill_recognition_under_visual_d.md)
- [Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Method](traceable_evidence_enhanced_visual_grounded_reasoning_evaluation_and_methodology.md)
- [Visual-RFT: Visual Reinforcement Fine-Tuning](../../ICCV2025/object_detection/visual-rft_visual_reinforcement_fine-tuning.md)
- [CineSRD: Leveraging Visual, Acoustic, and Linguistic Cues for Open-World Visual Media Speaker Diarization](../../CVPR2026/object_detection/cinesrd_leveraging_visual_acoustic_and_linguistic_cues_for_open-world_visual_med.md)
- [Zero-shot HOI Detection with MLLM-based Detector-agnostic Interaction Recognition](zero-shot_hoi_detection_with_mllm-based_detector-agnostic_interaction_recognitio.md)

<!-- RELATED:END -->
