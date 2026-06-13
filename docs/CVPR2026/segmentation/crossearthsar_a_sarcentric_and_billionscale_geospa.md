---
title: >-
  [论文解读] CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation
description: >-
  [CVPR 2026][语义分割][SAR] 提出首个十亿参数级 SAR 视觉基础模型 CrossEarth-SAR，在 DINOv2 ViT backbone 上将 FFN 替换为物理引导的稀疏 MoE（用方向熵、等效视数、局部粗糙度三个 SAR 物理描述符引导路由选择），配套 200K 级跨域预训练数据集及覆盖 8 种域差异的 22 个基准，在 20/22 个跨域语义分割评测上达到 SOTA。
tags:
  - "CVPR 2026"
  - "语义分割"
  - "SAR"
  - "基础模型"
  - "物理引导MoE"
  - "域泛化"
---

# CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation

**会议**: CVPR 2026  
**arXiv**: [2603.12008](https://arxiv.org/abs/2603.12008)  
**代码**: [GitHub](https://github.com/VisionXLab/CrossEarth-SAR)  
**领域**: 遥感 / SAR 基础模型 / 域泛化语义分割  
**关键词**: SAR, 基础模型, 物理引导MoE, 域泛化, 语义分割  

## 一句话总结

提出首个十亿参数级 SAR 视觉基础模型 CrossEarth-SAR，在 DINOv2 ViT backbone 上将 FFN 替换为物理引导的稀疏 MoE（用方向熵、等效视数、局部粗糙度三个 SAR 物理描述符引导路由选择），配套 200K 级跨域预训练数据集及覆盖 8 种域差异的 22 个基准，在 20/22 个跨域语义分割评测上达到 SOTA。

## 研究背景与动机

- **核心矛盾**: SAR 成像具备全天候全天时能力，对地球观测不可替代，但其域特异性极端——不同传感器平台（Sentinel-1、ALOS-2、Capella）、波段（C/L/X）、极化模式（HH/HV/VV/VH）和入射角产生碎片化域，导致跨域泛化极难。
- **SAR 三重挑战**: (1) 相干成像产生乘性散斑噪声，破坏纹理特征；(2) 侧视几何引起叠掩（layover）、透视缩短（foreshortening）和阴影，扭曲空间拓扑；(3) 后向散射由表面粗糙度和介电常数决定，导致同类异貌（同一地物因含水量不同差异巨大）和异类同貌（不同地物呈现相同暗背景）。
- **现有方法瓶颈**: 现有 SAR 基础模型（SARATR-X 90M、SatMAE 300M）要么聚焦目标检测，要么非为跨域泛化设计；光学基础模型（DINOv2、DINOv3）直接迁移到 SAR 域表现有限。缺乏一个同时具备大容量与域泛化能力的 SAR 语义分割基础模型。
- **关键洞察**: 解锁 SAR 大规模泛化需要：(1) 十亿级参数容量以吸收极端域多样性；(2) 稀疏激活以控制推理成本；(3) 物理先验引导以稳定跨域路由。

## 方法详解

### 整体框架

以 DINOv2 ViT 为 backbone，将每个 Transformer block 中的标准 FFN 替换为物理引导的稀疏 MoE 层，每层包含路由器 $R_\psi$ 和 $n$ 个专家 $\{E_k\}_{k=1}^n$（从 DINOv2 FFN 权重初始化）。输入 SAR 图像复制为 3 通道 $X \in \mathbb{R}^{3 \times H \times W}$ 送入 backbone，同时计算 3 个物理描述符 $s \in \mathbb{R}^3$ 辅助路由。最终 token 嵌入送入 Mask2Former 解码器生成分割预测。整个模型先在自建的 CrossEarth-SAR-200K 上做持续预训练，再冻结 backbone 微调解码器；提供 S（20M 激活）/B（80M 激活）/L（300M 激活，总参数 1.3B）三个版本。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    DATA["CrossEarth-SAR-200K 大规模数据集<br/>40K 真标注 + 163K 伪标注（光学模型迁移）"] -.持续预训练.-> BK
    X["SAR 图像<br/>复制为 3 通道"] --> BK["DINOv2 ViT backbone<br/>每个 block 的 FFN 换成物理引导 MoE"]
    X --> PD["SAR 物理描述符<br/>方向熵 H_DE / 等效视数 ENL / 局部粗糙度 R_LR"]
    subgraph MOE["物理引导稀疏 MoE 层"]
        direction TB
        RT["路由器 softmax(W_r·[Z ‖ S])<br/>token 嵌入 Z 拼接物理描述符 S"] --> TK["选 top-k 专家归一化加权聚合"]
    end
    BK --> RT
    PD --> RT
    TK --> DEC["Mask2Former 解码器"]
    DEC --> OUT["域泛化语义分割预测"]
    TK -.训练时约束.-> LB["负载均衡约束 L_BC<br/>防专家坍缩"]
```

### 关键设计

**1. SAR 物理描述符 — 为路由提供稳定的物理锚点**

- **功能**: 解决标准 MoE 路由器仅依赖 token 嵌入、在异构 SAR 数据下路由不稳定（"Routing Instability"）的问题，提供稳定的域级先验信号。
- **核心思路**: 对输入图像先做 log 变换 $X' = \log(1 + |X|)$ 保证数值稳定，然后计算三个互补的物理量：
    - (a) **方向熵** $H_{DE}$：对 Sobel 梯度方向做直方图后计算熵 $H_{DE} = -\sum_i p_i \ln p_i$，刻画成像几何特征（低值=强线性结构，高值=不规则纹理）
    - (b) **等效视数** ENL = $(\mu / \sigma)^2$：反映散斑强度/雷达系统特性（高值=弱散斑，低值=强噪声）
    - (c) **局部粗糙度** $R_{LR} = \text{Var}(\mu_j)_{j=1}^M$：空间块均值的方差，刻画目标散射的纹理变异性（高值=复杂纹理，低值=平滑区域）
- **设计动机**: 三个描述符分别对应 SAR 的三重物理挑战——成像几何、雷达系统噪声、目标散射特性，拼接为 $s = [H_{DE}, \text{ENL}, R_{LR}] \in \mathbb{R}^3$，为跨域路由提供物理锚点。

**2. 物理引导稀疏 MoE — 大容量低推理成本的域适应架构**

- **功能**: 以稀疏激活方式将模型扩展到 1.3B 参数，同时保持与标准 FFN 可比的推理开销。
- **核心思路**: 将物理描述符 $s$ 沿 token 维度 tile 为 $S \in \mathbb{R}^{B \times N \times 3}$，与 token 嵌入 $Z \in \mathbb{R}^{B \times N \times C}$ 拼接后送入路由器：$\pi = \text{softmax}(W_r[Z \| S] + b_r)$，计算每个 token 对 $n$ 个专家的得分，选择 top-$k$ 专家激活并做归一化加权聚合 $\tilde{z} = \sum_{k \in \mathcal{I}} g_k \cdot E_k(z)$。
- **设计动机**: 不同专家可以专化于不同 SAR 成像条件（极化、波段等），而物理描述符的引入使路由选择与底层物理机制对齐，避免了仅靠学习嵌入在域间剧烈波动的问题。

**3. 负载均衡约束 — 防止专家坍缩**

- **功能**: 确保所有专家被均匀利用，避免路由器退化为总是选择少数专家。
- **损失函数**: $\mathcal{L}_{BC} = \lambda_{BC} \cdot n \cdot \sum_{k=1}^n f_k p_k$，其中 $f_k$ 为分配给专家 $k$ 的 token 比例，$p_k$ 为平均路由概率，$\lambda_{BC} = 0.005$。总训练目标为 $\mathcal{L} = \mathcal{L}_{seg} + \mathcal{L}_{BC}$。

**4. CrossEarth-SAR-200K 大规模数据集 — 支撑全球尺度持续预训练**

- **功能**: 构建首个 20 万级 SAR 语义分割数据集，覆盖 109 个地区、6 大洲。
- **核心思路**: 整合 40K 有真实标注的公开 SAR 数据 + 163K 伪标注数据（用 CrossEarth 光学模型在配对光学图像上生成标签后迁移给 SAR），7 个语义类别（建筑/道路/水体/裸地/森林/农田/背景），所有图像裁剪/缩放至 512×512。伪标注质量经 4 个模型验证，Mean Agreement 达 75.88%（超过 OpenEarthMap-SAR 的 63.20%）。

### 训练策略

- **持续预训练（CPT）**: 在 CrossEarth-SAR-200K 上训练 18 epochs，batch size 4，AdamW lr=3e-5，16×A100 (80GB)
- **下游微调**: 冻结 backbone，仅训练 Mask2Former 解码器，40k iterations，batch size 2，lr=1e-4，单卡 4090
- **Earth-Adapter (PEFT)**: 在冻结 backbone 上加轻量适配器进一步提升，标记为 CrossEarth-SAR-L*

## 实验关键数据

### 主实验：单域差异（12 个基准）

| 方法 | Backbone | 参数 | 区域(N2S) | 区域(S2N) | 极化(VV2F) | 极化(HH2F) | 复数(C(r)2R) | 复数(C(i)2R) | Avg. |
|---|---|---|---|---|---|---|---|---|---|
| DINOv2 | ViT-L | 300M | 32.3 | 43.8 | 65.7 | 56.8 | 71.3 | 71.7 | 55.5 |
| DINOv3 | ViT-L | 300M | 33.7 | 42.8 | 48.3 | 50.6 | 69.9 | 69.2 | 53.0 |
| SARATR-X | HiViT-B | 90M | 34.6 | 43.2 | 71.3 | 68.5 | 74.5 | 74.2 | 59.7 |
| **CrossEarth-SAR-L** | ViT-L | **1.3B(300M)** | **38.0** | **46.7** | **73.9** | **72.3** | **76.9** | **76.7** | **62.7** |
| **CrossEarth-SAR-L*** | ViT-L | **1.3B(300M)** | 38.0 | 46.7 | 73.9 | 71.8 | 76.9 | 76.7 | **62.7** |

CrossEarth-SAR-L 相比 DINOv2 基线平均提升 **+7.2 mIoU**，极化域（HH2F）最高提升 **+15.5 mIoU**。

### 主实验：多域差异（10 个基准）

| 方法 | 区域+极化(A2F) | 区域+平台(O2D) | 区域+波段(S2A) | 区域+极化+波段(D2F) | 区域+平台+波段(W2D) | Avg. |
|---|---|---|---|---|---|---|
| DINOv2 | 15.5 | 17.8 | 55.9 | 26.0 | 16.7 | 24.3 |
| SARATR-X | 21.3 | 19.0 | 53.1 | 22.6 | 16.1 | 24.8 |
| **CrossEarth-SAR-L** | **25.0** | **23.7** | **59.1** | 25.1 | **22.2** | **27.7** |
| **CrossEarth-SAR-L*** | 27.0 | 23.1 | 57.9 | 26.5 | 25.6 | **28.5** |

多域差异场景下 CrossEarth-SAR-L* 平均 28.5 mIoU，比基线 +4.2。

### 消融实验

| 消融项 | 配置 | mIoU | 增益 |
|---|---|---|---|
| 仅 40K 真标注 | DINOv2 + 40K real | 45.1 | — |
| 200K 含伪标注 | DINOv2 + 200K | 59.4 | **+14.3** |
| 纯 MoE（无约束） | MoE only | 61.1 | +1.7 |
| + 负载均衡 $\mathcal{L}_{BC}$ | +BC | 62.2 | +2.8 |
| + 物理描述符 $S$ | +S | 61.6 | +2.2 |
| + 两者结合 | +BC+S | **62.4** | **+3.0** |
| 专家数 n=3 | top-1 | 60.9 | +1.5 |
| 专家数 n=6 | top-1 | **62.4** | **+3.0** |
| 激活 top-2 | n=6 | 61.7 | +2.3 |
| 激活 top-3 | n=6 | 61.3 | +0.9 |

### 关键发现

- **伪标注规模效用显著**: 200K 数据比仅 40K 真标注高 14.3% mIoU；40K 伪标注甚至略优于 40K 真标注（全球覆盖更广），两者结合进一步 +3.6%
- **6 专家 top-1 最优**: 增大 top-k 反而下降，说明在 200K 数据规模下单专家专化比多专家混合更有效
- **物理描述符敏感性各异**: $H_{DE}$ 对极化（73.47）和微波波段（59.18）敏感，ENL 对复数值（75.97）敏感，$R_{LR}$ 对区域（37.49）和平台（19.83）敏感
- **层级专化涌现**: 可视化显示 Expert 3/4 主导浅层（散斑统计），Expert 1/2/5/6 活跃中层（几何纹理），Expert 1/5 集中深层（高级语义）

## 亮点与洞察

- 将 SAR 的三重物理先验（散斑/几何/散射）编码为可微分物理描述符引导 MoE 路由，实现了"物理先验 + 数据驱动"的优雅结合，比纯学习路由稳定且可解释
- 22 个子基准覆盖 8 种域差异组合（区域/极化/复数值/平台/波段的单独及组合），为 SAR 社区建立了首个统一 DG 评测标准
- 仅 20M 激活参数的 CrossEarth-SAR-S 已超越 300M 的 DINOv2 和 DINOv3，证明物理引导 MoE 的参数效率优势

## 局限性

- 1.3B 参数量虽然激活仅 300M，但存储和部署到资源受限的遥感平台（星载/无人机）仍是挑战
- 伪标注依赖 CrossEarth 光学模型质量，Mean Agreement 仅 75.88%，部分类别（如道路 vs 裸地）混淆严重
- 仅评测语义分割任务，未验证在目标检测、变化检测等其他 SAR 下游任务的泛化性
- 训练需 16×A100 (80GB)，资源门槛较高

## 相关工作与启发

- **vs SARATR-X** (90M HiViT)：CrossEarth-SAR-L 在单域差异上平均高 3.0 mIoU，20M 的 -S 版本在极化域（HH2F）提升 11.7%
- **vs DINOv2/v3** (300M)：同等激活参数下 CrossEarth-SAR-L 单域差异 +7.2 / +9.7 mIoU
- **vs SatMAE/ScaleMAE/MTP**: 光学预训练模型在 SAR 域上全面落后，MTP 最优也仅 44.7 vs CrossEarth-SAR-L 的 62.7
- **物理引导路由**可推广到其他传感器特异性模态（红外/多光谱/高光谱），稀疏 MoE 在域碎片化场景下优于密集扩展的经验值得通用视觉借鉴

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 物理描述符引导 MoE 路由的设计创新且有物理依据，首个十亿级 SAR VFM
- **实验充分度**: ⭐⭐⭐⭐⭐ — 22 个基准、16 种对比方法、5 组消融、层级专化和损失曲线可视化
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，物理先验与工程设计的动机链完整
- **价值**: ⭐⭐⭐⭐ — 对遥感/SAR 社区贡献突出，数据集和基准具有长期价值；对通用视觉社区中等

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] CA-LoRA: Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation](ca-lora_concept-aware_lora_for_domain-aligned_segmentation_dataset_generation.md)
- [\[CVPR 2026\] SARMAE: Masked Autoencoder for SAR Representation Learning](sarmae_masked_autoencoder_for_sar_representation_learning.md)
- [\[CVPR 2026\] GKD: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](gkd_generalizable_knowledge_distillation_vfm.md)
- [\[CVPR 2026\] Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains](promptdriven_lightweight_foundation_model_for_inst.md)
- [\[CVPR 2026\] Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions](heuristic_self-paced_learning_for_domain_adaptive_semantic_segmentation_under_ad.md)

</div>

<!-- RELATED:END -->
