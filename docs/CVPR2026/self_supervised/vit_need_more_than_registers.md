---
title: >-
  [论文解读] Vision Transformers Need More Than Registers
description: >-
  [CVPR 2026][自监督学习][ViT artifacts] 系统分析了 ViT 中广泛存在的 artifact 现象（跨全监督、文本监督、自监督），揭示其根本原因是"lazy aggregation"——ViT 利用语义无关的背景 patch 作为捷径来表示全局语义，提出 LaSt-ViT（LazyStrike ViT）通过频率感知的选择性通道聚合将 CLS token 锚定到前景区域，在 12 个 benchmark 上一致消除 artifact 并提升性能。
tags:
  - CVPR 2026
  - 自监督学习
  - ViT artifacts
  - lazy aggregation
  - patch score
  - foreground aggregation
  - register tokens
---

# Vision Transformers Need More Than Registers

**会议**: CVPR 2026  
**arXiv**: [2602.22394](https://arxiv.org/abs/2602.22394)  
**代码**: [GitHub](https://github.com/ChengShiest/LAST-ViT)  
**领域**: 视觉表征学习 / Vision Transformer 分析  
**关键词**: ViT artifacts, lazy aggregation, patch score, foreground aggregation, register tokens

## 一句话总结

系统分析了 ViT 中广泛存在的 artifact 现象（跨全监督、文本监督、自监督），揭示其根本原因是"lazy aggregation"——ViT 利用语义无关的背景 patch 作为捷径来表示全局语义，提出 LaSt-ViT（LazyStrike ViT）通过频率感知的选择性通道聚合将 CLS token 锚定到前景区域，在 12 个 benchmark 上一致消除 artifact 并提升性能。

## 研究背景与动机

**领域现状**：ViT 已成为图像识别的事实标准，更重要的是作为通用特征提取器（frozen foundation model）应用于各种下游任务。不同监督方式训练的 ViT 各有所长：全监督/文本监督（如 CLIP）适用于开放词汇任务和 LVLM 视觉编码器，自监督（如 DINO）擅长无监督分割和物体发现。

**现有痛点**：
1. DINO 发现全监督 ViT 存在 attention deficit 问题
2. CLIPSelf 发现文本监督 ViT 的 dense feature 与文本 cue 不对齐
3. Register 论文发现自监督 ViT（DINOv2）生成 high-norm token artifact，影响物体定位
4. 这些现象暗示 ViT 存在共性的底层问题，但此前无统一解释和解决方案

**核心矛盾**：ViT 在 image-level 分类上表现优异，但 patch-level 的 dense feature 质量差——高分 patch 落在背景而非前景，且 Register token 只能移除 high-norm 现象但无法解决根本问题（PiB 反而下降）。

**本文目标**：从第一性原理出发，统一定义、分析和解决不同监督范式下 ViT 的 artifact 问题。

**切入角度**：提出 Patch Score（CLS-patch 相似度）和 Point-in-Box（PiB）统一度量 artifact，发现根因是 lazy aggregation——全局注意力 + 粗粒度监督 → ViT 走捷径用背景 patch 编码全局语义。

**核心 idea**：通过频率域稳定性分析区分前景/背景 patch，选择性聚合稳定 patch 到 CLS token，消除 lazy aggregation。

## 方法详解

### 整体框架

LaSt-ViT 的核心是替换 ViT 原有的 CLS token 聚合方式（Attention Pooling 或 GAP），改为基于通道级频率稳定性的 Top-K 选择性聚合。直觉是：前景 patch 的特征在通道维度上更同质（语义一致），因此对低通滤波更稳定。

### 关键设计

1. **Patch Score 和 Point-in-Box (PiB) 统一度量**:
    - **Patch Score** 定义为每个 patch 特征与 CLS token 的余弦相似度：$\mathcal{S}_p = \frac{\mathbf{x}_{\text{patch}} \cdot Q_{\text{CLS}}}{\|\mathbf{x}_{\text{patch}}\|_2 \|Q_{\text{CLS}}\|_2}$
    - **PiB** 度量最高分 patch 是否落在前景标注框内，作为 artifact 严重程度的指标
    - 实验发现：ViT 的 PiB 仅 42.7，远低于 ResNet 的 68.4；Register 虽消除 high-norm 但 PiB 反而降至 41.5
    - 该度量与监督方式无关，统一适用于全监督、文本监督和自监督 ViT

2. **Lazy Aggregation 假说验证**:
    - **masking probe**：移除 ViT 中 Patch Score 最高的 50% patch 对 ImageNet 精度几乎无影响甚至略有提升（+1.2%），而移除低分 patch 导致精度骤降（-60% at 70% masking）→ 高分 patch 是语义无关的背景
    - **训练动态**：ViT 的 PiB 从训练初期就很低（~0.42）且几乎不变，说明 lazy aggregation 是训练早期就形成的内在行为
    - **因素分离**：(1) 增大 patch size（28→减少背景 token 比例 10%）→ PiB 从 0.44 升至 0.52 但精度降；(2) 用窗口注意力替换全局注意力 → PiB 升至 59.8 但精度从 72.3 降至 63.9
    - 结论：粗粒度监督（image-level label）+ 全局依赖（long-range attention）共同导致 lazy aggregation

3. **LaSt-ViT：频率感知选择性聚合**:
    - **Stability Score 计算**：对 patch 特征做通道维 1D FFT → 乘以高斯低通滤波器 $\mathbf{g}$ → 逆 FFT 得到低频特征 $\hat{\mathbf{x}}_{\text{patch}}$
    - 稳定性分数：$\mathbf{S}_{i,j} = \frac{\hat{\mathbf{x}}_{\text{patch}}[i,j]}{|\hat{\mathbf{x}}_{\text{patch}}[i,j] - \mathbf{x}_{\text{patch}}[i,j]| + \varepsilon}$，分数越高表示该通道该 patch 越稳定（越可能是前景）
    - **Channel-wise Top-K Pooling**：对每个通道 $j$，选择稳定性最高的 $K$ 个 patch 取平均作为 CLS token 的该通道值：$\mathcal{Q}_{\text{CLS}}[j] = \frac{1}{K} \sum_{i \in \mathcal{I}_K(j)} \mathbf{x}_{\text{patch}}[i,j]$
    - **Vote Count** 可视化：定义 $v_i = \sum_{j=1}^D \mathbf{1}\{i \in \mathcal{I}_K(j)\}$，高投票 patch 与前景区域高度一致

### 损失函数 / 训练策略

- LaSt-ViT 不引入额外损失函数，仅替换 CLS token 的聚合方式
- 适用于任何 ViT 预训练流程（全监督、CLIP、DINO），作为 drop-in replacement
- 超参数 $K$ 控制每通道选择的 patch 数量，最优值约为总 patch 数的 50%（如 ViT-B/16 的 196 个 patch 中选 98 个）
- 训练流程与原始 ViT 完全一致，无需额外数据或调参

## 实验关键数据

### 主实验

Artifact 消除（Patch Score / PiB）：

| 方法 | High Norm | Point-in-Box (PiB) |
|------|:---------:|---:|
| ResNet | ✗ | 68.4 |
| ViT | ✓ | 42.7 |
| ViT + Register | ✗ | 41.5 |
| ViT + **LazyStrike** | ✗ | **55.1** (+12.4) |
| DINO-ViT | ✗ | 44.5 |
| DINO + **LazyStrike** | ✗ | **69.7** (+25.2) |
| CLIP-ViT | ✓ | 39.8 |
| CLIP + **LazyStrike** | ✗ | **50.1** (+10.3) |

零样本语义分割（mIoU %，CLIP ViT-L/14）：

| 方法 | VOC20 | ADE20K | Cityscapes | COCO-Stf. |
|------|---:|---:|---:|---:|
| CLIP | 17.1 | 1.6 | 2.7 | 3.2 |
| CLIP + **LazyStrike** | **72.4** (+55.3) | **8.4** (+6.8) | **12.3** (+9.6) | **11.9** (+8.7) |

### 消融实验

CLS 聚合方式对比（OpenCLIP ViT-B/16）：

| 方法 | ImageNet Top-1 | VOC20 (seg) | COCO-Stf. (seg) |
|------|---:|---:|---:|
| Attention-Pool | 55.8 | 49.0 | 7.2 |
| Max-Pool | 53.1 | 71.9 | 12.2 |
| LazyStrike K=1 | 53.5 | 72.7 | 13.5 |
| LazyStrike K=49 | 55.8 | **75.8** | **18.5** |
| LazyStrike K=98 | **56.2** | 75.9 | 18.0 |
| LazyStrike K=196 (Full) | 55.3 | 13.5 | 4.8 |

无监督物体发现（CorLoc，DINO ViT-S）：

| 方法 | VOC07 | VOC12 | COCO | FPS |
|------|---:|---:|---:|---:|
| DINO-seg | 45.8 | 46.2 | 42.1 | 29.4 |
| LOST | 61.9 | 64.0 | 50.7 | 29.4 |
| DINO + **LazyStrike** | **64.4** | **67.6** | **51.6** | **55.9** |

### 关键发现

1. Register token 只是把 high-norm 现象从 feature map 移到 register 上，PiB 反而下降（41.5 < 42.7），说明"Vision Transformers Need More Than Registers"
2. LazyStrike 同时消除 high-norm 和 patch score artifact，因为两者都是 lazy aggregation 的不同表现
3. 在 CLIP ViT-L/14 上 VOC20 零样本分割 mIoU 从 17.1% 跃升至 72.4%（+55.3%），证明 artifact 消除后 dense feature 质量大幅提升
4. LazyStrike 使全监督 ViT 展现出"涌现性分割"能力（PCA 可视化），此前被认为是自监督（DINO）独有的特性
5. $K$ 过大（全部 patch）等价于 GAP，性能退化；$K$ 过小则丢失太多信息；最优 $K \approx N/2$

## 亮点与洞察

1. **分析深度出色**：从第一性原理出发，通过 masking probe、训练动态追踪、因素分离等实验层层推进，假说验证严谨
2. **统一视角**: 将三种监督下的不同 artifact 现象归因为同一个根因（lazy aggregation），提供了理解 ViT 行为的新框架
3. **方法简洁有效**：仅替换 CLS 聚合方式，无需额外模块/数据/损失，即可在 12 个 benchmark 上一致提升
4. **颠覆性发现**：涌现性分割不是自监督独有的——消除 lazy aggregation 后全监督 ViT 也能展现

## 局限与展望

1. 频率域稳定性假设（前景 patch 更稳定）可能在某些场景下不成立（如纹理丰富的前景 vs 均匀背景）
2. Channel-wise Top-K 需要额外的 FFT/IFFT 计算，虽然轻量但对实时推理仍有开销
3. 仅在 ViT-S/B/L 上验证，未测试更大规模模型（如 ViT-G）
4. 训练时 $K$ 值固定，未探索自适应 $K$ 值或 learnable 选择机制
5. 论文标题暗示 Register 不够好，但未充分讨论 LazyStrike 和 Register 结合使用的效果

## 相关工作与启发

- **Register Tokens (Darcet et al.)**：通过额外 token 吸收 high-norm artifact，但不解决根因
- **CLIPSelf**：通过额外对齐训练修复 CLIP dense feature，属于后处理方案
- **MaskCLIP**：首次展示 CLIP 特征可用于零样本语义分割
- **F-ViT**：frozen CLIP backbone 做开放词汇检测，直接受益于 LazyStrike
- **启发**：预训练模型的"捷径学习"是一个普遍问题，理解模型内部行为后用简单方法即可大幅改善

## 评分

| 维度 | 评分 |
|------|------|
| 创新性 | ⭐⭐⭐⭐⭐ |
| 实用性 | ⭐⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐⭐ |
| 综合 | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Zero-Ablation Overstates Register Content Dependence in DINO Vision Transformers](zero_ablation_overstates_register_content_dependence_in_dino_vision_transformers.md)
- [\[CVPR 2026\] DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers](diversedit_towards_diverse_representation_learning_in_diffusion_transformers.md)
- [\[CVPR 2026\] Robustness of Vision Foundation Models to Common Perturbations](robustness_of_vision_foundation_models_to_common_perturbations.md)
- [\[CVPR 2026\] TALO: Pushing 3D Vision Foundation Models Towards Globally Consistent Online Reconstruction](talo_pushing_3d_vision_foundation_models_towards_globally_consistent_online_reco.md)
- [\[CVPR 2026\] Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models](com_pt_chain_of_models_pretraining.md)

<!-- RELATED:END -->
