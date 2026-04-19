---
title: >-
  [论文解读] GroupMamba: Efficient Group-Based Visual State Space Model
description: >-
  [CVPR 2025][state space model] 提出 Modulated Group Mamba 层，将输入通道分为四组并按不同方向扫描，配合通道亲和调制和蒸馏损失，构建了参数高效且训练稳定的视觉状态空间模型。
tags:
  - CVPR 2025
  - state space model
  - Mamba
  - 分组扫描
  - 知识蒸馏
  - 图像分类
---

# GroupMamba: Efficient Group-Based Visual State Space Model

**会议**: CVPR 2025  
**arXiv**: [2407.13772](https://arxiv.org/abs/2407.13772)  
**代码**: [GitHub](https://github.com/Amshaker/GroupMamba)  
**领域**: segmentation  
**关键词**: state space model, Mamba, group convolution, channel modulation, knowledge distillation, ImageNet

## 一句话总结

提出 Modulated Group Mamba 层，将输入通道分为四组分别按四个方向执行单向 SSM 扫描，通过 Channel Affinity Modulation（CAM）增强跨组通道交互，配合蒸馏训练目标解决大模型不稳定问题，在 ImageNet-1K 上以 23M 参数达到 83.3% Top-1 精度。

## 研究背景与动机

**领域现状**: 视觉状态空间模型（Visual SSM）如 VMamba、Vision Mamba 等借鉴 Mamba 的线性复杂度处理长序列的能力，在视觉任务中展现潜力。

**现有痛点**:
- **参数效率低**: 标准 VSS block 对所有通道执行 4 方向全扫描，输入/输出投影和深度卷积的参数量与通道数成正比，导致参数冗余
- **训练不稳定**: Mamba-based 模型在扩展到大模型时训练不稳定（如 SiMBA-L MLP 仅 49% 精度）
- **扫描冗余**: 每个方向都对全部通道扫描，计算浪费

**核心矛盾**: 提高视觉 SSM 的参数效率和训练稳定性，同时保持对局部/全局信息的有效建模。

**本文切入角度**: 受分组卷积（Group Convolution）启发，将通道分为四组，每组只沿一个方向扫描，大幅减少参数；同时设计 CAM 机制弥补分组带来的通道交互不足。

## 方法详解

### 整体框架

采用类似 Swin-Transformer 的四阶段层次结构：
1. Patch Embedding（两个 3×3 卷积，stride=2）生成 H/4×W/4 的初始特征
2. 每阶段包含 N 个 Modulated Group Mamba block + 下采样层
3. 四阶段特征分辨率依次为 H/4、H/8、H/16、H/32

### 关键设计

**1. Visual Single Selective Scan (VSSS) Block**
- **功能**: 基于 Mamba 的 token-channel 混合器，由一个 Mamba block + FFN 组成，各前接 LayerNorm
- **核心思路**: 对输入 $\mathbf{Z}_{in}$ 先经 Mamba SSM 做 token mixing（序列建模），再经 FFN 做 channel mixing，均带残差连接
- **设计动机**: 作为分组扫描的基本单元，每个 VSSS block 仅处理 $C/4$ 个通道的单方向扫描

**2. Grouped Mamba Operator（分组扫描）**
- **功能**: 将输入 $C$ 通道分为 4 组（各 $C/4$），分别按左→右、右→左、上→下、下→上四个方向展平为 1D 序列，各自独立通过一个 VSSS block 处理后拼接
- **核心思路**: 
$$\mathbf{X}_{GM} = \text{Concat}(\text{VSSS}(\mathbf{X}_{LR}), \text{VSSS}(\mathbf{X}_{RL}), \text{VSSS}(\mathbf{X}_{TB}), \text{VSSS}(\mathbf{X}_{BT}))$$
- **设计动机**: 每组只处理 $C/4$ 通道和单方向扫描，参数量和计算量大幅降低（参数减少约 26%）；四个方向覆盖完整空间依赖

**3. Channel Affinity Modulation (CAM)**
- **功能**: 对分组 Mamba 输出进行通道重标定，增强跨组通道信息交换
- **核心思路**: 
    - 全局平均池化 → 两层 FC（类似 SE block）→ Sigmoid 得到通道权重
    - $\mathbf{X}_{CAM} = \mathbf{X}_{GM} \cdot \text{Affinity}(\mathbf{X}_{in})$
- **设计动机**: 分组操作限制了通道间交互（每组只看 $C/4$ 通道），CAM 通过输入特征计算的亲和力权重重新校准输出，弥补信息隔离

### 损失函数 / 训练策略

蒸馏联合损失：

$$\mathcal{L}_{total} = \alpha \mathcal{L}_{CE}(Z_s, y) + (1-\alpha) \mathcal{L}_{CE}(Z_s, y_t)$$

- $Z_s$: 学生模型 logits，$y$: ground-truth 标签，$y_t$: 教师硬标签
- 教师模型: RegNetY-16G（84M 参数，82.9% Top-1）
- 蒸馏目标是缓解大模型训练不稳定（SiMBA 已证明 MLP channel mixer + 大 Mamba 会发散）
- Label smoothing 0.1，300 epochs，AdamW，初始 lr=1e-3

## 实验关键数据

### 主实验（ImageNet-1K 分类）

| 模型 | 参数量 | FLOPs | Top-1 |
|---|---|---|---|
| Swin-T | 28M | 4.6G | 81.3 |
| VMamba-T | 31M | 4.9G | 82.5 |
| LocalVMamba-T | 26M | 5.7G | 82.7 |
| **GroupMamba-T** | **23M** | **4.5G** | **83.3** |
| VMamba-S | 50M | 8.7G | 83.6 |
| **GroupMamba-S** | **34M** | **7.0G** | **83.9** |
| VMamba-B | 89M | 15.4G | 83.9 |
| **GroupMamba-B** | **57M** | **14G** | **84.5** |

下游任务：
- COCO 检测 (Mask R-CNN): AP^b = 47.6, AP^m = 42.9（超越 Swin-T、ConvNeXt-T）
- ADE20K 语义分割 (UperNet): mIoU = 48.6 (SS) / 49.2 (MS)

### 消融实验

| 配置 | Params | Throughput | Top-1 |
|---|---|---|---|
| 4-D scanning (baseline) | 22M | 803 | 82.30 |
| + Grouped 1-D scanning | 22M | 1125 | 82.20 |
| + CAM | 22M | 1069 | 82.50 |
| + Distillation loss | 23M | 1069 | **83.30** |

### 关键发现

1. **分组扫描几乎无精度损失**: 从 4-D full scanning 到 grouped 1-D scanning 仅降 0.1%，但吞吐量提升 40%（803→1125）
2. **CAM 有效弥补通道隔离**: +0.3% 精度，开销极小
3. **蒸馏是稳定训练的关键**: +0.8% 精度，解决大 SSM 模型发散问题
4. **参数效率显著**: GroupMamba-T 用 23M 超越 VMamba-T（31M），减少 26% 参数
5. **GroupMamba-B vs VMamba-B**: 36% 更少参数，+0.6% 精度

## 亮点与洞察

- 分组扫描思想简洁有效：用分组卷积的成熟理念解决 SSM 的通道冗余
- CAM 的设计虽类似 SE block，但在 SSM 分组上下文中有独特价值
- 蒸馏解决 SSM 训练不稳定的方案具有通用性
- 三个变体（T/S/B）形成完整的精度-效率 tradeoff 系列

## 局限与展望

- 蒸馏依赖于外部教师模型（RegNetY-16G），增加了训练复杂度
- 仅验证了图像分类、检测、分割，未扩展到视频理解或时序任务
- 四组固定分配可能不是最优，可探索自适应分组策略
- CAM 本质是 SE block 的应用，创新增量有限
- 未与同期 Mamba-2 等新架构对比

## 相关工作与启发

- VMamba 首创四方向 2D 扫描但计算冗余，本文通过分组有效解决
- DeiT 的蒸馏 token 思想被本文简化为蒸馏损失
- 启发：SSM 方法的参数效率和训练稳定性是关键瓶颈，分组+蒸馏的组合策略值得推广

## 评分

⭐⭐⭐⭐ — 分组扫描设计简洁优雅，实验充分覆盖多个下游任务；蒸馏+CAM 的组合虽非全新但实用有效，参数效率优势显著。

<!-- RELATED:START -->

## 相关论文

- [DefMamba: Deformable Visual State Space Model](defmamba_deformable_visual_state_space_model.md)
- [2DMamba: Efficient State Space Model for Image Representation with Applications on Giga-Pixel Whole Slide Image Classification](2dmamba_efficient_state_space_model_for_image_representation_with_applications_o.md)
- [Exploiting Temporal State Space Sharing for Video Semantic Segmentation](exploiting_temporal_state_space_sharing_for_video_semantic_segmentation.md)
- [EdgeTAM: On-Device Track Anything Model](edgetam_on-device_track_anything_model.md)
- [MambaOut: Do We Really Need Mamba for Vision?](mambaout_do_we_really_need_mamba_for_vision.md)

<!-- RELATED:END -->
