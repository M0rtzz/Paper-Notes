---
title: >-
  [论文解读] RAW-Adapter: Adapting Pre-trained Visual Model to Camera RAW Images
description: >-
  [ECCV 2024][RAW 图像] 提出 RAW-Adapter，通过输入级适配器（可学习 ISP 阶段）和模型级适配器（ISP 中间特征注入骨干网络），以极小参数量（0.2-0.8M）将 sRGB 预训练模型高效适配到 Camera RAW 图像，在正常光/暗光/过曝等多种光照条件下的检测和分割任务上达到 SOTA。
tags:
  - ECCV 2024
  - RAW 图像
  - 图像信号处理器
  - 适配器
  - 目标检测
  - 语义分割
---

# RAW-Adapter: Adapting Pre-trained Visual Model to Camera RAW Images

**会议**: ECCV 2024  
**arXiv**: [2408.14802](https://arxiv.org/abs/2408.14802)  
**代码**: [有](https://github.com/cuiziteng/RAW-Adapter)  
**领域**: 信号通信  
**关键词**: RAW 图像, 图像信号处理器, 适配器, 目标检测, 语义分割

## 一句话总结

提出 RAW-Adapter，通过输入级适配器（可学习 ISP 阶段）和模型级适配器（ISP 中间特征注入骨干网络），以极小参数量（0.2-0.8M）将 sRGB 预训练模型高效适配到 Camera RAW 图像，在正常光/暗光/过曝等多种光照条件下的检测和分割任务上达到 SOTA。

## 研究背景与动机

当前计算机视觉模型主要在 sRGB 图像上预训练，但 RAW 图像具有独特优势：**未经 ISP 压缩**，保留了丰富的物理信息（如噪声分布），与辐射能量呈线性关系，拥有更高的动态范围。在真实世界的极端光照条件下（日照强度从 $1.3 \times 10^3 W/m^2$ 到 $2.0 \times 10^{-6} W/m^2$），RAW 图像的优势尤为明显。

然而，现有将 RAW 用于视觉任务的方法存在三个核心问题：

**ISP 目标不匹配**：传统 ISP 旨在生成"视觉上令人愉悦"的图像，而**非优化下游视觉任务**。手动设计的 ISP 有时甚至不如直接使用 RAW 数据效果好。

**联合训练方法的局限**：现有方案要么用进化策略优化传统 ISP 参数（如 Hardware-in-the-loop），要么用神经网络替代 ISP（如 Dirty-Pixel 用残差 UNet），但前者不可微分，后者引入大量计算负担（Dirty-Pixel 增加 4.28M 参数）。

**缺乏交互**：上述方法将 ISP 和后端网络视为两个独立模块，缺少 ISP 阶段与下游网络之间的**信息交互**。同时，从头在 RAW 数据上训练会放弃 sRGB 大规模预训练模型的知识（Fig.2 显示使用预训练权重可显著提升性能）。

**核心问题**：如何高效地将信息丰富的 RAW 图像适配到知识丰富的 sRGB 预训练模型中？

## 方法详解

### 整体框架

RAW-Adapter 包含两层适配器：**输入级适配器**（Input-level Adapters）将 RAW 图像 $\mathbf{I}_1$ 转换为机器视觉友好的图像 $\mathbf{I}_5$；**模型级适配器**（Model-level Adapters）将 ISP 各阶段的中间特征注入骨干网络的不同 stage，增强两者之间的信息交互。整体框架保持 ISP 的模块化设计，但通过 Query Adaptive Learning (QAL) 使所有关键参数可学习、可微分。

### 关键设计

1. **Query Adaptive Learning (QAL)**：用于自适应预测 ISP 各阶段的关键参数。设计灵感来自 Transformer 的注意力机制：输入图像经两层下采样卷积提取特征，生成 key $\mathbbm{k}$ 和 value $\mathbbm{v}$，而 query $\mathbbm{q}$ 是一组**可学习的动态参数**，通过自注意力计算预测 ISP 参数：

$$parameters = FFN\left(softmax\left(\frac{\mathbbm{q} \cdot \mathbbm{k}^T}{\sqrt{d_k}}\right) \cdot \mathbbm{v}\right)$$

这使得每张图像可根据自身内容自适应预测不同的 ISP 参数。定义了两个 QAL 块 $\mathbb{P_K}$（预测增益/去噪参数）和 $\mathbb{P_M}$（预测白平衡/CCM 参数）。

2. **输入级适配器的 ISP 阶段**：保留传统 ISP 的模块化设计，但每个阶段的关键参数由 QAL 自适应预测：

    - **Gain & Denoise**：$\mathbb{P_K}$ 预测增益比 $g$（适应不同光照）和各向异性高斯核参数 $\{r_1, r_2\}$（自适应去噪），以及锐化参数 $\sigma$：
    $\mathbf{I}_2' = (g \cdot \mathbf{I}_1) \circledast k, \quad \mathbf{I}_2 = \mathbf{I}_2' + (g \cdot \mathbf{I}_1 - \mathbf{I}_2') \cdot \sigma$
    - **White Balance & CCM**：$\mathbb{P_M}$ 预测 Minkowski 距离参数 $\rho$（统一 gray-world 和 Max-RGB 白平衡）和 $3\times3$ 颜色转换矩阵 $\mathbf{E}_{ccm}$
    - **Color Manipulation**：使用 NILUT（Neural Implicit 3D LUT）进行端到端可学习的颜色映射

3. **模型级适配器**：ISP 中间阶段 $\mathbf{I}_1 \sim \mathbf{I}_4$ 含有丰富的先验信息，但此前方法完全忽略了这些信息。模型级适配器通过卷积层提取各 ISP 阶段特征并拼接：$\mathbb{C}(\mathbf{I}_{1\sim4}) = \mathbb{C}(c_1(\mathbf{I}_1), c_2(\mathbf{I}_2), c_3(\mathbf{I}_3), c_4(\mathbf{I}_4))$，经残差块处理后通过 merge block（拼接 + 残差连接）逐级注入骨干网络的 stage 1-3。这使得下游任务网络能够**利用 ISP 各阶段的先验知识**。

### 损失函数 / 训练策略

- 无需额外的人类视觉导向损失函数——仅使用下游任务本身的损失（检测用 focal loss / segmentation 用交叉熵等），ISP 参数通过任务损失端到端优化
- 所有比较方法使用相同的数据增强和训练设置
- 参数量极小：输入级适配器 $\mathbb{P_K}$ (37.57K) + $\mathbb{P_M}$ (37.96K) + $\mathbb{L}$ (1.97K)，模型级适配器 $\mathbb{M}$ (114.87K ~ 687.59K)，总计 **0.2M ~ 0.8M**，远小于 SID (11.99M) 和 Dirty-Pixel (4.28M)

## 实验关键数据

### 主实验

**PASCAL RAW 目标检测（RetinaNet, mAP）**：

| 方法 | ResNet-18 正常光 | ResNet-18 暗光 | ResNet-50 正常光 | ResNet-50 暗光 |
|------|-----------------|---------------|-----------------|---------------|
| Default ISP | 88.3 | - | 89.6 | - |
| Demosaicing | 87.7 | 80.3 | 89.2 | 82.6 |
| Dirty-Pixel | 88.6 | 80.8 | 89.7 | 83.6 |
| **RAW-Adapter** | **88.7** | **82.5** | **89.7** | **86.6** |

**LOD 暗光检测（ResNet-50, mAP）**：

| 方法 | RetinaNet | Sparse-RCNN |
|------|-----------|-------------|
| Demosaicing | 58.5 | 57.7 |
| Dirty-Pixel | 61.6 | 58.8 |
| **RAW-Adapter** | **62.1** | **59.2** |

**ADE20K RAW 语义分割（Segformer, mIoU）**：

| 方法 | Backbone | Params(M) | 正常光 | 过曝 | 暗光 |
|------|----------|-----------|-------|------|------|
| Demosaicing | MIT-B5 | 82.01 | 47.47 | 45.69 | 37.55 |
| Dirty-Pixel | MIT-B5 | 86.29 | 47.86 | 46.50 | 38.02 |
| **RAW-Adapter** | **MIT-B5** | **82.31** | **47.95** | **46.62** | **38.75** |
| **RAW-Adapter** | **MIT-B3** | **45.16** | 46.57 | 44.19 | **37.62** |

### 消融实验

**模型级适配器的效果（LOD 数据集, mAP）**：

| 配置 | RetinaNet | Sparse-RCNN | 说明 |
|------|-----------|-------------|------|
| RAW-Adapter (w/o $\mathbb{M}$) | 61.6 | 58.6 | 仅输入级适配器 |
| **RAW-Adapter (完整)** | **62.1** | **59.2** | 加上模型级适配器 |

**ADE20K RAW 消融（MIT-B5, mIoU）**：

| 配置 | 正常光 | 过曝 | 暗光 | 说明 |
|------|-------|------|------|------|
| w/o $\mathbb{M}$ | 47.83 | 46.48 | 38.41 | 仅输入级适配器 |
| **完整** | **47.95** | **46.62** | **38.75** | 完整 RAW-Adapter |

### 关键发现

- **暗光条件提升最显著**：PASCAL RAW (dark) 上 ResNet-50 从 Dirty-Pixel 的 83.6 提升至 86.6（+3.0），说明 RAW 的高动态范围优势在极端光照下更为突出
- **小 backbone 超越大 backbone**：RAW-Adapter + ResNet-18 的暗光检测 (82.5) 超过部分 ISP 方法 + ResNet-50 的性能
- **MIT-B3 + RAW-Adapter 的暗光分割 (37.62) 超过 Dirty-Pixel + MIT-B5 (38.02) 接近的性能**，参数量不到一半
- 传统 ISP 有时反而**降低**下游任务性能，尤其在非正常光照条件下
- 无需人类视觉损失约束，RAW-Adapter 仍然生成了视觉上令人满意的中间图像

## 亮点与洞察

- **设计哲学精准**：不追求复杂的输入处理，而是"简化输入阶段 + 增强 ISP 与网络的连接"
- **参数效率极高**：仅 0.2-0.8M 额外参数，远小于现有方案，且推理速度更快
- **通用性强**：框架可适配各种检测器（RetinaNet、Sparse-RCNN）和分割器（Segformer），以及不同大小的 backbone
- 将 NLP/CV 领域的 adapter 思想巧妙应用于 RAW-sRGB 域适配问题

## 局限与展望

- 不同光照条件（正常/暗/过曝）需要**分别训练**，无法用统一模型处理多种光照
- 仅在 CNN 骨干（ResNet、MIT）上验证，ViT 等 Transformer 架构的适配效果未探索
- ISP 阶段的选择和简化是手工设计的，更自动化的 ISP 阶段筛选可能进一步提升效果
- 合成暗光/过曝数据的噪声模型较为简单，在真实极端光照下的鲁棒性有待验证

## 相关工作与启发

- 与 Dirty-Pixel 的核心区别：DP 用 UNet 完全替换 ISP，是"黑盒"方法；RAW-Adapter 保留 ISP 模块化设计，是"白盒"方法，可解释性更强
- QAL 的设计思路可推广至其他需要自适应预测参数的场景（如自适应增强、自适应核预测）
- 模型级适配器启示：中间处理阶段的特征不应被丢弃，Multi-level 特征融合在域适配中很重要

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将 adapter 范式应用于 RAW-sRGB 域适配是新颖切入点，输入级+模型级双层设计有原创性
- **实验充分度**: ⭐⭐⭐⭐ — 涵盖检测+分割、多种光照、多个 backbone，但消融实验可以更详细
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，动机阐述充分，公式规范
- **价值**: ⭐⭐⭐⭐ — 对 RAW 图像的实际应用有重要推动作用，参数高效的设计具有工程价值
