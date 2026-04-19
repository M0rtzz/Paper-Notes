---
title: >-
  [论文解读] LabelAny3D: Label Any Object 3D in the Wild
description: >-
  [NeurIPS 2025][自动驾驶][3D标注] 提出 LabelAny3D，一个基于分析合成（analysis-by-synthesis）的自动 3D 标注流水线，从单目图像重建完整 3D 场景以获取高质量 3D 包围框标注；基于此构建了 COCO3D 基准，覆盖 80 类日常物体，在开放词汇单目 3D 检测上显著提升性能。
tags:
  - NeurIPS 2025
  - 自动驾驶
  - 3D标注
  - 单目3D检测
  - 开放词汇
  - 自动标注
  - 基础模型
---

# LabelAny3D: Label Any Object 3D in the Wild

**会议**: NeurIPS 2025  
**arXiv**: [2601.01676](https://arxiv.org/abs/2601.01676)  
**代码**: [项目页面](https://uva-computer-vision-lab.github.io/LabelAny3D/)  
**领域**: 自动驾驶  
**关键词**: 3D标注, 单目3D检测, 开放词汇, 自动标注, 基础模型

## 一句话总结

提出 LabelAny3D，一个基于分析合成（analysis-by-synthesis）的自动 3D 标注流水线，从单目图像重建完整 3D 场景以获取高质量 3D 包围框标注；基于此构建了 COCO3D 基准，覆盖 80 类日常物体，在开放词汇单目 3D 检测上显著提升性能。

## 研究背景与动机

单目 3D 目标检测是机器人、自动驾驶和 AR/VR 的基础任务。尽管 Cube R-CNN、OVMono3D 等方法取得进展，但核心瓶颈在于**大规模 3D 数据集的匮乏**：

**场景单一**：Omni3D（最大公开3D数据集）主要覆盖室内和自动驾驶场景，缺乏野外（in-the-wild）日常物体（如动物、食物）

**标注代价高**：3D 包围框标注比 2D 标注困难得多，且野外图像缺乏深度传感器

**现有自动标注方法受限**：
   - OVM3D-Det 依赖度量深度估计 + LLM 物体尺寸先验，对高类内变异物体（幼象 vs 成年象）失效
   - 3D Copy-Paste 插入合成3D模型，面临 sim-to-real 鸿沟

核心问题：**如何以最少人工监督从自然图像生成高质量 3D 标注？**

## 方法详解

### 整体框架

LabelAny3D 采用**分析合成范式**：从单目图像重建完整 3D 场景，再从重建场景中提取 3D 标注。

关键观察驱动设计：
1. 相对深度估计比度量深度更可靠和一致
2. 基于大规模 3D 数据集（Objaverse）的物体重建已足够精确
3. SAM、Grounding DINO 等 2D 基础模型在野外视觉场景泛化性极强

### 关键设计

#### 1. 图像超分辨率

MS-COCO 中许多物体因小尺度/压缩而分辨率低。使用 InvSR（扩散模型超分）提升4倍，$I^{SR} \in \mathbb{R}^{4H \times 4W \times 3}$，恢复细节以支持后续 3D 重建。

消融显示：去掉超分导致 $AP_{3D}$ 从 43.17 降至 28.13（-15.04），这是影响最大的组件。

#### 2. 2D 实例分割与过滤

使用 COCONut 数据集的高质量分割掩码（修正了 COCO 原始标注中的错误）。过滤规则：
- 排除截断物体（掩码与图像边界交集过大）
- 排除过小物体（掩码面积低于阈值）

#### 3. Amodal 补全 + 3D 重建

对被遮挡物体，使用 Gen3DSR 的扩散模型进行 amodal 补全（inpaint 缺失区域）。然后使用 **TRELLIS** 单视图 3D 重建，恢复归一化尺度的完整 3D 网格。

TRELLIS 比 DreamGaussian 提升 6.33 AP，生成更高保真度的重建。

#### 4. 场景几何估计

**双深度策略**：
- MoGe：仿射不变的相对深度估计（更准确的相对布局）
- Depth Pro：度量深度估计（提供绝对尺度参考）

将 MoGe 深度对齐到 Depth Pro 的度量尺度，再通过相机内参反投影为 3D 点云。

去掉 MoGe 导致 AP 从 43.17 降至 22.77（-20.4），证明相对深度是几何精度的核心。

#### 5. 位姿估计（2D-3D 对齐）

使用 MASt3R 计算真实图像与物体网格渲染视角之间的 2D-2D 稠密对应。通过已知渲染参数将 2D 匹配点反投影为 3D 点，然后用 PnP+RANSAC 求解相对位姿 $(R, T)$，将重建物体变换到图像坐标系。

PnP 方案优于 ICP（43.17 vs 24.28），因为 2D 匹配模型更鲁棒。

#### 6. 尺度估计与 3D 标注生成

通过掩码重叠区域的深度中值比 $s = \text{median}(D_{real}(\Omega) / D_{render}(\Omega))$ 恢复度量尺度。

最终均匀采样网格表面点云，PCA 估计主轴朝向，拟合紧凑 3D 包围框，得到中心位置、朝向和尺寸。

### 损失函数 / 训练策略

LabelAny3D 本身是标注流水线，不涉及训练。下游检测器 OVMono3D 的训练使用：

$$\mathcal{L} = \sqrt{2} \exp(-\mu) \mathcal{L}_{3D} + \mu$$

其中 $\mathcal{L}_{3D}$ 包含解耦的 3D 属性损失（2D 中心偏移、深度、尺寸、旋转）+ Chamfer 整体损失。$\mu$ 为不确定度评分。

训练数据：15,869 张 MS-COCO 训练图（LabelAny3D 自动标注，无人工精修）。

## 实验关键数据

### 主实验

**COCO3D 基准上的检测性能**（OVMono3D）：

| 训练数据 | AP₃D↑ | AR₃D↑ | AP₃D^Rel↑ | AR₃D^Rel↑ |
|----------|-------|-------|-----------|-----------|
| Omni3D (baseline) | 5.87 | 10.51 | 20.86 | 30.06 |
| OVM3D-Det* | 2.69 | 5.25 | 7.98 | 12.25 |
| LabelAny3D | 7.78 | 15.41 | 24.66 | 34.54 |
| Omni3D + LabelAny3D | **10.92** | **20.10** | **32.02** | **43.82** |

联合训练带来 +5.05 AP₃D 提升（Omni3D 自身仅 5.87）。

**伪标注质量对比（COCO3D）**：

| 方法 | AP₃D | AP₃D^15 | AP₃D^50 | AR₃D |
|------|------|---------|---------|------|
| OVM3D-Det | 10.03 | 16.88 | 1.44 | 17.82 |
| **LabelAny3D** | **64.17** | **82.11** | **57.34** | **73.57** |

在高 IoU 阈值（AP₃D^50）下优势更加显著（57.34 vs 1.44），说明标注精度远超 OVM3D-Det。

### 消融实验

| 配置 | AP₃D↑ |
|------|-------|
| 完整 LabelAny3D | **43.17** |
| 去掉超分辨率 | 28.13 (-15.04) |
| 去掉 MoGe | 22.77 (-20.40) |
| 去掉 Amodal 补全 | 39.22 (-3.95) |
| 用 DreamGaussian 替换 TRELLIS | 36.84 (-6.33) |
| 用 ICP 替换 PnP | 24.28 (-18.89) |
| 基线 Gen3DSR | 1.95 (-41.22) |

### 关键发现

1. **MoGe 相对深度是精度核心**：去掉后 AP 跌幅最大（-20.4），远超度量深度
2. **超分辨率意外重要**：小物体/压缩物体的细节恢复对 3D 重建至关重要
3. **2D 匹配+PnP 远优于 ICP**：几何对齐精度差异巨大（43.17 vs 24.28）
4. 在 KITTI Truck 类别上，LabelAny3D（32.74）大幅超越 OVM3D-Det（13.46），体现对高类内变异的优势
5. LabelAny3D 伪标注与人工精修之间的 AP₃D 为 64.17，意味着多数标注只需极少人工修正

## 亮点与洞察

1. **分析合成范式**：不直接预测 3D 框，而是先重建 3D 场景再提取标注，精度远超直接回归方案
2. **基础模型编排**：将 SAM、MoGe、Depth Pro、TRELLIS、MASt3R 等多个基础模型组装成流水线，充分利用各自优势
3. **COCO3D 基准填补空白**：首个覆盖 80 类日常野外物体的 3D 检测基准，推动开放词汇 3D 检测评估
4. **实用价值**：生成的伪标注已可直接用于训练检测器并获得显著提升

## 局限与展望

1. 基础模型在严重遮挡、无纹理、极小物体场景下仍会失败
2. TRELLIS 可能生成沿观测方向深度模糊的网格，导致与 RGBD 点云不对齐
3. 未穷举标注所有物体（排除了严重遮挡/截断的），不适合评估端到端检测器
4. 所有微调模型在 Omni3D 基础类别上出现性能退化（灾难性遗忘）
5. 度量深度仍依赖模型预测（非 GT），可能引入系统偏差

## 相关工作与启发

- **自动3D标注**：OVM3D-Det 用 LLM 尺寸先验，LabelAny3D 用 3D 重建，后者对类内变异更鲁棒
- **Model-in-the-loop 标注**：类似 Stereo4D、Cap3D，但 LabelAny3D 聚焦于单目 3D 框标注
- **开放词汇3D检测**：OVMono3D、DetAny3D 等受限于训练数据多样性，LabelAny3D 从数据端解决
- **启发**：基础模型的组合编排是扩展标注规模的可行路径，可推广到 6D 位姿、场景补全等任务

## 评分

- 新颖性: ⭐⭐⭐⭐ — 分析合成范式 + 多基础模型流水线，思路清晰有效
- 实验充分度: ⭐⭐⭐⭐ — COCO3D 基准 + 标注质量 + 下游检测 + 充分消融
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，流水线步骤详细
- 价值: ⭐⭐⭐⭐⭐ — 填补了野外3D标注和评估的空白，极具实用价值

<!-- RELATED:START -->

## 相关论文

- [OpenBox: Annotate Any Bounding Boxes in 3D](openbox_annotate_any_bounding_boxes_in_3d.md)
- [Detect Anything 3D in the Wild](../../ICCV2025/autonomous_driving/detect_anything_3d_in_the_wild.md)
- [Towards Predicting Any Human Trajectory in Context](towards_predicting_any_human_trajectory_in_context.md)
- [Difficulty-Aware Label-Guided Denoising for Monocular 3D Object Detection](../../AAAI2026/autonomous_driving/difficulty-aware_label-guided_denoising_for_monocular_3d_object_detection.md)
- [MonoSOWA: Scalable Monocular 3D Object Detector Without Human Annotations](../../ICCV2025/autonomous_driving/monosowa_scalable_monocular_3d_object_detector_without_human_annotations.md)

<!-- RELATED:END -->
