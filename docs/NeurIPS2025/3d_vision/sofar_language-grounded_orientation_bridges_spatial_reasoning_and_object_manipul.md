---
title: >-
  [论文解读] SoFar: Language-Grounded Orientation Bridges Spatial Reasoning and Object Manipulation
description: >-
  [NeurIPS 2025 Spotlight][3D视觉][semantic orientation] 提出"语义朝向"(Semantic Orientation)概念，用自然语言描述物体方向（如 USB 的"插入方向"、杯子的"把手方向"），构建 OrienText300K 大规模数据集训练 PointSO 模型实现零样本朝向预测，并集成为 SoFar 系统实现 6-DoF 场景理解与机器人操作。
tags:
  - "NeurIPS 2025 Spotlight"
  - "3D视觉"
  - "semantic orientation"
  - "6-DoF manipulation"
  - "spatial reasoning"
  - "3D scene graph"
  - "点云"
---

# SoFar: Language-Grounded Orientation Bridges Spatial Reasoning and Object Manipulation

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2502.13143](https://arxiv.org/abs/2502.13143)  
**作者**: Zekun Qi, Wenyao Zhang, Yufei Ding, Runpei Dong 等 (清华大学, 上海交大, Galbot, 北京大学, UIUC, 上海科技大学)
**代码**: [SoFar](https://github.com/SoFar-LGO/SoFar)  
**领域**: 3D视觉  
**关键词**: semantic orientation, 6-DoF manipulation, spatial reasoning, 3D scene graph, point cloud

## 一句话总结
提出"语义朝向"(Semantic Orientation)概念，用自然语言描述物体方向（如 USB 的"插入方向"、杯子的"把手方向"），构建 OrienText300K 大规模数据集训练 PointSO 模型实现零样本朝向预测，并集成为 SoFar 系统实现 6-DoF 场景理解与机器人操作。

## 研究背景与动机

现有空间推理方法主要关注物体**位置关系**（左/右/前/后），而忽略了物体**朝向**——这是 6-DoF 精细操作的关键因素。例如：
- 用刀切面包需要知道刀刃方向
- 扶正倾倒的酒杯需要知道杯口朝上的方向
- 插充电器需要对准插孔方向

传统朝向表示使用四元数/欧拉角，依赖预定义参考系或模板模型，存在显著局限：

**缺乏语义关联**：旋转矩阵无法表达"把手方向""插入方向"等功能性语义

**泛化性差**：需要为每类物体预定义坐标框架，无法处理开放世界中的新物体

**VLM 能力缺失**：当前视觉语言模型（GPT-4V、LLaVA 等）难以理解和输出精确的物体朝向

核心洞察：人类用自然语言直觉地描述朝向（如"刀刃朝下"），这种**语义化的朝向表示**可以桥接几何推理与功能语义，实现无参考系的开放词汇朝向理解。

## 方法详解

### 语义朝向定义
对物体 $X$ 和语言描述 $\ell$，语义朝向定义为单位方向向量：
$$\mathbf{s}_{\ell}^{X} = \mathcal{F}(X, \ell) \in S(2)$$
其中 $\ell$ 可以是通用方向（前/上）、物体部件（把手/瓶盖）或交互动作（倒出/插入）。一个物体可关联多个语义朝向，形成集合 $S_X = \{\mathbf{s}_{\ell_1}^X, \ldots, \mathbf{s}_{\ell_n}^X\}$。

### OrienText300K 数据集
从 Objaverse（约 800K 个 3D 模型）出发，经过严格过滤和标注构建：

**数据过滤**（6 条准则）：标准正交视图、无地面辅助、合理物体、高质量、可辨识、非场景级。使用 GPT-4o 自动过滤，过滤准确率 88.3%。

**数据标注**：利用 GPT-4o 作为判别器，解读六视图语义内容生成语义-视图配对。标注准确率 97.1%。

最终得到 350K+ 干净样本，渲染 8M+ 高质量图像。

### PointSO 模型架构
基于 plain Transformer 的跨模态 3D-语言融合模型：

1. **3D 编码**：FPS 采样种子点 → KNN 分组 → 轻量 PointNet 提取局部几何特征
2. **语言编码**：CLIP（冻结）提取全局文本特征
3. **跨模态融合**：将文本 token 逐元素加到每层的点 token 上（简单但效果最佳）
4. **预测头**：MLP 将 [CLS] token 映射为方向向量
5. **损失函数**：负余弦相似度 $\mathcal{L}_{\text{cos}}$

### SoFar 系统框架
整合 PointSO 与基础模型（SAM、Florence-2）的完整推理系统：

**6-DoF 场景图构建**：
1. VLM 从语言查询提取任务相关物体短语
2. Florence-2 + SAM 执行语言条件分割，获取 3D 点云
3. VLM 生成朝向描述，PointSO 预测语义朝向
4. 构建场景图 $\mathcal{G} = (\mathbf{V}, \mathbf{E})$，节点含物体 ID、3D 位置、包围盒、语义朝向集合

**链式思维空间推理**：
1. 分析场景与查询，识别相关物体
2. 计算目标位置和朝向
3. 平移 $\mathbf{t}_i = \tilde{\mathbf{c}}_i - \mathbf{c}_i$，旋转通过 Kabsch-Umeyama 算法从初始和目标语义朝向估计

**低层运动执行**：GSNet 生成抓取候选 → 选择最优抓取 → OMPL 规划无碰撞轨迹

## 实验关键数据

### Table 1: Open6DOR 6-DoF 物体重排评估

| 方法 | 位置 Level 0 | 位置 Level 1 | 位置总体 | 旋转 Level 0 | 旋转 Level 1 | 旋转 Level 2 | 旋转总体 | 6-DoF 位置 | 6-DoF 旋转 | 6-DoF 总体 | 耗时 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-4V | 46.8 | 39.1 | 45.2 | 9.1 | 6.9 | 11.7 | 9.2 | - | - | - | - |
| Dream2Real | 17.2 | 11.0 | 15.9 | 37.3 | 27.6 | 26.2 | 31.3 | 26.2 | 18.7 | 13.5 | 358.3s |
| Open6DOR-GPT | 78.6 | 60.3 | 74.9 | 45.7 | 32.5 | 49.8 | 41.1 | 84.8 | 40.0 | 35.6 | 126.3s |
| SoFar-LLaVA | 86.3 | 57.9 | 78.7 | 62.5 | 30.2 | 67.1 | 48.6 | 83.0 | 48.2 | 40.3 | 9.6s |
| **SoFar** | **96.0** | **81.5** | **93.0** | **68.6** | **42.2** | **70.1** | **57.0** | **92.7** | **52.7** | **48.7** | **8.5s** |

SoFar 在感知任务上全面领先：位置总体 93.0%（vs Open6DOR-GPT 74.9%，+18.1%）；旋转总体 57.0%（+15.9%）；6-DoF 综合 48.7%（+13.1%）；推理时间仅 8.5s（vs 126.3s，加速约 15 倍）。

执行任务（Open6DOR V2）上，SoFar 同样超越 Octo 和 OpenVLA，6-DoF 总体成功率 18.4% vs 8.0%/8.2%。

### Table 4: SimplerEnv Google Robot 评估（成功率）

| 方法 | 训练数据 | Pick Coke Can Avg | Move Near Avg | Open/Close Drawer Avg | 总体平均 |
|---|---|---|---|---|---|
| RT-1-X | OXE | 0.490 | 0.323 | 0.294 | 0.397 |
| RT-2-X | OXE | 0.823 | 0.792 | 0.353 | 0.661 |
| OpenVLA | OXE | 0.545 | 0.477 | 0.177 | 0.411 |
| **SoFar** | **Zero-Shot** | **0.907** | **0.740** | **0.297** | **0.676** |

Visual Matching 设置下 SoFar 达到总体 **74.9%** 成功率，超越所有基线（含使用 OXE 训练数据的方法），且为零样本迁移。Pick Coke Can 任务上 SoFar 达 92.3%。

### Table 2: PointSO 语义朝向预测精度

| 模型 | 45° | 30° | 15° | 5° | 平均 |
|---|---|---|---|---|---|
| PointSO-S | 77.34 | 74.22 | 67.97 | 60.94 | 70.12 |
| PointSO-B | 79.69 | 77.34 | 70.31 | 62.50 | 72.46 |
| PointSO-L | 81.25 | 78.13 | 72.66 | 65.63 | 74.42 |

即使在 5° 严格阈值下，PointSO-L 仍有 65.63% 准确率。在鲁棒性测试中（单视图、噪声、随机旋转），模型保持较好性能。

### Table 6: 6-DoF SpatialBench 空间推理评估

| 方法 | 位置相对 | 位置绝对 | 朝向相对 | 朝向绝对 | 总体 |
|---|---|---|---|---|---|
| GPT-4o | 49.4 | 28.4 | 44.2 | 25.8 | 36.2 |
| SpatialBot | 50.9 | 21.6 | 39.6 | 22.9 | 32.7 |
| RoboPoint | 43.8 | 30.8 | 33.8 | 25.8 | 33.5 |
| **SoFar** | **59.6** | **33.8** | **54.6** | **31.3** | **43.9** |

SoFar 总体 43.9%，较次优方法提升超过 **18%**（相对增幅），在朝向推理上优势尤为明显。

## 亮点
- **语义朝向概念**：首次将物体朝向与自然语言语义绑定，提出无参考系、开放词汇的朝向表示，填补了位置推理和朝向推理之间的空白
- **大规模数据+自动标注**：OrienText300K（350K+ 模型、8M+ 图像）通过 GPT-4o 自动过滤和标注，避免昂贵的机器人数据采集，标注准确率 97.1%
- **零样本泛化**：SoFar 在 Open6DOR、SimplerEnv、真实世界任务上均为零样本，却超越使用大规模机器人轨迹训练的 VLA 模型（Octo、OpenVLA）
- **跨具身泛化**：支持夹爪、吸盘、灵巧手等不同末端执行器，以及操作、导航、VQA 等多任务
- **推理效率**：8.5s 推理时间，比 Open6DOR-GPT 快 15 倍，比 Dream2Real 快 42 倍

## 局限性
- **解耦系统的级联误差**：各子模块（分割、抓取、朝向预测）的误差会累积，如不稳定抓取导致放置时物体旋转偏移
- **执行成功率仍有提升空间**：Open6DOR V2 执行任务中 6-DoF 成功率仅 18.4%，部分物体本身难以操作
- **依赖深度输入**：需要 RGB-D 图像，纯 RGB 场景下需额外深度估计模块
- **闭环能力有限**：当前通过 VLM 错误检测+重试（最多 3 次）近似闭环，非真正的闭环控制
- **OrienText300K 来源单一**：数据来源于 Objaverse，可能存在物体类别分布偏差

## 相关工作
- **空间推理**：SpatialVLM、SpatialBot、RoboPoint 等侧重位置关系，未建模朝向；SoFar 将推理扩展到完整 6-DoF
- **机器人操作**：RT-1/RT-2（端到端策略）、VoxPoser（3D 值图）、CoPa（可泛化操作）均未显式建模语义朝向
- **VLA 模型**：Octo、OpenVLA、SpatialVLA 等需要大规模机器人数据训练，泛化性受限；SoFar 零样本即优于它们
- **物体姿态估计**：传统方法依赖 CAD 模型或类别级模板，无法处理开放世界新物体；语义朝向提供了更灵活的替代方案
- **场景图**：传统场景图编码物体关系但缺少朝向信息；SoFar 的 6-DoF 场景图首次融入语义朝向

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 语义朝向概念具有原创性，将自然语言与3D方向向量绑定的思路新颖且优雅
- 实验充分度: ⭐⭐⭐⭐⭐ — 覆盖仿真(Open6DOR V1/V2, SimplerEnv)+真实世界(60任务100+物体)+VQA(SpatialBench)，多基线对比，消融完整
- 写作质量: ⭐⭐⭐⭐ — 概念清晰，图示丰富，系统描述完整；但内容量大导致部分细节压缩
- 价值: ⭐⭐⭐⭐⭐ — 为6-DoF机器人操作提供了新的表示范式，数据集和基准有望推动后续研究

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Orientation Matters: Making 3D Generative Models Orientation-Aligned](orientation_matters_making_3d_generative_models_orientation-aligned.md)
- [\[CVPR 2026\] Scalable Object Relation Encoding for Better 3D Spatial Reasoning in Large Language Models](../../CVPR2026/3d_vision/scalable_object_relation_encoding_for_better_3d_spatial_reasoning_in_large_langu.md)
- [\[NeurIPS 2025\] Orientation-anchored Hyper-Gaussian for 4D Reconstruction from Casual Videos](orientation-anchored_hyper-gaussian_for_4d_reconstruction_from_casual_videos.md)
- [\[NeurIPS 2025\] Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations](concerto_joint_2d-3d_self-supervised_learning_emerges_spatial_representations.md)
- [\[CVPR 2026\] Masking Matters: Unlocking the Spatial Reasoning Capabilities of LLMs for 3D Scene-Language Understanding](../../CVPR2026/3d_vision/masking_matters_unlocking_the_spatial_reasoning_capabilities_of_llms_for_3d_scen.md)

</div>

<!-- RELATED:END -->
