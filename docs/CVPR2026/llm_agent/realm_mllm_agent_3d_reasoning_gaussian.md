---
title: >-
  [论文解读] REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting
description: >-
  [CVPR 2026][LLM Agent][3D推理分割] 提出 REALM 框架，利用 MLLM 的推理能力通过全局到局部空间定位策略在 3DGS 上进行开放世界 3D 推理分割，无需 3D 后训练即可处理隐式指令，在 LERF 上 mIoU 达 92.88%，远超基线方法 40+ 个百分点，并支持物体移除、替换和风格迁移等编辑任务。
tags:
  - CVPR 2026
  - LLM Agent
  - 3D推理分割
  - MLLM-Agent
  - 3D高斯溅射
  - 全局到局部空间定位
  - 3D场景编辑
---

# REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting

**会议**: CVPR 2026  
**arXiv**: [2510.16410](https://arxiv.org/abs/2510.16410)  
**代码**: [https://ChangyueShi.github.io/REALM](https://ChangyueShi.github.io/REALM)  
**领域**: LLM Agent / 3D视觉  
**关键词**: 3D推理分割, MLLM-Agent, 3D高斯溅射, 全局到局部空间定位, 3D场景编辑

## 一句话总结
提出 REALM 框架，利用 MLLM 的推理能力通过全局到局部空间定位策略在 3DGS 上进行开放世界 3D 推理分割，无需 3D 后训练即可处理隐式指令，在 LERF 上 mIoU 达 92.88%，远超基线方法 40+ 个百分点，并支持物体移除、替换和风格迁移等编辑任务。

## 研究背景与动机

**领域现状**：让 AI 系统理解复杂人类指令并在 3D 场景中精确定位目标物体，是机器人和人机协作的基础能力。现有 3D 开放词汇分割方法（如 LERF、LangSplat、GS-Grouping）已能处理显式类别查询（如"分割杯子"），但面对需要推理的隐式指令（如"分割灯和书之间的物体"、"让桌子变得更整洁"）时表现不佳。

**现有痛点**：(1) 3D 分割方法缺乏推理能力——只能做显式关键词匹配，无法理解空间关系、语义属性或常识推理；(2) 2D MLLM 虽然擅长推理，但天然缺乏 3D 空间理解——直接将渲染视图输入 MLLM 对视角选择高度敏感，不同角度可能产生矛盾结果；(3) 已有尝试（如 ScanReason、ReasonGrounder）受限于预测 bounding box 或依赖俯视图，精度和适用性不足。

**核心矛盾**：MLLM 具备强大的 2D 推理能力但缺乏 3D 空间感知，如何在不对 MLLM 做 3D 特定后训练的情况下，稳定地将 2D 推理结果提升到 3D 空间并获得精确的分割 mask？

**本文目标**：实现一个能理解隐式推理指令、不需要 3D 后训练、且能生成精确 3D 分割 mask 的开放世界框架。

**切入角度**：以 3DGS 作为 3D 世界的高保真代理——它能渲染逼真的新视角供 MLLM 理解；通过全局到局部的两阶段多视角推理策略聚合不同视角的 MLLM 响应，消除单视角敏感性。

**核心 idea**：用 3DGS 渲染多视角给 MLLM 做推理，通过全局粗定位+局部精分割的两阶段策略获得精确 3D mask。

## 方法详解

### 整体框架
REALM 的 pipeline 包含三个核心模块：(1) **3D 特征场构建**——从 SAM 的 2D 实例 mask 出发，通过跨视角跟踪建立一致的实例 ID，然后为每个 3D Gaussian 学习实例特征 $f_i \in \mathbb{R}^D$，使其能通过分类器 $\mathcal{C}LS$ 映射到实例 ID；(2) **LMSeg（MLLM-Based Visual Segmenter）**——给定一张图像和语言查询，调用 MLLM 获取推理结果（bounding box + 类别 + 解释），再通过 SAM 生成 2D mask，结合特征场获得目标实例 ID；(3) **GLSpaG（全局到局部空间定位）**——先在全局视角进行粗定位，再在局部视角进行精细分割和 3D mask 优化。分割完成后可直接用于物体移除、替换、风格迁移等 3D 编辑任务。

### 关键设计

1. **3D 特征场与实例标识**:

    - 功能：为每个 Gaussian 原语学习实例特征，使 2D 分割结果可以稳定地映射到 3D 空间
    - 核心思路：利用 SAM 逐帧提取实例 mask，通过时序传播模型跨视角关联实例获得一致 ID。为每个 Gaussian $G_i$ 分配特征 $f_i$，通过 alpha blending 渲染 2D 特征图 $F = \sum_i f_i \alpha_i \prod_{j<i}(1-\alpha_j)$，再用分类器预测实例 ID：$\hat{id}(u,v) = \arg\max_k (CLS(F)_{u,v,k})$
    - 设计动机：建立 2D 到 3D 的稳定桥梁——LMSeg 在 2D 上推理得到目标，通过特征场可以直接确定对应的 3D Gaussian 集合，无需复杂的多视角 3D 融合

2. **全局空间定位（GLSpaG-Global）**:

    - 功能：从多个全局视角并行推理，通过投票聚合确定目标实例的粗定位
    - 核心思路：先用 K-means 聚类训练相机位姿得到 $N^{cluster}$ 个代表视角，再选取包含最多实例的 $N^{global}=8$ 个视角作为全局相机。对每个全局视角调用 LMSeg 获得目标实例 ID，通过多数投票确定最终目标 $ID^q = \arg\max_{c} |\{i: ID_i^q = c\}|$，据此在 3D 特征场中生成粗分割 mask $M^{3D}$
    - 设计动机：单视角推理对视角选择极其敏感（Fig.2 展示了 10 次随机视角的结果方差很大），多视角投票显著降低了随机性

3. **局部空间定位与精细化（GLSpaG-Local）**:

    - 功能：基于全局定位结果，在目标物体附近采样局部相机，获取精细 2D mask，通过优化将 3D mask 与局部 mask 对齐
    - 核心思路：从聚类代表相机中选取包含目标 ID 的视角作为局部相机；对每个局部视角调用 LMSeg 获得 2D mask；通过可微分光栅化将 3D mask $M^{3D}$ 渲染到各局部视角，用 L1 损失 $\mathcal{L}_{local} = \|\hat{M}_i - M_i^{2D\text{-Local}}\|_1$ 优化 3D mask 的边界精度，50 步迭代
    - 设计动机：全局阶段得到的粗 mask 边界不够精确，局部特写视角能提供更精细的分割信息，通过优化对齐可以显著改善 mask 质量

### 损失函数 / 训练策略
特征场训练阶段用交叉熵损失对齐渲染的实例 ID 与 SAM 的 ground truth ID。推理阶段的局部精细化用 L1 损失对齐 3D 渲染 mask 与 LMSeg 的 2D mask，仅需 50 步优化（3.67s）。整体框架无需 3D 特定的后训练，MLLM（Qwen-2.5-VL）和 SAM 均为现成预训练模型。

## 实验关键数据

### 主实验

| 数据集 | 指标 | REALM | 之前最佳 | 提升 |
|--------|------|-------|----------|------|
| LERF | mIoU | 92.88% | 44.82% (Gaga) | +48.06% |
| LERF | mBIoU | 90.12% | 42.37% (Gaga) | +47.75% |
| 3D-OVS | mIoU | 93.68% | 58.46% (GAGS) | +35.22% |
| 3D-OVS | mBIoU | 86.02% | 50.34% (GAGS) | +35.68% |
| REALM3D | mIoU | 82.30% | 65.55% (GS-Group) | +16.75% |
| REALM3D | mBIoU | 70.37% | 55.99% (GS-Group) | +14.38% |

注：以上对比均在 **隐式查询** 条件下。基线方法主要依赖 CLIP 关键词匹配，无法有效处理需要推理的隐式指令。

### 消融实验

| 配置 | mIoU | mBIoU | 说明 |
|------|------|-------|------|
| GS-Group (Baseline) | 0.32 | 0.30 | 无推理能力 |
| +Qwen2.5-VL | 0.78 | 0.77 | 加入MLLM推理但不稳定 |
| +Global Reasoning | 0.89 | 0.88 | 多视角投票消除随机性 |
| +Local Refinement | 0.95 | 0.94 | 边界精细化 |

全局相机采样策略消融（LERF Figurines）：

| 策略 | mIoU |
|------|------|
| 无K-means | 0.38 |
| K-means+Random选择 | 0.76 |
| 完全随机 | 0.59 |
| K-means+Top-K-ID (最终) | 0.95 |

推理效率：渲染速度 354.72 FPS；总推理时间 <10s（全局 MLLM 2.53s + 局部 MLLM 2.48s + 局部精细化 3.67s）。

### 关键发现
- REALM 在隐式查询上的优势极为显著（LERF 上 mIoU 超基线 48%），因为基线方法根本无法处理推理型指令
- 全局相机采样策略至关重要：K-means 聚类确保视角多样性 + Top-K-ID 选择确保覆盖更多物体，两步缺一不可
- 局部精细化步数 50 为最优——过少(10)精度不够，过多(500/1000)导致过拟合退化
- 聚类数 $N^{cluster}=24$ 表现最佳，过少(2)覆盖不足，过多(128)引入噪声

## 亮点与洞察
- 将 3DGS 作为"视角工厂"的思路非常优雅——MLLM 最擅长理解 2D 照片级图像，3DGS 恰好能渲染这种图像，两者天然互补
- 全局到局部的两阶段策略本质上是一种 coarse-to-fine 的空间注意力机制，从全场景的粗检索到局部的精分割，逐步缩小不确定性
- 投票聚合机制简单但有效，将单视角的高方差转化为多视角的低方差稳定输出
- 新提出的 REALM3D 基准（100+ 场景、1444 prompt-mask pairs、含隐式指令）填补了 3D 推理分割评估的空白
- 一个 Agent 框架同时支持分割、移除、替换、风格迁移等多种 3D 交互任务

## 局限与展望
- 依赖 3DGS 重建质量——如果 3DGS 重建不佳（纹理缺失、几何错误），渲染视角质量下降会级联影响 MLLM 理解
- MLLM 的推理上限决定了系统天花板——对于极复杂的推理链或高度抽象的指令可能失效
- 多视角推理增加延迟（8 个全局视角 + N 个局部视角），总时间约 8.68s，难以实时交互
- 局部精细化是逐视角的 L1 优化，视角数量有限时 3D mask 可能在未覆盖区域不够精确
- 当前 REALM3D 数据集虽然较大（100+ 场景），但隐式指令的难度和多样性仍有提升空间

## 相关工作与启发
- **vs LERF/LangSplat**: 这些方法在辐射场/3DGS 中嵌入 CLIP 语言特征，只能处理显式类别查询（"杯子"），无法理解"帮我找到会发光的东西"这类推理指令；REALM 通过 MLLM 推理突破了这个限制
- **vs GS-Grouping/Gaga**: 基于对比学习分组 3D 实例，查询也局限于显式词汇；在隐式查询上 REALM mIoU 比 Gaga 高 48%
- **vs ReasonGrounder**: 概念上最接近，但 ReasonGrounder 依赖俯视图且只预测 bounding box，REALM 提供精细 mask 且不受限于特定视角
- **vs 2D 推理分割 (LISA)**: LISA 等方法在 2D 上做推理分割但缺乏 3D 一致性；REALM 通过多视角聚合+特征场保证 3D 一致

## 评分
- 新颖性: ⭐⭐⭐⭐ MLLM-Agent + 3DGS 的组合新颖，全局到局部空间定位策略设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 三个基准+详尽消融+效率分析+多种下游任务，极为全面
- 写作质量: ⭐⭐⭐⭐ 动机清晰，Fig.2 的视角敏感性可视化直观有力
- 价值: ⭐⭐⭐⭐ 开创了 3D 推理分割方向，REALM3D 基准对社区有长期价值

<!-- RELATED:START -->

## 相关论文

- [SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation](sceneassistant_a_visual_feedback_agent_for_open-vocabulary_3d_scene_generation.md)
- [WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning](worldmm_dynamic_multimodal_memory_agent_for_long_video_reasoning.md)
- [Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code](nerfify_a_multi-agent_framework_for_turning_nerf_papers_into_code.md)
- [CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare](carepilot_a_multi-agent_framework_for_long-horizon_computer_task_automation_in_h.md)
- [Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding](think_then_verify_a_hypothesis-verification_multi-agent_framework_for_long_video.md)

<!-- RELATED:END -->
