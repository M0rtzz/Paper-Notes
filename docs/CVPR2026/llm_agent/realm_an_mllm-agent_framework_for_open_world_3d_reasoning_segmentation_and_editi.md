---
title: >-
  [论文解读] REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting
description: >-
  [CVPR 2026][LLM Agent][3D推理分割] 提出 REALM 框架，通过 MLLM agent 对 3D 高斯泼溅(3DGS)渲染的视图进行推理分割，设计全局-局部空间接地策略(GLSpaG)聚合多视角MLLM推理结果，在隐式指令下的3D分割中大幅超越现有方法（LERF上mIoU 92.88% vs 基线44.82%），并支持3D编辑。
tags:
  - CVPR 2026
  - LLM Agent
  - 3D推理分割
  - 多模态大语言模型
  - 高斯泼溅
  - 全局-局部接地
  - 3D编辑
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting

**会议**: CVPR 2026  
**arXiv**: [2510.16410](https://arxiv.org/abs/2510.16410)  
**代码**: [项目页面](https://ChangyueShi.github.io/REALM)  
**领域**: LLM Agent  
**关键词**: 3D推理分割, 多模态大语言模型, 高斯泼溅, 全局-局部接地, 3D编辑

## 一句话总结
提出 REALM 框架，通过 MLLM agent 对 3D 高斯泼溅(3DGS)渲染的视图进行推理分割，设计全局-局部空间接地策略(GLSpaG)聚合多视角MLLM推理结果，在隐式指令下的3D分割中大幅超越现有方法（LERF上mIoU 92.88% vs 基线44.82%），并支持3D编辑。

## 研究背景与动机

**领域现状**：3D开放词汇分割方法（如LERF、GS-Group）能处理显式查询（"分割杯子"），但无法理解需要推理的隐式指令（"分割灯和书之间的物体"）。MLLM在2D推理上表现优秀但缺乏3D空间理解。

**现有痛点**：直接将一张或几张渲染视图输入MLLM对视角选择高度敏感——次优视角可能遮挡目标物体。同时输入大量视图又会压垮MLLM，无法建立一致的3D理解。

**核心矛盾**：MLLM有强大的2D推理能力但无3D感知；3D分割方法有空间理解但无推理能力。如何桥接？

**本文目标** 利用现成MLLM的推理能力实现开放世界的3D推理分割，无需3D特定后训练。

**切入角度**：用3DGS作为3D世界的高保真代理，渲染光照逼真的新视角供MLLM理解。通过全局-局部两阶段策略聚合多视角推理结果。

**核心 idea**：全局多视角粗定位+局部特写精分割的两阶段策略，将MLLM的2D推理能力提升到3D领域。

## 方法详解

### 整体框架
输入：3DGS场景 + 自然语言指令（可以是隐式的推理指令）。输出：3D分割mask + 可选的3D编辑结果。方法分三部分：(1) 3D特征场构建（SAM+跨视图传播→Gaussian实例特征）；(2) LMSeg（MLLM-based图像级推理分割agent）；(3) GLSpaG（全局-局部空间接地策略）。

### 关键设计

1. **3D特征场 (3D Feature Field)**:

    - 功能：为每个Gaussian primitive分配一致的实例ID
    - 核心思路：用SAM提取每帧实例mask，用时序传播模型跨视图关联实例得到一致$id_i$。给每个Gaussian附加特征$f_i \in \mathbb{R}^D$，通过alpha blending渲染到2D特征图$F$，训练分类器$\mathcal{CLS}$将特征映射到实例ID
    - 设计动机：MLLM的推理结果需要链接回3D空间，实例特征场是2D推理与3D分割的桥梁

2. **MLLM-Based Visual Segmenter (LMSeg)**:

    - 功能：对单视图进行语言引导的推理分割
    - 核心思路：给定渲染图像$\mathcal{I}$和查询$q$，MLLM返回 (bounding box $\mathcal{B}$, 类别$\mathcal{C}$, 解释$\mathcal{E}$)。$\mathcal{B}$输入SAM得到2D mask $M^{2D}$。将mask与实例特征场中的$\hat{id}$交叉，确定目标实例ID
    - 设计动机：将MLLM的推理能力与SAM的精确分割能力结合——MLLM负责"理解"，SAM负责"画mask"

3. **全局阶段 (GLSpaG - Global)**:

    - 功能：从多个全局视角粗定位目标物体
    - 核心思路：用K-means聚类训练相机位姿得到候选视角，选择包含最多实例的top-$N^{\text{global}}$个视角。对每个全局视角并行运行LMSeg，通过投票聚合得到目标实例ID：$ID^q = \arg\max_c |\{i: ID_i^q = c\}|$。用分类器将匹配的3D Gaussian分组得到粗3D mask
    - 设计动机：并行多视角推理+投票比依赖单一视角鲁棒得多。K-means选视角确保覆盖性

4. **局部阶段 (GLSpaG - Local)**:

    - 功能：用目标物体的近景视图精细化3D mask
    - 核心思路：从聚类视角中选择包含目标ID的视角作为局部相机，运行LMSeg得到精细2D mask $M_i^{2D-Local}$。通过可微渲染将3D mask投影到2D，与LMSeg mask对齐优化：$\mathcal{L}_{\text{local}} = \|\hat{M}_i - M_i^{2D-Local}\|_1$，迭代50步
    - 设计动机：全局阶段的粗mask可能包含噪声，局部特写视图的精细分割+渲染对齐可以修正边界

### 损失函数 / 训练策略
特征场训练：2D实例标签监督。局部精细化：L1渲染mask对齐loss, 50步优化。全程不需要MLLM微调（使用Qwen-2.5-VL）。可在单RTX 3090上运行。

## 实验关键数据

### 主实验

**隐式指令3D分割 (mIoU %)**

| 方法 | LERF | 3D-OVS | REALM3D |
|------|------|--------|---------|
| Gaga | 44.82 | 42.53 | 58.56 |
| GAGS | 17.84 | 58.46 | 52.24 |
| GS-Group | 42.43 | 41.79 | 65.55 |
| **REALM** | **92.88** | **93.68** | **82.30** |

### 消融实验

| 配置 | mIoU | 说明 |
|------|------|------|
| GS-Group 基线 | ~43% | 无推理能力 |
| + LMSeg (单视图) | 有提升但不稳定 | 视角敏感 |
| + Global阶段 | 显著提升 | 多视角投票鲁棒 |
| + Local阶段 | **92.88%** | 精细化修正边界 |

### 关键发现
- REALM在隐式指令上mIoU达到92.88%（LERF），是基线方法的2倍+。说明MLLM的推理能力对3D分割有巨大价值
- 全局-局部策略比直接输入多视图到MLLM效果好得多——直接输入多视图时MLLM无法解决歧义
- 新提出的REALM3D数据集含100+场景、1444个隐式prompt-mask对，填补了3D推理分割评估的空白
- 支持下游3D编辑（移除/替换/风格迁移），展示了实用性

## 亮点与洞察
- **全局-局部两阶段设计极其直觉**：先"远看"确定是哪个物体（投票鲁棒），再"近看"精细分割，完美模拟人类的视觉搜索策略
- **无需3D特定训练**：完全利用现成MLLM+SAM，通过系统设计而非模型训练解决问题。意味着随着VLM能力提升，REALM的性能自动提升
- **REALM3D数据集是重要贡献**：100+场景1444个隐式prompt-mask对，首次为3D推理分割提供定量评估基准

## 局限与展望
- 依赖预训练的3DGS质量——如果3DGS重建失败或质量差，整个流程会受影响
- MLLM推理需要多次调用（全局$N^{\text{global}}$次 + 局部$N^{\text{local}}$次），延迟较高
- 投票机制假设多数视角能正确识别目标，对遮挡严重的物体可能失败
- 未测试在大规模户外场景中的表现

## 相关工作与启发
- **vs ReasonGrounder**: ReasonGrounder依赖俯视图，限制了在复杂3D环境中的适用性。REALM用多视角全局-局部策略更通用
- **vs ScanReason/VGMamba**: 这些方法只能预测3D bounding box，REALM输出精细的3D mask
- **vs SceneAssistant (本批)**: 两者都用VLM agent处理3D场景，但任务不同——REALM做分割+编辑，SceneAssistant做生成

## 评分
- 新颖性: ⭐⭐⭐⭐ 全局-局部两阶段策略设计优雅，但整体框架基于现有组件的组合
- 实验充分度: ⭐⭐⭐⭐⭐ 三个基准（含新提出的REALM3D）、消融完整、编辑演示丰富
- 写作质量: ⭐⭐⭐⭐ 清晰系统，公式化规范
- 价值: ⭐⭐⭐⭐⭐ 性能碾压（2倍+mIoU），新数据集，实用的3D编辑能力

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation](sceneassistant_a_visual_feedback_agent_for_open-vocabulary_3d_scene_generation.md)
- [\[CVPR 2026\] WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning](worldmm_dynamic_multimodal_memory_agent_for_long_video_reasoning.md)
- [\[CVPR 2026\] Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code](nerfify_a_multi-agent_framework_for_turning_nerf_papers_into_code.md)
- [\[CVPR 2026\] CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare](carepilot_a_multi-agent_framework_for_long-horizon_computer_task_automation_in_h.md)
- [\[CVPR 2026\] Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding](think_then_verify_a_hypothesis-verification_multi-agent_framework_for_long_video.md)

<!-- RELATED:END -->
