---
title: >-
  [论文解读] RAGNet: Large-scale Reasoning-based Affordance Segmentation Benchmark towards General Grasping
description: >-
  [ICCV 2025][图像分割][affordance 分割] 构建了首个大规模推理式 affordance 分割基准 RAGNet（273k 图像、180 类别、26k 推理指令），并提出 AffordanceNet 框架，将 VLM 预训练的 affordance 预测与抓取姿态生成相结合，展现出强大的开放世界泛化和推理能力。
tags:
  - ICCV 2025
  - 图像分割
  - affordance 分割
  - 机器人抓取
  - 推理指令
  - 大规模基准
  - 视觉语言模型
---

# RAGNet: Large-scale Reasoning-based Affordance Segmentation Benchmark towards General Grasping

**会议**: ICCV 2025  
**arXiv**: [2507.23734](https://arxiv.org/abs/2507.23734)  
**代码**: [GitHub](https://github.com/DongmingWu/RAGNet)  
**领域**: segmentation  
**关键词**: affordance 分割, 机器人抓取, 推理指令, 大规模基准, 视觉语言模型

## 一句话总结

构建了首个大规模推理式 affordance 分割基准 RAGNet（273k 图像、180 类别、26k 推理指令），并提出 AffordanceNet 框架，将 VLM 预训练的 affordance 预测与抓取姿态生成相结合，展现出强大的开放世界泛化和推理能力。

## 研究背景与动机

通用机器人抓取系统需要在多样化的开放世界场景中，根据人类指令准确感知物体的 affordance（可供性）区域。然而，现有研究存在两大瓶颈：

**数据规模与多样性不足**：现有 affordance 数据集通常局限于特定领域（机器人、第一人称视角、野外场景），类别有限（UMD 仅 17 类，AGD20k 仅 50 类），且图像来源单一，导致模型在开放世界中泛化能力有限

**缺乏推理式指令**：现有方法使用固定格式的模板化语言提示（如"请分割 xx 的 affordance 区域"），无法处理类人的高层指令（如"我需要能切面包的东西"），缺乏复杂推理能力

关键洞察：机器人在真实场景中接收到的指令往往不会直接指明目标物体类别，而是描述功能需求。这要求 affordance 预测模型具备从功能描述推理到物体识别再到区域分割的完整能力链。

## 方法详解

### 整体框架

本文贡献分为两部分：
1. **RAGNet 基准**：大规模多源数据收集 + 多工具 affordance 标注 + 三级推理指令构建
2. **AffordanceNet 模型**：AffordanceVLM（affordance 区域预测）+ Pose Generator（抓取姿态生成）

### 关键设计一：多源数据收集与标注

**数据来源覆盖四大领域**：
- **Wild 数据**：HANDAL（17 类五金/厨房工具），真实多样场景
- **Robot 数据**：Open-X（124 类，机器人操作场景）、GraspNet（32 类）
- **Ego-centric 数据**：EgoObjects（74 类室内第一人称）
- **Simulation 数据**：RLBench（10 类模拟环境）

总计 **273k 图像、180 类别**，远超此前最大数据集。

**五级 affordance 标注工具链**：
- ❶ **原始掩码**：直接使用数据集已有精细标注（如 HANDAL）
- ❷ **SAM2**：对无把手物体，用 GT bounding box 引导 SAM2 生成掩码
- ❸ **Florence2 + SAM2**：利用语言指令通过 Florence2 生成多边形框，再用 SAM2 精细化
- ❹ **VLPart + SAM2**：利用 VLPart 的部件级识别能力（如刀柄、杯柄），结合 SAM2 分割 affordance 区域
- ❺ **人工标注（+ SAM2）**：上述工具失败时的兜底方案

标注策略根据物体特性自适应：如苏打罐需整体标注（抓取整个物体），而炒锅只需标注把手。

### 关键设计二：三级推理指令体系

1. **模板式指令**：固定模板如 "Please segment the affordance map of \<category\> in this image"，可覆盖全部数据
2. **简单推理式指令（Easy）**：包含物体名称的推理指令，如 "Can you find a mug for tea"
3. **困难推理式指令（Hard）**：不包含物体名称，仅用功能描述，如 "I need something to drink coffee"

利用 GPT-4 生成推理指令，共生成 26k 条（HANDAL 8.5k hard + EgoObjects 12.7k easy + 4.7k hard）。

### 关键设计三：AffordanceNet 模型

**AffordanceVLM**：基于 VLM 的 affordance 区域预测模块
- 输入：RGB 图像 + 人类指令（模板式/推理式）
- 输出：affordance 分割掩码
- 在 RAGNet 大规模数据上预训练，学习从多样化指令到 affordance 区域的映射

**Pose Generator**：抓取姿态生成模块
- 输入：2D affordance 掩码 + 深度图像
- 输出：3D 抓取姿态（gripper pose）
- 将 VLM 的 2D 预测转化为可执行的机器人动作

### 验证基准设计

构建四类验证集：
- **零样本类别识别**：测试在未见类别上的泛化能力
- **跨领域 affordance 预测**：测试在未见图像领域上的预测能力
- **推理式指令验证**：使用不含类别名的困难指令测试推理能力
- **真实机器人抓取闭环测试**：完全跨域的实物操作验证

## 实验关键数据

### 数据规模对比

| 数据集 | 图像数 | 类别 | Wild | Robot | Ego | Sim | 推理 |
|--------|--------|------|------|-------|-----|-----|------|
| UMD (2015) | 10k | 17 | - | ✓ | - | - | - |
| AGD20k (2020) | 20k | 50 | ✓ | - | - | - | - |
| HANDAL (2023) | 200k | 17 | ✓ | - | - | - | - |
| ManipVQA (2024) | 84k | - | - | ✓ | ✓ | - | ✓ |
| **RAGNet** | **273k** | **180** | **✓** | **✓** | **✓** | **✓** | **✓** |

RAGNet 是首个同时覆盖所有四大领域并支持推理指令的大规模 affordance 基准。

### 标注工具组合（按数据源）

| 数据源 | 领域 | 标注工具 | 推理指令数 | 类别数 |
|--------|------|----------|-----------|--------|
| HANDAL | Wild | ❶ | 8.5k (hard) | 17 |
| Open-X | Robot | ❸❹❺ | - | 124 |
| GraspNet | Robot | ❶❺ | - | 32 |
| EgoObjects | Ego | ❷❹❺ | 17.4k | 74 |
| RLBench | Sim | ❺ | - | 10 |

### 关键发现

- 大规模多源数据的预训练显著提升了 affordance 预测的开放世界泛化能力
- 困难推理指令（不含类别名）虽然增加了任务难度，但有效提升了模型的功能推理能力
- 跨域验证表明模型可以从 wild 数据泛化到 robot 场景，反之亦然
- 真实机器人闭环实验验证了从 VLM affordance 预测到抓取执行的完整流程可行性
- 五级标注工具链的自适应设计大幅降低了人工标注成本，同时保证了标注质量

## 亮点与洞察

1. **基准贡献远大于方法贡献**：RAGNet 数据集本身（273k 图像、180 类别、26k 推理指令、四大领域覆盖）是该领域缺失的关键基础设施，将推动后续研究
2. **推理指令的分级设计**：模板式→简单推理→困难推理的三级体系，精准对应了从学术研究到真实部署的不同需求层次
3. **标注工具链的工程价值**：五级自适应标注策略（原始掩码 → SAM2 → Florence2+SAM2 → VLPart+SAM2 → 人工）是一套可复用的大规模标注方法论
4. **闭环验证思路**：从 VLM affordance 预测到 depth-based 姿态生成再到真实机器人执行，打通了感知-规划-执行的完整链路
5. **数据动机清晰**：通过系统对比现有数据集在领域、类别、推理能力三个维度的不足，构建了令人信服的数据需求论证

## 局限性

- AffordanceNet 模型架构相对简单，主要是 VLM 微调 + 姿态生成的串联，方法创新性有限
- 推理指令仅由 GPT-4 生成，未验证真实用户指令的多样性和复杂性
- 部分数据源的标注依赖自动化工具（SAM2/Florence2/VLPart），标注质量未经系统化定量评估
- 模拟数据（RLBench）仅 10 类，与其他数据源相比规模较小
- 真实机器人实验的场景和物体种类有限，未全面覆盖 180 个类别的抓取能力

## 相关工作

- **Affordance 数据集**：UMD 率先定义 affordance 分割并提供 10k RGB-D 图像；AGD20k 扩展到 36 affordance 类别的外部视角数据；HANDAL 提供精细的把手标注；AED 和 3DOI 引入更多场景；ManipVQA 首次引入推理但数据规模有限
- **Affordance 算法**：从监督学习到迁移学习再到自监督学习，逐步解决跨域泛化问题；近期 VLM 方法（AffordanceLLM、ManipVQA）引入语言推理但受限于数据规模
- **机器人抓取**：从传统的几何方法到学习方法，affordance 预测在抓取规划中的作用日益重要；本文首次在统一框架中打通 VLM affordance 预测到真实抓取的完整链路

## 评分

- 新颖性：⭐⭐⭐⭐ — 大规模推理式 affordance 基准填补了领域空白；困难推理指令（无类别名）的评估设置有创意
- 技术深度：⭐⭐⭐ — 模型部分（AffordanceNet）相对直接，主要贡献在数据和基准构建
- 实验充分度：⭐⭐⭐⭐ — 零样本/跨域/推理/真实机器人四类验证较全面；但部分实验细节需查看补充材料
- 写作质量：⭐⭐⭐⭐ — 数据构建的叙述系统完整，对比表格清晰
- 推荐度：⭐⭐⭐⭐ — 数据集资源对 affordance 和机器人抓取社区有重要价值

## 亮点与洞察

## 局限与展望

## 相关工作与启发

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Towards Affordance-Aware Robotic Dexterous Grasping with Human-like Priors](../../AAAI2026/segmentation/towards_affordance-aware_robotic_dexterous_grasping_with_human-like_priors.md)
- [\[CVPR 2025\] Scale Efficient Training for Large Datasets](../../CVPR2025/segmentation/scale_efficient_training_for_large_datasets.md)
- [\[CVPR 2026\] RealVLG-R1: A Large-Scale Real-World Visual-Language Grounding Benchmark for Robotic Perception and Manipulation](../../CVPR2026/segmentation/realvlg-r1_a_large-scale_real-world_visual-language_grounding_benchmark_for_robo.md)
- [\[ICCV 2025\] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation](towards_omnimodal_expressions_and_reasoning_in_referring_audio-visual_segmentati.md)
- [\[ICCV 2025\] HiMTok: Learning Hierarchical Mask Tokens for Image Segmentation with Large Multimodal Model](himtok_learning_hierarchical_mask_tokens_for_image_segmentation_with_large_multi.md)

<!-- RELATED:END -->
