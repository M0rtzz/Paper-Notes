---
title: >-
  [论文解读] Novel Architecture of RPA in Oral Cancer Lesion Detection
description: >-
  [CVPR 2026][医学图像][口腔癌检测] 本文对比了低代码 RPA 平台（UiPath、Automation Anywhere）与基于 Python 设计模式（Singleton + Batch Processing）的口腔癌检测自动化方案，后者 (OC-RPAv2) 将单图推理时间从 2.5 秒压缩到 0.06 秒，实现 60-100 倍加速。
tags:
  - CVPR 2026
  - 医学图像
  - 口腔癌检测
  - RPA
  - EfficientNetV2B1
  - 设计模式
  - UiPath
---

# Novel Architecture of RPA in Oral Cancer Lesion Detection

**会议**: CVPR 2026  
**arXiv**: [2603.10928](https://arxiv.org/abs/2603.10928)  
**代码**: 无  
**领域**: 医学图像 / 自动化工作流  
**关键词**: 口腔癌检测, RPA, EfficientNetV2B1, 设计模式, UiPath

## 一句话总结

本文对比了低代码 RPA 平台（UiPath、Automation Anywhere）与基于 Python 设计模式（Singleton + Batch Processing）的口腔癌检测自动化方案，后者 (OC-RPAv2) 将单图推理时间从 2.5 秒压缩到 0.06 秒，实现 60-100 倍加速。

## 研究背景与动机

**领域现状**：口腔癌的早期精确检测对提高患者生存率至关重要。RPA（机器人流程自动化）已被应用于医疗工作流自动化，如影像处理、实验数据管理。UiPath 等低代码平台提供了易用的自动化接口。

**现有痛点**：低代码 RPA 平台在处理深度学习推理时效率低下——约 78% 的时间花在模型重加载、活动切换、数据序列化等开销上，仅 22% 用于真正的推理计算。处理 31 张口腔病变图像时 UiPath 需约 80 秒。

**核心矛盾**：RPA 平台的易用性与计算密集型 AI 推理的高效执行之间的矛盾。低代码环境的顺序执行模式和重复模型加载造成严重瓶颈。

**本文目标** 通过软件设计模式优化 Python 推理流水线，消除 RPA 平台的执行开销。

**切入角度**：将 Singleton 模式（模型只加载一次）和 Batch Strategy 模式（批量处理图像）组合到 Python 推理流程中。

**核心 idea**：用 Singleton 避免重复模型加载 + Batch Processing 实现批量推理，将口腔癌检测从 RPA 平台的 2.5s/图降至 0.06s/图。

## 方法详解

### 整体框架

使用在约 3000 张口腔临床图像上训练的 EfficientNetV2B1 进行 16 类口腔病变多分类。比较四种部署方案：UiPath RPA、Automation Anywhere RPA、OC-RPAv1（Python 逐图处理）、OC-RPAv2（Python + Singleton + Batch Processing）。

### 关键设计

1. **EfficientNetV2B1 分类模型**:

    - 功能：将口腔临床图像分为 4 大类（Healthy、Benign、OPMD、Oral Cancer）共 16 子类
    - 核心思路：以预训练 ImageNet 的 EfficientNetV2B1 为 backbone，输入 224×224×3。第一阶段冻结 base 层训练 15 个 epoch (lr=1e-3)，第二阶段解冻深层微调 10 个 epoch (lr=1e-5)。使用 Adam + categorical cross-entropy，配合 early stopping、ReduceLROnPlateau、checkpoint saving
    - 设计动机：EfficientNetV2 在轻量和精度间平衡好，适合部署在实际临床环境

2. **Singleton + Batch Processing 设计模式 (OC-RPAv2)**:

    - 功能：优化 Python 推理流水线以消除重复模型加载和单图串行瓶颈
    - 核心思路：Singleton 模式确保 EfficientNetV2B1 模型在整个推理过程中只加载一次并常驻内存。Batch Strategy 模式将所有待处理图像组织为批次，利用 GPU 并行推理。处理后的文件移至独立目录确保数据完整性
    - 设计动机：RPA 平台（UiPath/AA）每次预测都重新加载模型，这是 78% 开销的根源。Singleton 消除这一瓶颈，Batch 进一步利用 GPU 并行性

3. **RPA-Python 混合工作流**:

    - 功能：RPA 负责工作流编排（文件管理、日志、错误处理），Python 负责计算密集任务
    - 核心思路：UiPath 管理自动化流水线，调用 Python 函数执行模型推理。Try-Catch 块处理运行时异常，本地安全工作站处理以保护患者隐私
    - 设计动机：结合 RPA 的流程标准化/错误追踪/部署便利与 Python 的优化计算/并行处理能力

### 损失函数 / 训练策略

分类模型使用 categorical cross-entropy，数据增强包括翻转、仿射变换、旋转、色彩调整，每个样本 5 次增强。对少于 200 样本的类做随机重复。层次化增强处理类间不平衡。

## 实验关键数据

### 主实验

| 方案 | 31 图总时间 | 单图平均时间 | vs 本文最优 |
|------|-----------|------------|-----------|
| UiPath | 80 s | 2.58 s | 43× 慢 |
| Automation Anywhere | 75 s | 2.42 s | 40× 慢 |
| OC-RPAv1 (Python 逐图) | 8.65 s | 0.28 s | 4.7× 慢 |
| **OC-RPAv2 (Singleton+Batch)** | **1.96 s** | **0.06 s** | **基准** |

### 消融实验（性能分解）

| 优化手段 | 效果 | 说明 |
|---------|------|------|
| 仅消除模型重加载 (Singleton) | ~0.28 s/图 | OC-RPAv1，消除了主要开销 |
| + Batch 并行 | ~0.06 s/图 | OC-RPAv2，GPU 利用率进一步提升 |
| RPA 开销分析 | 78% 开销 | 模型加载、活动切换、数据序列化 |

### 关键发现
- RPA 平台 78% 时间花在非推理开销上，真正的模型推理只占 22%
- Singleton 模式是最关键优化（从 2.5s 降到 0.28s/图，消除 ~90% 开销）
- Batch Processing 在此基础上再提升 ~4.7 倍，但增益来自 GPU 并行化
- 2500 图场景：UiPath 需 1.8 小时，OC-RPAv2 不到 3 分钟

## 亮点与洞察
- **问题真实实用**：RPA 平台与深度学习推理的效率矛盾在实际医疗部署中普遍存在
- **解决方案简洁有效**：Singleton + Batch 是经典设计模式，但指出其在 RPA-AI 混合系统中的价值是有意义的

## 局限与展望
- 论文技术贡献偏工程化，设计模式应用本身不是新方法，更像系统优化报告
- 仅用 31 张测试图评估，规模太小，统计可信度有限
- 未报告分类精度指标（如 F1、AUC），只关注执行效率
- 未与其他高效推理方案（如 ONNX Runtime、TensorRT）对比
- 安全性和隐私合规讨论不充分

## 相关工作与启发
- **vs UiPath/Automation Anywhere 原生**: 低代码 RPA 平台适合非程序员但计算效率差，混合方案是更好折中
- **vs CLASEG 框架**: 本文使用的 EfficientNetV2B1 管线基于 CLASEG 框架的 16 类口腔病变分类
- **vs LMV-RPA**: Abdellaif 等的工作也探索了 RPA+Python 混合自动化，本文聚焦于设计模式的量化效果

## 评分
- 新颖性: ⭐⭐ 设计模式应用不算创新，主要价值在工程实践
- 实验充分度: ⭐⭐ 测试集仅31图，缺乏分类精度评估
- 写作质量: ⭐⭐ 存在重复段落，结构不够精炼
- 价值: ⭐⭐⭐ 对医疗 AI 部署中 RPA 效率瓶颈有实用参考价值

<!-- RELATED:START -->

## 相关论文

- [Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset](instruction-guided_lesion_segmentation_for_chest_x-rays_with_automatically_gener.md)
- [Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts](association_of_radiologic_ppfe_change_with_mortali.md)
- [The Invisible Gorilla Effect in Out-of-distribution Detection](the_invisible_gorilla_effect_in_out-of-distribution_detection.md)
- [MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection](moeclip_patch-specialized_experts_for_zero-shot_anomaly_detection.md)
- [InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](invad_inversion-based_reconstruction-free_anomaly_detection_with_diffusion_model.md)

<!-- RELATED:END -->
