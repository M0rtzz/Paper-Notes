---
title: >-
  [论文解读] CAD-Recode: Reverse Engineering CAD Code from Point Clouds
description: >-
  [ICCV 2025][3D视觉][CAD逆向工程] CAD-Recode 将 3D CAD 逆向工程问题转化为"点云→Python 代码"翻译任务，利用预训练 LLM 的 Python 代码理解能力作为解码器，结合轻量级点云投影器和百万级程序化生成数据集，在多个 CAD 数据集上显著超越现有方法，并支持 LLM 驱动的 CAD 编辑和问答。
tags:
  - ICCV 2025
  - 3D视觉
  - CAD逆向工程
  - 点云
  - Python代码生成
  - LLM解码器
  - sketch-extrude
---

# CAD-Recode: Reverse Engineering CAD Code from Point Clouds

**会议**: ICCV 2025  
**arXiv**: [2412.14042](https://arxiv.org/abs/2412.14042)  
**代码**: 有  
**领域**: 3D视觉 / CAD重建  
**关键词**: CAD逆向工程, 点云, Python代码生成, LLM解码器, sketch-extrude

## 一句话总结
CAD-Recode 将 3D CAD 逆向工程问题转化为"点云→Python 代码"翻译任务，利用预训练 LLM 的 Python 代码理解能力作为解码器，结合轻量级点云投影器和百万级程序化生成数据集，在多个 CAD 数据集上显著超越现有方法，并支持 LLM 驱动的 CAD 编辑和问答。

## 研究背景与动机

**领域现状**：CAD 逆向工程旨在从 3D 表示（如点云）重建参数化 sketch-extrude 操作序列。现有方法通常将 CAD 序列表示为离散 token 序列，使用自回归模型预测。

**现有痛点**：(1) 现有 CAD 序列表示（如命令序列）对预训练模型不友好，需要从头训练；(2) 生成的序列不可解释，难以进行后续编辑；(3) 训练数据有限（DeepCAD 仅约 18K）。

**核心矛盾**：CAD 序列本质是结构化程序，但现有方法将其视为任意 token 序列，未利用 LLM 对结构化代码的先验知识。

**本文目标**：将 CAD 序列表示为可执行的 Python 代码，利用 LLM 的代码理解能力实现更好的点云到 CAD 代码翻译。

**切入角度**：预训练 LLM 已大量接触 Python 代码，将 CAD 操作表示为 Python 可以天然利用这种先验。

**核心 idea**：用 Python 代码作为 CAD 序列的统一表示，使用预训练 LLM 解码器（带轻量级点云投影器）实现点云到 CAD 代码的端到端翻译。

## 方法详解

### 整体框架
输入：3D 点云。输出：可执行的 Python 代码（执行后重建 CAD 模型）。架构：点云编码器 → 轻量级投影器 → 预训练 LLM 解码器 → Python 代码。

### 关键设计

1. **Python 代码表示**:

    - 功能：用 Python 函数表示 CAD sketch-extrude 序列
    - 核心思路：每个 sketch-extrude 操作被编码为一个 Python 函数调用，包含草图顶点坐标、拉伸方向和距离等参数。整个 CAD 模型是一系列函数调用的组合。生成的代码可以直接被 Python CAD 库执行以重建 3D 模型
    - 设计动机：Python 代码对 LLM 来说是"母语"，利用 LLM 的代码理解能力；代码输出可被人类和其他 LLM 直接理解和编辑

2. **LLM 解码器 + 点云投影器**:

    - 功能：将点云信息注入 LLM 进行代码生成
    - 核心思路：使用较小的预训练 LLM（如 CodeLlama 等）作为解码器，在 LLM 前添加轻量级点云投影器将点云 embedding 映射到 LLM 的输入空间。仅训练投影器和 LoRA 适配器
    - 设计动机：直接利用 LLM 的代码生成先验，避免从头训练

3. **程序化生成百万级数据集**:

    - 功能：提供大规模训练数据
    - 核心思路：程序化生成 100 万个 CAD 序列及其对应 Python 代码和点云。通过控制 sketch 复杂度和 extrude 参数范围来覆盖多样化的 CAD 模式
    - 设计动机：现有 CAD 数据集规模小（DeepCAD 18K），百万级数据可充分训练模型

### 损失函数 / 训练策略
标准的自回归交叉熵损失（next-token prediction），在 Python 代码 token 上计算。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 | 之前SOTA | 说明 |
|--------|------|------|----------|------|
| DeepCAD | Coverage↑ | 显著提升 | - | 超越所有方法 |
| Fusion360 | 重建质量 | 最优 | - | 真实工业CAD |
| CC3D | 重建质量 | 最优 | - | 真实世界CAD |

### 消融实验

| 配置 | 表现 | 说明 |
|------|------|------|
| 随机初始化 LLM | 差 | LLM 预训练至关重要 |
| 无程序化数据预训练 | 中等 | 大规模数据显著提升 |
| 完整模型 | 最优 | LLM先验+大数据+Python表示 |

### 关键发现
- Python 代码表示比传统 token 序列更适合 LLM 解码
- 预训练 LLM 的 Python 代码先验对 CAD 生成至关重要
- 输出代码可被现成 LLM 直接编辑和问答——实现了 CAD 交互式编辑

## 亮点与洞察
- **Python 代码作为 CAD 表示的创意绝妙**：将几何重建问题转化为代码生成问题，天然利用 LLM 的海量代码训练数据。这种表示转换思路可迁移到其他结构化生成任务
- **可解释性优势**：生成的 Python 代码对人类可读，支持通过自然语言与 LLM 交互来编辑 CAD 模型
- **百万级程序化数据集**：解决了 CAD 逆向工程的数据瓶颈

## 局限与展望
- 目前仅支持 sketch-extrude 操作，未覆盖更复杂的 CAD 操作（如 fillet、chamfer）
- 对极复杂的工业 CAD 模型（数百个特征）可能力不从心
- LLM 解码器增加了推理成本

## 相关工作与启发
- **vs DeepCAD**: 使用自定义 token 表示和专用 Transformer。CAD-Recode 用 Python 代码利用了 LLM 先验，更高效且可解释
- **vs Point-E/Shape-E**: 通用 3D 生成方法，不生成可编辑的 CAD 参数。CAD-Recode 输出精确参数化表示

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Python 代码表示+LLM 解码的创意极具启发性
- 实验充分度: ⭐⭐⭐⭐ 三个数据集验证，含真实世界数据
- 写作质量: ⭐⭐⭐⭐ 思路清晰，方法简洁
- 价值: ⭐⭐⭐⭐⭐ 对 CAD 逆向工程和 LLM for 3D 领域都有重要意义

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] RayletDF: Raylet Distance Fields for Generalizable 3D Surface Reconstruction from Point Clouds or Gaussians](rayletdf_raylet_distance_fields_for_generalizable_3d_surface_reconstruction_from.md)
- [\[ICCV 2025\] Zero-Shot Inexact CAD Model Alignment from a Single Image](zero-shot_inexact_cad_model_alignment_from_a_single_image.md)
- [\[ICCV 2025\] DAP-MAE: Domain-Adaptive Point Cloud Masked Autoencoder for Effective Cross-Domain Learning](dap-mae_domain-adaptive_point_cloud_masked_autoencoder_for_effective_cross-domai.md)
- [\[ICCV 2025\] CMT: A Cascade MAR with Topology Predictor for Multimodal Conditional CAD Generation](cmt_a_cascade_mar_with_topology_predictor_for_multimodal_conditional_cad_generat.md)
- [\[ICCV 2025\] Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation](bridging_3d_anomaly_localization_and_repair_via_high-quality_continuous_geometri.md)

<!-- RELATED:END -->
