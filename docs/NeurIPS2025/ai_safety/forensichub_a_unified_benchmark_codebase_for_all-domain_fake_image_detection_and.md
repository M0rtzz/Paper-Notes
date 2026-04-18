---
title: >-
  [论文解读] ForensicHub: A Unified Benchmark & Codebase for All-Domain Fake Image Detection and Localization
description: >-
  [NeurIPS 2025][AI安全][图像伪造检测] ForensicHub 提出首个统一所有域（Deepfake/IMDL/AIGC/文档篡改）的假图检测与定位基准平台，包含 4 个任务、23 个数据集、42 个模型、6 个骨干网络和 11 个 GPU 加速评估指标，通过模块化架构和适配器设计打破领域孤岛，并进行了 16 种跨域评估得出 8 条关键洞察。
tags:
  - NeurIPS 2025
  - AI安全
  - 图像伪造检测
  - 统一基准
  - Deepfake
  - AIGC检测
  - 文档篡改
---

# ForensicHub: A Unified Benchmark & Codebase for All-Domain Fake Image Detection and Localization

**会议**: NeurIPS 2025  
**arXiv**: [2505.11003](https://arxiv.org/abs/2505.11003)  
**代码**: [GitHub](https://github.com/scu-zjz/ForensicHub)  
**领域**: ai_safety  
**关键词**: 图像伪造检测, 统一基准, Deepfake, AIGC检测, 文档篡改

## 一句话总结
ForensicHub 提出首个统一所有域（Deepfake/IMDL/AIGC/文档篡改）的假图检测与定位基准平台，包含 4 个任务、23 个数据集、42 个模型、6 个骨干网络和 11 个 GPU 加速评估指标，通过模块化架构和适配器设计打破领域孤岛，并进行了 16 种跨域评估得出 8 条关键洞察。

## 研究背景与动机

**领域现状**：假图检测与定位 (FIDL) 已分化为四个相对独立的子领域：Deepfake 检测（人脸篡改）、IMDL（自然图像操纵检测/定位）、AIGC 检测（AI 生成图像检测）、文档图像篡改定位 (Doc)。

**现有痛点**：各领域独立构建数据集、模型和评估协议，形成严重的"领域孤岛"。虽有 DeepfakeBench 和 IMDLBenCo 等单域基准，但缺乏跨域统一基准。现有基准不兼容——DeepfakeBench 绑定人脸预处理，IMDLBenCo 要求像素级 mask，AIGCDetectBenchmark 不支持多 GPU 评估。

**核心矛盾**：四个域在数据格式、模型输出、评估标准上差异巨大，但它们共享许多底层技术（如预训练骨干、低级视觉特征、对比学习），割裂研究导致冗余和发展受阻。

**本文目标**：构建足够灵活可扩展的统一基准，兼容现有基准，填补 AIGC 和 Doc 缺少基准的空白，并提供跨域评估协议。

**切入角度**：模块化 + 配置驱动架构，适配器模式兼容已有基准。

**核心 idea**：通过可组合的 Dataset-Transform-Model-Evaluator 四模块架构和适配器设计，首次实现四个 FIDL 域的统一训练、测试和跨域评估。

## 方法详解

### 整体框架

ForensicHub 采用模块化架构，由四个核心组件构成：Datasets（数据加载）→ Transforms（预处理/增强）→ Models（检测模型）→ Evaluators（评估指标）。用户通过 YAML 配置即可构建训练/测试流水线，无需编码。

### 关键设计

1. **模块化配置驱动架构**：

    - 功能：将取证流水线分解为可互换的组件，支持任意域的 Dataset + Model + Evaluator 组合
    - 核心思路：每个 Dataset 返回统一的字段规范，Models 通过统一输出接口对接，Evaluators 覆盖图像级和像素级指标，11 个指标全部 GPU 加速
    - 设计动机：克服各域数据格式、模型输出类型和评估标准不统一的问题

2. **适配器设计 (Adapter-based Integration)**：

    - 功能：无缝集成 DeepfakeBench 和 IMDLBenCo 两个现有基准
    - 核心思路：通过适配器层将已有基准的模型和数据集转换为 ForensicHub 统一格式，无需重写代码
    - 具体覆盖：DeepfakeBench 的 27/34 个检测器（22 个支持跨域评估）+ IMDLBenCo 的全部 10 个模型
    - 设计动机：避免重复造轮子，让社区已有资产可以直接进入统一评估框架

3. **IFF-Protocol（图像取证融合协议）**：

    - 功能：建立跨域统一训练和测试协议
    - 核心思路：训练集混合四个域的数据（FaceForensics++, CASIAv2, GenImage, 多个文档数据集），每个 epoch 从各域等量采样（以最小数据集 CASIAv2 的 12,641 为基准）；测试时直接在各域数据集上评估，不做 fine-tuning
    - 设计动机：探索通用取证模型的可行性，评估模型在未见过的域上的泛化能力

4. **新增 AIGC 和 Document 基准**：

    - AIGC 基准：在 DiffusionForensics 上训练，在 GenImage 的 8 个生成模型上跨域测试
    - Document 基准：提供域内评估和 Doc Protocol（仅 Doctamper 训练→四个外部数据集测试）

### 平台规模
全平台：4 个任务、23 个数据集、42 个模型（含 3 个从头复现）、6 个骨干网络、11 个 GPU 加速指标、16 种跨域评估。

## 实验关键数据

### AIGC 检测基准：AUC (训练于 DiffusionForensics)

| 方法 | 域内 | ADM | BigGAN | Midjourney | SD V1.4 | 跨域平均 |
|------|------|-----|--------|-----------|---------|---------|
| DualNet | 1.000 | 0.999 | 0.917 | 0.850 | 0.813 | 0.890 |
| IML-ViT (IMDL) | 1.000 | 0.959 | 0.915 | 0.809 | 0.892 | **0.896** |
| HiFiNet | 1.000 | 1.000 | 0.841 | 0.721 | 0.677 | 0.816 |
| UnivFD | 0.995 | 0.840 | 0.969 | 0.543 | 0.708 | 0.792 |
| Trufor (IMDL) | 1.000 | 0.975 | 0.932 | 0.708 | 0.868 | 0.875 |

### 文档篡改检测基准：Binary-F1 (Doc Protocol, 仅 Doctamper 训练)

| 方法 | Doctamper 域内 | T-SROIE 跨域 | OSTF 跨域 | TPIC-13 跨域 | 跨域平均 |
|------|-------------|-------------|----------|------------|---------|
| FFDN | **0.813** | 0.533 | 0.241 | 0.357 | **0.300** |
| DTD | 0.743 | 0.525 | 0.124 | 0.284 | 0.247 |
| Cat-Net (IMDL) | 0.722 | **0.609** | 0.178 | 0.343 | 0.298 |
| PSCC-Net (IMDL) | 0.425 | 0.517 | **0.441** | **0.550** | **0.408** |

### 关键发现（8 条核心洞察精选）
- AIGC 模型在同类扩散模型上表现优秀，但对 Midjourney、Wukong 等不同体系生成模型泛化较差
- IMDL 模型（如 IML-ViT, Trufor）在 AIGC 跨域检测中竟然可以匹敌甚至超越专门的 AIGC 检测器，说明低级伪造痕迹具有跨域通用性
- 文档取证中 JPEG 压缩先验（DCT 系数/量化表）至关重要——FFDN、DTD、Cat-Net 都利用了此类特征
- PSCC-Net 的渐进式空间建模在文档跨域泛化中表现最佳
- 域内性能和跨域泛化往往不成正比，高域内 F1 不保证好的跨域迁移

## 亮点与洞察
- **破壁之作**：首次统一四个 FIDL 子领域，适配器设计让已有基准资产零改动接入，大幅降低社区迁移成本
- **IMDL 模型的跨域惊喜**：IMDL 检测器在 AIGC 和文档域上表现意外出色，暗示存在跨域通用的低级伪造特征空间，这为构建通用取证模型指明了方向
- **配置驱动的零代码评估**：YAML 配置即可组合任意 Dataset+Model+Evaluator，极大降低了基准实验的工程门槛

## 局限与展望
- 目前未包含视频级 Deepfake 检测（仅帧级）
- IFF-Protocol 的等量采样策略可能不是最优的域混合比例
- 部分域的跨域泛化仍然较差（如 AIGC 对 Midjourney），需要更好的域泛化方法
- 未探索多任务学习框架下的联合检测+定位模型

## 相关工作与启发
- **vs DeepfakeBench (Yan et al., 2023)**：专注 Deepfake 单域，与人脸预处理强耦合。ForensicHub 通过适配器兼容它并扩展到四个域
- **vs IMDLBenCo (Ma et al., 2024)**：专注 IMDL 单域，要求 mask 输出。ForensicHub 同样通过适配器集成
- **vs AIGCDetectBenchmark**：仅仓库级别的实验集合，缺乏跨域设计和统一评估。ForensicHub 提供了完整的模块化框架

## 评分
- 新颖性: ⭐⭐⭐⭐ 系统工程方面贡献巨大，但方法论创新有限（主要是基准构建）
- 实验充分度: ⭐⭐⭐⭐⭐ 42 个模型、23 个数据集、16 种跨域评估，规模空前
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但表格和数据量大导致阅读负担较重
- 价值: ⭐⭐⭐⭐⭐ 对 FIDL 领域有里程碑意义，将成为社区标准基础设施
