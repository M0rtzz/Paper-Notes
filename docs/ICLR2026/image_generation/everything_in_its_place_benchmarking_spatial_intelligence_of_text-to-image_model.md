---
title: >-
  [论文解读] Everything in Its Place: Benchmarking Spatial Intelligence of Text-to-Image Models
description: >-
  [ICLR 2026][图像生成][空间智能] 提出 SpatialGenEval 基准，通过 1,230 条长且信息密集的提示覆盖 10 个空间子领域，系统评估 23 个 SOTA T2I 模型的空间智能，揭示空间推理是主要瓶颈；同时构建 SpatialT2I 数据集实现数据中心的空间智能提升。
tags:
  - ICLR 2026
  - 图像生成
  - 空间智能
  - 文本到图像生成
  - 基准评测
  - 信息密集提示
  - 数据中心范式
---

# Everything in Its Place: Benchmarking Spatial Intelligence of Text-to-Image Models

**会议**: ICLR 2026  
**arXiv**: [2601.20354](https://arxiv.org/abs/2601.20354)  
**代码**: 有 ([GitHub](https://github.com/AMAP-ML/SpatialGenEval))  
**领域**: 图像生成  
**关键词**: 空间智能, 文本到图像生成, 基准评测, 信息密集提示, 数据中心范式

## 一句话总结

提出 SpatialGenEval 基准，通过 1,230 条长且信息密集的提示覆盖 10 个空间子领域，系统评估 23 个 SOTA T2I 模型的空间智能，揭示空间推理是主要瓶颈；同时构建 SpatialT2I 数据集实现数据中心的空间智能提升。

## 研究背景与动机

当前 T2I 模型在生成高保真图像方面表现优异，能够准确渲染场景中"是什么"（what），但在精确描绘物体"在哪里"（where）、"如何排列"（how）和"为什么交互"（why）等空间关系方面频繁失败。即使是 GPT-Image-1、Qwen-Image 等 SOTA 模型也会出现物体错位、方向错误、数值比较失败或因果交互渲染失败等问题。

现有基准的不足：

**提示信息稀疏**：T2I-CompBench、GenEval 等使用简短提示，仅能验证物体存在和简单属性

**评估粒度粗糙**：多采用分类或 Yes/No 问答，无法捕捉高阶空间能力

**缺少系统的空间智能分层**：未区分感知、推理和交互等不同层次的空间能力

## 方法详解

### 整体框架

SpatialGenEval 的设计围绕四个核心原则：

1. **长且信息密集的提示**：每条提示约 60 词，密集包含 10 个互相关联的空间约束
2. **全维度多选题评估**：每条提示配 10 道多选题，覆盖所有空间子领域
3. **图像依赖答案（无答案泄露）**：不将生成提示发送给评估器
4. **拒绝回答选项**：每道多选题含 "E: None" 选项，避免强制选择错误答案

### 关键设计

#### 1. 空间智能分层体系（10 个子领域）

SpatialGenEval 将空间智能分为 4 个层级、10 个子领域：

**空间基础（S1/S2）**：

| 子领域 | 评测内容 |
|--------|----------|
| S1 物体类别 | 组合完整性——是否生成了所有提到的物体 |
| S2 物体属性 | 属性绑定——颜色/形状/材质是否正确关联 |

**空间感知（S3/S4/S5）**：

| 子领域 | 评测内容 |
|--------|----------|
| S3 空间位置 | 绝对/相对位置定位 |
| S4 空间方向 | 旋转对齐（如面朝左、倒置） |
| S5 空间布局 | 多物体排列（线性序列、圆形等） |

**空间推理（S6/S7/S8）**：

| 子领域 | 评测内容 |
|--------|----------|
| S6 空间比较 | 相对定量属性（如大三倍） |
| S7 空间邻近 | 精细物理距离（接触、最近、远离） |
| S8 空间遮挡 | 3D 深度和物体层叠 |

**空间交互（S9/S10）**：

| 子领域 | 评测内容 |
|--------|----------|
| S9 运动交互 | 动态状态或运动中的时刻 |
| S10 因果交互 | 因果物理关系 |

#### 2. 基准构建流程

**提示生成**：使用 Gemini 2.5 Pro，输入 25 个真实世界场景和 10 个空间子领域定义，生成信息密集的提示。每条提示无缝融合所有 10 个空间约束。

**人工审核**（Human-in-the-loop）：
- 合并不自然的短句（如 "There is a robot. It is rusty." → "A rusty robot"）
- 修正逻辑矛盾（如循环布局约束）
- 替换生僻词（如 vermilion → bright red）

**QA 生成与验证**：
- 每条提示自动生成 10 道多选题
- 人工检查答案泄露：确保问题中不含显式答案
- 程序化添加 "E: None" 选项

#### 3. SpatialT2I 数据集（超越评估的数据中心范式）

- 额外构建 1,100 条提示，由 14 个 top 开源模型生成图像
- Qwen2.5-VL-72B 评估质量，Gemini 2.5 Pro 重写提示确保一致性
- 最终得到 15,400 个文本-图像对
- 用于微调 SDXL、UniWorld-V1、OmniGen2

### 损失函数 / 训练策略

评估流程：
- 主评估器：Qwen2.5-VL-72B（开源，保证可复现性）
- 5 轮投票机制：仅当 MLLM 在 5 轮中至少 4 轮选择正确答案时算正确
- 最终分数：每个空间子领域的正确率

## 实验关键数据

### 主实验

**表2：SpatialGenEval 排行榜（23 个模型）**

| 模型 | 规模 | Overall | 基础 (S1/S2) | 感知 (S3-S5) | 推理 (S6-S8) | 交互 (S9/S10) |
|------|------|---------|------------|------------|------------|-------------|
| SD-1.5 | 0.86B | 28.5 | 8.5/33.7 | 19.5/29.2/38.2 | 12.8/37.7/15.6 | 42.0/47.6 |
| FLUX.1-dev | 12B | 56.5 | 51.7/73.8 | 50.0/55.5/66.7 | 28.2/62.9/28.9 | 73.1/73.8 |
| Qwen-Image | 20B | 60.6 | 61.0/77.2 | 55.6/56.7/69.7 | 28.6/67.7/30.8 | 78.1/80.2 |
| GPT-Image-1 | - | 60.5 | 56.3/74.1 | 53.3/58.9/70.4 | 31.4/66.8/30.2 | 80.9/82.2 |
| **Seed Dream 4.0** | - | **62.7** | 59.9/80.2 | 57.2/58.9/70.1 | 32.1/68.3/33.8 | 83.0/83.8 |

**表6：SpatialT2I 微调效果**

| 模型 | 微调前 Overall | 微调后 Overall | 提升 |
|------|--------------|--------------|------|
| SD-XL | 41.2 | 45.4 | +4.2% |
| UniWorld-V1 | 54.2 | 59.9 | +5.7% |
| OmniGen2 | 56.4 | 60.8 | +4.4% |

### 消融实验

**评估器一致性**：GPT-4o 和 Qwen2.5-VL-72B 的模型排名完全一致，验证评估鲁棒性。

**人类对齐研究**：Gemini-2.5-Pro 达到 84.2% 平衡准确率，Qwen2.5-VL-72B 为 80.4%。

### 关键发现

1. **空间推理是主要瓶颈**：比较（Comparison）和遮挡（Occlusion）子任务得分常低于 30%，接近随机选择的 20%
2. **开源模型正在追赶闭源**：Qwen-Image (60.6%) vs Seed Dream 4.0 (62.7%)
3. **文本编码器是关键**：使用 LLM 编码器的模型（如 Qwen-Image）显著优于仅用 CLIP 的模型
4. **统一架构更参数高效**：7B 的 Bagel (57.0%) 接近 12B 的 FLUX.1-krea (58.5%)
5. **数据中心范式有效**：SpatialT2I 微调一致提升 4-6 个百分点

## 亮点与洞察

1. **信息密集提示设计**：将 10 个空间约束融合到单条 60 词提示中，避免了简单提示无法区分能力的问题
2. **分层式空间智能定义**：从基础→感知→推理→交互的递进结构清晰且可扩展
3. **"E: None" 选项的工程设计**：避免强制选择，提高评估准确性
4. **SpatialT2I 的数据飞轮**：评估基准的副产品可直接用于模型改进，形成闭环

## 局限与展望

1. 最高分仅 ~63%，面向及格线水平——说明任务仍然极具挑战性
2. 提示约 60 词可能超出某些 CLIP 编码器的有效处理长度（77 tokens）
3. 场景覆盖25类，可进一步扩展到更多复杂交互场景
4. 评估依赖 MLLM 判断，可能引入评估模型自身偏差
5. SpatialT2I 数据集质量受限于生成模型的当前能力

## 相关工作与启发

- **T2I-CompBench**：短提示 + Yes/No 评估，覆盖不足
- **DPG-Bench**：长提示但用评分方式，区分度有限
- **TIIF-Bench**：长短混合但 Yes/No 评估
- 启发：信息密集提示 + 全维度评估的范式可推广到视频生成、3D 生成等领域的空间智能评测

## 评分

- 新颖性：⭐⭐⭐⭐ — 首个系统性空间智能 T2I 评测基准
- 技术贡献：⭐⭐⭐⭐⭐ — 基准设计 + 数据集构建 + 大规模评测三位一体
- 实验充分度：⭐⭐⭐⭐⭐ — 23 个模型、多评估器验证、人类对齐
- 写作质量：⭐⭐⭐⭐ — 结构清晰，但表格较多
- 总体推荐：⭐⭐⭐⭐⭐ — 对 T2I 空间能力的全面透视，高影响力工作

## 背景与动机

## 核心问题

## 方法详解

## 实验关键数据

## 亮点

## 局限与展望

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

## 相关论文

- [Blueprint-Bench: Comparing Spatial Intelligence of LLMs, Agents and Image Models](blueprint-bench_comparing_spatial_intelligence_of_llms_agents_and_image_models.md)
- [Six-CD: Benchmarking Concept Removals for Text-to-Image Diffusion Models](../../CVPR2025/image_generation/six-cd_benchmarking_concept_removals_for_text-to-image_diffusion_models.md)
- [CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models](../../ICCV2025/image_generation/compass_enhancing_spatial_understanding_in_text-to-image_diffusion_models.md)
- [Asynchronous Denoising Diffusion Models for Aligning Text-to-Image Generation](asynchronous_denoising_diffusion_models_for_aligning_text-to-image_generation.md)
- [GeoDiv: Framework for Measuring Geographical Diversity in Text-to-Image Models](geodiv_framework_for_measuring_geographical_diversity_in_text-to-image_models.md)

<!-- RELATED:END -->
