---
title: >-
  [论文解读] Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training
description: >-
  [CVPR 2026][多模态][文档解析] 提出数据-训练协同设计框架 DocHumming：通过 Realistic Scene Synthesis 构建 DocMix-3M 大规模合成数据集，结合渐进学习和结构 token 加权的 Document-Aware Training Recipe，在仅 1B 参数的 MLLM 上实现 OmniDocBench Overall 93.75（超越 Qwen3-VL-235B 的 89.15），且在真实拍摄场景下仅退化 6.72 分（模块化方法退化 18-20 分）。
tags:
  - CVPR 2026
  - 多模态
  - 文档解析
  - 合成数据
  - 渐进式训练
  - 结构token加权
  - 真实场景鲁棒性
---

# Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training

**会议**: CVPR 2026  
**arXiv**: [2603.23885](https://arxiv.org/abs/2603.23885)  
**代码**: 待开源  
**领域**: 文档理解 / 端到端文档解析  
**关键词**: 文档解析, 合成数据, 渐进式训练, 结构token加权, 真实场景鲁棒性

## 一句话总结

提出数据-训练协同设计框架 DocHumming：通过 Realistic Scene Synthesis 构建 DocMix-3M 大规模合成数据集，结合渐进学习和结构 token 加权的 Document-Aware Training Recipe，在仅 1B 参数的 MLLM 上实现 OmniDocBench Overall 93.75（超越 Qwen3-VL-235B 的 89.15），且在真实拍摄场景下仅退化 6.72 分（模块化方法退化 18-20 分）。

## 研究背景与动机

**领域现状**：文档解析已从传统模块化管线（布局分析→OCR→元素解析）发展到端到端 MLLM 直接映射图像到结构化输出。模块化方法在数字/扫描文档上表现优秀（如 MinerU2.5 OmniDocBench 90.67），但端到端方法在真实场景下仍面临严重挑战。

**现有痛点**：(1) 模块化方法依赖精确布局分析，在随意拍摄条件下布局错误向下游传播（退化 18-20 分）；(2) 端到端方法在真实拍摄场景下产生重复内容、幻觉和结构不一致；(3) 缺乏大规模高质量的页面级端到端解析训练数据（SynthDog 布局简单，GOT 的 PDF-to-LaTeX 缺乏视觉多样性）。

**核心矛盾**：端到端范式无需显式布局分割、天然更鲁棒，但受限于数据稀缺和缺乏结构感知训练策略，潜力未被释放。

**本文目标** 通过数据-训练协同设计释放端到端文档解析在真实场景下的潜力。

**切入角度**：同时解决数据瓶颈（大规模合成）和训练瓶颈（结构感知优化），而非只攻其一。

**核心 idea**：576K 布局模板 + 9M 原子元素合成 3M 页面级数据，配合短到长渐进训练和结构 token 加权损失，让 1B 模型达到 235B 模型水平。

## 方法详解

### 整体框架

数据-训练协同设计：数据层面通过 Realistic Scene Synthesis（RSS）生成大规模多样化端到端解析数据（DocMix-3M），训练层面通过 Document-Aware Training Recipe（DATR：渐进学习 + 结构 token 加权）提升结构保真度和解码稳定性。最终在 InternVL2-1B 上训练得到 DocHumming。

### 关键设计

1. **Realistic Scene Synthesis（真实场景合成）**:

    - 功能：从原子元素和布局模板合成大规模页面级端到端解析数据
    - 核心思路：(a) **原子元素仓库**：整合表格识别、公式解析、段落理解等多源数据集，格式归一化，用 Qwen2.5-72B 改写增强（重组表格、扰动公式符号、创建混合元素、生成多语言段落组），通过 LaTeX 管线渲染为图像+标注对；(b) **布局模板库**：公开数据集+网页挖掘+补充欠表示风格，共 576K+ 布局模式，带阅读顺序标注；(c) **组合合成**：在空间/结构约束下将采样元素放置到模板中；(d) **拍摄增强**：模拟透视/弯曲/褶皱/光照变化/相机旋转/环境背景，约 20% 样本经增强
    - 产出：DocMix-3M（~3M 高质量合成文档），基于 ~9M 原子元素 + 576K 布局模板
    - 设计动机：底层合成而非 PDF-to-LaTeX 转换，能控制布局多样性和视觉条件

2. **Document-Aware Training Recipe（文档感知训练策略）**:

    - 功能：设计专门针对文档解析的训练策略，解决上下文长度跨度大和结构化输出不稳定的问题
    - 核心思路：(a) **渐进学习范式**——阶段 1 训练解析单个元素（表格/公式/段落），用异构提示获取类型特定能力，同时扩展词表加入布局结构 token；阶段 2 以 DocMix-3M 为主体 + 1M 阶段 1 样本 + 100K 人工标注数据，统一提示格式做端到端全文档训练。(b) **结构 Token 加权优化**——对结构化 token（`<table>`...`</table>` 内）施加更高损失权重
    - 损失公式：$L = -\sum_t \alpha_t y_t \log P(x_t|x_{<t})$，结构 token $\alpha_t = \lambda = 4$，其他 $\alpha_t = 1$
    - 设计动机：渐进学习避免直接在长上下文上训练的收敛不稳定；结构 token 加权解决表格等结构化内容的重复和不一致问题

3. **Wild-OmniDocBench（真实场景基准）**:

    - 功能：构建真实拍摄条件下的文档解析评估基准
    - 核心思路：手动将 OmniDocBench 全部转化为真实拍摄形式。(a) 打印→物理变形（折叠/弯曲/揉皱）→多种光照下拍照；(b) 屏幕显示→拍照（引入摩尔纹/反射/亮度变化）
    - 设计动机：现有基准仅评估数字/扫描文档，无法反映真实场景挑战

### 损失函数 / 训练策略

结构 token 加权交叉熵损失。阶段 1：batch=512, lr=4e-5, 2 epochs；阶段 2：batch=256, lr=2e-5, 2 epochs。余弦学习率衰减，最大输出长度 8192 tokens。基座模型 InternVL2-1B，全参数微调。16x NVIDIA H20 GPU。

## 实验关键数据

### 主实验：OmniDocBench 文档解析

| 类型 | 方法 | 参数量 | Overall↑ | TextEdit↓ | FormulaCDM↑ | TableTEDS↑ | ReadOrder↓ |
|------|------|-------|---------|-----------|-------------|------------|------------|
| Pipeline | PP-StructureV3 | - | 86.73 | 0.073 | 85.79 | 81.68 | 0.073 |
| 通用MLLM | Qwen2.5-VL-72B | 72B | 87.02 | 0.094 | 88.27 | 82.15 | 0.102 |
| 通用MLLM | Qwen3-VL-235B | 235B | 89.15 | 0.069 | 88.14 | 86.21 | 0.068 |
| E2E专用 | dots.ocr | 3B | 88.41 | 0.048 | 83.22 | 86.78 | 0.053 |
| 模块化专用 | MinerU2.5 | 1.2B | 90.67 | 0.047 | 88.46 | 88.22 | 0.044 |
| 模块化专用 | PaddleOCR-VL | 0.9B | 91.93 | 0.039 | 88.67 | 91.01 | 0.043 |
| **E2E专用** | **DocHumming** | **1B** | **93.75** | **0.035** | **93.27** | **91.49** | **0.041** |

### Wild-OmniDocBench 真实场景鲁棒性

| 类型 | 方法 | Origin | Wild | 退化↓ | 说明 |
|------|------|--------|------|-------|------|
| 通用MLLM | Qwen3-VL-235B | 89.15 | 79.69 | -9.46 | 大模型也退化 |
| 模块化 | MonkeyOCR-3B | 88.85 | 70.00 | -18.85 | 布局错误传播 |
| 模块化 | MinerU2.5 | 90.67 | 70.91 | -19.76 | 模块化退化最严重 |
| 模块化 | PaddleOCR-VL | 91.93 | 72.19 | -19.74 | 退化约20分 |
| E2E | DeepSeek-OCR | 87.01 | 74.23 | -12.78 | E2E退化较小 |
| E2E | dots.ocr | 88.41 | 78.01 | -10.40 | E2E退化较小 |
| **E2E** | **DocHumming** | **93.75** | **87.03** | **-6.72** | **退化最小** |

### 消融实验

| # | RSS | 渐进(PTP) | 结构加权(ST) | OmniDoc↑ | Repeat↓ | Wild↑ | Wild Repeat↓ |
|---|-----|----------|------------|---------|---------|------|-------------|
| 1 | X | Y | Y | 89.96 | 4.7% | 78.82 | 8.6% |
| 2 | Y | Y | X | 88.74 | 4.6% | 84.90 | 5.4% |
| 3 | Y | X | Y | 91.24 | 4.2% | 85.39 | 4.9% |
| 4 | Y | Y | Y | **93.75** | **2.1%** | **87.03** | **4.3%** |

数据规模曲线：DocMix-1M(85.41) -> 2M(88.14) -> 3M(89.96) -> 4M(89.31, 趋于饱和)。3M 超越 100K 人工标注数据（89.96 vs 89.26）。

### 关键发现

- 端到端 vs 模块化在真实场景下分化显著：模块化退化 18-20 分 vs DocHumming 仅退化 6.72 分
- 1B 超越 235B：通过正确的数据+训练策略，1B 模型（93.75）超越 Qwen3-VL-235B（89.15）
- 合成数据 3M 超越 100K 人工标注（89.96 vs 89.26），但 4M 时趋于饱和——元素仓库和模板池多样性是瓶颈
- 结构 token 加权是减少重复的关键：移除后重复率从 2.1% 升至 4.6%
- XFUND 多语言测试全面领先（德语 85.15、日语 87.99、西语 84.39）

## 亮点与洞察

- 数据-训练协同设计的框架思路值得推广：不只做数据或只做训练，联合优化效果显著
- 渐进训练借鉴 LLM 短到长上下文课程——文档解析的元素->页面与 LLM 的短->长上下文高度对应
- 定义重复率指标（连续结构模式重复>10次 + 达到最大长度）是量化解码稳定性的实用工具
- 数据规模效应的饱和点（~3M）为合成数据投入产出比提供了实用指导

## 局限与展望

- 不规则交错布局（报纸、海报）仍表现不佳，文本块嵌套/交错时阅读顺序和结构边界模糊
- 超高分辨率页面需下采样/切片，可导致长表格/密集公式重复或丢失
- 3M 后数据规模效应饱和，根本原因是元素仓库和模板池有限
- 推理效率：文本密集页面约需 ~3s，限制交互式使用
- 结构 token 加权 lambda=4 较为启发式，未探索自适应策略

## 相关工作与启发

- **vs GOT/SmolDocling**：这些 E2E 方法使用 PDF-to-LaTeX 数据，布局单一；DocHumming 通过底层合成实现布局多样性
- **vs MinerU2.5/PaddleOCR-VL**：标准文档上接近，但 Wild 场景退化 3 倍以上，模块化范式在真实场景下本质脆弱
- **训练策略启发**：结构 token 加权可推广到任何结构化生成任务（代码生成、HTML/JSON 生成）
- **数据合成方法论**：原子元素仓库 + 布局模板 + 组合合成的范式可迁移到其他多模态数据生成

## 评分

⭐⭐⭐⭐ (4/5)

- **新颖性** ⭐⭐⭐⭐：数据-训练协同设计思路系统性强，结构 token 加权简单但有效
- **实验充分度** ⭐⭐⭐⭐⭐：OmniDocBench + Wild + XFUND 三基准，完整消融（RSS/ST/PTP），数据规模曲线
- **写作质量** ⭐⭐⭐⭐：框架图清晰，数据构建流程详尽，消融设计严谨
- **价值** ⭐⭐⭐⭐：1B 超越 235B 的结论令人信服，Wild 基准填补评估空白

<!-- RELATED:START -->

## 相关论文

- [Efficient Document Parsing via Parallel Token Prediction](efficient_document_parsing_via_parallel_token_prediction.md)
- [World-Env: Leveraging World Model as a Virtual Environment for VLA Post-Training](rehearsevla_simulated_posttraining_world_model.md)
- [GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)
- [Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](fine-grained_post-training_quantization_for_large_vision_language_models_with_qu.md)
- [Index-Preserving Lightweight Token Pruning for Efficient Document Understanding](../../ICLR2026/multimodal_vlm/index-preserving_lightweight_token_pruning_for_efficient_document_understanding_.md)

<!-- RELATED:END -->
