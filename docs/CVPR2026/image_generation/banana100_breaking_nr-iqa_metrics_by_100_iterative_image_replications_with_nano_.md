---
title: >-
  [论文解读] Banana100: Breaking NR-IQA Metrics by 100 Iterative Image Replications with Nano Banana Pro
description: >-
  [CVPR 2026][图像生成][Image Quality Assessment] Banana100 通过让 Nano Banana Pro 迭代复制图像 100 次来系统性研究多轮编辑中的质量退化问题，构建了包含 28,000 张退化图像的数据集，并揭示了一个惊人发现：21 种主流无参考图像质量评估（NR-IQA）指标均无法可靠检测迭代退化——大多数指标甚至给噪声图像打出比干净图像更高的分数。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "Image Quality Assessment"
  - "NR-IQA"
  - "Iterative Degradation"
  - "image editing"
  - "Multi-turn"
---

# Banana100: Breaking NR-IQA Metrics by 100 Iterative Image Replications with Nano Banana Pro

**会议**: CVPR 2026  
**arXiv**: [2604.03400](https://arxiv.org/abs/2604.03400)  
**代码**: [https://huggingface.co/datasets/kenantang/Banana100](https://huggingface.co/datasets/kenantang/Banana100)  
**领域**: 图像生成 / 图像质量评估  
**关键词**: Image Quality Assessment, NR-IQA, Iterative Degradation, image editing, Multi-turn

## 一句话总结
Banana100 通过让 Nano Banana Pro 迭代复制图像 100 次来系统性研究多轮编辑中的质量退化问题，构建了包含 28,000 张退化图像的数据集，并揭示了一个惊人发现：21 种主流无参考图像质量评估（NR-IQA）指标均无法可靠检测迭代退化——大多数指标甚至给噪声图像打出比干净图像更高的分数。

## 研究背景与动机

**领域现状**：多模态 Agent 系统的兴起使迭代图像编辑成为常态——用户生成图像后反复修改和精炼。现代模型（如 Nano Banana Pro）在单轮编辑中表现出色。

**被忽视的失败模式——迭代退化**：
   - 每次编辑都引入微小、几乎不可察觉的伪影
   - 当输出再次作为输入时，这些伪影**累积放大**
   - 约 5-10 步后，出现明显静态噪声、色偏（绿色色调）和散点
   - 不仅是视觉质量退化，模型的**指令遵循能力也随之下降**（如无法完成简单的"在桌上加苹果"）

**更严重的问题——评估器也失效**：
   - 23 种 NR-IQA 指标中仅 2 种能一致检测退化
   - **惊人案例**：简单复制图像使 BRISQUE 分数从 34.1 降到 -9.8（越低越好），但人眼明显看到严重噪声
   - 生成器失效 + 评估器失效 = 低质量合成数据可能**静默泄漏**到训练集中

**潜在后果**：
   - **训练侧**：如果评估器无法过滤噪声数据，未来模型训练数据越来越脏，加速模型坍塌
   - **推理侧**：Agent 系统已知在长序列操作后脆弱，质量检查失败进一步加剧碎片化

**核心贡献**：构建系统性数据集，分类失败模式，揭露 NR-IQA 的系统性缺陷。

## 方法详解

### 整体框架

Banana100 不是提方法，而是设计了一个极简却尖锐的压力测试：让最新的图像编辑模型（主力是 Nano Banana Pro）对同一张图「原样复制」100 次，看微小伪影怎么在迭代里累积放大，并借此质问主流的无参考图像质量评估（NR-IQA）指标到底能不能察觉这种退化。整套工作由三块构成——一个 28000 张退化图的数据集、一套从子对象到整图的失败模式分类、以及一个对 21 种 NR-IQA 指标的失效性检验。

### 关键设计

**1. 迭代复制协议与数据集：把「复制一张图」做成退化放大器**

单轮编辑看不出问题，迭代才是放大镜——每次编辑都引入几乎不可察的伪影，输出再喂回输入就会累积，5–10 步后就出现明显静态噪声、绿色色偏和散点，连指令遵循也跟着垮（连「在桌上加个苹果」都做不到）。数据集从 13 张至少 2K 分辨率的高质量初始图出发（11 张 Nano Banana Pro + 2 张 SPICE 生成，覆盖摩天楼、食物、苔藓、孔雀羽毛、沙丘等，刻意排除人脸以免变形引起不适），主设置是 "Produce an exact replica of the provided image, with no alterations."，外加换措辞、镜像翻转、去色再上色、改 seed/temperature/resolution 等变体，以及「逐步加苹果 / 加 100 种水果」的对象添加测试。每种设置跑 100 步 × 5 次重复，共 28000 张、成本约 \$4000，并在 Nano Banana 2 Fast、FLUX.2 [dev]、Qwen Image Edit 上各补 1400 张做交叉验证。

**2. 失败模式分类体系：按子对象 / 对象 / 整图三级归因**

光说「变差了」没用，得指明坏在哪。**子对象级**表现为简化偏好——复制角色表情时把红橙紫蓝 4 种眼色简化成红蓝 2 种，推理摘要只抓最显著的颜色；**对象级**包括计数失败（要求加 1 个苹果却不加或多加一排）、替换而非添加（用新水果换掉旧水果）、以及新增对象虽清晰但背景噪声持续恶化；**图像级**则有宽高比不匹配（复制几乎总裁剪）、去噪指令也消不掉的持久噪声、即便对话里有早期干净图也无法参考、以及镜像/旋转失败（动画风格图镜像成功率远低于写实图，旋转两者都低）。这套分级为后续改进指明了靶点。

**3. NR-IQA 失效判定协议：用单调性把指标「将军」**

判定一个指标是否真能测退化，标准很硬：对比 Step 1 与 Step 5/10/20 的归一化分数差 $\Delta_i$，要求 $\Delta_5, \Delta_{10}, \Delta_{20}$ 全为负（即随退化分数单调变差）才算合格。在 pyiqa 库的 21 种传统 NR-IQA 指标上检验，结果无一能在所有图像上一致检测退化——典型如简单复制竟让 BRISQUE 从 34.1 掉到 -9.8（越低越好），人眼却明显看到严重噪声；只有两种基于大 VLM 的最新指标（VisualQuality-R1、RALI）零失败通过。

## 实验关键数据

### 主实验：NR-IQA 指标失败统计

| 指标类型 | 代表 | 能否检测退化 | 说明 |
|----------|------|-------------|------|
| 手工特征 | BRISQUE, NIQE, PIQE | ❌ | 多数图像上给噪声图更高分 |
| CNN-based | CNNIQA, DBCNN, HyperIQA | ❌ | 不同图像失败模式不同 |
| CLIP-based | CLIPIQA, LIQE, QualiCLIP | ❌ | 被模型生成的纹理模式混淆 |
| Transformer-based | MUSIQ, MANIQA, TReS | ❌ | 同样无法一致检测 |
| **大 VLM-based** | **VisualQuality-R1, RALI** | **✅** | 两者 0 失败案例 |

### 消融：不同模型的噪声累积

| 模型 | 噪声类型 | 说明 |
|------|---------|------|
| Nano Banana Pro | 静态噪声 + 绿色色调 | 最常见 |
| Nano Banana 2 Fast | 沿物体轮廓的皱痕 | 无推理模式 |
| FLUX.2 [dev] | 散点 | 开源模型 |
| Qwen Image Edit | 纹理简化 + 物体重复 | 将右侧放到左侧 |

### 关键发现
- **所有测试的图像编辑模型**都存在迭代退化问题，且噪声模式各不相同
- Nano Banana Pro 的推理摘要存在严重幻觉——声称去噪成功但实际未去噪
- 模型的自我评估（reasoning summary）平均到第 20-37 步才首次提及噪声，远滞后于人眼感知（5-10 步）
- 开源模型（FLUX.2, Qwen）同样受影响 → 非水印导致的专属问题
- 仅基于大 VLM 的最新指标（VisualQuality-R1, RALI）成功检测所有退化

## 亮点与洞察
- **揭示了一个系统性而非个案性的问题**：所有主流编辑模型都退化 + 所有主流 NR-IQA 都失效 = 生态级风险
- **"生成器失效 × 评估器失效"的组合效应**非常危险：低质量数据可能不被过滤就流入训练集
- 失败模式分类（子对象/对象/图像级）为未来改进提供了清晰靶向
- 仅大 VLM-based 指标成功检测 → NR-IQA 领域需要范式更新

## 局限与展望
- 初始图像仅 13 张，且全为 AI 生成——真实照片（有压缩伪影）的结论可能不同
- 未包含 Mean Opinion Score（MOS）标注，因此无法用 PLCC/SRCC 评估 NR-IQA 排名
- 黑盒 API 模型的可重复性有限
- 虽指出问题但未提出解决方案——如何修复编辑模型或 NR-IQA 指标留为未来工作

## 相关工作与启发
- Pico-Banana-400K 数据集的多步子集已出现明显退化（第 5-6 步后），与本文发现一致
- REED-VAE 尝试在扩散模型中减少迭代编辑退化，但效果有限
- 对 Agent 系统评估框架的重大警示——如果底层评估指标不可靠，Agent 的自主决策也不可靠
- 呼吁 NR-IQA 社区关注模型生成噪声（vs 传统人工添加噪声）作为训练数据

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统性揭示迭代编辑退化 + NR-IQA 失效的双重问题
- 实验充分度: ⭐⭐⭐⭐⭐ 4 个模型 × 23 种指标 × 多种编辑方式，覆盖极为全面
- 写作质量: ⭐⭐⭐⭐ 分类清晰，案例丰富，但论文结构略散
- 价值: ⭐⭐⭐⭐⭐ 对整个图像生成和评估生态系统有重要警示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] EMMA: Concept Erasure Benchmark with Comprehensive Semantic Metrics and Diverse Categories](emma_concept_erasure_benchmark_with_comprehensive_semantic_metrics_and_diverse_c.md)
- [\[ECCV 2024\] ReNoise: Real Image Inversion Through Iterative Noising](../../ECCV2024/image_generation/renoise_real_image_inversion_through_iterative_noising.md)
- [\[NeurIPS 2025\] Evaluating the Evaluators: Metrics for Compositional Text-to-Image Generation](../../NeurIPS2025/image_generation/evaluating_the_evaluators_metrics_for_compositional_text-to-image_generation.md)
- [\[ICML 2026\] Shifting the Breaking Point of Flow Matching for Multi-Instance Editing](../../ICML2026/image_generation/shifting_the_breaking_point_of_flow_matching_for_multi-instance_editing.md)
- [\[ICLR 2026\] Test-Time Iterative Error Correction for Efficient Diffusion Models](../../ICLR2026/image_generation/test-time_iterative_error_correction_for_efficient_diffusion_models.md)

</div>

<!-- RELATED:END -->
