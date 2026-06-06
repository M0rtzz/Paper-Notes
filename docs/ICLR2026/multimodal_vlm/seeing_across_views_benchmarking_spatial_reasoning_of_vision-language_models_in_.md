---
title: >-
  [论文解读] Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes
description: >-
  [ICLR2026][多模态VLM][multi-view spatial reasoning] 提出 MV-RoboBench，首个整合多视角空间推理与机器人操作执行评测的 benchmark，包含 1.7K 人工标注 QA…
tags:
  - "ICLR2026"
  - "多模态VLM"
  - "multi-view spatial reasoning"
  - "robotic manipulation"
  - "VLM benchmark"
  - "embodied AI"
  - "MV-RoboBench"
---

# Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes

**会议**: ICLR2026  
**arXiv**: [2510.19400](https://arxiv.org/abs/2510.19400)  
**代码**: [项目页面](https://github.com/) (已开源)  
**领域**: 多模态VLM  
**关键词**: multi-view spatial reasoning, benchmark, embodied AI, VLM evaluation, robotic manipulation  

## 一句话总结
提出 MV-RoboBench，首个结合多视角空间推理与机器人操作任务的基准，系统评估了 40+ 个 VLM（开源+闭源+推理增强），发现最强模型 GPT-5 仅达 56.4% 准确率，远低于人类的 91.0%，并揭示空间与机器人推理正相关、单视角基准表现无法可靠迁移至多视角场景。

## 背景与动机
- VLM 是 Embodied AI 的核心组件，为 VLA（Vision-Language-Action）模型提供感知和推理能力
- 然而大多数 VLM 评估集中于**单视角**设置，对多视角信息整合的评估严重不足
- 多相机配置在机器人平台中已成为标准，可提供互补视角以缓解遮挡和深度模糊
- 现有空间推理基准（EmbSpatial-Bench、RoboSpatial 等）主要关注单视角，ERQA 和 MMSI-Bench 仅包含部分多视角数据
- All-Angles Bench 和 Ego3D-Bench 虽然使用多视角输入，但任务局限于照片对齐或导航感知，缺乏面向操作的具身推理

## 方法详解

### 整体框架：MV-RoboBench 基准设计
基于 AgiWorld 和 BridgeV2 数据集，构建包含 1,708 道人工标注多选题的基准，涵盖**空间理解**和**机器人执行**两大类共 8 个子任务：

- **空间理解（4 个子任务）**：
    - Cross-View Matching：跨视角目标匹配
    - Distance Judgement：物体间相对距离判断
    - Viewpoint Identification：视角变换推理
    - 3D Spatial Consistency：3D 空间一致性维护

- **机器人执行（4 个子任务）**：
    - Action Planning：多步动作序列规划
    - Step Execution：单步动作正确性验证
    - Trajectory Selection：候选运动路径可行性评估
    - Affordance Recognition：物体交互可行性判断

### 关键设计 1：多阶段人工质控流水线
- **数据收集**：规则筛选 + GPT-4.1 辅助过滤 + 人工验证（GPT-4.1 仅用于分诊，不生成 QA 内容）
- **QA 生成**：任务特定模板 + 训练有素的标注员构建五选一 QA，确保干扰项合理但可区分
- **迭代审核**：trained annotators 多轮审核、修订、答案分布平衡，消除偏差

### 关键设计 2：CoT-inspired 增强探索
系统探索三种 CoT 风格的增强方案：
1. **文本 CoT（w text）**：GPT-4.1 生成场景描述作为额外文本上下文
2. **视觉 CoT（w vggt）**：VGGT 进行新视角合成提供额外视觉证据
3. **结构 CoT（w depth）**：MoGe-2 深度估计提供几何约束

### 关键设计 3：双轴相关性分析
- **内部相关轴**：多视角场景中空间推理 vs. 机器人执行的相关性
- **外部迁移轴**：单视角空间基准（OmniSpatial）表现 → 多视角具身推理的迁移性

## 实验

### 主实验结果

| 模型类型 | 代表模型 | 平均准确率 |
|---------|---------|-----------|
| 随机猜测 | - | 19.7% |
| 闭源 VLM | GPT-4.1 | 30.9% |
| 开源 VLM | Qwen2.5-vl-72B | 24.3% |
| 开源 MoE | Llama-4-Maverick | 26.1% |
| 推理模型 | GPT-5 | **56.4%** |
| 推理模型 | Gemini-2.5-pro | 49.5% |
| 人类 | - | **91.0%** |

### CoT 增强消融实验

| 模型 | 原始 | w cot | w text | w vggt | w depth |
|------|------|-------|--------|--------|---------|
| Qwen2.5-vl-7B | 20.84 | 20.49 | 20.90 | 20.02 | **21.14** |
| Gemma-3-12B | 20.49 | **24.19** | 18.43 | 18.31 | 20.41 |
| GPT-4.1 | 29.87 | 29.84 | 31.66 | 28.02 | **33.12** |

### 关键发现
1. **推理能力是核心差异来源**：推理增强模型（GPT-5、o4-mini）大幅领先感知型模型，但仍远低于人类
2. **3D Spatial Consistency 极具挑战**：大部分非推理模型在该子任务上接近甚至低于随机水平（19.07%）
3. **CoT 增强效果因模型而异**：合成新视角通常有害；深度先验仅在高容量模型上有效；CoT prompting 对中等规模开源模型最有效
4. **空间与机器人推理正相关**：但仅在具备足够多视角融合能力的模型上成立
5. **单视角到多视角迁移失败**：OmniSpatial 上的强表现不能可靠预测多视角具身推理能力

## 亮点
- 首个系统性的多视角机器人操作空间推理基准，填补空白
- 评估覆盖 40+ 个模型（5 大类），实验规模完整
- 双轴分析揭示了重要的 negative result：单视角能力不可靠迁移
- 人工标注质量高（1.7K QA 全部人工curated），数据来源涵盖单臂和双臂操作

## 局限性
- 仅使用 2D 图像作为输入，未探索显式 3D 表示（点云、网格）的效果
- 多视角相机配置固定（依赖现有数据集的相机布局），未探索不同相机布局的影响
- 基准以多选题形式评估，未覆盖开放式推理能力
- CoT 增强方案较基础，未探索如 active view selection 等更高级策略

## 相关工作
- **空间推理基准**：EmbSpatial-Bench、Visual Spatial、RoboSpatial、Spatial-MM、SpatialVLM、VSI-Bench 等均限于单视角
- **多视角基准**：All-Angles Bench（照片对齐）、Ego3D-Bench（导航感知），但都不涉及机器人操作
- **机器人场景评估**：ShareRobot（单视角）、ERQA（部分多视角但规模小）
- **VLM 空间增强**：SpatialRGPT、SpatialLLM、3D-LLM 等尝试注入几何先验

## 评分
⭐⭐⭐⭐ (4/5)

扎实的基准工作，评估规模大且系统性强。双轴分析提供了有价值的 insight。但作为 benchmark 论文，方法贡献有限，CoT 增强部分的探索较浅。
---
title: >-
  [论文解读] Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes
description: >-
  [ICLR2026][多模态][空间推理] 提出 MV-RoboBench，首个面向机器人多视角空间推理能力的 VLM 评测基准，包含 1.7K 人工标注 QA，覆盖空间理解与机器人执行两大类八个子任务。实验发现当前最强 VLM 远低于人类水平，且单视角空间基准上的性能无法可靠迁移到多视角机器人场景。
tags:
  - ICLR2026
  - 多模态
  - 空间推理
  - benchmark
  - multi-view
  - robotic manipulation
  - VLM evaluation
  - embodied AI
---

# Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes

**会议**: ICLR2026  
**arXiv**: [2510.19400](https://arxiv.org/abs/2510.19400)  
**代码**: [GitHub](https://github.com/) (项目页面已发布)  
**领域**: multimodal_vlm  
**关键词**: multi-view spatial reasoning, robotic manipulation, VLM benchmark, embodied AI, MV-RoboBench  

## 一句话总结
提出 MV-RoboBench，首个整合多视角空间推理与机器人操作执行评测的 benchmark，包含 1.7K 人工标注 QA，揭示当前最强 VLM（GPT-5 仅 56.4%）与人类（91.0%）之间存在巨大差距。

## 研究背景与动机
- 视觉语言模型（VLM）是具身 AI 和视觉-语言-动作（VLA）模型的核心基础，在机器人感知、推理和决策中扮演关键角色
- 大多数 VLM 评测聚焦于单视角设定，但多相机配置在机器人平台上已日益普及，可提供互补视角以缓解遮挡和深度模糊
- 现有空间推理基准（EmbSpatial-Bench、Visual Spatial、RoboSpatial 等）主要关注单视角关系推理，缺乏多视角+机器人操作的结合
- 少数多视角基准（All-Angles Bench、Ego3D-Bench）仅关注照片对齐或导航感知，未涉及操作导向的具身推理
- **核心 gap**: 缺少一个系统评测 VLM 在多视角机器人操作场景中空间推理能力的基准

## 方法详解

### 整体框架：MV-RoboBench
基于 AgiWorld 和 BridgeV2 两个真实机器人数据集构建，覆盖单臂和双臂操作场景，共 1,708 道五选一 QA 题，来源于 980 个操作 episode。

### 关键设计 1：两大类八任务的系统化评测体系
**空间理解（Spatial Understanding）**四个子任务：
1. **Cross-View Matching**：跨视角识别同一物体
2. **Distance Judgement**：判断物体间相对距离
3. **Viewpoint Identification**：推理视角变换关系
4. **3D Spatial Consistency**：维护物体在 3D 空间中的一致相对位置

**机器人执行（Robotic Execution）**四个子任务：
1. **Action Planning**：选择合适的多步操作序列
2. **Step Execution**：验证下一步单步动作是否正确
3. **Trajectory Selection**：评估候选运动路径的可行性
4. **Affordance Recognition**：评估物体特定交互的可行性

### 关键设计 2：高质量人工构建流程
采用三阶段 pipeline：
1. **数据收集**：规则过滤 + GPT-4.1 辅助筛选（仅用于候选分流，不生成 QA）+ 人工验证
2. **QA 生成**：任务特定模板 + 训练有素的标注员构建五选一 QA
3. **Human-in-the-loop 质量审查**：迭代审查、修订、答案分布平衡

### 关键设计 3：CoT 增强探索
探索三类 CoT 式增强：
- **文本 CoT**：GPT-4.1 生成场景描述作为补充文本
- **视觉 CoT**：VGGT 进行新视角合成，提供额外视觉证据
- **结构 CoT**：MoGe-2 估计深度先验，添加几何约束

### 相关性分析
设计两个分析轴：
- **内部相关性**：空间推理与机器人执行在多视角场景中的关联
- **外部迁移性**：单视角空间基准表现是否能迁移到多视角具身推理

## 实验

### 主实验表：多模型多类别评测

| 模型 | 平均准确率 | 空间理解 | 机器人执行 |
|------|-----------|---------|-----------|
| Random Choice | 19.71% | ~19% | ~20% |
| GPT-4.1 | 30.90% | 26.8% avg | 32.8% avg |
| GPT-5 (最强) | **56.41%** | 52.7% avg | 60.4% avg |
| Gemini-2.5-pro | 49.52% | 45.8% avg | 53.2% avg |
| o4-mini | 46.47% | 40.4% avg | 52.5% avg |
| Qwen2.5-vl-72B (开源最强) | 24.29% | 21.9% avg | 26.7% avg |
| InternVL3-78B | 23.25% | 20.9% avg | 25.6% avg |
| 人类 | **91.04%** | 93.7% avg | 88.2% avg |

### CoT 增强消融

| 增强方式 | Qwen2.5-vl-7B | Gemma-3-12B | GPT-4.1 |
|---------|---------------|-------------|---------|
| 无增强 (baseline) | 20.84% | 20.49% | 29.87% |
| + CoT prompting | 20.49 (-0.35) | **24.19 (+3.70)** | 29.84 (-0.03) |
| + 文本描述 | 20.90 (+0.06) | 18.43 (-2.06) | **31.66 (+1.79)** |
| + 新视角合成 | 20.02 (-0.82) | 18.31 (-2.18) | 28.02 (-1.85) |
| + 深度先验 | 21.14 (+0.30) | 20.41 (-0.08) | **33.12 (+3.25)** |

### 关键发现
1. **3D 空间一致性最具挑战性**：大多数非推理模型在此子任务上接近甚至低于随机（~19%），推理增强模型可提升至 49-82%
2. **空间与机器人推理正相关**：但仅在模型具备充足跨视角融合能力时成立
3. **单视角表现不可靠迁移**：OmniSpatial 上表现优秀的模型在 MV-RoboBench 上仍可能接近随机
4. **CoT 增强效果混合**：合成新视角倾向于降低性能，深度先验仅对高容量模型有效
5. **推理优化架构显著优于感知模型**：GPT-5 vs GPT-4.1 提升约 25 个百分点

## 亮点
- 首个将多视角空间推理与机器人操作执行系统整合的评测基准，填补重要空白
- 1,708 道人工精标 QA 质量高，覆盖八个子任务维度，评测粒度细
- 发现了两个重要结论：空间-机器人推理正相关 + 单视角不可靠迁移，对后续研究有指导意义
- 系统性地探索了 CoT 增强方式在多视角场景中的效果，发现简单堆叠几何线索不够

## 局限性
- 基准规模相对小（1.7K QA），可能不足以覆盖所有操作场景的多样性
- 所有任务均为五选一 MCQ 格式，未涉及开放式空间推理
- 仅使用两个数据源（AgiWorld + BridgeV2），场景多样性有限
- CoT 增强探索较初步，未深入结合几何编码器等更深层方法
- 未包含动态/视频场景的多视角推理

## 相关工作
- 单视角空间基准：EmbSpatial-Bench、Visual Spatial、RoboSpatial、SpatialVLM、VSI-Bench、OmniSpatial
- 多视角基准：All-Angles Bench、Ego3D-Bench、ERQA、MMSI-Bench
- 机器人评测：ShareRobot
- 3D 理解方法：SpatialRGPT、3D-LLM、SpatialBot、VLM-3R
- VLA 模型：π0、CogAct、OpenVLA

## 总体评价

| 维度 | 评分 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 影响力 | ⭐⭐⭐⭐ |
| **综合** | **⭐⭐⭐⭐** |

> 评测类工作，核心贡献在于识别了多视角+机器人操作这一关键 gap 并构建了高质量基准。实验覆盖 30+ 个模型非常全面，内外部相关性分析有洞见。但技术方法上偏数据构建，无模型创新。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] 3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models](../../ICML2026/multimodal_vlm/3viewsense_spatial_and_mental_perspective_reasoning_from_orthographic_views_in_v.md)
- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[ICLR 2026\] FRIEDA: Benchmarking Multi-Step Cartographic Reasoning in Vision-Language Models](frieda_benchmarking_multi-step_cartographic_reasoning_in_vision-language_models.md)
- [\[ICLR 2026\] Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation](spatial_captcha_generatively_benchmarking_spatial_reasoning_for_human-machine_di.md)
- [\[ICLR 2026\] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)

</div>

<!-- RELATED:END -->
