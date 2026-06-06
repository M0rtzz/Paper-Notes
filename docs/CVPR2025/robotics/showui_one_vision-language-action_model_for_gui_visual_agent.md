---
title: >-
  [论文解读] ShowUI: One Vision-Language-Action Model for GUI Visual Agent
description: >-
  [CVPR 2025][机器人][GUI视觉代理] ShowUI 基于 Qwen2-VL-2B，通过 UI 连通图引导的视觉 token 选择减少 33% 冗余 token 并加速 1.4 倍，配合交错式视觉-语言-动作流和精选 256K 训练数据…
tags:
  - "CVPR 2025"
  - "机器人"
  - "GUI视觉代理"
  - "视觉token选择"
  - "交错式VLA流"
  - "UI连通图"
  - "轻量级模型"
---

# ShowUI: One Vision-Language-Action Model for GUI Visual Agent

**会议**: CVPR 2025  
**arXiv**: [2411.17465](https://arxiv.org/abs/2411.17465)  
**代码**: [https://github.com/showlab/ShowUI](https://github.com/showlab/ShowUI) (开源)  
**领域**: 机器人  
**关键词**: GUI视觉代理、视觉token选择、交错式VLA流、UI连通图、轻量级模型

## 一句话总结
ShowUI 基于 Qwen2-VL-2B，通过 UI 连通图引导的视觉 token 选择减少 33% 冗余 token 并加速 1.4 倍，配合交错式视觉-语言-动作流和精选 256K 训练数据，仅 2B 参数即在零样本 ScreenSpot 上达到 75.1% 的 SOTA 精度。

## 研究背景与动机

**领域现状**：当前 GUI 自动化主要分两派——语言型代理依赖 GPT-4 等闭源 API 配合 HTML/DOM 树等元信息；视觉型代理直接处理截图但需要训练大模型。CogAgent (18B)、SeeClick (9.6B) 等视觉代理模型体量庞大且需要大量训练数据。

**现有痛点**：(1) UI 截图分辨率极高（如 2K），产生大量视觉 token（1000+），self-attention 计算代价高昂且存在大量冗余（空白区域、纯色背景）；(2) GUI 任务中的动作与自然语言不同——不同设备的动作空间各异（Web 有 CLICK、Mobile 有 PRESS HOME），如何统一建模不清楚；(3) 导航任务需要管理截图-动作的历史序列（交错多模态），现有 VLM 不擅长这种格式；(4) GUI 数据类型多样且不平衡，如何选取高质量子集也是问题。

**核心矛盾**：UI 截图中存在大量视觉冗余（纯色背景、空白区域），但也包含极其重要的小元素（按钮、图标），需要在大幅压缩 token 的同时保留关键细节的定位精度。

**本文目标** 如何在极低参数量和数据量下构建一个高效且精准的 GUI 视觉代理？

**切入角度**：UI 截图与自然图像的本质区别在于"结构化冗余"——大片相同颜色的区域对应背景/空白，而颜色变化的地方才是功能元素。因此可以在 RGB 空间构建 patch 级连通图来识别冗余，指导 token 裁剪。

**核心 idea**：用 UI 截图的 RGB 颜色相似性构建连通图来识别冗余 patch，在 self-attention 中按连通分量稀疏采样 token，配合交错 VLA 流统一导航和定位任务。

## 方法详解

### 整体框架
ShowUI 基于 Qwen2-VL-2B 构建。输入为用户查询 + 动作空间说明（JSON 格式的 README）+ 截图观测；输出为 JSON 格式的动作预测（动作类型 + 坐标/文本参数）。训练数据仅 256K 样本，涵盖 Web/Mobile/Desktop 三类设备的定位和导航数据。推理时以循环方式执行：预测动作→更新截图→预测下一动作。

### 关键设计

1. **UI 引导的视觉 Token 选择（UI-Guided Visual Token Selection）**:

    - 功能：根据 UI 截图的视觉冗余性自适应裁剪 token，减少计算量同时保留关键元素
    - 核心思路：将截图按 patch 划分后，每个 patch 作为图节点，如果相邻 patch 的 RGB 值差异小于阈值 $\delta$ 则用 Union-Find 合并为一个连通分量。大面积纯色区域会形成大分量（高冗余），而文字/图标区域的分量较小（低冗余）。训练时在每个连通分量内随机采样部分 token（设定比例如 50%），被采样的 token 保留原始位置编码。这样做的效果：Google 搜索页 1296→291 token，Overleaf 文档 1296→986 token，自适应于内容复杂度
    - 设计动机：与 Token Merging 不同（将分量内所有 token 池化为一个 → 丢失位置信息 → 定位任务崩溃），Token Selection 保留了被选 token 的原始位置，因此不影响 self-attention 的位置感知。零额外参数，训练时用、推理时可选用/不用

2. **交错式视觉-语言-动作流（Interleaved VLA Streaming）**:

    - 功能：统一处理导航历史和单帧多查询两种场景，提高训练效率和导航性能
    - 核心思路：设计两种流模式——(a) Action-Visual Streaming 用于导航：截图1→动作1→截图2→动作2→...，模型看到完整的视觉-动作历史来推理下一步；(b) Action-Query Streaming 用于定位：一张截图配多个 query-action 对做多轮对话，避免每个 query 重复编码长截图。动作统一为 JSON 格式（{'action': 类型, 'value': 元素, 'position': [x,y]}），并在 system prompt 中提供动作空间的 README 文档
    - 设计动机：截图 token 通常 1000+，而 query/action 不到 10 个 token。一图一动作太浪费；多轮对话可以复用截图 encoding。对于不同设备的动作差异（Web 无 TAP、Mobile 无 CLICK），用 README 文档让模型"读规则"而非"记动作"，支持测试时遇到新动作

3. **精选小规模高质量训练数据**:

    - 功能：在仅 256K 样本下达到与百万级训练数据可比的性能
    - 核心思路：Web 数据过滤掉 40% 的"静态文本"标签（VLM 本身 OCR 能力已很强），只保留 Button/Checkbox 等视觉元素（22K 截图）；Desktop 数据太少（仅 100 张），用 GPT-4o 对原始标注做逆向工程，生成外观/空间/意图三种查询来扩充（100→6K 元素）；Mobile 引入 AMEX 的功能性描述。训练时用重采样策略平衡不同数据类型
    - 设计动机：数据不在多而在精——过滤低价值数据 + 增强稀缺类型 + 平衡采样，比简单堆量更有效

### 损失函数 / 训练策略
标准自回归 next-token prediction loss，预测 JSON 格式的动作输出。Token Selection 比例 0.5，交叉层插入（cross-layer，隔层使用），学习率等遵循 Qwen2-VL 微调配置。

## 实验关键数据

### 主实验

| 基准 | 指标 | ShowUI (2B, 256K) | 之前最佳 | 说明 |
|--------|------|------|----------|------|
| ScreenSpot 零样本 | 平均精度 | **75.1%** | 73.3% (UGround-7B, 1.3M) | 小 3.5 倍、少 5 倍数据 |
| ScreenSpot Mobile Icon | 精度 | **75.5%** | 60.3% (UGround) | +15.2% |
| AITW Mobile | 成功率 | **70.0%** | 67.2% (Qwen2-VL-2B baseline) | +2.8% |
| Mind2Web Cross-Domain | Step SR | 35.2% | 30.7% (Qwen2-VL-2B baseline) | +4.5% |
| MiniWob Online | 得分 | **71.5** | 67.0 (SeeClick-9.6B) | +4.5 |

### 消融实验

| 配置 | 视觉Token数 | 训练加速 | ScreenSpot |
|------|---------|---------|---------|
| Baseline（无压缩） | 1344.0 | 1× | 70.8 |
| Token Merge (UI-Graph) | 852.8 | 1.6× | 42.3 |
| Token Selection (Random) | 941.5 | 1.5× | 65.3 |
| **Token Selection (UI-Graph)** | **947.4** | **1.5×** | **70.4** |

### 关键发现
- **Token Merge 会严重损害定位性能**（70.8→42.3），因为池化丢失了位置信息；而 Token Selection 通过保留位置编码几乎无损（70.8→70.4），证明位置保留是 GUI grounding 的关键
- **交叉层插入（cross-layer）远优于全层/前半/后半插入**：同样 14 层，cross 70.5% vs early 68.2% vs late 67.6%，说明需要在全深度范围内交替保留完整信息和压缩信息
- **视觉历史对 Mobile 导航很重要（+1.7%）但对 Web 导航帮助有限**：因为 Mobile 跨应用切换导致视觉变化大，而 Web 在同一页面操作截图变化小
- **文本定位能力可跨平台迁移，图标定位不行**：在所有方法上 Text 准确率 >> Icon 准确率，且 Desktop/Web 的图标定位明显弱于 Mobile，暗示需要更多视觉丰富的跨平台训练数据

## 亮点与洞察
- **RGB 连通图是一个极其简洁高效的 UI 理解 prior**：无需任何学习参数，仅用像素颜色就能区分"有信息"和"无信息"的区域，这个思路可以迁移到任何涉及结构化文档/UI 的视觉理解任务
- **Token Selection > Token Merge 的洞察很有价值**：保留位置编码的稀疏采样比丢失位置的池化要好得多，这对所有视觉 token 压缩方法都有启示
- **用 README 文档描述动作空间**而非硬编码动作集，让模型具备处理新动作类型的函数调用能力，非常优雅
- **仅 2B + 256K 就能打 7B + 1.3M**：说明数据质量和模型设计比蛮力堆参数堆数据重要

## 局限与展望
- 在线环境（MiniWob）零样本性能与微调差距巨大（27.1% vs 71.5%），说明离线训练不足以应对 OOD 错误，需要 online learning 策略
- Desktop 训练数据极少（仅 100 张截图），Desktop Icon 的 grounding 准确率偏低（61.1%），需要更多跨平台视觉型训练数据
- 2B 模型的语言推理能力有限，对复杂多步 planning 的能力可能不足
- Mind2Web 的跨网站/跨域泛化仍有提升空间——瓶颈在视觉感知而非文本理解

## 相关工作与启发
- **vs CogAgent (18B, 400K)**: ShowUI 用 9 倍少的参数和 1.6 倍少的数据，在 ScreenSpot 上大幅超越（75.1% vs 47.4%），证明专门针对 UI 的设计比堆大模型有效
- **vs UGround (7B, 1.3M)**: 几乎持平的精度（75.1% vs 73.3%），但 ShowUI 小 3.5 倍且数据少 5 倍，token selection 带来的效率优势明显
- **vs OmniParser**: OmniParser 依赖 GPT-4V 做推理，ShowUI 是独立的端到端模型，更实用
- **vs Magma**: Magma 用 SoM 标注 + 大规模预训练，ShowUI 走轻量化路线 + 数据精选，两者理念互补

## 评分
- 新颖性: ⭐⭐⭐⭐ UI 连通图 token selection 是很巧妙的 UI-specific 设计，交错 VLA 流也有新意
- 实验充分度: ⭐⭐⭐⭐ 覆盖 Web/Mobile/Desktop/Online 四类环境，消融充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题驱动的实验分析很好
- 价值: ⭐⭐⭐⭐ 开源轻量级 GUI agent，实用性强，效率优化思路可广泛迁移

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent](../../CVPR2026/robotics/mergevla_cross-skill_model_merging_toward_a_generalist_vision-language-action_ag.md)
- [\[NeurIPS 2025\] ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning](../../NeurIPS2025/robotics/thinkact_vision-language-action_reasoning_via_reinforced_visual_latent_planning.md)
- [\[CVPR 2025\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_towards_active_perception_and_manipulation_in_vision-language-action_mode.md)
- [\[CVPR 2025\] MoManipVLA: Transferring Vision-Language-Action Models for General Mobile Manipulation](momanipvla_transferring_vision-language-action_models_for_general_mobile_manipul.md)
- [\[ICCV 2025\] CombatVLA: An Efficient Vision-Language-Action Model for Combat Tasks in 3D Action Role-Playing Games](../../ICCV2025/robotics/combatvla_an_efficient_vision-language-action_model_for_combat_tasks_in_3d_actio.md)

</div>

<!-- RELATED:END -->
