---
title: >-
  [论文解读] UIPro: Unleashing Superior Interaction Capability for GUI Agents
description: >-
  [ICCV 2025][LLM Agent][GUI 智能体] 提出 UIPro，通过构建 2060 万 GUI 理解样本进行预训练并提出统一动作空间整合异构 GUI agent 任务数据，实现跨移动端、Web 端和桌面端的 SOTA GUI 交互性能。
tags:
  - ICCV 2025
  - LLM Agent
  - GUI 智能体
  - 统一动作空间
  - GUI grounding
  - 视觉语言模型
  - 多平台交互
---

# UIPro: Unleashing Superior Interaction Capability for GUI Agents

**会议**: ICCV 2025  
**arXiv**: [2509.17328](https://arxiv.org/abs/2509.17328)  
**代码**: [GitHub](https://github.com/ZJULiHongxin/UIPro)  
**领域**: llm_agent  
**关键词**: GUI 智能体, 统一动作空间, GUI grounding, 视觉语言模型, 多平台交互

## 一句话总结

提出 UIPro，通过构建 2060 万 GUI 理解样本进行预训练并提出统一动作空间整合异构 GUI agent 任务数据，实现跨移动端、Web 端和桌面端的 SOTA GUI 交互性能。

## 研究背景与动机

构建能像人类一样操作图形界面的自主 GUI agent 是 AI 的长期愿景。GUI 交互的核心能力包括：(1) GUI 元素的视觉理解和定位（grounding）；(2) 规划和执行符合用户目标的动作序列。

现有方法面临两个关键瓶颈：

**数据规模不足**：现有 GUI 交互数据集通常缺乏足够的规模和场景多样性。大规模训练的优势在小规模下无法显现（涌现能力），但 CogAgent（2.47 亿）和 ScreenAI（4.21 亿）等大规模数据集未公开

**训练流程缺陷**：不同 GUI 轨迹数据集采用**异构动作空间**（如 AITW 定义 swipe 为 DUAL_POINT(start, end)，AndroidControl 用 scroll(direction)），直接混合训练会导致动作冲突和性能下降

**核心思路**：(1) 构建最大规模开源 GUI 理解数据集（2060 万样本），为 agent 奠定强 grounding 基础；(2) 设计统一动作空间整合异构数据源，释放多源数据的潜力。

## 方法详解

### 整体框架

UIPro 采用两阶段训练：
- **阶段 1：GUI 理解预训练** — 用 2060 万多平台多任务 GUI 理解样本训练，获得强 grounding 能力
- **阶段 2：GUI agent 任务微调** — 用统一动作空间整合后的 agent 轨迹数据微调，获得动作预测能力

两个基座模型：UIPro-SLiME（3B，从零训练）和 UIPro-Qwen2VL（7B，基于 Qwen2-VL 微调）。

### 关键设计

1. **大规模 GUI 理解数据构建**：从多来源采集并清洗 GUI 数据（Common Crawl 网页、Android 模拟器、RICO、MobileViews 等），生成 13 种任务类型的 <截图, 指代表达, 坐标> 三元组：

    - **元素描述（elemgnd/elemref）**：描述视觉外观、元素类型和位置
    - **用户意图（intentgnd）**：描述用户如何与元素交互，如"点击密码输入框"
    - **上下文功能（funcgnd/funcref）**：描述交互可供性，如"此元素使用户能分享内容"
    - **文本定位（textgnd/OCR）**、**图标分类（icongnd/iconref）**、widget 列表、GUI 问答和 GUI 摘要
    - 最终 2060 万样本关联 250 万唯一截图，67% 新标注、33% 清洗自开源数据

2. **统一动作空间设计**：针对异构动作定义的冲突，设计**动作超集**：

    - 统一 swipe 为 `swipe(start, direction, distance)`，兼容 AITW 的 DUAL_POINT 和 AndroidControl 的 scroll(direction)
    - 为移动端、Web 端、桌面端分别定义统一动作空间（移动端含 tap, long_press, drag, input_text, swipe, navigate 等 12 种动作）
    - 统一为 JSON 格式输出，如 `{"action_type": "click", "target": (x, y)}`
    - 不在 prompt 中包含动作定义（实验发现排除后训练更高效）

3. **系统化去噪流程**：因原始 GUI 数据噪声严重（95.9% 主页有可访问性错误，某数据源噪声率达 29%），设计七步去噪：

    - 检测空白元素（颜色标准差 < 5）
    - OCR 检测不可见元素
    - 移除无效/过大/过小边界框
    - 移除重复框和不匹配元素

### 损失函数 / 训练策略

- 预训练阶段：坐标归一化至 [0, 1000]，UIPro-SLiME 全量融合ViT 冻结训练 1 epoch，UIPro-Qwen2VL 用 440 万子集 LoRA 微调
- Agent 微调阶段：6 epochs 直到性能饱和，prompt 包含任务描述和动作历史，GT action 格式化为 JSON 对象
- 移动端混合 6 个数据源（38 万样本），Web 端混合 3 个数据源（14.5 万样本）

## 实验关键数据

### 主实验（表格）

**AITW 移动端基准（Step SR%）**：

| 方法 | 规模 | General | Install | GoogleApps | Single | WebShop | Overall |
|------|------|---------|---------|------------|--------|---------|---------|
| GPT-4V-OmniParser | - | 48.3 | 57.8 | 51.6 | 77.4 | 52.9 | 57.7 |
| SeeClick | 10B | 54.0 | 66.4 | 54.9 | 63.5 | 57.6 | 59.3 |
| OS-ATLAS | 7B | 57.9 | 63.4 | 55.5 | 79.1 | 59.7 | 63.1 |
| **UIPro-Qwen2VL** | **7B** | **64.4** | **74.6** | **67.9** | **79.4** | **67.6** | **70.4** |
| **UIPro-SLiME** | **3B** | **67.0** | **71.4** | **65.4** | 73.2 | 62.9 | **68.0** |

**Mind2Web Web 端基准（Step SR%）**：

| 方法 | 规模 | Cross-Task | Cross-Website | Cross-Domain |
|------|------|------------|---------------|--------------|
| OmniParser (GPT-4V) | - | 39.4 | 36.5 | 42.0 |
| OS-ATLAS | 7B | 36.7 | 35.7 | 37.2 |
| **UIPro-Qwen2VL** | **7B** | **48.4** | **43.6** | **45.5** |

### 消融实验（表格）

**GUI 理解预训练数据量影响**：

| 预训练数据量 | 平均 Grounding 准确率 | AITW Step SR | AndroidControl Step SR |
|-------------|---------------------|-------------|----------------------|
| 0 | ~30% | ~52% | ~40% |
| 5.9M | ~55% | ~63% | ~55% |
| 20.6M | ~60% | ~68% | ~61% |

**统一动作空间影响**：混合数据源但不统一动作空间导致所有基准显著性能下降，主要原因是动作类型准确率大幅降低和 swipe 方向预测不一致。

### 关键发现

- 3B 的 UIPro-SLiME 超越 18B 的 CogAgent 和 GPT-4V-OmniParser
- Grounding 准确率与下游 agent 任务性能呈正相关——grounding 是 agent 的基础
- 统一动作空间不仅提升共有动作准确率，也提升特有动作（如 Wait）准确率，说明跨任务知识迁移和数据多样性的正则化效果
- GUI 理解数据和 agent 任务数据均展现清晰的 scaling law
- 去噪带来的提升在所有 6 个 grounding 基准上一致显著

## 亮点与洞察

- 最大规模开源 GUI 理解数据集（2060 万），涵盖 13 种任务类型
- 统一动作空间的设计理念简洁有效——用超集兼容不同定义，不同平台共享相似交互原则
- 系统化去噪流程揭示了 GUI 数据质量问题的严重性（某数据源 29% 噪声率）
- 功能性 grounding 任务（funcgnd）的加入是一个重要贡献——让模型理解元素"能做什么"而非仅"是什么"

## 局限与展望

- 桌面环境训练数据远少于移动和 Web，限制了 UIPro 在 Windows/MacOS 上的表现
- 当前仅支持离线评估（offline evaluation），on-device 实时交互评估待探索
- 动作空间统一仍是手动设计，未来可探索自动学习跨平台动作对齐
- AITW 等基准未考虑替代解决方案，导致评估可能偏低

## 相关工作与启发

- 相比 SeeClick（5.3M 数据），UIPro 4x 数据量且加入功能性标注
- 相比 OS-ATLAS（13.6M 数据），UIPro 多 50% 且在多数基准上性能更优
- 统一动作空间思路可推广到其他多源混合训练场景（如机器人操作的 action space 统一）

## 评分

- 新颖性: ⭐⭐⭐⭐ （统一动作空间 + 大规模数据工程的系统性贡献）
- 实验充分度: ⭐⭐⭐⭐⭐ （5 个基准 + 6 个 grounding 基准 + 全面消融 + 迁移实验）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，细节完整）
- 价值: ⭐⭐⭐⭐⭐ （对 GUI agent 社区的数据和方法论贡献均很重要）
