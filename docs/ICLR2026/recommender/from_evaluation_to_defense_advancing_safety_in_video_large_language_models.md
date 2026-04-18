---
title: >-
  [论文解读] From Evaluation to Defense: Advancing Safety in Video Large Language Models
description: >-
  [ICLR2026][video LLM safety] 构建 VideoSafetyEval（11.4k 视频-查询对覆盖 19 种风险类别）揭示视频模态使安全性能下降 34.2%，提出 VideoSafety-R1 三阶段框架（报警 Token+SFT+Safety-guided GRPO）在 VSE-HH 上提升 71.1% 防御成功率。
tags:
  - ICLR2026
  - video LLM safety
  - benchmark
  - alarm token
  - GRPO
  - safety alignment
---

# From Evaluation to Defense: Advancing Safety in Video Large Language Models

**会议**: ICLR2026  
**arXiv**: [2505.16643](https://arxiv.org/abs/2505.16643)  
**代码**: 待确认  
**领域**: multimodal_vlm  
**关键词**: video LLM safety, benchmark, alarm token, GRPO, safety alignment

## 一句话总结
构建 VideoSafetyEval（11.4k 视频-查询对覆盖 19 种风险类别）揭示视频模态使安全性能下降 34.2%，提出 VideoSafety-R1 三阶段框架（报警 Token+SFT+Safety-guided GRPO）在 VSE-HH 上提升 71.1% 防御成功率。

## 研究背景与动机
**领域现状**：图像 LLM 的安全风险已被广泛研究（MMBench、SIUO、SafeVLM 等），但视频 LLM 的安全对齐严重不足。视频的时间动态、视觉线索和演化上下文引入了比静态图像更微妙且更有效的风险。

**现有痛点**：对 21 个主流视频 LLM 的系统测试发现，引入视频模态后防御成功率（DSR）平均下降 34.2%，暴露了多模态攻击利用中的系统性风险。VideoLLaMA3-2B 的 DSR 降幅高达 79.4%。

**安全研究空白**：现有防御方法（SafeVLM、SPA-VL、MM-RLHF）均聚焦静态图像，忽略了视频安全。视频异常检测（VAD）虽相关但目标不同——VAD 关注检测异常事件，而安全对齐关注控制模型在有害输入下的行为响应。

**核心设计理念**：安全对齐应从单纯的"危害感知"升级为"主动推理"——模型不仅要识别有害内容，还要通过推理链分析视频-文本对的有害性并生成有帮助的安全响应。

## 方法详解

### 整体框架
VideoSafety-R1 是一个后训练框架，包含三个创新组件：VideoSafetyThinking 数据集 → AT-SFT（报警 Token 引导的安全微调） → Safety-guided GRPO（安全引导的强化学习）。

### 关键设计

1. **VideoSafetyEval (VSE) 基准**

    - 11.4k 视频-查询对，覆盖 6 大风险类别（暴力、管制物品、色情等）、19 个子类别、10 种语言社区
    - 三个子集：VSE-HH（有害视频+有害查询，最强对抗），VSE-SH（安全视频+有害查询），VSE-SafeQ（安全查询，评估误拒率）
    - 数据来源：YouTube，经 DINOv2 静态过滤 → 商业视频理解模型标注 → 模板驱动查询生成

2. **报警 Token 引导安全微调 (AT-SFT)**

    - 在视觉序列末尾注入可学习报警 Token $\mathbf{h}_v^{\text{alarm}}$，文本序列末尾注入 $\mathbf{h}_t^{\text{alarm}}$
    - 多任务训练目标：$\mathcal{L}_{\text{AT-SFT}} = \mathcal{L}_{\text{base}} + \lambda_1 \mathcal{L}_{\text{ATC}}^v + \lambda_2 \mathcal{L}_{\text{ATC}}^t$
    - ATC（报警 Token 分类）对视觉和文本分别进行二分类（有害/安全），使报警 Token 的隐藏状态与安全信号对齐
    - 作为安全机制的"预激活"步骤，为后续 GRPO 训练奠定基础

3. **Safety-guided GRPO**

    - 冷启动阶段：用 15k 样本训练结构化思维链（`<think>` 安全推理 + `<answer>` 响应 + `<vidType>`/`<textType>` 双模态标签）
    - 复合奖励函数：$r = r_{\text{format}} + \alpha \cdot r_{\text{ROUGE}} + \gamma_1 \cdot r_v + \gamma_2 \cdot r_t$
    - 动态奖励适应（DRA）：当双模态分类均正确时降低 ROUGE 权重（鼓励多样性），分类错误时增强 ROUGE（强制对齐安全参考）
    - $\alpha = \alpha_{\min} + (1 - \text{Correct}_v \cdot \text{Correct}_t)(\alpha_{\max} - \alpha_{\min})$

### VideoSafetyThinking 数据集
46k 视频-查询-思维链三元组：6k 用于 AT-SFT，15k 用于冷启动 SFT，25k 用于 GRPO 训练。

## 实验关键数据

### 主实验：21 个视频 LLM 在 VSE-HH 上的表现
| 模型 | DSR(有视频)↑ | DSR(无视频) | DSR 降幅↓ | 帮助度↑ |
|------|------------|-----------|----------|--------|
| Gemini-2.5-Pro | 86.7% | 99.5% | 12.8% | 1.6 |
| GPT-4o | 73.0% | 98.4% | 25.9% | 2.2 |
| VideoLLaMA3-2B | 18.4% | 89.3% | **79.4%** | 2.3 |
| InternVideo2.5-8B | 16.5% | 53.5% | 69.2% | 1.0 |

### VideoSafety-R1 效果
| 指标 | 基线(VideoLLaMA3-2B) | VideoSafety-R1 | 提升 |
|------|------|------|------|
| VSE-HH DSR | 18.4% | — | **+71.1%** |
| MMBench DSR | — | — | **+59.1%** |
| VLGuard | — | — | **+44.3%** |
| FigStep | — | — | **+15.0%** |

### 关键发现
- 视频模态引入使所有模型的安全性显著退化——即使是 GPT-4o 也下降 25.9%
- 越依赖高效视频编码（1fps）的模型退化越严重（VideoLLaMA3 降 79.4% vs VideoLLaMA2 降 7.3%）
- VideoSafety-R1 在 19 个子类别中的 18 个上达到最高 DSR
- 安全提升的同时不显著损害通用能力——帮助度评分保持合理水平
- 模型可泛化到图像安全基准（MMBench/VLGuard/FigStep），说明安全推理能力可迁移

## 亮点与洞察
- 首个大规模真实世界视频 LLM 安全基准——基于 YouTube 社区准则，贴合实际场景
- 从感知（AT-SFT 报警 Token）到推理（Safety-guided GRPO 思维链）的渐进式安全对齐设计——不是简单拒绝而是生成有帮助的安全响应
- 动态奖励适应机制优雅地平衡了安全性和响应质量——分类正确时放松 ROUGE 约束鼓励自然回复
- 双模态独立标注（视频有害性 vs 文本有害性）的设计使模型能区分不同来源的风险

## 局限与展望
- 安全分类的二值标签（有害/安全）可能过于粗糙，细粒度风险等级未考虑
- 过度防御（误拒率）需要与安全性做权衡——VSE-SafeQ 子集可评估但论文未深入分析
- 基线模型为 VideoLLaMA3-2B（2B 参数），对更大模型（7B+）的效果未充分验证
- 46k 训练数据的标注质量依赖商业 LLM，存在标注偏差风险
- 评估依赖 Qwen-Long API 作为判断器，可能引入评估偏差

## 相关工作与启发
- **vs SafeVLM/SPA-VL**: 聚焦静态图像安全，本文首次系统处理视频安全
- **vs 视频异常检测 (UCF-Crime/XD-Violence)**: VAD 检测异常事件，本文控制模型行为响应——目标不同
- **vs MM-RLHF**: 用 DPO 做视觉安全对齐，本文用 GRPO+规则奖励——更可控
- **vs SafeWatch-Bench**: 关注视频内容安全理解，本文关注模型反应安全对齐——互补方向

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个系统的视频 LLM 安全工作，填补关键空白
- 实验充分度: ⭐⭐⭐⭐⭐ 21 个模型评估 + 4 个安全基准 + 多组件消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，三组件层层递进
- 价值: ⭐⭐⭐⭐⭐ 为视频 LLM 安全研究奠定基准和方法基础
