---
title: >-
  [论文解读] MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation
description: >-
  [ICLR 2026][机器人][VLA] 受认知科学双重记忆系统启发，提出MemoryVLA框架，在VLA模型中引入感知-认知记忆库（PCMB），通过记忆检索、门控融合和整合机制捕捉长时序依赖，在SimplerEnv/LIBERO/真实世界150+任务上全面超越CogACT和π₀。
tags:
  - ICLR 2026
  - 机器人
  - VLA
  - 记忆机制
  - 长时序操作
  - 扩散策略
  - 认知科学
---

# MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation

**会议**: ICLR 2026  
**arXiv**: [2508.19236](https://arxiv.org/abs/2508.19236)  
**代码**: [项目页面](https://shihao1895.github.io/MemoryVLA)  
**领域**: 机器人/VLA  
**关键词**: VLA, 记忆机制, 长时序操作, 扩散策略, 认知科学

## 一句话总结
受认知科学双重记忆系统启发，提出MemoryVLA框架，在VLA模型中引入感知-认知记忆库（PCMB），通过记忆检索、门控融合和整合机制捕捉长时序依赖，在SimplerEnv/LIBERO/真实世界150+任务上全面超越CogACT和π₀。

## 研究背景与动机

**领域现状**：VLA模型（OpenVLA、π₀、CogACT）在机器人操作中取得显著进展，但主流方法仅依赖当前观测，忽略时序依赖——在长时序任务上表现差。如Push Buttons任务中，按压前后视觉几乎无差异，无法判断动作是否完成。

**现有痛点**：(1) 拼接多帧→自注意力二次复杂度+与单帧预训练分布不匹配；(2) RoboFlamingo用LSTM压缩→丢失细粒度信息；(3) TraceVLA画轨迹→丢失语义细节；(4) UniVLA加过去动作→只是CoT不是真正的记忆利用。

**核心矛盾**：机器人操作本质是非马尔可夫的（过去动作影响未来决策），但当前VLA模型是马尔可夫的（只看当前帧）。

**切入角度**：认知科学中人类通过工作记忆（短期）+情景记忆（长期，含verbatim细节和gist语义）来处理操作任务。据此设计PCMB存储感知细节和认知语义两个层次的记忆。

## 方法详解

### 整体框架
视觉-语言认知模块(7B VLM) → 感知tokens($p$) + 认知token($c$) = 工作记忆 → PCMB存储/检索/融合/整合 → 记忆条件化扩散动作专家生成动作序列。

### 关键设计

1. **视觉-语言认知模块**:

    - 功能：从当前RGB+语言指令提取感知tokens和认知token
    - 核心思路：并行DINOv2+SigLIP视觉编码 → SE瓶颈压缩为256个感知tokens $p$；LLaMA-7B处理视觉+语言 → EOS位置输出认知token $c$
    - 设计动机：感知tokens保留细粒度视觉信息，认知token编码高层语义理解

2. **感知-认知记忆库 (PCMB)**:

    - **记忆检索**：当前tokens用时间位置编码作为query，对PCMB做cross-attention，检索决策相关历史 $H^p, H^c$
    - **门控融合**：$\tilde{x} = g^x \odot H^x + (1-g^x) \odot x$，$g^x = \sigma(\text{MLP}(\text{concat}[x, H^x]))$，自适应混合当前和历史信息
    - **记忆整合**：容量满时，计算相邻条目余弦相似度，合并最相似的一对 → 保留关键信息的同时控制大小

3. **记忆条件化扩散动作专家**:

    - 功能：以记忆增强的感知+认知tokens为条件，用扩散Transformer生成未来N步7DoF动作
    - 设计动机：扩散策略能捕捉多模态动作分布，记忆条件化使其时序感知

### 损失函数 / 训练策略
- 端到端训练，7B VLM在OXE数据集上预训练
- 扩散动作专家用DDPM训练
- 感知压缩模块SE-bottleneck，认知用EOS token

## 实验关键数据

### 仿真主实验

| 基准 | MemoryVLA | CogACT | π₀ | 提升 |
|------|-----------|--------|-----|------|
| SimplerEnv-Bridge | **71.9%** | 57.3% | 低于 | **+14.6** |
| SimplerEnv-Fractal | **72.7%** | 68.1% | 低于 | +4.6 |
| LIBERO-5 | **96.5%** | 次优 | 次优 | 超越两者 |
| Mikasa-Robo | **41.2%** | — | 29.4% | **+11.8** |

### 真实世界实验（12任务）

| 任务类型 | MemoryVLA | CogACT | π₀ |
|---------|-----------|--------|-----|
| 通用技能(6任务) | **85%** | 76% | 低 |
| 长时序依赖(6任务) | **83%** | 57% | 低→**+26** |

### 关键发现
- 长时序任务提升最显著(+26 vs CogACT)→证明记忆机制对时序依赖至关重要
- 门控融合中 $g$ 的值随任务需要动态变化——简单任务主要用当前信息，复杂任务更多依赖历史
- 记忆整合通过合并相似邻居控制大小，比固定窗口或FIFO更高效
- 在OOD条件（不同背景/干扰物/光照/遮挡）下展现强鲁棒性

## 亮点与洞察
- **认知科学驱动的设计**：工作记忆+情景记忆的双重系统映射到感知tokens+认知token+PCMB。不是简单堆叠帧或LSTM，而是有认知理论支撑的记忆架构。
- **感知vs认知的分离**：感知tokens(256个)保留空间细节，认知token(1个)压缩高层语义。PCMB分两个流存储和检索→不同任务需要不同层次的历史信息。
- **+26在长时序任务上**：这个提升量说明记忆不是锦上添花而是必要条件——没有记忆的VLA在需要时序理解的任务上根本做不好。

## 局限与展望
- 7B VLM推理开销大——实时性受限
- PCMB容量L需要手动设置——自适应容量管理值得探索
- 余弦相似度做整合可能不够精细——更复杂的记忆选择策略可能更好
- 仅用第三人称RGB——多视角+触觉等多模态记忆未探索

## 相关工作与启发
- **vs π₀**: π₀无记忆机制，在长时序任务上差距巨大；MemoryVLA的PCMB填补了这个空白
- **vs CogACT**: CogACT也用扩散动作头但无时序建模，MemoryVLA加了记忆后全面超越
- **vs RoboFlamingo**: RoboFlamingo用LSTM粗粒度记忆，MemoryVLA的双流记忆更细致

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 认知科学启发的双流记忆架构在VLA中首次出现
- 实验充分度: ⭐⭐⭐⭐⭐ 3个机器人、150+任务(仿真+真实)、多baseline、OOD测试
- 写作质量: ⭐⭐⭐⭐ 认知科学动机清晰，架构图直观
- 价值: ⭐⭐⭐⭐⭐ 解决了VLA领域的关键缺失（时序记忆），实验效果convincing

<!-- RELATED:START -->

## 相关论文

- [TwinVLA: Data-Efficient Bimanual Manipulation with Twin Single-Arm Vision-Language-Action Models](twinvla_data-efficient_bimanual_manipulation_with_twin_single-arm_vision-languag.md)
- [SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](../../CVPR2026/robotics/sapave_active_perception_manipulation_vla_roboti.md)
- [Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)](../../CVPR2026/robotics/lada_robotic_manipulation.md)
- [Language-Grounded Decoupled Action Representation for Robotic Manipulation](../../CVPR2026/robotics/language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [When would Vision-Proprioception Policies Fail in Robotic Manipulation?](when_would_vision-proprioception_policies_fail_in_robotic_manipulation.md)

<!-- RELATED:END -->
