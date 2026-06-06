---
title: >-
  [论文解读] AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention
description: >-
  [CVPR 2026][机器人][VLA模型] 从POMDP视角重新审视VLA模型的视觉处理，提出AVA-VLA框架通过循环状态和主动视觉注意力模块，根据历史上下文动态调制当前帧的视觉token重要性，在LIBERO和CALVIN等基准上达到SOTA。
tags:
  - "CVPR 2026"
  - "机器人"
  - "VLA模型"
  - "主动视觉注意力"
  - "POMDP"
  - "循环状态"
  - "视觉token调制"
---

# AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention

**会议**: CVPR 2026  
**arXiv**: [2511.18960](https://arxiv.org/abs/2511.18960)  
**代码**: [项目页面](https://liauto-dsr.github.io/AVA-VLA-Page)  
**领域**:机器人
**关键词**: VLA模型, 主动视觉注意力, POMDP, 循环状态, 视觉token调制

## 一句话总结

从POMDP视角重新审视VLA模型的视觉处理，提出AVA-VLA框架通过循环状态和主动视觉注意力模块，根据历史上下文动态调制当前帧的视觉token重要性，在LIBERO和CALVIN等基准上达到SOTA。

## 研究背景与动机

视觉-语言-动作（VLA）模型在机器人操作任务中展现了显著进展，但大多数方法在每个时间步独立处理视觉观测，隐式地将机器人操作建模为马尔可夫决策过程（MDP）。这种无历史设计存在根本缺陷：

1. 真实机器人控制本质上是部分可观测的（POMDP），当前帧无法完整描述环境状态
2. 视觉注意力仅由静态语言指令引导，无法根据历史动作抑制时间冗余信息
3. 模型无法预判"接下来应该关注什么"，视觉系统是被动的而非主动的

例如，在"打开炉灶并把摩卡壶放上去"的任务中，vanilla OpenVLA-OFT无法定位任务关键的"炉灶开关"，而AVA-VLA通过利用历史上下文可以稳定聚焦。

## 方法详解

### 整体框架

当前观测 + 上一步循环状态 → AVA模块计算视觉token软权重 → 调制LLM骨干各层注意力矩阵 → 循环状态初始化动作placeholder → 并行解码动作块 → 输出动作 + 更新循环状态。

### 关键设计

1. **循环状态（Recurrent State）**:
    - 功能：作为POMDP中信念状态的神经近似，编码历史上下文
    - 核心思路：从前一时间步LLM最后一层的动作相关隐藏状态通过MLP投影得到，同时用于初始化当前步的动作placeholder
    - 设计动机：直接计算理论信念状态不可行，用循环结构的压缩表示近似

2. **主动视觉注意力（AVA）模块**:
    - 功能：根据历史信息动态调制视觉token的重要性
    - 核心思路：先用FiLM将语言指令特征条件化视觉特征，再以视觉token为Query、循环状态为Key/Value做交叉注意力+自注意力，最终输出每个视觉token的软权重（增强/削弱二分类后的加权分数）
    - 设计动机：使视觉系统从"被动看到什么处理什么"转变为"根据历史经验主动聚焦关键区域"

3. **软注意力矩阵调制**:
    - 功能：将AVA输出的软权重应用到LLM骨干各层的注意力计算中
    - 核心思路：构建软注意力矩阵U，对视觉token位置施加权重，在Softmax之前乘以注意力分数
    - 设计动机：层共享权重确保一致的视觉聚焦，且不改变LLM骨干的基本结构

### 损失函数 / 训练策略

- 动作预测MAE损失 + L2正则化（约束软权重均值接近目标值c，避免过于分散）
- 截断时间反向传播（T=4步），平衡计算可行性与时间动态学习
- 初始循环状态为零向量，每个episode开始时重置

## 实验关键数据

### 主实验

| 基准 | 指标 | AVA-VLA | OpenVLA-OFT | 提升 |
|------|------|---------|-------------|------|
| LIBERO (全部4套) | 平均SR | 98.0% | 96.8% | +1.2% |
| LIBERO-Long | SR | 97.6% | 95.3% | +2.3% |
| CALVIN ABC→D | 平均长度 | 4.65 | 4.28 | +0.37 |
| 真实机器人 | 平均SR | 最高 | 次高 | 多任务提升 |

### 消融实验

| 配置 | LIBERO平均SR | 说明 |
|------|-------------|------|
| OpenVLA-OFT基线 | 96.8% | 无历史信息 |
| + 状态初始化 | 97.5% | 循环状态注入动作placeholder |
| + AVA模块 | 97.5% | 视觉token重加权 |
| + 两者结合 | 98.0% | 互补效果 |

### 关键发现

- 视觉token裁剪实验：裁剪70%视觉token后性能仍超过基线OpenVLA-OFT（97.3 vs 96.8），验证AVA模块有效识别了关键区域
- 不同骨干实验：在OpenVLA-7B、LLaMA2-7B、Qwen2.5-0.5B上均有提升，通用性好
- 可视化显示AVA权重一致聚焦于机器人接触区域和目标物体

## 亮点与洞察

- POMDP理论视角为VLA模型的历史建模提供了优雅的理论基础
- AVA模块轻量且即插即用，不改变LLM骨干结构
- 软权重的副产品——视觉token裁剪潜力，为VLA效率优化提供方向
- 在最具挑战性的LIBERO-Long和CALVIN长序列任务上改进最显著

## 局限与展望

- 截断反向传播（T=4）限制了长期依赖的学习
- 循环状态仅来自上一步，未探索更长记忆窗口
- 软权重仅调制注意力矩阵，未直接修改视觉特征表示
- 真实机器人实验数据量较少（30-450条演示）

## 相关工作与启发

- **vs OpenVLA/UniVLA**: 自回归解码动作，无历史建模；AVA-VLA通过循环状态保留时间上下文
- **vs CoT-VLA**: 使用思维链进行推理但不显式建模视觉注意力的时间动态
- **vs SP-VLA/FLOWER**: 关注视觉token效率裁剪，但不基于历史上下文做主动聚焦

## 评分

- 新颖性: ⭐⭐⭐⭐ POMDP视角+主动视觉注意力的结合在VLA领域新颖
- 实验充分度: ⭐⭐⭐⭐⭐ LIBERO/CALVIN/真实机器人全覆盖，消融/可视化/裁剪分析充分
- 写作质量: ⭐⭐⭐⭐ 动机清晰，理论推导简洁，实验呈现规范
- 价值: ⭐⭐⭐⭐ 为VLA模型提供了时间感知的视觉处理新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_active_perception_manipulation_vla_roboti.md)
- [\[CVPR 2026\] HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for Vision-Language-Action Models](hif-vla_hindsight_insight_and_foresight_through_motion_representation_for_vision.md)
- [\[CVPR 2026\] Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[ICML 2026\] Latent Reasoning VLA: Latent Thinking and Prediction for Vision-Language-Action Models](../../ICML2026/robotics/latent_reasoning_vla_latent_thinking_and_prediction_for_vision-language-action_m.md)

</div>

<!-- RELATED:END -->
