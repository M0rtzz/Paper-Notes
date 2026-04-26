---
title: >-
  [论文解读] CityWalker: Learning Embodied Urban Navigation from Web-Scale Videos
description: >-
  [CVPR 2025][城市导航] 利用互联网上超过 2000 小时的城市步行和驾驶视频，通过视觉里程计 (VO) 自动提取动作标签进行大规模模仿学习，训练出能在复杂动态城市环境中导航的具身智能体，真实部署成功率达 77.3%，显著超越现有方法。
tags:
  - CVPR 2025
  - 城市导航
  - 模仿学习
  - 视觉里程计
  - 网络视频训练
  - 具身智能
---

# CityWalker: Learning Embodied Urban Navigation from Web-Scale Videos

**会议**: CVPR 2025  
**arXiv**: [2411.17820](https://arxiv.org/abs/2411.17820)  
**代码**: https://ai4ce.github.io/CityWalker/  
**领域**: 具身导航 / 强化学习  
**关键词**: 城市导航, 模仿学习, 视觉里程计, 网络视频训练, 具身智能

## 一句话总结
利用互联网上超过 2000 小时的城市步行和驾驶视频，通过视觉里程计 (VO) 自动提取动作标签进行大规模模仿学习，训练出能在复杂动态城市环境中导航的具身智能体，真实部署成功率达 77.3%，显著超越现有方法。

## 研究背景与动机

**领域现状**：视觉导航在室内模拟器中已取得近乎完美的表现（point-goal navigation 被认为是"已解决"的问题），但在城市户外场景中仍是未解决难题。现有方法主要在静态或简单环境中工作。

**现有痛点**：城市导航面临行人交互、交通信号灯、障碍物绕行、人行道规范等复杂约束，这些难以在模拟器中建模。遥操作收集专家数据成本高、规模小、多样性不足。部分工作依赖大语言模型/VLM 生成动作标签，成本高且难以规模化。

**核心矛盾**：要在真实城市环境中学会导航，需要大规模多样的训练数据，但传统方法（遥操作/模拟器）难以提供。互联网上有海量的城市步行视频，但缺乏动作标签。

**本文目标** 如何从无标注的网络视频中自动提取动作监督信号，实现大规模模仿学习。

**切入角度**：作者发现现成的视觉里程计 (VO) 工具虽然全局轨迹不精确，但短时窗口内的相对位姿足够可靠，可以作为模仿学习的动作伪标签。

**核心 idea**：用 VO 从网络城市步行视频中提取动作伪标签，通过大规模模仿学习训练城市导航策略。

## 方法详解

### 整体框架
输入：过去 k=5 帧的 RGB 图像 + 过去 k 步轨迹坐标 + 目标位置。图像通过冻结的 DINOv2 编码器提取特征，坐标通过可学习编码器嵌入。Transformer 处理时序 token 序列，输出通过 action head 预测未来 5 步动作（2D 位移），通过 arrival head 预测是否到达子目标。训练数据来自 2000+ 小时的网络城市视频。

### 关键设计

1. **VO-based 动作标签提取**:

    - 功能：从无标注视频中自动生成动作监督信号
    - 核心思路：使用 DPVO 从视频中提取帧间相对位姿作为动作标签。虽然 VO 存在全局累积误差和尺度歧义，但模型只需预测短时窗口（5 步）的相对动作，累积误差影响极小。尺度歧义通过对每条轨迹按平均步长归一化来消除——这 simultaneously 解决了不同视频来源（步行 vs 驾驶）和不同机器人的步长不一致问题
    - 设计动机：相比 VLM prompting 方案（如 LeLaN），VO 方案可以完全并行化处理，2000 小时视频的处理成本几乎可忽略

2. **Feature Hallucination Loss**:

    - 功能：辅助训练目标，让模型学会预测未来视觉特征
    - 核心思路：计算 Transformer 输出的 image token 与未来帧真实特征之间的 MSE 损失。引导模型生成能模拟未来观测的信息性 token，间接提升动作预测质量。注意：在 zero-shot 推理时该损失反而有害（因为模型会倾向预测人类视角的未来帧），但微调后问题消失
    - 设计动机：受 feature learning 启发，预测未来特征迫使模型建模环境动态

3. **跨域跨具身训练**:

    - 功能：通过混合步行和驾驶视频提升泛化能力
    - 核心思路：驾驶视频虽然来自不同域和不同具身形式，但经过步长归一化后可以统一到相同的抽象动作空间中。实验发现仅 250 小时混合数据就接近 1000 小时纯步行数据的效果，显示跨域数据的互补价值
    - 设计动机：充分利用互联网上更丰富的驾驶视频资源

### 损失函数 / 训练策略
总损失为四项加权和：L1 动作损失 + 方向损失（预测与 GT 动作的负余弦相似度）+ 到达状态的 BCE 损失 + 特征幻觉 MSE 损失。方向损失权重设为 5.0，其他为 1.0。预训练用 2000 小时网络视频，微调用 6 小时遥操作数据（纽约城市场景）。

## 实验关键数据

### 主实验

| 方法 | MAOE↓ (场景均值) | 真实部署成功率 | 前进 | 左转 | 右转 |
|------|-----------------|--------------|------|------|------|
| ViNT (zero-shot) | 17.5° | 37.7% | 62.5% | 0.0% | 50.0% |
| ViNT (fine-tuned) | 16.5° | 57.1% | 100% | 25.0% | 25.0% |
| NoMaD (fine-tuned) | 19.1° | 42.9% | 75.0% | 16.7% | 28.6% |
| CityWalker (zero-shot) | 16.5° | - | - | - | - |
| **CityWalker (fine-tuned)** | **15.2°** | **77.3%** | **100%** | **62.5%** | **66.7%** |

### 消融实验

| 配置 | MAOE (场景均值) |
|------|---------------|
| 基线（无 ori loss / 无 feat hall / 无微调） | 17.03° |
| + 方向损失 | 17.00° |
| + 方向损失 + 特征幻觉 | 17.02° |
| + 微调 | 15.23° |
| + 方向损失 + 微调 | 15.21° |
| + 全部组件 | **15.16°** |

### 关键发现
- 微调是最大的性能提升来源（17.03→15.23），方向损失和特征幻觉的边际贡献较小
- 数据规模效应显著：超过 1000 小时训练数据后，zero-shot 模型就能超越微调的 ViNT
- 跨域训练（步行+驾驶混合）效果惊人：250 小时混合数据 ≈ 1000 小时纯步行数据
- CityWalker 在转弯场景（左转 62.5%、右转 66.7%）远超基线（最高仅 25-50%），说明大规模数据帮助模型学到复杂操控策略

## 亮点与洞察
- **VO 替代 VLM 做标签**：用简单高效的 VO 工具替代昂贵的 VLM prompting 获取动作标签，是一个极具实用价值的工程决策。处理 2000 小时视频的成本几乎为零
- **数据规模的 scaling law**：明确展示了导航性能随数据量增长的趋势，1000 小时是一个关键拐点。这个发现可以指导未来数据收集策略
- **步长归一化统一异构数据**：一个简单的归一化技巧就消除了跨域（步行/驾驶）和跨具身（人/四足机器人）的差异，优雅且实用

## 局限与展望
- iPhone GPS 定位噪声敏感，实际部署依赖 GPS 精度
- 微调仍需遥操作数据（6 小时），未实现完全 zero-shot 的真实部署
- 只在纽约城市测试，未验证跨城市泛化能力
- "绕行"(detour) 场景表现较弱，因为训练视频中此类数据比例低
- 未考虑语义地图或高层规划，仅做 waypoint 间的局部导航

## 相关工作与启发
- **vs ViNT**: ViNT 是视觉导航基础模型，主要在郊区/越野场景训练，城市环境泛化差。CityWalker 专注城市场景，利用网络视频获得更大更多样的训练集
- **vs NoMaD**: NoMaD 用扩散模型预测动作，但在复杂城市场景中表现不佳（成功率仅 42.9%）
- **vs LeLaN**: 并发工作，依赖 VLM prompting 和预训练导航模型生成标签，成本高且难以规模化
- 对自动配送机器人、自动驾驶末端导航等应用有直接参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 用 VO 从网络视频提取动作标签的思路简单有效
- 实验充分度: ⭐⭐⭐⭐ 有真实部署实验和 scaling law 分析，但测试规模偏小
- 写作质量: ⭐⭐⭐⭐ 问题驱动的写作风格清晰
- 价值: ⭐⭐⭐⭐ 开启了利用网络视频训练城市导航的新范式

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Embodied Navigation with Auxiliary Task of Action Description Prediction](../../ICCV2025/reinforcement_learning/embodied_navigation_with_auxiliary_task_of_action_description_prediction.md)
- [\[ICCV 2025\] NavQ: Learning a Q-Model for Foresighted Vision-and-Language Navigation](../../ICCV2025/reinforcement_learning/navq_learning_a_q-model_for_foresighted_vision-and-language_navigation.md)
- [\[NeurIPS 2025\] DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/deepdiver_adaptive_search_intensity_scaling_via_open-web_reinforcement_learning.md)
- [\[NeurIPS 2025\] Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/memo_training_memory-efficient_embodied_agents_with_reinforcement_learning.md)
- [\[ICCV 2025\] RoboFactory: Exploring Embodied Agent Collaboration with Compositional Constraints](../../ICCV2025/reinforcement_learning/robofactory_exploring_embodied_agent_collaboration_with_compositional_constraint.md)

<!-- RELATED:END -->
