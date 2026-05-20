---
title: >-
  [论文解读] DONUT: A Decoder-Only Model for Trajectory Prediction
description: >-
  [ICCV 2025][自动驾驶][轨迹预测] DONUT受LLM中decoder-only架构启发，提出用统一的自回归模型处理历史和未来轨迹，配合"过预测（overprediction）"策略让模型更好预判未来，在Argoverse 2基准上取得SOTA。
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "轨迹预测"
  - "decoder-only"
  - "自回归模型"
  - "运动预测"
---

# DONUT: A Decoder-Only Model for Trajectory Prediction

**会议**: ICCV 2025  
**arXiv**: [2506.06854](https://arxiv.org/abs/2506.06854)  
**代码**: [https://vision.rwth-aachen.de/donut](https://vision.rwth-aachen.de/donut)  
**领域**: 自动驾驶  
**关键词**: 轨迹预测, decoder-only, 自回归模型, 运动预测, 自动驾驶

## 一句话总结
DONUT受LLM中decoder-only架构启发，提出用统一的自回归模型处理历史和未来轨迹，配合"过预测（overprediction）"策略让模型更好预判未来，在Argoverse 2基准上取得SOTA。

## 研究背景与动机
运动预测是自动驾驶的核心任务——通过预测场景中其他智能体的未来轨迹，自动驾驶车辆可以提前规划。当前主流方法采用编码器-解码器架构：编码器嵌入历史轨迹，解码器预测未来轨迹。

**现有痛点**：

**单次预测全部未来**的方式不了解远期时间步周围的场景元素，导致远期预测不准确

**循环解码器**虽能迭代预测，但存在不一致性——输入有时是学习embedding，有时是上一步输出，复杂化了解码器任务
3. 循环解码器只能获取编码器提供的"过时"历史信息，预测远期时其他智能体状态已严重不同步
4. 编码器和解码器使用不同模块，历史和未来轨迹的处理方式存在结构性割裂

**切入角度**：类比LLM的decoder-only成功范式，用同一个自回归模型统一处理历史和未来轨迹序列，确保一致性和信息实时性。受LLM多token预测的启发，引入overprediction策略帮助模型预判更远的未来。

## 方法详解

### 整体框架
将所有智能体轨迹分割为 $T_{\text{sub}}=10$ 个时间步的子轨迹（1秒）。使用同一个decoder网络按自回归方式处理历史子轨迹和逐步预测未来子轨迹。每步包含：proposer生成初始预测和过预测 → 更新参考点 → refiner修正预测。

### 关键设计

1. **统一的Decoder-Only架构**:

    - 功能：用单一模型同时编码历史轨迹和预测未来轨迹
    - 核心思路：
        - 历史轨迹通过proposer和refiner处理，产生历史token
        - 未来子轨迹逐步预测，每步使用相同的网络结构
        - 每次预测后更新参考点到预测终点，重新计算与周围场景元素的相对位置编码
        - 使用query-centric方案（每个场景元素有自己的局部参考系），可复用前一步特征
    - 设计动机：避免encoder-decoder的不一致性（不同输入类型、过时的其他智能体信息），让模型自然地将历史轨迹的学习迁移到未来预测

2. **过预测策略（Overprediction）**:

    - 功能：预测当前子轨迹时，同时预测"下一个"子轨迹作为辅助任务
    - 核心思路：proposer输出预测 $\hat{Y}'_{\{0;T_{\text{sub}}\}}$ 和过预测 $\hat{Y}'^{\text{over}}_{\{T_{\text{sub}};2T_{\text{sub}}\}}$。过预测在训练时有 ground truth 监督，推理时丢弃
    - 设计动机：灵感来自LLM的多token预测。强迫模型考虑更远的时间范围，为当前步预测提供更好的未来感知。实验证明overprediction帮助训练更稳定收敛，并释放refiner的潜力

3. **Proposer与Refiner**:

    - 功能：分别负责初始预测和修正
    - 核心结构：共享相同的architecture
        - Tokenizer：提取每个子轨迹的位置、朝向、运动向量、速度的Fourier特征，MLP融合为单token
        - 四种attention：(1) 时序自注意力（同一智能体的历史token），(2) 地图注意力（半径 $r=50$m 内的道路token），(3) 社交注意力（其他智能体token），(4) 模态注意力（同一智能体不同模态间）
        - Detokenizer：MLP预测下一子轨迹和过预测
    - 设计动机：refiner可在更新参考点后获取其他智能体的最新预测轨迹和更精确的场景信息

### 损失函数 / 训练策略
- 位置使用Laplace混合分布参数化：$p(\hat{Y}_n^{\text{pos}}) = \sum_k P_{n,k} \prod_t \text{Laplace}(\hat{Y}_{n,t}^{\text{pos}} | \mu, b)$
- 朝向使用von-Mises分布参数化
- 仅对最近模态（endpoint距离最小）计算损失
- proposed和refined轨迹、main prediction和overprediction分别独立应用损失
- 训练：AdamW优化器，4×H100 GPU，batch size 64（梯度累积），60个epoch

## 实验关键数据

### 主实验（Argoverse 2 Test Leaderboard，非集成方法）

| 方法 | b-minFDE₆↓ | minFDE₆↓ | minADE₆↓ | MR₆↓ |
|------|-----------|---------|---------|------|
| QCNet | 1.91 | 1.29 | 0.65 | 0.16 |
| DeMo | 1.84 | 1.17 | 0.61 | 0.13 |
| SmartRefine | 1.86 | 1.23 | 0.63 | 0.15 |
| SEPT* | 1.74 | 1.15 | 0.61 | 0.14 |
| QCNet* (集成) | 1.78 | 1.19 | 0.62 | 0.14 |
| DeMo* (集成) | 1.73 | 1.11 | 0.60 | 0.12 |
| **DONUT** | **1.79** | **1.16** | 0.63 | 0.14 |

DONUT在主指标b-minFDE₆上取得非集成方法SOTA，在MR₁指标上取得整体SOTA（0.54）。

### 消融实验

| 配置 | b-minFDE₆↓ | minFDE₆↓ | minADE₆↓ | MR₆↓ | 说明 |
|------|-----------|---------|---------|------|------|
| Encoder-decoder基线 | 1.874 | 1.253 | 0.720 | 0.157 | QCNet |
| Decoder-only | 1.838 | 1.198 | 0.745 | 0.145 | 单独decoder-only提升FDE |
| + Overprediction | 1.838 | 1.193 | 0.728 | 0.146 | 小幅提升 |
| + Refinement | 1.835 | 1.218 | 0.751 | 0.150 | 单独使用训练不稳定 |
| + 两者同时 | **1.807** | **1.176** | **0.722** | **0.144** | 协同效应显著 |

### 关键发现
- Decoder-only架构在b-minFDE₆和minFDE₆上均优于encoder-decoder基线，特别是在长期预测（6秒）中优势明显
- Overprediction和refinement单独使用提升有限，但两者结合产生显著协同效应
- 远期预测精度提升最大：decoder-only在3-6秒时间段的FDE显著低于encoder-decoder
- DONUT在MR₁（单模态miss rate）取得SOTA（0.54），说明最佳模态的预测非常准确
- minADE₆上encoder-decoder基线略优，可能因为其编码器为每个时间步生成独立token（粒度更细）

## 亮点与洞察
- 类比LLM成功范式的核心洞察：运动预测本质上是序列预测任务，decoder-only最适合
- Overprediction策略巧妙借鉴LLM多token预测，为轨迹预测任务量身改编
- 参考点实时更新确保模型始终了解当前位置附近的场景信息
- 定性分析非常直观：在复杂路口场景中DONUT明显优于QCNet

## 局限与展望
- minADE₆上略逊于encoder-decoder基线，粗粒度的子轨迹tokenization可能牺牲了中间步精度
- 当前仅验证了单智能体预测，多智能体联合预测的extension值得探索
- 推理速度分析缺失——自回归方式是否比单次预测慢?
- 对不同 $T_{\text{sub}}$ 的敏感性分析不足

## 相关工作与启发
- 延续QCNet的query-centric场景编码，在其基础上替换decoder结构
- 与LLM领域的Decoder-only趋势（GPT系列）高度呼应
- 与运动仿真领域的GPT-style模型（MotionLM等）不同：运动预测需要固定数量的多模态预测覆盖未来分布
- 启发：NLP领域的成功范式可通过巧妙改编迁移到其他序列预测任务

## 评分
- 新颖性: ⭐⭐⭐⭐ decoder-only + overprediction的组合对轨迹预测领域是新颖的贡献
- 实验充分度: ⭐⭐⭐⭐ 消融充分，leaderboard验证严格，定性分析直观
- 写作质量: ⭐⭐⭐⭐⭐ 动机阐述清晰，与LLM的类比恰到好处
- 价值: ⭐⭐⭐⭐ 在竞争激烈的Argoverse 2上取得SOTA，验证了decoder-only范式在轨迹预测中的有效性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Generative Active Learning for Long-tail Trajectory Prediction via Controllable Diffusion Model](generative_active_learning_for_long-tail_trajectory_prediction_via_controllable_.md)
- [\[ICCV 2025\] Foresight in Motion: Reinforcing Trajectory Prediction with Reward Heuristics](foresight_in_motion_reinforcing_trajectory_prediction_with_reward_heuristics.md)
- [\[ICCV 2025\] LangTraj: Diffusion Model and Dataset for Language-Conditioned Trajectory Simulation](langtraj_diffusion_model_and_dataset_for_language-conditioned_trajectory_simulat.md)
- [\[CVPR 2025\] Tra-MoE: Learning Trajectory Prediction Model from Multiple Domains for Adaptive Policy Conditioning](../../CVPR2025/autonomous_driving/tra-moe_learning_trajectory_prediction_model_from_multiple_domains_for_adaptive_.md)
- [\[CVPR 2025\] Certified Human Trajectory Prediction](../../CVPR2025/autonomous_driving/certified_human_trajectory_prediction.md)

</div>

<!-- RELATED:END -->
