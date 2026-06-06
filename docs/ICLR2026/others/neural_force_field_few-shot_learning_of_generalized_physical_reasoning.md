---
title: >-
  [论文解读] Neural Force Field: Few-shot Learning of Generalized Physical Reasoning
description: >-
  [ICLR 2026][neural force field] 提出Neural Force Field（NFF），将物体交互建模为连续力场，通过神经算子学习力场函数并用ODE积分器解码轨迹，在I-PHYRE（100条轨迹）、N-body（200条轨迹）、PHYRE（0.012M数据…
tags:
  - "ICLR 2026"
  - "neural force field"
  - "Neural ODE"
  - "few-shot physical reasoning"
  - "ODE solver"
  - "interactive planning"
---

# Neural Force Field: Few-shot Learning of Generalized Physical Reasoning

**会议**: ICLR 2026  
**arXiv**: [2502.08987](https://arxiv.org/abs/2502.08987)  
**代码**: [项目页面](https://neuralforcefield.github.io/)  
**领域**: 其他  
**关键词**: neural force field, Neural ODE, few-shot physical reasoning, ODE solver, interactive planning

## 一句话总结

提出Neural Force Field（NFF），将物体交互建模为连续力场，通过神经算子学习力场函数并用ODE积分器解码轨迹，在I-PHYRE（100条轨迹）、N-body（200条轨迹）、PHYRE（0.012M数据,比SOTA少267倍）三个基准上实现少样本SOTA，跨场景RMSE降低32-64%,规划任务接近人类水平。

## 研究背景与动机

**领域现状**：物理推理是AI核心能力之一。人类能从少量物理现象中快速抽象核心原理并泛化到新环境，但现有AI模型即使在海量数据上训练仍在OOD场景中挣扎。

**现有痛点**：
    - 现有GNN/Transformer方法（IN、SlotFormer）通过隐式向量表示物体交互→倾向于过拟合观测轨迹而非捕获物理原理→OOD泛化差
    - 离散隐空间解码→无法解释物体如何穿越障碍物（如绿球穿过黑墙）→物理不一致
    - 少样本设置下过拟合风险更大→需要强物理归纳偏置
    - 交互式推理需要主动实验和反馈适应→现有方法缺乏反向规划能力

**核心矛盾**：需要一种既能从极少样本学习、又能在OOD场景泛化的物理表示——这要求表示本身编码物理原理而非统计模式。

**本文目标** 开发具有人类式少样本物理学习能力的agent，在多样化环境中实现鲁棒泛化。

**切入角度**：力场是物理学的自然抽象——力是运动变化的因果原因。将交互表示为力场而非状态转换，天然可组合、可泛化。

**核心 idea**：用神经算子学习连续力场函数，通过ODE积分保证物理一致性，力场的低维性使少样本学习成为可能。

## 方法详解

### 整体框架

NFF框架三步走：(1) 构建动态交互图，物体为节点，接触/吸引关系为边 → (2) 神经算子预测连续力场 $\mathbf{F}(\mathbf{z}^q(t))$ → (3) ODE积分器（Runge-Kutta/Euler）将力场积分为速度和位移轨迹。训练时将长轨迹分段进行自回归预测，最小化MSE损失。

### 关键设计

1. **神经算子力场（Neural Operator Force Field）**:
    - 功能：从物体状态和交互图预测连续力场
    - 核心思路：基于DeepONet框架，力场函数为 $\mathbf{F}(\mathbf{z}^q(t)) = \sum_{i \in \mathcal{G}(q)} \mathbf{W}(f_\theta(\mathbf{z}^i(t)) \odot f_\phi(\mathbf{z}^q(t))) + \mathbf{b}$，其中 $\mathcal{G}(q)$ 是查询物体的邻居集合，$f_\theta$ 和 $f_\phi$ 是neural networks，$\odot$ 是逐元素乘积，$\mathbf{W} \in \mathbb{R}^{d_\text{hidden} \times d_\text{force}}$ 将隐特征映射到低维力空间
    - 设计动机：力场是低维的（2D/3D力向量）→比高维隐向量更容易从少量数据学习；神经算子的函数空间学习能力使力场模式可泛化到新交互图

2. **ODE积分轨迹解码**:
    - 功能：将学习到的力场转换为物理一致的轨迹
    - 核心思路：二阶ODE描述运动——$\mathbf{a}^q(t) = \frac{d^2 x^q(t)}{dt^2} = \frac{\mathbf{F}(\mathbf{z}^q(t))}{m^q}$，通过积分得到 $\mathbf{x}(t) = \mathbf{x}(0) + \int_0^t \mathbf{v}(t)dt$, $\mathbf{v}(t) = \mathbf{v}(0) + \int_0^t \frac{\mathbf{F}(\mathbf{z}^q(t))}{m^q}dt$
    - 设计动机：ODE积分保证轨迹的连续性和物理一致性——不会出现离散解码中"物体穿墙"的问题；高精度积分（步长$1e\text{-}3$）提升细粒度碰撞建模

3. **前向-后向交互规划**:
    - 功能：利用学习到的力场进行目标导向的规划任务
    - 核心思路：前向规划——采样500个action候选，用NFF作为心理模拟器评估，选最优序列执行；后向规划——反转ODE时间方向，从目标状态反演初始条件：$\mathbf{x}(0) = \mathbf{x}(t) + \int_t^0 \mathbf{v}(t)dt$
    - 设计动机：ODE的可逆信息流使后向计算天然高效；5轮交互学习协议（执行→观察偏差→更新模型→重新规划）模拟人类trial-and-error学习

### 损失函数 / 训练策略

训练使用MSE损失最小化预测与真实轨迹的差异。关键策略：将长轨迹分段为小单元训练（alleviates accumulated error in teacher forcing），评估时仅给初始状态预测全部未来动态。

## 实验关键数据

### 主实验：轨迹预测（RMSE↓为主指标）

| 基准 | 设置 | IN | SlotFormer | SEGNO | **NFF** | 提升 |
|------|------|-----|-----------|-------|---------|------|
| I-PHYRE | Within | 0.124 | 0.067 | 0.203 | **0.048** | 28%↓ vs SlotFormer |
| I-PHYRE | Cross | 0.194 | 0.206 | 0.314 | **0.131** | 32%↓ vs IN |
| N-body | Within [0,T] | 0.200 | 0.214 | 0.079 | **0.097** | — |
| N-body | Cross [0,3T] | 6.942 | 2.533 | 2.759 | **1.226** | 52%↓ vs SlotFormer |
| PHYRE | Cross AUCCESS↑ | — | 21.04 | — | **49.22** | +134% vs SlotFormer |

### 消融实验（N-body Cross RMSE↓）

| 配置 | Cross RMSE | 说明 |
|------|-----------|------|
| NFF (1e-3精度) | 1.226 | 完整模型 |
| NFF (5e-3精度) | 1.251 | 精度降低→性能下降 |
| NFF (自适应) | 1.788 | 自适应积分不如固定高精度 |
| w/o ODE (退化为IN) | 3.518 | ODE grounding至关重要 |
| w/o NOL (用MLP替代DeepONet) | 1.347 | 神经算子提升泛化 |

### 关键发现

- **数据效率惊人**：I-PHYRE仅100条轨迹（10个游戏×10个样本），N-body仅200条，PHYRE仅12K条（比RPIN的3.2M少267倍）
- **力场可视化验证**：学习到的重力场与真实引力场高度吻合（图5b），碰撞/滑动/摩擦力场也被正确捕获（图5a）
- **ODE grounding是泛化关键**：去掉ODE后Cross RMSE从1.226暴增到3.518（2.87×）
- **规划接近人类**：在I-PHYRE交互规划中，NFF经5轮精化后的累积成功概率接近人类水平，而IN和SlotFormer甚至低于随机采样
- **物体一致性**：PHYRE视觉任务中，RPIN会错误地将灰色杯子变形为灰色球，SlotFormer出现物体消失，NFF保持物体一致性

## 亮点与洞察

- **"力场 = 物理的正确抽象层次"**：不是学"状态如何转换"而是学"为什么转换"——力是运动变化的因果原因，因果表示天然可泛化
- **连续 vs 离散的本质差异**：离散解码无法解释物体穿墙（图2），连续ODE积分自然避免物理不一致
- **少样本的物理直觉对应**：人类也是从少量经验抽取物理规律（如婴儿的直觉物理），NFF的低维力场表示模拟了这一认知过程
- **后向规划的优雅**：反转ODE时间方向即可从目标反演初始条件，比基于梯度的迭代优化效率高数量级（表A3）

## 局限与展望

- 仅在合成/抽象推理数据集上测试，未验证真实物理场景
- 假设确定性刚体环境，未探索随机环境或软体/流体
- 训练单一模型时变化摩擦力和弹性可能带来额外挑战
- 视觉输入版本依赖物体掩码，未实现端到端从像素到力场的学习

## 相关工作与启发

- **vs IN (Battaglia et al., 2016)**：IN用隐向量+离散转换，Cross RMSE比NFF高2.87×，规划低于随机采样
- **vs SlotFormer (Wu et al., 2023)**：SlotFormer用Transformer+slot attention，在PHYRE Cross中AUCCESS仅21.04（NFF=49.22），且有物体消失问题
- **vs SEGNO (Liu et al., 2024b)**：SEGNO也用ODE但无力场表示，Within性能有时优于NFF但Cross泛化差（2.759 vs 1.226）
- **vs Kofinas et al. (2023)**：也使用"场"概念但学latent field而非显式力场，NFF更物理grounded

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将物理学的力场概念引入学习系统，ODE积分保证物理一致性，是物理推理表示学习的范式创新
- 实验充分度: ⭐⭐⭐⭐ 3个基准（I-PHYRE/N-body/PHYRE）+ 预测/规划多设置 + 详尽消融 + 力场可视化
- 写作质量: ⭐⭐⭐⭐⭐ 物理直觉和方法设计完美结合，图示清晰（尤其图2的连续vs离散对比），motivation有力
- 价值: ⭐⭐⭐⭐ 对物理推理、认知AI和few-shot learning有基础贡献，力场表示可能启发更广泛的物理世界模型研究

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Is Meta-Learning Out? Rethinking Unsupervised Few-Shot Classification with Limited Entropy](../../ICCV2025/others/is_meta-learning_out_rethinking_unsupervised_few-shot_classification_with_limite.md)
- [\[CVPR 2026\] DirPA: Addressing Prior Shift in Imbalanced Few-shot Crop-type Classification](../../CVPR2026/others/dirpa_addressing_prior_shift_in_imbalanced_fewshot.md)
- [\[ICLR 2026\] Learning on a Razor's Edge: Identifiability and Singularity of Polynomial Neural Networks](learning_on_a_razors_edge_identifiability_and_singularity_of_polynomial_neural_n.md)
- [\[ICML 2026\] Amortized Simulation-Based Inference in Generalized Bayes via Neural Posterior Estimation](../../ICML2026/others/amortized_simulation-based_inference_in_generalized_bayes_via_neural_posterior_e.md)
- [\[ICCV 2025\] Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection](../../ICCV2025/others/doodle_your_keypoints_sketch-based_few-shot_keypoint_detection.md)

</div>

<!-- RELATED:END -->
