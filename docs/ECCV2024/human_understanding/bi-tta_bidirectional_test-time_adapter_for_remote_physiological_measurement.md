---
title: >-
  [论文解读] Bi-TTA: Bidirectional Test-Time Adapter for Remote Physiological Measurement
description: >-
  [ECCV2024][人体理解][rPPG] 提出 Bi-TTA 框架，首次将 Test-Time Adaptation 引入远程光电容积脉搏波 (rPPG) 任务，通过时空一致性自监督先验和前瞻-回溯双向适应策略，在推理时仅用无标注单实例数据即可完成模型域适应。
tags:
  - ECCV2024
  - 人体理解
  - rPPG
  - 测试时自适应
  - 远程生理信号测量
  - 自监督先验
  - 域适应
---

# Bi-TTA: Bidirectional Test-Time Adapter for Remote Physiological Measurement

**会议**: ECCV2024  
**arXiv**: [2409.17316](https://arxiv.org/abs/2409.17316)  
**代码**: [bi-tta.github.io](https://bi-tta.github.io)  
**领域**: human_understanding  
**关键词**: rPPG, 测试时自适应, 远程生理信号测量, 自监督先验, 域适应

## 一句话总结

提出 Bi-TTA 框架，首次将 Test-Time Adaptation 引入远程光电容积脉搏波 (rPPG) 任务，通过时空一致性自监督先验和前瞻-回溯双向适应策略，在推理时仅用无标注单实例数据即可完成模型域适应。

## 背景与动机

远程光电容积脉搏波 (rPPG) 通过普通摄像头从人脸视频中提取血容量脉冲 (BVP) 信号，可非接触式测量心率 (HR)、心率变异性 (HRV) 和呼吸频率 (RF) 等生理指标。相比传统心电设备 (ECG) 和指夹式传感器，rPPG 无需佩戴任何装置，成本更低、使用更便捷。

然而，rPPG 信号极其微弱——皮肤因心跳产生的颜色变化在视频中几乎不可察觉，容易被光照变化、头部运动、相机参数差异等环境因素干扰。现有深度学习方法虽然在受控实验室条件下表现优异，但在部署到新场景 (unseen domain) 时性能显著下降。为了适应新域，Domain Adaptation 需要目标域标注数据，Domain Generalization 则不针对特定目标域优化，均有局限。考虑到隐私约束（不能访问源数据和目标标签），Test-Time Adaptation (TTA) 成为最合适的范式——在推理时利用无标注目标数据自适应调整模型。

## 核心问题

将 TTA 直接应用于 rPPG 面临两个关键挑战：

1. **监督信号缺失**：现有 TTA 方法主要面向分类任务，依赖熵最小化或伪标签，不适用于 rPPG 这类回归任务，缺乏有效的自监督信号。
2. **单实例学习的不稳定性**：实际部署中模型逐帧处理目标域视频，每次仅用单个样本微调。单实例学习带来的偏差和噪声使模型难以区分域相关/域无关特征，且容易遗忘已学知识（灾难性遗忘）或过拟合到噪声特征。

## 方法详解

### 整体框架

Bi-TTA 包含两个正交维度的设计：(1) 基于专家知识的自监督先验提供适应梯度；(2) 前瞻-回溯双向适应策略保证适应过程的有效性和稳定性。

### 时空特征图 (STMap) 构建

输入为人脸视频帧序列。首先进行人脸对齐与裁剪，从不同面部区域提取局部颜色信号，拼接为二维 STMap $\boldsymbol{x} \in \mathbb{R}^{W \times H}$，其中 $W$ 为滑动窗口时序长度 (256帧)，$H$ 为空间维度 (25个面部区域)。STMap 经 resize 至 $256 \times 64 \times 3$ 送入 ResNet-18 网络。

### 时间一致性损失 (TCL)

BVP 信号在短时间内变化平滑连续，因此对原始样本施加随机时间偏移 $\delta_T$（均匀采样自 $(0, 59]$），分别预测原始和偏移样本的心率，用 L1 正则化约束二者差异不超过容忍阈值 $\xi_T$：

$$L_t = \sum_i^W \max(0, \|\text{HR}(\boldsymbol{x}_t) - \text{HR}(\boldsymbol{x}_{t-\delta_T})\|_1 - \xi_T)$$

### 空间一致性损失 (SCL)

心率变化引起的皮肤色变在不同面部区域具有一致性。SCL 在 ResNet-18 四个残差块的多尺度隐层特征图上计算相邻空间位置的 L1 差异：

$$L_s = \sum_i^4 \sum_j^{W_i - \delta_S} \|F_{i,j} - F_{i,j+\delta_S}\|_1$$

多尺度策略使监督信号从浅层纹理到深层语义全面覆盖。总自监督损失为 $L_p = \lambda_s L_s + \lambda_t L_t$。

### 前瞻适应模块 (Prospective Adaptation, PA)

单实例学习中噪声样本容易干扰模型。PA 借鉴 Sharpness-Aware Minimization (SAM) 思想，不直接最小化当前参数点的损失，而是寻找**邻域内最大损失也较低**的平坦区域：

$$L_p'(\boldsymbol{w}) = \max_{\|\boldsymbol{\epsilon}\|_2 \leq \rho} L_p(\boldsymbol{w} + \boldsymbol{\epsilon})$$

通过一阶泰勒展开近似最优扰动方向 $\hat{\epsilon}$，在扰动后的参数点计算梯度 $\boldsymbol{g}_t^{PA}$。这使模型对单实例噪声更鲁棒，过滤掉域无关信息。

### 回溯稳定模块 (Retrospective Stabilization, RS)

PA 虽能抵御单步噪声，但长期适应中模型仍可能捕获有害噪声特征导致性能退化。RS 引入"趋势梯度" $\boldsymbol{g}^*$，按自监督损失加权累积历史梯度。每步计算当前梯度在趋势梯度上的投影：

- 若投影方向与趋势梯度**相反**（检测到振荡），说明当前更新可能损害泛化性能，RS 启用回溯机制，以回溯系数 $k$ 缩放投影梯度；
- 若方向**一致**，则按退火系数 $\lambda_t^{RS}$ 混合当前梯度与趋势梯度。

退火系数 $\lambda_t^{RS}$ 随样本数增加从 0 趋近 1，保证趋势梯度在积累足够样本后（$\Omega = 4000$）主导优化方向。

## 实验关键数据

在 VIPL、V4V、PURE、UBFC、BUAA 五个数据集上建立大规模 TTA 评测基准，评估指标为 MAE↓、RMSE↓、Pearson r↑：

| 方法 | VIPL MAE↓ | PURE MAE↓ | UBFC MAE↓ | BUAA MAE↓ |
|------|-----------|-----------|-----------|-----------|
| NEST (DG) | 7.86 | 6.71 | 4.67 | 2.88 |
| Tent (TTA) | 8.09 | 6.86 | 4.57 | 2.37 |
| EATA (TTA) | 7.69 | 6.13 | 4.25 | 1.89 |
| SHOT (TTA) | 7.75 | 5.81 | 4.05 | 1.87 |
| ConPhys | 7.43 | 6.09 | 3.92 | 1.75 |
| **Bi-TTA (仅先验)** | 7.31 | 5.56 | 3.64 | 1.68 |
| **Bi-TTA (完整)** | **7.09** | **5.02** | **3.53** | **1.49** |

消融实验关键发现：

- 仅用先验（无双向策略）已超越所有基线，验证 TCL+SCL 的有效性
- 去掉 PA：PURE MAE 从 5.02 升至 5.39；去掉 RS：从 5.02 升至 5.24
- PA 和 RS 协同效果优于单独使用，兼顾收敛速度和长期稳定性
- 约 5000-6000 样本后，仅用先验的方法出现明显性能退化，而 Bi-TTA 保持稳定

## 亮点

1. **首次将 TTA 引入 rPPG**：建立了完整的 TTA 评测协议和大规模基准，填补了该交叉领域的空白
2. **巧妙的自监督设计**：TCL 和 SCL 从生理信号的固有时空一致性出发，无需标签即可提供有效监督，且收敛速度甚至快于全监督方法
3. **双向适应策略互补性强**：PA 保证单步鲁棒性（过滤噪声），RS 保证长期稳定性（防遗忘/过拟合），二者正交互补
4. **对预训练模型无额外要求**：不要求特殊网络结构或预训练阶段的辅助任务，适配性强

## 局限与展望

1. **仅验证了 ResNet-18 骨干**：未探索 Transformer 等更现代架构的适应效果
2. **超参数较多**：$\lambda_s, \lambda_t, \xi_T, \rho, k, \Omega$ 共 6 个超参需要调整，虽有消融但跨数据集泛化性未充分讨论
3. **趋势梯度需要累积阶段**：前 $\Omega=4000$ 个样本内 RS 效果有限，冷启动问题存在
4. **计算开销**：SAM 需要两次前向+反向传播，推理时延增加，对实时应用可能是瓶颈
5. **仅评估心率估计**：未验证 HRV 和呼吸频率等其他生理指标的适应效果

## 与相关工作的对比

- **vs DG 方法 (NEST/Coral/VREx)**：DG 不针对特定目标域，且需要所有源数据，Bi-TTA 无需源数据且针对性适应，全面优于 DG 方法
- **vs 通用 TTA (Tent/SAR/EATA/SHOT)**：这些方法依赖分类任务的熵/伪标签，直接用于回归任务效果有限，Bi-TTA 的领域先验提供更合适的监督
- **vs ConPhys**：ConPhys 也利用了时空一致性先验，但 Bi-TTA 在多尺度隐层特征上计算 SCL（而非仅在输出层），且加入双向适应策略，MAE 在 VIPL 上从 7.43 降至 7.09
- **vs AdaODM**：AdaODM 在 V4V 上略优（MAE 9.1 vs 9.1 持平），但其他数据集均不如 Bi-TTA

## 启发与关联

- **自监督先验的设计范式**：从任务固有属性（时间平滑性、空间一致性）出发构造损失函数的思路，可推广到其他生理信号或时序回归 TTA 场景
- **SAM 用于 TTA**：将 Sharpness-Aware Minimization 从训练阶段迁移到测试时适应是有启发性的组合
- **趋势梯度与振荡检测**：RS 模块的梯度方向一致性检测机制对其他在线学习/持续学习场景也有借鉴价值
- 可结合 LoRA 等参数高效微调方法降低 TTA 计算量

## 评分
- 新颖性: ⭐⭐⭐⭐ (首次将 TTA 引入 rPPG，双向策略设计新颖)
- 实验充分度: ⭐⭐⭐⭐ (五数据集全面评测+详细消融)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，可视化丰富)
- 价值: ⭐⭐⭐⭐ (建立新基准，方法可推广)

<!-- RELATED:START -->

## 相关论文

- [Human Motion Forecasting in Dynamic Domain Shifts: A Homeostatic Continual Test-Time Adaptation Framework](human_motion_forecasting_in_dynamic_domain_shifts_a_homeostatic_continual_test-t.md)
- [CRISP: Object Pose and Shape Estimation with Test-Time Adaptation](../../CVPR2025/human_understanding/crisp_object_pose_and_shape_estimation_with_test-time_adaptation.md)
- [FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation](../../CVPR2026/human_understanding/fozo_forward-only_zeroth-order_prompt_optimization_for_test-time_adaptation.md)
- [P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling](../../ICLR2026/human_understanding/p-genrm_personalized_generative_reward_model_with_test-time_user-based_scaling.md)
- [Mingle: Mixture of Null-Space Gated Low-Rank Experts for Test-Time Continual Model Merging](../../NeurIPS2025/human_understanding/mingle_mixture_of_null-space_gated_low-rank_experts_for_test-time_continual_mode.md)

<!-- RELATED:END -->
