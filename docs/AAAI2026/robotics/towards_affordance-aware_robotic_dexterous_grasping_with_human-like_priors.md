---
title: >-
  [论文解读] Towards Affordance-Aware Robotic Dexterous Grasping with Human-like Priors
description: >-
  [AAAI 2026][机器人][灵巧抓取] 提出AffordDex，一个两阶段框架：第一阶段通过模仿学习预训练人类手部运动先验（自然的运动轨迹），第二阶段通过残差模块和VLM引导的负可供性分割（NAA）进行强化学习精炼，实现既像人类一样自然、又功能正确的灵巧机器人抓取（如避开刀刃抓握刀柄）…
tags:
  - "AAAI 2026"
  - "机器人"
  - "灵巧抓取"
  - "功能可供性"
  - "人类运动先验"
  - "负可供性分割"
  - "强化学习"
---

# Towards Affordance-Aware Robotic Dexterous Grasping with Human-like Priors

**会议**: AAAI 2026  
**arXiv**: [2508.08896](https://arxiv.org/abs/2508.08896)  
**代码**: [afforddex.github.io](https://afforddex.github.io/)  
**领域**: 分割  
**关键词**: 灵巧抓取, 功能可供性, 人类运动先验, 负可供性分割, 强化学习

## 一句话总结

提出AffordDex，一个两阶段框架：第一阶段通过模仿学习预训练人类手部运动先验（自然的运动轨迹），第二阶段通过残差模块和VLM引导的负可供性分割（NAA）进行强化学习精炼，实现既像人类一样自然、又功能正确的灵巧机器人抓取（如避开刀刃抓握刀柄），在多个泛化级别上显著超越SOTA。

## 研究背景与动机

灵巧抓取是机器人操作的基础能力。与简单的平行夹爪相比，五指灵巧手更接近人类手部结构，具有更强的灵活性和任务适应性。

**现有方法的核心缺陷**：只关注**低级抓取稳定性指标**（能否抬起物体），忽视了两个关键维度：

**功能可供性感知**（Affordance-aware）：抓取不仅仅是抬起物体。必须考虑功能正确性——例如刀的刀刃虽然几何上适合抓握，但任何接触刀刃的抓取都是功能错误且不安全的。

**类人运动姿态**（Human-like）：现有RL方法可能产生虽然成功但运动学上不自然的关节配置，这些配置在下游任务中效率低下、不可预测，且有害于需要流畅人机交互的场景。

**核心思路**：将自然性和功能正确性解耦后再协同——用人类数据约束运动先验，用视觉语言模型理解物体的功能属性。

## 方法详解

### 整体框架

AffordDex分为两阶段训练：

**第一阶段（Human Hand Trajectory Imitating, HTI）**：在大规模人类手部运动数据集（OakInk2, ~2200条序列）上预训练基础策略 $\pi^H$，建立自然运动的强先验。

**第二阶段（Affordance-aware Residual Learning）**：冻结 $\pi^H$ 的权重，训练轻量级残差模块通过RL将通用的类人运动适配到特定物体交互。受两个关键组件引导：
- **负可供性分割（NAA）**：识别功能上不应接触的物体区域
- **教师-学生蒸馏**：利用特权状态信息提升最终视觉策略

### 关键设计

#### 1. **人类手部轨迹模仿（HTI）**

将任务形式化为RL问题，策略 $\pi^H(a_t|S_t^H)$ 基于当前状态生成灵巧手动作。状态包含机器人状态 $R_t$、物体状态 $O_t$ 和物体点云 $P_t$。

**奖励函数** $r^H$ 包含两项：

**手指模仿奖励**：鼓励灵巧手跟踪参考人类手指姿态：

$$r^H_{finger} = \sum_{f=1}^F w_f \cdot \exp(-\lambda_f \|\mathbf{j}_{d,f} - \mathbf{j}_{h,f}\|_2^2)$$

其中 $\mathbf{j}_{d,f}$ 是灵巧手上第 $f$ 个关键点位置，$\mathbf{j}_{h,f}$ 是参考人类轨迹的对应目标位置。

**平滑性奖励**：通过惩罚关节速度与力矩的乘积来鼓励节能运动。

设计动机：通过大量人类运动数据将策略约束在自然运动的流形上，为后续精炼提供良好初始化。

#### 2. **负可供性分割（NAA）**

核心创新在于将分割问题转换为**分类问题**，巧妙地绕过了VLM在细粒度空间定位上的不足。

**流程**：
1. **程序化纹理化**：对无纹理3D网格进行纹理化（TextPainter），确保VLM可解读
2. **多视图渲染**：从6个基本方向渲染带纹理物体，获得全视图图像集 $I$
3. **VLM查询**：通过GPT-4V获取物体负可供性的详细描述（如"刀刃部分"）
4. **掩码候选生成**：对每张图像用SAM加密集点网格进行穷举分割，NMS去重得到候选掩码集 $M_i$：

$$M_i = \text{NMS}(\text{SAM}(I_i, G_i))$$

5. **CLIP分类选择**：对每个掩码区域制作视觉高亮图像（模糊掩码外区域），用CLIP计算与负可供性文本描述的相似度，选择最高分掩码
6. **3D投影**：将选中的2D掩码投影到3D空间获得负可供性点云 $N_t$

设计动机：直接让CLIP在图像中找"刀刃部分"效果差（VLM擅长图像级理解但难以精确空间定位），但让SAM先生成精确掩码候选，再让CLIP从中选出最匹配的，将困难的分割问题降维为简单的分类问题。NAA是离线一次性过程，每个物体约160秒。

#### 3. **可供性感知的残差学习与蒸馏**

**状态基教师策略**：输入 $S_t^T = \{R_t, O_t, P_t, N_t\}$（含特权物体状态），学习残差动作：

$$a_t = \pi^H(S_t^T) + \pi^T(S_t^T)$$

使用PPO训练，奖励函数 $r^T$ 包含四项：
- $r_d^T$：抓取距离惩罚（手与物体的距离）
- $r_g^T$：目标距离惩罚（物体与目标位置的距离）
- $r_s^T$：成功奖励（物体到达目标时的奖金）
- $r_n^T$：**负可供性惩罚**（手接近负可供性区域的惩罚）

**视觉基学生策略**：仅使用真实世界可获取的信息 $S_t^S = \{R_t, P_t, N_t\}$（无特权物体状态），通过DAgger从教师蒸馏：

$$\pi^S = \arg\min_{\pi^S} \|\pi^T(S_t^T) - \pi^S(S_t^S)\|$$

### 损失函数 / 训练策略

- 第一阶段：PPO + 模仿奖励函数
- 第二阶段：PPO + 多项奖励函数（含负可供性惩罚）→ DAgger蒸馏
- 模拟器：IsaacGym，4096并行环境，RTX 4090
- 策略网络：4层MLP (1024,1024,512,512) + PointNet+Transformer（视觉模式）
- Shadow Hand：24个主动自由度

## 实验关键数据

### 主实验

UniDexGrasp和OakInk2数据集上的表现：

| 方法 | 已见物体 Succ↑ | HLS↑ | AS↓ | 未见物体(已见类) Succ↑ | HLS↑ | AS↓ | 未见类 Succ↑ | HLS↑ | AS↓ |
|------|--------------|------|-----|---------------------|------|-----|-------------|------|-----|
| UniDexGrasp | 73.7 | 6.2 | 16 | 68.6 | 6.1 | 18 | 65.1 | 6.0 | 17 |
| UniDexGrasp++ | 85.4 | 5.4 | 29 | 79.6 | 5.1 | 25 | 76.7 | 4.8 | 28 |
| **AffordDex** | **87.0** | **8.3** | **10** | **82.8** | **7.8** | **14** | **79.2** | **8.0** | **15** |

（视觉基设定，上表为部分数据）

状态基设定中AffordDex达到89.2% Succ / 8.6 HLS / 4 AS，每个维度都大幅领先。

### 消融实验

| 配置 | Succ↑ | HLS↑ | AS↓ | 说明 |
|------|-------|------|-----|------|
| 基线（无HTI/NAA/蒸馏） | 70.1 | 5.0 | 27 | 视觉基，纯RL |
| +HTI | 84.9 | 5.6 | 28 | 成功率大幅提升 |
| +HTI+蒸馏 | 85.8 | 7.2 | 13 | 蒸馏显著提升HLS |
| +HTI+NAA | 86.9 | 8.1 | 20 | NAA降低AS |
| **+HTI+NAA+蒸馏** | **87.0** | **8.3** | **10** | 三者协同最优 |

**模块可迁移性**（应用于UniDexGrasp++）：

| 配置 | Succ↑ | HLS↑ | AS↓ |
|------|-------|------|-----|
| UniDexGrasp++ | 87.9 | 5.4 | 28 |
| +HTI | 88.2 | **7.8** | 23 |
| +NAA | 88.0 | 5.9 | **19** |
| +HTI+NAA | **88.8** | 8.0 | **12** |

HTI和NAA模块可即插即用地增强其他RL方法。

**NAA vs 朴素GPT+SAM**：GPT+SAM直接用MLLM粗定位+SAM分割，常导致整个物体被分割而非特定部位。NAA通过将分割转化为分类问题，实现精确的部位级分割。

### 关键发现

1. HTI提供的人类运动先验对成功率提升最大（从70.1到84.9），说明自然运动不仅更美观，还更有效
2. NAA的AS从27降至10，证明负可供性约束成功引导策略避开功能不当区域
3. 教师-学生蒸馏对HLS贡献显著（7.2→8.3），特权信息帮助学习更精确的抓取
4. AffordDex的模块可迁移到其他方法（如UniDexGrasp++），具有良好的通用性
5. UniDexGrasp++虽然Succ高但AS极高（28-29），说明成功≠功能正确

## 亮点与洞察

1. **问题定义升级**：将灵巧抓取从"能否抬起"扩展到"是否安全+是否自然+是否利于后续操作"，更贴近真实应用需求
2. **负可供性的巧妙建模**：不去学"应该在哪抓"（正面困难且模糊），而是学"不应该碰哪里"（负面约束清晰且明确），大幅简化学习问题
3. **分割→分类的降维思路**：绕过VLM空间定位的弱点，利用SAM的精确分割+CLIP的语义理解，组合出超越二者独立能力的结果
4. **两阶段训练的优雅设计**：先学通用运动先验再学特定物体适配，残差模块保证不破坏已学的自然运动
5. **HLS评估创新**：用Gemini 2.5 Pro作为自动评估器评估抓取的类人程度，虽然不完美但提供了一个可扩展的评估方案

## 局限与展望

1. **固定6视图渲染**：对几何复杂或凹陷物体可能因遮挡导致负可供性分割不精确
2. **仿真到真实的差距**：所有实验在IsaacGym中进行，未展示真实世界部署结果
3. **NAA依赖GPT-4V**：需要商业API且无法本地部署，限制了可重复性和扩展性
4. **仅考虑抓取任务**：负可供性是任务相关的（如切菜时需要握刀刃），未来需根据具体任务动态调整
5. 未来可探索基于隐式3D表示的体积化可供性学习，天然对视角遮挡鲁棒

## 相关工作与启发

- **UniDexGrasp / UniDexGrasp++**：AffordDex的主要对比基线，提出几何感知课程学习但忽视可供性
- **DexGrasp Anything**：基于扩散模型生成静态抓取姿态，但无运动轨迹因此无法评估类人度
- **OakInk2**：提供人类手部操作序列数据集，是HTI阶段的训练数据来源
- **GEAL**：双分支架构做跨模态可供性预测，但任务/类别特定
- NAA的分割→分类思路可推广到其他需要VLM做细粒度空间推理的任务

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （将可供性和类人性引入灵巧抓取，NAA的分割→分类转换很聪明）
- 实验充分度: ⭐⭐⭐⭐ （多设置评估，详尽消融，但缺少真实世界实验）
- 写作质量: ⭐⭐⭐⭐⭐ （问题动机清晰，框架图精美，消融逻辑严密）
- 价值: ⭐⭐⭐⭐⭐ （从任务层面重新定义灵巧抓取评价标准，对具身智能发展意义重大）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] D-REX: Differentiable Real-to-Sim-to-Real Engine for Learning Dexterous Grasping](../../ICLR2026/robotics/d-rex_differentiable_real-to-sim-to-real_engine_for_learning_dexterous_grasping.md)
- [\[CVPR 2025\] DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness](../../CVPR2025/robotics/dexgrasp_anything_towards_universal_robotic_dexterous_grasping_with_physics_awar.md)
- [\[CVPR 2026\] PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation](../../CVPR2026/robotics/palm_progress-aware_policy_learning_via_affordance_reasoning_for_long-horizon_ro.md)
- [\[AAAI 2026\] RLSLM: A Hybrid Reinforcement Learning Framework Aligning Rule-Based Social Locomotion Model with Human Social Norms](rlslm_a_hybrid_reinforcement_learning_framework_aligning_rule-based_social_locom.md)
- [\[AAAI 2026\] GRIM: Task-Oriented Grasping with Conditioning on Generative Examples](grim_task-oriented_grasping_with_conditioning_on_generative_examples.md)

</div>

<!-- RELATED:END -->
