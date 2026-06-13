---
title: >-
  [论文解读] Generative Modeling of Shape-Dependent Self-Contact Human Poses
description: >-
  [ICCV 2025][图像生成][Self-Contact] 构建首个大规模精确形状标注的自接触姿态数据集Goliath-SC（383K姿态/130个subject），提出形状条件的部件感知潜在扩散模型PAPoseDiff来建模体型依赖的自接触姿态分布，并利用学到的扩散先验进行单视角姿态refinement，在unseen subject上超越BUDDI和SMPLer-X等SOTA方法。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "Self-Contact"
  - "Body Shape"
  - "扩散模型"
  - "SMPL-X"
  - "Pose Generation"
  - "Pose Refinement"
---

# Generative Modeling of Shape-Dependent Self-Contact Human Poses

**会议**: ICCV 2025  
**arXiv**: [2509.23393](https://arxiv.org/abs/2509.23393)  
**代码**: [https://tkhkaeio.github.io/projects/25-scgen](https://tkhkaeio.github.io/projects/25-scgen)  
**领域**: 图像生成 / 人体姿态建模 / 自接触姿态  
**关键词**: Self-Contact, Body Shape, diffusion model, SMPL-X, Pose Generation, Pose Refinement

## 一句话总结
构建首个大规模精确形状标注的自接触姿态数据集Goliath-SC（383K姿态/130个subject），提出形状条件的部件感知潜在扩散模型PAPoseDiff来建模体型依赖的自接触姿态分布，并利用学到的扩散先验进行单视角姿态refinement，在unseen subject上超越BUDDI和SMPLer-X等SOTA方法。

## 研究背景与动机

### 问题定义
自接触姿态（如摸脸、交叉手臂、手放在身体上）在日常生活中极为常见，且与心理状态表达、手语交流密切相关。关键挑战：自接触姿态本质上受体型约束——同一个"揉肚子"的动作，瘦体型和胖体型的人呈现完全不同的姿态和接触区域。

### 现有方法的局限

**数据集不足**：
   - HumanSC3D：仅1K自接触姿态/6个subject
   - MTP：1.6K姿态/148个subject，但无配对RGB图像导致annotation不准确
   - InterHand2.6M/Decaf：聚焦于手-手或手-脸的局部交互，忽略全身姿态对接触的影响

**方法局限**：
   - 回归方法（SMPLer-X）：通过ViT直接回归姿态，缺乏接触先验
   - BUDDI：学习两个body的联合分布，但非潜在扩散且无部件感知注意力
   - 现有方法建模pose-shape联合分布，未显式建模"pose依赖于shape"的条件关系

### 核心洞察
体型（BMI、骨骼比例等）决定了自接触姿态的可行空间：低BMI的人搓肚子时手臂角度与高BMI的人完全不同。需要显式建模 $p(\theta | \beta)$ 而非 $p(\theta, \beta)$。

## 方法详解

### 整体框架
PAPoseDiff包含三个阶段：
1. **数据构建**：从Goliath多相机dome捕获 → SMPL-X拟合 → 接触图分析 → 筛选自接触帧
2. **生成训练**：部件式SMPL-X参数 → 各部件自动编码器编码为潜在空间 → 形状条件+时间步条件的自注意力Transformer → 去噪预测
3. **单视角refinement**：初始SMPL-X估计 → 加噪到中间步 → 以2D关键点fitting为引导的条件去噪 → 输出refined姿态

### 关键设计1：部件感知潜在扩散（PAPoseDiff）

**数据表示**：
- 目标（去噪对象）：$\mathbf{X} = [\theta_f, \theta_{rh}, \theta_{lh}, \theta_b]$，分别为脸+表情、右手、左手、身体的姿态参数
- 条件：SMPL-X的shape参数 $\mathbf{I} \in \mathbb{R}^{300}$

**潜在扩散**：
- 为每个部件训练自动编码器，将高维姿态参数编码为低维潜在表示
- 在潜在空间进行扩散，因为自接触姿态被限制在身体表面附近的低维流形上

**部件感知自注意力**：
- 将face/right_hand/left_hand/body/time/shape的embedding拼接为序列
- 自注意力使模型能学习部件间交互（如手与身体的接触关系）
- 添加部件可学习embedding以促进部件感知的关系学习

### 关键设计2：形状条件扰动

直接条件dropout（$c=\varnothing$，shape=0）对应"平均体型"，缺乏个体特异信号。替代方案——shape参数微扰动：

$$\mathbf{c} = \mathbf{I} + s_I \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)$$

以30%概率替换，$s_I = 10^{-4}$。假设：相似体型的人执行相似自接触姿态。这比conditional dropout更好地增强subject多样性。

### 关键设计3：训练损失

$$\mathcal{L}_D = \lambda_\theta L_\theta + \lambda_v L_v + \lambda_{col} L_{col}$$

- $L_\theta$：姿态参数L1损失
- $L_v$：SMPL-X网格顶点L1损失
- $L_{col}$：碰撞损失——使用射线追踪碰撞检测器（仅检测手相关碰撞，避免腋下误检）

### 关键设计4：单视角姿态refinement（Algorithm 1）

给定初始估计 $\mathbf{X}_0^{init}$，仅从最后10%步长开始去噪：
1. 加噪初始姿态到步数$n_r=100$
2. 每步去噪时加入2D关键点fitting梯度引导
3. 可选：Blended Pose Denoising——仅refine感兴趣的身体部位（如上半身），其余保持初始值

关键优势：无需额外fine-tuning（对比InterHandGen的SDS方法），可直接用于任何2D/3D估计器的输出。

## 实验关键数据

### 数据集：Goliath-SC
- 383K自接触姿态，130个subject（70F/56M/4NB）
- 220台RGB相机的多相机dome + 3D全身扫描
- 脚本化动作指令（如"揉脖子"），30Hz捕获
- 训练集313K，评估集9.7K（unseen subjects）

### 自接触姿态生成结果

| 方法 | FID↓ | KID(×10⁻³)↓ | Diversity↑ | Precision↑ | Recall↑ | Col.ratio↓ |
|------|------|-------------|------------|------------|---------|------------|
| VPoser*(shape-cond) | 9.16 | 0.882 | 3.20 | 1.0 | 0.005 | 1.37 |
| BUDDI*(shape-cond) | 2.66 | 1.12 | 5.59 | 0.995 | 0.488 | 1.47 |
| **PAPoseDiff(Ours)** | **1.25** | **0.430** | 5.98 | 0.985 | **0.708** | 1.52 |

FID比BUDDI*降低53%，Recall（覆盖度）从0.488提升到0.708。

### 消融实验

| 模型变体 | FID↓ | Diversity↑ | Col.ratio↓ |
|----------|------|------------|------------|
| w/o Shape cond. | 2.18 | 5.52 | 1.92 |
| w/o PASA | 1.42 | 5.74 | 1.62 |
| w/o Shape rand. | 1.27 | 5.89 | 1.41 |
| w/o Anti-col. | 1.28 | 6.01 | 1.85 |
| **Full model** | **1.25** | 5.98 | 1.52 |

关键发现：
1. Shape条件是最重要因素：移除后FID从1.25升到2.18（+74%）
2. 部件感知自注意力(PASA)贡献显著（FID 1.42→1.25）
3. Anti-collision guidance有效降低碰撞率（1.85→1.52）同时保持FID

### 单视角姿态估计refinement（MPJPE mm）

| 初始估计器 | 初始 | +2D fitting | +BUDDI* | +Ours(w/o shape) | +Ours |
|------------|------|-------------|---------|------------------|-------|
| Hand4Whole | 126.3 | 89.5 | 74.5 | 37.9 | **35.3** |
| HybrIK-X | 82.3 | 51.8 | 65.0 | 45.9 | **32.4** |
| SMPLer-X | 58.0 | 41.7 | 71.7 | 33.7 | **31.8** |
| SMPLer-X† | 42.0 | 41.7 | - | - | - |

关键发现：
1. PAPoseDiff先验在所有初始估计器上都显著改善结果，初始越差改善越大
2. Shape条件refine手部误差尤为明显（如HybrIK-X：85.5→58.7 mm）
3. BUDDI*在初始估计较好时反而有限（SMPLer-X：58.0→71.7），说明精细接触先验需要更精确的建模
4. 形状条件先验优于无条件先验，尤其在手和身体部位

## 亮点与洞察

1. **数据集的里程碑意义**：Goliath-SC比现有数据集大2个数量级（383K vs 1-4K），首次提供精确body shape标注的全身自接触数据，覆盖130个不同体型的subject
2. **条件生成 vs 联合生成**：首次提出将自接触姿态建模为 $p(\theta|\beta)$ 而非 $p(\theta,\beta)$——体型是"约束"而非"自由度"，这一假设被实验有力验证
3. **Shape perturbation > Conditional dropout**：对于连续条件变量（如SMPL-X shape），微扰动比置零更能增强泛化——因为零shape对应"模板体型"，而微扰对应"相似体型"
4. **高效refinement**：仅用最后10%步长(100步)去噪，无需额外训练，可插拔到任何SMPL-X估计器之后
5. **碰撞检测的实用优化**：限制在手相关碰撞，避免腋下等解剖特殊区域的误检

## 局限性

1. 手-手交互的精细碰撞仍难以完全消除（高自由度区域）
2. 仅处理有接触的场景，扩展到非接触的一般姿态建模（如MTP）是有待探索的方向
3. 数据集仅包含脚本化动作，自然场景中的自发自接触（如无意识摸脸）未覆盖
4. 仅建模局部姿态（忽略全局朝向和平移），限制了部分应用场景
5. 时间维度和语言内容的融入留待未来工作

## 相关工作与启发

- **Blended Latent Diffusion**的思想被巧妙用于姿态refinement——只refine感兴趣的身体部位，其余保持原始估计
- 与BUDDI的核心差异：(1)潜在空间扩散 vs 参数空间扩散；(2)部件级自注意力 vs 整体处理；(3)条件生成 vs 联合生成
- 射线追踪碰撞检测器（follow Müller et al.）的效率优势，可用于训练时在线损失计算
- 形状插值实验（Fig.4）展示了平滑的shape-pose流形——从大体型到瘦体型的渐变中，生成姿态持续保持合理接触

## 评分 ⭐⭐⭐⭐
- 创新性：⭐⭐⭐⭐（shape条件扩散+部件感知注意力+高效refinement的组合设计新颖）
- 实验：⭐⭐⭐⭐⭐（包含生成评估+估计refinement+消融+形状插值定性分析，数据集本身是重要贡献）
- 写作：⭐⭐⭐⭐（逻辑清晰，问题动机明确，Algorithm 1的refinement流程描述精炼）
- 实用性：⭐⭐⭐⭐（先验模型可插拔到任何SMPL-X估计器后，无需重训练，但数据集未开源限制复现）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] DreamDance: Animating Human Images by Enriching 3D Geometry Cues from 2D Poses](dreamdance_animating_human_images_by_enriching_3d_geometry_cues_from_2d_poses.md)
- [\[ICCV 2025\] SummDiff: Generative Modeling of Video Summarization with Diffusion](summdiff_generative_modeling_of_video_summarization_with_diffusion.md)
- [\[ICLR 2026\] Direct Reward Fine-Tuning on Poses for Single Image to 3D Human in the Wild](../../ICLR2026/image_generation/direct_reward_fine-tuning_on_poses_for_single_image_to_3d_human_in_the_wild.md)
- [\[ECCV 2024\] NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model](../../ECCV2024/image_generation/nl2contact_natural_language_guided_3d_hand-object_contact_modeling_with_diffusio.md)
- [\[ICCV 2025\] Efficient Autoregressive Shape Generation via Octree-Based Adaptive Tokenization](efficient_autoregressive_shape_generation_via_octree-based_adaptive_tokenization.md)

</div>

<!-- RELATED:END -->
