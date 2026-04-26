---
title: >-
  [论文解读] OmniGaze: Reward-inspired Generalizable Gaze Estimation in the Wild
description: >-
  [NeurIPS 2025][多模态][注视估计] 提出OmniGaze，一个半监督3D注视估计框架，利用融合视觉嵌入、MLLM生成的语义注视描述和几何方向向量的奖励模型来评估伪标签质量，在140万无标签人脸数据上训练，在5个数据集的域内/跨域设置下达到SOTA，并在4个未见数据集上展示零样本泛化能力。
tags:
  - NeurIPS 2025
  - 多模态
  - 注视估计
  - 半监督学习
  - 伪标签
  - 奖励模型
  - 跨域泛化
---

# OmniGaze: Reward-inspired Generalizable Gaze Estimation in the Wild

**会议**: NeurIPS 2025  
**arXiv**: [2510.13660](https://arxiv.org/abs/2510.13660)  
**代码**: [GitHub](https://github.com/quhongyu/OmniGaze)  
**领域**: 多模态VLM  
**关键词**: 注视估计, 半监督学习, 伪标签, 奖励模型, 跨域泛化

## 一句话总结

提出OmniGaze，一个半监督3D注视估计框架，利用融合视觉嵌入、MLLM生成的语义注视描述和几何方向向量的奖励模型来评估伪标签质量，在140万无标签人脸数据上训练，在5个数据集的域内/跨域设置下达到SOTA，并在4个未见数据集上展示零样本泛化能力。

## 研究背景与动机

3D注视估计旨在从人脸图像直接预测注视方向，是VR、人机交互、医学诊断等领域的基础技术。然而，现有方法面临两大核心挑战：

**标注数据稀缺且多样性不足**：现有标注数据集（如ETH-XGaze、Gaze360）通过预定义假设简化了真实场景（如使用全景相机、多视角摄影），但数据量和场景多样性仍然有限，导致模型在新域上性能显著下降。

**现有泛化方法受限于标注覆盖**：跨域泛化方法（域适应、域泛化）虽然有效，但始终受制于标注源域数据的覆盖范围。而海量无标签人脸图像可以通过网络爬取或生成模型轻松获取。

**半监督框架在注视估计中的空白**：弱监督方法受限于需要注视相关域的标注，自监督预训练的前置任务（如图像重建、注视重定向）与注视估计的语义相关性弱，利用无标签数据效率低。关键的是，现有SSL的伪标签选择策略（如FixMatch的固定阈值、FlexMatch的类别感知阈值）**专为分类任务设计**，无法直接应用于注视方向这样的连续回归任务。

这一空白驱动了OmniGaze的设计：一个专门为连续回归任务设计的半监督框架，通过奖励模型评估伪标签质量，同时利用大规模无标签数据提升泛化能力。

## 方法详解

### 整体框架

OmniGaze采用标准的三阶段半监督训练流程：（1）教师模型在标注数据上监督训练；（2）教师模型为无标签数据生成伪标签，奖励模型筛选高质量伪标签；（3）学生模型在标注数据+筛选后伪标签数据上联合训练。核心创新在于奖励模型的设计和多模态线索的融合。

### 关键设计

1. **多模态线索融合的奖励模型**：奖励模型 $h_G$ 从三个层面评估伪标签的可靠性：

    - **视觉线索**：使用CLIP视觉编码器提取输入人脸图像的视觉嵌入 $\boldsymbol{f}_k^v$
    - **语义线索**：使用MLLM（InstructBLIP）对图像提问"这个人在3D空间中看向哪里"，获得自然语言描述，通过CLIP文本编码器编码为 $\boldsymbol{f}_k^t$
    - **几何线索**：将伪标签的yaw/pitch角度转换为3D方向向量

   视觉和语义线索通过交叉注意力融合为语义感知注视表征：
    $\hat{\boldsymbol{f}}_k^v = \text{AvgPool}(\text{LN}(\text{CrossAttn}(\boldsymbol{f}_k^v, \boldsymbol{f}_k^t)))$

   其中视觉特征为query，文本特征为key/value，使奖励模型能动态关注相关语义线索来解读视觉特征。

2. **球坐标到笛卡尔坐标的几何表征**：将伪标签 $(θ_k, ψ_k)$ 转换为3D方向向量：
    $\boldsymbol{v}_k = [\cos(ψ_k)\cdot\sin(θ_k), \cos(ψ_k), \cos(ψ_k)\cdot\cos(θ_k)]$

   这种表征比角度更具表达性和连续性，有利于与语义感知表征的精确对齐。奖励模型通过交叉注意力+MLP预测置信度分数：
    $\hat{r}_k = \text{Sigmoid}(\text{MLP}(\text{CrossAttn}(\hat{\boldsymbol{f}}_k^v, \boldsymbol{v}_k)))$

3. **标签评分器与最终置信度**：将奖励模型的初始置信度 $\hat{r}_k$ 和学生模型预测与伪标签之间的余弦相似度一起输入轻量MLP，得到最终置信度：
    $r_k = \text{Sigmoid}(\text{MLP}([(\hat{r}_k, \text{sim}(\hat{y}_k, y_k))]))$

   该设计鼓励奖励模型和学生模型"相互校验"，提升评估鲁棒性。

### 损失函数 / 训练策略

- **监督损失**（有标签数据）：角度L2损失 $\mathcal{L}^s = \frac{1}{N_L}\sum\|\hat{y} - y_i^l\|_2$

- **奖励模型损失**：二元交叉熵，ground truth标签标记为可信（$c_k=1$），伪标签标记为不可信（$c_k=0$），让奖励模型学会区分：
  $$\mathcal{L}^g = \sum -(c_k\log(r_k) + (1-c_k)\log(1-r_k))$$

- **无监督损失**（伪标签数据）：带阈值过滤和置信度加权的角度损失：
  $$\mathcal{L}^u = \sum \mathbb{1}[r_j \geq 0.5] \cdot r_j \cdot \|h_S(x_j^u) - y_j^u\|_2$$

  阈值 $τ=0.5$：低于阈值的伪标签被丢弃，高置信伪标签获更大权重。

- **周期性伪标签更新**：每K=10个epoch将学生模型权重更新到教师模型，重新生成伪标签，避免早期过拟合噪声标签。

- **总训练目标**：$\mathcal{L}^s$ 和 $\mathcal{L}^u$ 的平均组合。

## 实验关键数据

### 主实验——域内注视估计（角度误差°↓）

| 方法 | MPIIFaceGaze | EyeDiap | RT-Gene | Gaze360 | IVGaze |
|------|:---:|:---:|:---:|:---:|:---:|
| GazeTR (ICPR'22) | 4.00 | 5.17 | 6.55 | 10.62 | 7.33 |
| AGE-Net (ICIP'24) | 3.61 | 4.78 | - | - | - |
| 3DGazeNet (ECCV'24) | 4.00 | - | - | 9.60 | - |
| **OmniGaze** | **2.97** | **4.07** | **5.40** | **9.12** | **6.72** |
| 提升 | -0.64 | -0.71 | -1.15 | -0.48 | -0.61 |

### 消融实验——核心组件（零样本设置）

| 配置 | MPIIGaze | EyeDiap | RT-Gene | 说明 |
|------|:---:|:---:|:---:|------|
| 仅有标签数据 | 6.17 | 8.45 | 20.73 | 基线 |
| +无标签数据（无筛选） | 4.97 | 5.42 | 13.75 | 数据扩大带来显著提升 |
| +奖励模型筛选 | **3.44** | **4.31** | **9.09** | 筛选进一步提升1.5°+ |

### 消融——奖励模型组件

| 组件 | MPIIGaze | EyeDiap | RT-Gene |
|------|:---:|:---:|:---:|
| 仅语义注视描述 | 3.69 | 4.78 | 10.23 |
| 仅3D方向向量 | 4.03 | 4.93 | 10.56 |
| 两者结合（完整） | **3.44** | **4.31** | **9.09** |

### 关键发现

- **数据规模+质量筛选协同生效**：从仅有标签→加无标签→加奖励筛选，三步递进每步都有显著提升
- **语义描述贡献大于几何向量**：语义注视描述作为单独组件的效果优于3D方向向量，表明MLLM的场景理解能力为回归任务提供了有价值的先验
- **零样本超越域内SOTA**：OmniGaze-Base的零样本性能（3.44°@MPIIFaceGaze）甚至优于之前需要域内训练的SOTA（3DGazeNet的4.00°）
- **模型规模可扩展**：ViT-S→ViT-B→ViT-L持续提升

## 亮点与洞察

1. **用MLLM做回归任务的语义先验**是本文最巧妙的设计：不直接用MLLM做注视估计，而是利用MLLM的场景理解能力来辅助伪标签质量评估——这是一种间接但有效的多模态知识迁移
2. **填补了SSL在连续回归领域的方法空白**：现有SSL的伪标签筛选策略几乎都为分类设计，奖励模型+置信度打分为回归任务提供了通用框架
3. **作为可扩展数据引擎的潜力**：OmniGaze可以为任意人脸图像生成可靠的注视标注，且随着解锁更多无标签数据持续提升

## 局限与展望

- 奖励模型训练依赖一定量的有标签数据来初始化教师模型，完全无监督场景下不适用
- MLLM调用（InstructBLIP生成描述）增加了数据预处理成本，对于百万级数据存在效率瓶颈
- 奖励模型的二元分类目标（ground truth vs 伪标签）可能过于简化——ground truth标签也可能有噪声
- 仅在ViT架构上验证，未探索CNN等其他骨干

## 相关工作与启发

- **SemiReward（SSL中的奖励打分）**：本文将SemiReward的思路从分类扩展到回归，核心差异在于引入了多模态线索和几何表征
- **对其他回归SSL任务的启发**：深度估计、姿态估计、光流估计等连续回归任务都可能从类似的奖励驱动伪标签筛选中获益
- **MLLM作为辅助先验**：MLLM不一定要直接做预测，其语义理解能力可以作为"元知识"辅助其他任务的训练

## 评分

- **新颖性**: ⭐⭐⭐⭐ 奖励模型+MLLM语义描述+几何表征的三模态融合新颖，SSL应用于回归任务的框架有方法论贡献
- **实验充分度**: ⭐⭐⭐⭐⭐ 5个数据集×3种设置（域内/跨域/零样本）+详尽消融，三种模型规模
- **写作质量**: ⭐⭐⭐⭐ 方法描述清晰，挑战→方案的对应关系明确
- **价值**: ⭐⭐⭐⭐ 可扩展数据引擎思路对其他视觉回归任务有很好的启示价值

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Reading Recognition in the Wild](reading_recognition_in_the_wild.md)
- [\[ICML 2025\] The Devil Is in the Details: Tackling Unimodal Spurious Correlations for Generalizable Multimodal Reward Models](../../ICML2025/multimodal_vlm/the_devil_is_in_the_details_tackling_unimodal_spurious_correlations_for_generali.md)
- [\[NeurIPS 2025\] Systematic Reward Gap Optimization for Mitigating VLM Hallucinations](systematic_reward_gap_optimization_for_mitigating_vlm_hallucinations.md)
- [\[NeurIPS 2025\] Enhancing Outcome Reward-Based RL Training of MLLMs with Self-Consistency Sampling](enhancing_the_outcome_reward-based_rl_training_of_mllms_with_self-consistency_sa.md)
- [\[ICCV 2025\] MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild](../../ICCV2025/multimodal_vlm/molparser_end-to-end_visual_recognition_of_molecule_structures_in_the_wild.md)

<!-- RELATED:END -->
