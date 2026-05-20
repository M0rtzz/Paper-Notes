---
title: >-
  [论文解读] Purify-then-Align: Towards Robust Human Sensing under Modality Missing with Knowledge Distillation from Noisy Multimodal Teacher
description: >-
  [CVPR 2026][多模态VLM][模态缺失] 本文提出PTA（Purify-then-Align）框架，通过元学习驱动的模态加权机制先"净化"噪声多模态教师，再用扩散模型驱动的知识蒸馏"对齐"每个单模态学生，使单模态编码器在模态缺失场景下保持强鲁棒性，在MM-Fi和XRF55上实现SOTA。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "模态缺失"
  - "知识蒸馏"
  - "元学习"
  - "扩散对齐"
  - "多模态融合"
---

# Purify-then-Align: Towards Robust Human Sensing under Modality Missing with Knowledge Distillation from Noisy Multimodal Teacher

**会议**: CVPR 2026  
**arXiv**: [2604.05584](https://arxiv.org/abs/2604.05584)  
**代码**: [https://github.com/Vongolia11/PTA](https://github.com/Vongolia11/PTA)  
**领域**: 多模态VLM / 人体感知  
**关键词**: 模态缺失, 知识蒸馏, 元学习, 扩散对齐, 多模态融合

## 一句话总结

本文提出PTA（Purify-then-Align）框架，通过元学习驱动的模态加权机制先"净化"噪声多模态教师，再用扩散模型驱动的知识蒸馏"对齐"每个单模态学生，使单模态编码器在模态缺失场景下保持强鲁棒性，在MM-Fi和XRF55上实现SOTA。

## 研究背景与动机

1. **领域现状**：多模态人体感知（结合深度相机、LiDAR、WiFi等）是人机交互和智能医疗的基础技术，多模态融合能克服单传感器的局限性。

2. **现有痛点**：两个核心挑战——(a) **表示鸿沟(Representation Gap)**：不同传感器的数据表示差异巨大（如图像的网格像素 vs LiDAR的稀疏点云），直接融合导致信息损失；(b) **污染效应(Contamination Effect)**：低质量/高噪声的模态在融合时会污染高质量模态，降低整体性能。

3. **核心矛盾**：这两个问题是**因果关联**的——低质量模态的污染（Contamination）从根本上阻碍了异质表示间差距的缩小（Representation Gap）。现有方法（生成式重建、共享表示学习、简单融合、传统知识蒸馏）各自只解决一个方面，忽略了这一因果链。

4. **本文目标** 构建一个统一框架，先解决因（污染效应） → 再解决果（表示鸿沟），使每个单模态编码器都能独立工作且蕴含跨模态知识。

5. **切入角度**：教师-学生范式——多模态共识作为教师指导每个单模态学生。但教师本身可能被噪声模态污染，因此必须先"净化"教师（Purify），再用净化后的教师"对齐"学生（Align）。

6. **核心 idea**：元学习自适应权重解决教师端的模态污染，扩散模型知识蒸馏解决学生端的表示对齐。

## 方法详解

### 整体框架

训练采用嵌套循环结构。外循环（Purify）：通过元学习在验证集上优化模态权重$\mathbf{w}$，控制每个模态对教师的贡献比例。内循环（Align）：固定$\mathbf{w}$，利用加权融合后的"干净教师"通过扩散蒸馏对齐每个单模态学生特征。推理时，每个单模态编码器独立工作，无需其他模态。

### 关键设计

1. **Purify阶段：元学习模态加权**:

    - 功能：自适应学习每个模态的重要性权重$\mathbf{w}$，抑制噪声/低贡献模态
    - 核心思路：嵌套优化——内循环在训练集$\mathcal{D}_{train}$上用固定$\mathbf{w}$优化模型参数$\Theta$（最小化 $\mathcal{L}_{inner} = \mathcal{L}_{task} + \lambda\mathcal{L}_{DiffKD}$）；外循环在验证集$\mathcal{D}_{val}$上评估$\Theta^*(\mathbf{w})$的性能，通过梯度$\nabla_\mathbf{w}\mathcal{L}_{outer}$更新$\mathbf{w}$。权重经Softmax归一化确保稳定性。训练时随机以均匀概率丢弃每个模态来模拟真实缺失场景
    - 设计动机：X-Fi需要手工调每个模态的dropout概率（WiFi/Radar/RFID可能需要不同的(0.5,0.5,0.8)等组合），模态数增多时调参不可行。元学习自动学习权重，完全避免这个问题

2. **Align阶段：扩散模型知识蒸馏**:

    - 功能：将净化后的多模态教师知识蒸馏到每个单模态学生
    - 核心思路：教师特征 $f_T = \sum_{i \in \mathcal{M}_{all}} \mathbf{w}_i f_i$（所有模态的加权和）。将$f_T$和$f_S$投影到压缩潜在空间得到$z_T$和$z_S$。训练一个噪声预测网络$\Phi_\phi$学习$z_T$的分布（标准扩散损失$\mathcal{L}_{Diff}$），然后将$z_S$视为$z_T$的"噪声版本"，通过反向去噪过程将$z_S$精炼为$\hat{z}_S$。总蒸馏损失 $\mathcal{L}_{DiffKD} = \mathcal{L}_{Diff} + \mathcal{L}_{KD}$，其中$\mathcal{L}_{KD} = MSE(\hat{z}_S, z_T)$
    - 设计动机：传统KL/MSE sttilllation难以跨越异质模态间的巨大表示差距。扩散模型的去噪过程天然适合将信息量不足的$z_S$渐进式地精炼到信息丰富的$z_T$附近

3. **Noise Adapter（自适应噪声匹配）**:

    - 功能：为每个样本动态确定学生特征的噪声水平
    - 核心思路：不同输入样本的$z_S$与$z_T$的差距不同，固定时间步$t$无法适应这种一对多映射。Noise Adapter是一个小辅助网络（1个Bottleneck + Global AvgPool + FC），预测融合系数$\gamma \in [0,1]$，用于混合学生特征和纯噪声：$z_{TS} = \gamma z_S + (1-\gamma)\epsilon_T$。然后用DDIM从$z_{TS}$出发做5步确定性去噪得到$\hat{z}_S$
    - 设计动机：解决扩散蒸馏中学生噪声水平未知的关键问题——如果$z_S$已相当接近$z_T$，不需要太多去噪；如果$z_S$很差，需要更多的噪声起点让扩散模型充分精炼

### 损失函数 / 训练策略

总内循环损失$\mathcal{L}_{inner} = \mathcal{L}_{task} + \lambda\mathcal{L}_{DiffKD}$，$\lambda=0.1$。外循环损失$\mathcal{L}_{outer} = \mathcal{L}_{task}(\Theta^*(\mathbf{w}))$。MM-Fi上使用Adam优化器（模型lr=5e-4, 元学习lr=1e-2），batch=16。XRF55上模型lr=2e-4, batch=32。均在RTX 3090上训练。

## 实验关键数据

### 主实验

MM-Fi人体姿态估计（MPJPE mm ↓）：

| 模态 | Base.1 | Base.2 | X-Fi | **PTA (Ours)** | 提升 |
|------|--------|--------|------|----------------|------|
| Depth | 102.4 | 102.4 | 96.40 | **84.81** | +12.0% |
| LiDAR | 161.5 | 161.5 | 130.06 | **68.30** | +47.5% |
| WiFi | 227.1 | 227.1 | 210.12 | **182.18** | +13.3% |
| D+L | 111.7 | 108.0 | 89.41 | **64.68** | +27.7% |
| L+W | 167.1 | 206.2 | 111.15 | **74.74** | +32.8% |
| D+L+W | 130.7 | 154.6 | 87.59 | **68.86** | +21.4% |

XRF55动作识别（准确率% ↑）：

| 模态 | Baseline | X-Fi | **PTA (Ours)** | 提升 |
|------|----------|------|----------------|------|
| Radar | 82.1 | 83.9 | **90.03** | +6.13% |
| WiFi | 77.8 | 55.7 | **82.34** | +26.64% |
| RFID | 42.2 | 42.5 | **55.04** | +12.54% |
| R+W+RF | 70.6 | 89.8 | **95.87** | +6.07% |

### 消融实验

MM-Fi上的模块消融（MPJPE mm ↓）：

| 模态 | Full | w/o Diff | w/o Meta |
|------|------|----------|----------|
| Depth | **84.81** | 89.66 | 157.98 |
| LiDAR | **68.30** | 76.27 | 183.04 |
| WiFi | **182.18** | 187.92 | 236.99 |
| D+L | **64.68** | 78.12 | 148.65 |
| D+L+W | **68.86** | 76.79 | 160.34 |

### 关键发现

- **单模态性能提升巨大**：PTA的核心价值在于大幅增强单模态编码器——LiDAR单模态MPJPE从X-Fi的130.06降至68.30（+47.5%），说明扩散蒸馏有效地将跨模态知识注入到了单模态特征中
- **Purify stage是关键**：去掉元学习权重后性能灾难性崩溃（D+L从64.68暴涨到148.65），证明不净化教师直接蒸馏会传播噪声模态的负面影响
- **WiFi的污染效应**：WiFi单模态MPJPE=182.18远差于Depth(84.81)和LiDAR(68.30)，但PTA在L+W融合时（74.74）只比LiDAR单模态（68.30）轻微退化，说明meta权重成功抑制了WiFi的污染
- **同质模态融合增益更大**：XRF55上三种RF模态（Radar+WiFi+RFID）全融合达95.87%，因为同类射频信号的表示鸿沟较小，更容易对齐
- X-Fi需要敏感的dropout率手工调优（WiFi准确率在不同设置下从29.1%到55.7%波动），PTA用均匀dropout完全避免了这个问题

## 亮点与洞察

- **因果问题分解**：首次明确指出Contamination Effect和Representation Gap的因果链，并设计"先因后果"的两阶段解决方案。这种问题分析方式可迁移到所有多源信息融合场景（如多模态大模型训练中的数据质量问题）
- **扩散模型用于特征对齐**：将学生特征视为教师特征的"噪声版本"进行去噪精炼，是知识蒸馏的新范式。5步DDIM确保了效率，Noise Adapter解决了噪声水平未知的关键问题
- **元学习替代手工调参**：用meta-learning自动学习模态权重，彻底避免了多模态系统中per-modality dropout概率的调参噩梦。这个技巧在模态数量多时尤其有价值

## 局限与展望

- MPJPE vs PA-MPJPE的trade-off：PTA在全局定位（MPJPE）上很强，但在消除位置因素后的骨架结构（PA-MPJPE）上部分退化，说明扩散蒸馏可能让学生倾向于预测"均值姿态"而非精确的骨骼细节
- 扩散去噪在极低质量模态（WiFi/RFID）条件下可能引入生成伪影（消融中个别L+W/W+RF场景去掉扩散反而更好）
- 元学习的嵌套优化增加了训练复杂度（内外循环+验证集评估）
- 仅在人体感知任务上验证，向其他多模态融合任务的迁移尚未测试
- 模态缺失是随机均匀丢弃的模拟，未考虑更现实的系统性缺失模式（如某传感器长时间故障）

## 相关工作与启发

- **vs X-Fi**: X-Fi构建模态不变表示，但牺牲了单模态最大性能（LiDAR MPJPE 130.06 vs PTA的68.30），且需要敏感的per-modality dropout调参。PTA通过提升单模态底座实现了更好的整体性能
- **vs 生成式方法(VAE/GAN重建)**: 这些方法试图重建缺失模态的原始数据，训练不稳定且容易产生幻觉。PTA在特征空间对齐，避免了原始数据重建的困难
- **vs 传统KD**: 标准知识蒸馏（如MSE距离）难以跨越异质模态的巨大表示差距，扩散蒸馏提供了渐进式的特征精炼路径

## 评分

- 新颖性: ⭐⭐⭐⭐ 因果分析视角和Purify-then-Align范式原创性强，扩散蒸馏+元学习的组合也是新的
- 实验充分度: ⭐⭐⭐⭐ 两个大数据集，7种模态组合全覆盖，消融分析深入（含单模块消融和edge case分析）
- 写作质量: ⭐⭐⭐⭐ 因果关系的motivate非常有说服力，但符号较多需要仔细阅读
- 价值: ⭐⭐⭐⭐ 对多模态人体感知社区有重要贡献，Purify-then-Align范式可推广到更广泛的多模态学习场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] EBMC: Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis](ebmc_multimodal_sentiment_analysis.md)
- [\[CVPR 2026\] Disentangle-then-Align: Non-Iterative Hybrid Multimodal Image Registration via Cross-Scale Feature Disentanglement](disentangle-then-align_non-iterative_hybrid_multimodal_image_registration_via_cr.md)
- [\[CVPR 2026\] Uncertainty-Aware Knowledge Distillation for Multimodal Large Language Models](uncertainty-aware_knowledge_distillation_for_multimodal_large_language_models.md)
- [\[CVPR 2026\] BALM: A Model-Agnostic Framework for Balanced Multimodal Learning under Imbalanced Missing Rates](balm_a_model-agnostic_framework_for_balanced_multimodal_learning_under_imbalance.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr_turbo_merged_checkpoint_free_teacher.md)

</div>

<!-- RELATED:END -->
