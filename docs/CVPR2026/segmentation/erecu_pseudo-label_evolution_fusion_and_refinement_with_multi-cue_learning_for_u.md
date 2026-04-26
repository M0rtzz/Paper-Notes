---
title: >-
  [论文解读] EReCu: Pseudo-label Evolution Fusion and Refinement with Multi-Cue Learning for Unsupervised Camouflage Detection
description: >-
  [CVPR 2026][图像分割][伪装目标检测] 提出统一的无监督伪装目标检测框架 EReCu，通过多线索原生感知(MNP)、伪标签进化融合(PEF)和局部伪标签精炼(LPR)三个协同模块，在不依赖人工标注的情况下实现了边界精确、细节丰富的伪装目标分割。
tags:
  - CVPR 2026
  - 图像分割
  - 伪装目标检测
  - 无监督分割
  - 伪标签学习
  - 教师-学生框架
  - 注意力融合
---

# EReCu: Pseudo-label Evolution Fusion and Refinement with Multi-Cue Learning for Unsupervised Camouflage Detection

**会议**: CVPR 2026  
**arXiv**: [2603.11521](https://arxiv.org/abs/2603.11521)  
**代码**: [GitHub](https://github.com/JSLiam94/EReCu)  
**领域**: 分割  
**关键词**: 伪装目标检测, 无监督分割, 伪标签学习, 教师-学生框架, 注意力融合

## 一句话总结

提出统一的无监督伪装目标检测框架 EReCu，通过多线索原生感知(MNP)、伪标签进化融合(PEF)和局部伪标签精炼(LPR)三个协同模块，在不依赖人工标注的情况下实现了边界精确、细节丰富的伪装目标分割。

## 研究背景与动机

伪装目标检测(COD)旨在分割与背景高度相似的隐匿目标，是一项极具挑战的视觉任务。全监督方法虽然效果好，但依赖昂贵且模糊的像素级标注，限制了数据规模和生态多样性。因此，无监督伪装目标检测(UCOD)成为重要研究方向。

现有 UCOD 方法主要分两类范式，各有瓶颈：

**伪标签引导型**：如 UCOS-DA 使用静态伪标签，UCOD-DPL 引入教师-学生动态融合，但它们过度依赖高维嵌入而忽视原始图像的感知线索，导致**边界溢出**和语义漂移。

**特征学习型**：如 SdalsNet 通过注意力机制解耦前景/背景特征，EASE 引入环境原型检索，但缺乏伪标签的显式监督，导致**边界模糊**和细节丢失。

核心洞察：语义可靠性和纹理保真度不应孤立优化，而应通过相互反馈回路协同进化。EReCu 正是基于这一思路，让原生感知线索持续引导伪标签进化，同时感知学习也从逐步去噪的监督中受益。

## 方法详解

### 整体框架

EReCu 基于 DINO 预训练的教师-学生架构。教师分支提供稳定的语义引导，学生分支在进化伪标签监督下逐步学习精炼的伪装掩码。三个核心模块形成协同流水线：
- **MNP** 提取原生纹理和语义线索 $F_{\text{MNP}}$ 和质量度量 $S_{\text{mc}}$
- **PEF** 利用这些线索通过教师-学生交互和谱张量注意力融合进化出全局伪标签
- **LPR** 利用原生线索从高置信区域生成局部伪标签，修复全局预测遗漏的边界和纹理细节

### 关键设计

1. **多线索原生感知模块(MNP)**：MNP 是整个框架的基石，为 PEF 和 LPR 提供原生感知引导。核心思想是，尽管伪装来源于与背景的高度视觉相似性，但原始图像中仍存在微妙但有区分力的纹理变化。MNP 融合低层纹理特征（LBP、DoG）和中层语义特征（冻结的 ResNet-18），构建多线索表示：$F_{\text{MNP}} = \mathcal{C}(F_{\text{text}}, F_{\text{sem}})$。同时提出多线索质量度量 $S_{\text{mc}}$，将图像按预测掩码划分为内部 $R_i$、边界 $R_s$、外部 $R_o$ 三个区域，通过随机采样 patch 计算余弦相似性来评估前景-背景可分离性。$S_{\text{mc}} = (D_{\text{io}} + D_{\text{is}} + S_{\text{so}})/3$，值越大说明前景-背景分离越明显。

2. **伪标签进化融合模块(PEF)**：包含两个互补子模块：

    - **进化伪标签学习(EPL)**：让学生浅层特征与教师深层特征通过深度可分离卷积(DSC)交互。DSC 将标准卷积分解为逐深度和逐点操作，降低计算成本的同时增强细粒度纹理和边界结构。迭代优化过程同时利用 Dice 损失（教师-学生一致性）和 $\mathcal{L}_{\text{MNP}}$（原生线索正则化），使伪标签在语义和感知双重引导下持续进化。
    - **谱张量注意力融合(STAF)**：将来自学生网络不同层的三个注意力图堆叠为三阶张量 $\mathcal{T}_s \in \mathbb{R}^{3 \times C \times HW}$，通过 Tucker 分解捕获层间、通道间、空间间的相关性，再用截断 SVD 保留主能量成分并过滤噪声，最终投影为融合预测 $M_s^{\text{fu}}$。复杂度仅为 $\mathcal{O}(r^2 d)$，其中 $r \ll d$。

3. **局部伪标签精炼模块(LPR)**：利用 DINO 多头自注意力(MHSA)的空间多样性来精炼伪标签。

    - **目标感知注意力选择(TAS)**：通过注意力熵 $E_k$ 和多线索度量 $S_{\text{mc}}$ 联合筛选出聚焦且语义一致的注意力头：$\mathcal{A}_{\text{sel}} = \{A_k \mid E_k < \tau_e \wedge S_{\text{mc}}(\hat{A}_k, F_{\text{MNP}}) > \tau_s\}$。
    - **局部伪标签生成(LPG)**：从选中注意力头的高置信区域（自适应阈值 $\tau_k = \mu_{A_k} + \alpha \cdot \sigma_{A_k}$）生成局部伪标签，通过 Dice + CE 联合损失引导学生网络精炼。

### 损失函数 / 训练策略

- **EPL 损失**：双 Dice 损失（学生-教师对齐）+ $\mathcal{L}_{\text{MNP}}$（原生线索正则化），迭代进化
- **LPR 损失**：$\mathcal{L}_{\text{LPR}} = \mathcal{L}_D(M_s^{\text{fu}}, \bigcup_k P_k) + \mathcal{L}_{\text{CE}}(M_s^{\text{fu}}, \bigcup_k P_k)$
- 教师模型通过 EMA ($\eta=0.99$) 更新，训练 25 个 epoch，batch size 32，AdamW + 余弦退火，AMP 混合精度
- 编码器采用 DINO-ViT-S/8，纹理提取器为 LBP 和 DoG

## 实验关键数据

### 主实验

| 数据集 | 指标 | EReCu | UCOD-DPL (之前SOTA) | 提升 |
|--------|------|-------|---------------------|------|
| CHAMELEON | $S_m\uparrow$ | .7321 | .7287 | +0.34% |
| CAMO | $S_m\uparrow$ | .7027 | .7013 | +0.14% |
| COD10K | $S_m\uparrow$ | .7221 | .7090 | +1.31% |
| COD10K | $F_\omega^\beta\uparrow$ | .5628 | .5481 | +1.47% |
| COD10K | $M\downarrow$ | .0613 | .0601 | -0.12% |
| NC4K | $S_m\uparrow$ | .7583 | .7538 | +0.45% |
| NC4K | $E_m^\phi\uparrow$ | .8498 | .8447 | +0.51% |

EReCu 在所有四个数据集上全面超越所有 UOS 和 UCOD 基线方法。

### 消融实验

| 配置 | CAMO $S_m\uparrow$ | COD10K $S_m\uparrow$ | 说明 |
|------|---------------------|----------------------|------|
| MNP+EPL+STAF+LPR (完整) | .7027 | .7221 | 最佳 |
| 去掉 MNP | .6887 | .7111 | 纹理线索缺失导致定位退化 |
| 去掉 EPL | .6758 | .7038 | 结构一致性显著下降 |
| 去掉 STAF | .6815 | .7179 | 区域不一致性出现 |
| 去掉 LPR | .6895 | .7109 | 局部细节恢复能力减弱 |
| 仅 DINO-ViT-S/8 | .6376 | .6400 | 基线 |

### 关键发现

- 每个模块都有正向贡献且互补，三个以上模块组合显著优于两个模块
- MNP + EPL 配对产生最显著的改进，验证了原生线索对齐与伪标签学习的协同效应
- 在挑战性场景中（深度伪影、极端纹理抑制），EReCu 表现出更清晰的边界和更完整的结构

## 亮点与洞察

1. **范式创新**：首次将伪标签进化与原生感知学习通过自进化教师-学生机制统一，打破了"伪标签精炼"和"特征学习"两类方法的壁垒
2. **多线索质量度量**：$S_{\text{mc}}$ 通过随机采样 patch 的余弦相似性计算前景-背景可分离性，既优雅又鲁棒，为伪标签质量评估提供了可迁移的工具
3. **谱张量融合**：用 Tucker 分解 + 截断 SVD 融合多层注意力图，比简单加权聚合更好地保留语义和结构信息，且计算高效

## 局限与展望

- 提升幅度在部分数据集/指标上相对有限（如 CHAMELEON 上 $S_m$ 仅提升 0.34%），说明 UCOD 任务在某些场景已接近性能瓶颈
- 框架复杂度较高（三个协同模块 + 教师-学生架构），推理效率和部署成本值得关注
- 仅在标准 COD 数据集上验证，未涉及更复杂的真实生态场景或跨域泛化测试
- 纹理提取器（LBP、DoG）为手工设计，可探索可学习的低层特征提取替代方案

## 相关工作与启发

- 与 UCOD-DPL 的教师-学生框架类似，但 EReCu 引入了原生线索的正则化，避免了纯语义引导导致的边界溢出
- SdalsNet 的注意力解耦思路与 EReCu 的 LPR 互补——前者做全局分离，后者做局部精炼
- STAF 的谱融合策略可推广至其他需要融合多层特征图的任务
- 多线索质量度量 $S_{\text{mc}}$ 的"内部-边界-外部"三区域划分可启发其他分割任务的质量评估

## 评分

- 新颖性: ⭐⭐⭐⭐ 三模块协同的统一框架设计有新意，原生感知引导伪标签进化的思路清晰
- 实验充分度: ⭐⭐⭐⭐ 四个数据集 + 全面消融 + 可视化分析，对比方法覆盖面广
- 写作质量: ⭐⭐⭐⭐ 框架图和可视化清晰，模块关系阐述明确，公式推导完整
- 价值: ⭐⭐⭐⭐ 无监督伪装检测的重要进展，原生感知引导伪标签的思路有启发性

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] SDDF: Specificity-Driven Dynamic Focusing for Open-Vocabulary Camouflaged Object Detection](sddf_specificity-driven_dynamic_focusing_for_open-vocabulary_camouflaged_object.md)
- [\[CVPR 2026\] DSS: Discover, Segment, and Select for Zero-shot Camouflaged Object Segmentation](discover_segment_and_select_a_progressive_mechanism_for_zero-shot_camouflaged_ob.md)
- [\[CVPR 2026\] Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](efficient_rgbd_scene_understanding_via_multitask_a.md)
- [\[CVPR 2026\] Love Me, Love My Label: Rethinking the Role of Labels in Prompt Retrieval for Visual In-Context Learning](love_me_love_my_label_rethinking_the_role_of_labels_in_prompt_retrieval_for_visu.md)
- [\[CVPR 2026\] FCL-COD: Weakly Supervised Camouflaged Object Detection with Frequency-aware and Contrastive Learning](fcl-cod_weakly_supervised_camouflaged_object_detection_with_frequency-aware_and_.md)

<!-- RELATED:END -->
