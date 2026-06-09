---
title: >-
  [论文解读] CGSA: Class-Guided Slot-Aware Adaptation for Source-Free Object Detection
description: >-
  [ICLR 2026][目标检测][域适应] 首次将 Object-Centric Learning（Slot Attention）引入无源域自适应目标检测（SF-DAOD），通过分层 Slot 感知模块提取域不变的目标级结构先验，并用类引导对比学习驱动域不变表征，在多个跨域基准上大幅超越现有方法。
tags:
  - "ICLR 2026"
  - "目标检测"
  - "域适应"
  - "object-centric learning"
  - "注意力机制"
  - "DETR"
  - "对比学习"
---

# CGSA: Class-Guided Slot-Aware Adaptation for Source-Free Object Detection

**会议**: ICLR 2026  
**arXiv**: [2602.22621](https://arxiv.org/abs/2602.22621)  
**代码**: [GitHub](https://github.com/Michael-McQueen/CGSA)  
**领域**: 目标检测  
**关键词**: source-free domain adaptation, object-centric learning, slot attention, DETR, contrastive learning

## 一句话总结

首次将 Object-Centric Learning（Slot Attention）引入无源域自适应目标检测（SF-DAOD），通过分层 Slot 感知模块提取域不变的目标级结构先验，并用类引导对比学习驱动域不变表征，在多个跨域基准上大幅超越现有方法。

## 研究背景与动机

**域偏移问题**：目标检测器部署时面临天气/摄像头/场景的域偏移，性能大幅下降

**无源域自适应（SF-DAOD）限制**：仅有源域预训练模型和无标注目标域数据，不可访问源数据（隐私/版权约束）

**现有方法局限**：主流 SF-DAOD 方法（SFOD/PETS/A2SFOD）专注于伪标签阈值调优或 teacher-student 框架改进，忽略了跨域数据中目标级结构的共性信息

**Slot Attention 的潜力**：OCL 将场景分解为离立的"slot"表征，每个 slot 绑定一个目标，天然隔离前景与背景。在分割/视频预测/机器人等任务中展示了强迁移性，但从未用于 SF-DAOD

**自然契合**：DETR 检测器已使用 object queries，将 slot 先验嵌入 query 空间是自然但未探索的方向

## 方法详解

### 整体框架

CGSA 建立在 RT-DETR 之上，分两个阶段运行：源域预训练时除了标准检测损失外额外挂上 HSA 重建损失，让模型学会把特征拆成目标级 slot；目标域自适应时切换成 Teacher-Student 自训练，再叠加 HSA 提供的结构先验与 CGSC 的类引导对比，把源域学到的目标级结构迁移到无标注的目标域上。两个核心模块——分层 slot 感知 HSA 与类引导 slot 对比 CGSC——分别负责"提取域不变结构"和"对齐域不变语义"。

### 关键设计

**1. 分层 Slot 感知 HSA：把场景拆成目标级结构先验。** SF-DAOD 之所以难，是因为伪标签噪声大、跨域只剩零散的低层特征可用；CGSA 的思路是改用 Object-Centric Learning 抽取的目标级结构，这种结构对天气、相机、画风等域偏移天然鲁棒。HSA 采用两阶段分解：第一阶段对 backbone 特征 $h$ 做迭代式 Slot Attention，提取 $n=5$ 个粗粒度 slot，再经空间广播 MLP 解码重建，解码时的 softmax 竞争迫使每个 slot 绑定不同的图像区域、把前景从背景里隔离出来；第二阶段把重建特征再喂回 Slot Attention，进一步细分出 $n^2=25$ 个细粒度 slot。两阶段重建都受监督，损失为 $\mathcal{L}_{rec} = \|\hat{h}^{(1)} - h\|_2^2 + \|\hat{h}^{(2)} - h\|_2^2$。最后把第二阶段 slot 投影后直接加到 DETR 的 object query 上，$Q_{aware} = Q_{obj} + f_{map}(z^{(2)})$，让 decoder 在解码每个目标时都带上这份域不变的结构先验。25 个 slot 远超传统 OCL ≤10 的惯例，但正是分层设计（先粗后细）保证了这种规模下仍能稳定收敛。

**2. 类引导 Slot 对比 CGSC：把结构先验对齐到统一语义空间。** 光有结构还不够，跨域时同一类目标在两域的特征分布仍可能错位，CGSC 用对比学习把它们拉到一起。模块维护一组 EMA 更新的全局类原型 $P_c$，每个原型由 decoder queries 按预测类别平均聚合而来，跨 batch 持续积累一个稳定的类语义锚点。对当前图像，先用第二阶段的注意力 mask $m_k^{(2)}$ 对原始特征做加权聚合，得到压抑了背景 slot 的 weighted slot；再用余弦相似度矩阵加匈牙利算法把这些 weighted slot 与 decoder queries 一一匹配，从而给每个 slot 借来一个伪类标签。有了伪标签就能算 InfoNCE 对比损失，把同类的 slot 原型 $\bar{z}_c$ 往对应的类原型 $P_c$ 上拉近、把异类推远，逼着不同域的同类目标共享同一套语义表征。

### 损失函数 / 训练策略

目标域自适应阶段的总损失把自训练、对比、重建三项加权相加：

$$\mathcal{L}_{total} = \mathcal{L}_{unsup} + \lambda_{con} \mathcal{L}_{con} + \lambda_{rec} \mathcal{L}_{rec}$$

其中 $\mathcal{L}_{unsup}$ 是 Teacher-Student 框架下基于伪标签的无监督检测损失，$\lambda_{con}$、$\lambda_{rec}$ 分别平衡对比与重建项。论文还给出了理论支撑，证明每步自适应后目标域风险存在下降界 $\mathbb{E}[\mathcal{R}_T(\theta_{t+1})] \le \mathbb{E}[\mathcal{R}_T(\theta_t)] - c_1 \Delta_t + c_2(\epsilon_{rec} + \sigma^2)$，说明只要重建误差 $\epsilon_{rec}$ 和噪声 $\sigma^2$ 足够小，slot-aware 设计带来的风险下降就有保证，而非纯经验调参。

## 实验关键数据

### 主实验

| 跨域设置 | SF | 方法 | mAP |
|---------|-----|------|-----|
| Cityscapes→BDD100K | ✗ | DATR (有源DAOD) | 43.3 |
| Cityscapes→BDD100K | ✓ | TITAN (SF-DAOD) | 38.3 |
| Cityscapes→BDD100K | ✓ | **CGSA** | **53.0** |
| Cityscapes→Foggy | ✓ | A2SFOD | 41.2 |
| Cityscapes→Foggy | ✓ | **CGSA** | **49.8** |

### 消融实验

| 配置 | Cityscapes→BDD100K mAP | 说明 |
|------|------------------------|------|
| 仅 Teacher-Student | 35.4 | 无任何结构先验 |
| +HSA | 45.2 | slot 结构先验有效 |
| +CGSC | 41.8 | 类引导对比有效 |
| **+HSA+CGSC (CGSA)** | **53.0** | 两者互补，最佳 |

### 关键发现

- CGSA 在 SF 设置下甚至超越多个有源 DAOD 方法（需要源数据的方法）
- 基于 RT-DETR 检测器，4×A100 训练
- 在多个跨域场景（正常→雾天、真实→卡通/水彩等）均显著领先
- 25 个 slot 在数量上超越传统 OCL 的 ≤10 限制，但分层设计保证收敛稳定

## 亮点与洞察

- **OCL + SF-DAOD 的首创结合**：开辟了目标级结构先验用于域适应的新范式
- 分层 slot 设计巧妙突破了传统 slot 数量限制（5→25），且保持训练稳定
- 提供理论泛化分析——slot-aware 设计不仅是经验有效，还有理论支撑
- **在无源设置下超越有源方法**是强有力的实验证据

## 局限与展望

- 仅在驾驶场景数据集验证，医疗/航拍/工业等领域的泛化性待测试
- Slot 数量 $n=5$ 是手动设定，自适应机制可能更好
- 匈牙利匹配依赖检测器预测质量，early stage 可能不稳定导致错误类标签
- HSA 的两阶段 Slot Attention + 重建目标增加了训练时间和内存开销

## 相关工作与启发

- **SFOD/PETS/A2SFOD**：聚焦伪标签过滤，忽略目标级结构
- **DATR/MRT** (有源 DAOD)：需要源数据，CGSA 无源下仍超越
- **Slot Attention/SAVi**：原用于分割/视频预测，首次引入域自适应检测

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ OCL + SF-DAOD 首创结合，开辟新方向
- 实验充分度: ⭐⭐⭐⭐ 5个数据集/跨域设置+完整消融+理论分析
- 写作质量: ⭐⭐⭐⭐ 动机清晰，理论+实验双支撑
- 价值: ⭐⭐⭐⭐ 为 SF-DAOD 提供了新的方法论基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SFUOD: Source-Free Unknown Object Detection](../../ICCV2025/object_detection/sfuod_source-free_unknown_object_detection.md)
- [\[AAAI 2026\] Beyond Boundaries: Leveraging Vision Foundation Models for Source-Free Object Detection](../../AAAI2026/object_detection/beyond_boundaries_leveraging_vision_foundation_models_for_so.md)
- [\[CVPR 2026\] Foundation Model Priors Enhance Object Focus in Feature Space for Source-Free Object Detection](../../CVPR2026/object_detection/foundation_model_priors_enhance_object_focus_in_feature_space_for_source-free_ob.md)
- [\[ICLR 2026\] Bootstrapping MLLM for Weakly-Supervised Class-Agnostic Object Counting (WS-COC)](bootstrapping_mllm_for_weakly-supervised_class-agnostic_object_counting.md)
- [\[CVPR 2026\] PaQ-DETR: Learning Pattern and Quality-Aware Dynamic Queries for Object Detection](../../CVPR2026/object_detection/paq-detr_learning_pattern_and_quality-aware_dynamic_queries_for_object_detection.md)

</div>

<!-- RELATED:END -->
