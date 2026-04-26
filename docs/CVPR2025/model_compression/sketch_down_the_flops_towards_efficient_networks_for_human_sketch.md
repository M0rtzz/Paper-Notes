---
title: >-
  [论文解读] Sketch Down the FLOPs: Towards Efficient Networks for Human Sketch
description: >-
  [CVPR 2025][模型压缩][FG-SBIR] 首次针对人类草图（sketch）数据的特有特性设计高效推理网络：通过跨模态知识蒸馏（SketchyNetV1）将大网络压缩到轻量级网络并保持 FG-SBIR 精度，再通过强化学习驱动的自适应画布尺寸选择器（SketchyNetV2）利用草图的稀疏抽象特性进一步减少 FLOPs，最终实现 99.37% 的 FLOPs 缩减（40.18G→0.254G）而几乎不损失精度。
tags:
  - CVPR 2025
  - 模型压缩
  - FG-SBIR
  - 知识蒸馏
  - 画布尺寸选择
  - 强化学习
  - 模型高效化
---

# Sketch Down the FLOPs: Towards Efficient Networks for Human Sketch

**会议**: CVPR 2025  
**arXiv**: [2505.23763](https://arxiv.org/abs/2505.23763)  
**代码**: 无  
**领域**: 模型压缩 / 草图检索  
**关键词**: FG-SBIR, 知识蒸馏, 画布尺寸选择, 强化学习, 模型高效化

## 一句话总结

首次针对人类草图（sketch）数据的特有特性设计高效推理网络：通过跨模态知识蒸馏（SketchyNetV1）将大网络压缩到轻量级网络并保持 FG-SBIR 精度，再通过强化学习驱动的自适应画布尺寸选择器（SketchyNetV2）利用草图的稀疏抽象特性进一步减少 FLOPs，最终实现 99.37% 的 FLOPs 缩减（40.18G→0.254G）而几乎不损失精度。

## 研究背景与动机

**领域现状**：模型高效化研究（MobileNet、EfficientNet 等）在照片数据上已经非常成熟，但针对人类草图（sketch）数据的高效推理几乎完全空白。细粒度草图图像检索（FG-SBIR）是草图领域研究最多、最接近商业化的任务，使用草图检索匹配的照片实例。

**现有痛点**：直接将为照片设计的轻量级网络（如 MobileNetV2）用于 FG-SBIR 会导致精度大幅下降——MobileNetV2 相比 VGG-16 基线精度下降 37%。这是因为草图的细粒度特性（用简洁线条表达精细视觉差异）需要足够的模型容量来建立跨模态的草图-照片匹配关系。

**核心矛盾**：草图数据有两个独特特性被现有高效方法忽略：(1) 草图是稀疏的黑白线条而非像素密集的照片，同样的语义信息可以在更低分辨率下保持；(2) 不同草图的抽象程度不同——有的简洁几笔就传达了足够信息，有的则需要更多细节。固定分辨率的方式对所有草图都是次优的。

**本文目标**：设计即插即用的草图特有组件，使任何照片高效网络都能适配草图数据，具体目标是大幅降低 FLOPs 同时保持 FG-SBIR 精度。

**切入角度**：(1) 通过知识蒸馏把大网络在 FG-SBIR 上的细粒度匹配能力迁移到小网络；(2) 利用草图的向量格式可以自由渲染到任意分辨率的特性，动态选择最优画布尺寸。

**核心 idea**：先用关系型知识蒸馏（保持跨模态距离结构）压缩模型，再用强化学习训练画布尺寸选择器，根据每个草图的抽象程度动态选择最低足够分辨率。

## 方法详解

方法分两阶段：第一阶段（SketchyNetV1）通过知识蒸馏将 VGG-16 教师网络的跨模态检索能力迁移到 MobileNetV2 学生网络；第二阶段（SketchyNetV2）在学生网络基础上训练一个基于 GRU 的画布尺寸选择器，用强化学习优化。

### 整体框架

输入为草图（向量格式或栅格化图像）和照片，输出为草图-照片的嵌入表征用于检索。SketchyNetV1：以 VGG-16 为教师、MobileNetV2 为学生，三分支 Siamese 网络架构，用 triplet loss + 关系蒸馏损失联合训练。SketchyNetV2：在 SketchyNetV1 前端增加画布尺寸选择器 $\psi_C$，接收向量格式草图，预测最优渲染分辨率，然后以该分辨率渲染并送入检索网络。

### 关键设计

1. **跨模态关系知识蒸馏（SketchyNetV1）**:

    - 功能：将大网络的细粒度跨模态检索能力迁移到轻量网络
    - 核心思路：不直接蒸馏教师和学生的特征向量（因为不同网络的嵌入空间维度和结构可能不兼容），而是蒸馏教师嵌入空间中三元组（草图-正照片-负照片）之间的**成对距离关系**。定义距离 $d_{sp}^T = \delta(f_s^T, f_p^T)$ 等，用 Huber 损失匹配教师和学生对应的距离：$\mathcal{L}_{RKD}^{sp} = \mathcal{L}_\delta(d_{sp}^T, d_{sp}^{st})$。总蒸馏损失 $\mathcal{L}_{RKD} = \mathcal{L}_{RKD}^{sp} + \mathcal{L}_{RKD}^{sn} + \mathcal{L}_{RKD}^{pn}$。学生总训练目标 $\mathcal{L}_{trn}^{st} = \lambda \mathcal{L}_{Tri} + (1-\lambda) \mathcal{L}_{RKD}$。此外，学生在多种画布尺寸上训练以确保尺度不变性。
    - 设计动机：标准的 logit 蒸馏适用于分类任务，但 FG-SBIR 是跨模态检索，输出是连续的 d 维嵌入而非离散类别。直接回归特征向量会受嵌入空间差异困扰。保持距离结构是更稳健的方式——只要学生空间中草图-正样本距离小、草图-负样本距离大，就足以完成检索任务。

2. **自适应画布尺寸选择器（SketchyNetV2）**:

    - 功能：根据每个草图的抽象程度动态选择最优渲染分辨率
    - 核心思路：使用 1 层 GRU 网络对草图的向量序列 $s_v = (v_1, ..., v_T)$ 编码（每个 $v_t = (x_t, y_t, q_t^1, q_t^2, q_t^3)$ 包含坐标和笔状态），取最终隐状态通过线性层输出 K 个画布尺寸的概率 $p(c|s_v)$。使用 Douglas-Peucker 算法限制序列长度。通过采样 $c_{pred} \sim \text{categorical}(p(c|s_v))$ 得到渲染尺寸，栅格化后送入 SketchyNetV1。
    - 设计动机：实验证明草图在低分辨率下仍保持较好检索精度（32×32 时精度仅下降 4% vs 照片 15%），且不同草图的最优分辨率不同——30% 的草图在 32×32 就能完美检索。由于渲染是不可微的离散操作，采用 RL 训练。向量模态编码器比图像更轻量且天然适合编码抽象程度。

3. **强化学习奖励设计**:

    - 功能：平衡检索精度和计算效率的训练信号
    - 核心思路：精度奖励 $R_{acc} = \lambda_r(1/r) + \lambda_{Tri}(-\mathcal{L}_{Tri})$，其中 $r$ 是检索排名。计算奖励 $R_{comp} = -\mathcal{L}_F$，$\mathcal{L}_F = \frac{\sum_j q_j \cdot \eta_j}{q_{max} - q_{min}}$ 是 FLOPs 加权的归一化开销。总奖励 $R_{Tot} = \lambda_F R_{comp} + (1-\lambda_F) R_{acc}$。使用 Policy Gradient 方法优化策略网络，$\mathcal{L}_{PG}(\theta) = -\frac{1}{B}\sum_i \log p(c|s_v^i) \cdot R_{Tot}^i$。
    - 设计动机：没有画布尺寸的标注真值（ill-posed），暴力搜索不可行。RL 奖励巧妙地将两个矛盾目标——高精度（倾向高分辨率）和低计算（倾向低分辨率）——统一为单一优化目标。

### 损失函数 / 训练策略

SketchyNetV1 训练：$\mathcal{L}_{trn}^{st*} = \frac{1}{4}\sum_{i=1}^4 \mathcal{L}_{trn}^{st(c_i)}$，在多个画布尺寸上平均。SketchyNetV2 训练：冻结 SketchyNetV1，仅用 Policy Gradient 训练 GRU 画布选择器。教师使用 ImageNet 预训练的 VGG-16，学生使用 MobileNetV2。

## 实验关键数据

### 主实验

| 方法 | Params | ShoeV2 Top1 | ShoeV2 FLOPs |
|------|--------|-------------|-------------|
| VGG-16 (Triplet-SN) | 14.71M | 28.71% | 40.18G |
| MobileNetV2 (直接训) | 2.22M | 20.85% | 0.83G |
| **SketchyNetV1** (KD) | 2.22M | 28.46% (↓0.25) | 0.833G |
| **SketchyNetV2** (KD+RL) | 2.27M | 27.89% (↓0.82) | **0.264G** |
| VGG-16 (HOLEF-SN) | 9.31M | 31.74% | 5.758G |
| **SketchyNetV1** (HOLEF) | 2.22M | 31.59% (↓0.15) | 0.833G |

### 消融实验

| 配置 | ShoeV2 Top1 | FLOPs |
|------|------------|-------|
| VGG-16 全分辨率 | 33.03% | 40.18G |
| MobileNetV2 直接训 256×256 | 20.85% | 0.833G |
| SketchyNetV1 (KD) | 28.46% | 0.833G |
| SketchyNetV2 (KD+RL) | 27.89% | 0.254G |
| FLOPs 缩减率 | - | **99.37%** |
| 参数缩减率 | - | **84.89%** |

### 关键发现

- 直接用 MobileNetV2 训 FG-SBIR 精度为 20.85%，但通过关系蒸馏可以恢复到 28.46%（接近 VGG-16 基线 28.71%），FLOPs 缩减 48 倍
- 画布选择器进一步将 FLOPs 从 0.833G 降到 0.254G（3.28 倍），精度仅额外损失 0.57 个点
- 草图与照片对低分辨率的容忍度差异显著：32×32 时草图精度降 4%，照片降 15%
- 约 30% 的草图在最低分辨率 32×32 就能完美检索，验证了自适应选择的必要性
- 方法对不同基线（Triplet-SN、HOLEF-SN）和数据集（ShoeV2、ChairV2、Sketchy、FSCOCO）均有效

## 亮点与洞察

- 填补了草图高效推理的研究空白——照片领域的高效方法不能直接用于草图
- 关键洞见：草图是稀疏数据，天然适合低分辨率推理，这一特性此前未被利用
- 利用草图的向量格式实现"免费的"分辨率变换——不需要像照片那样做双线性插值
- 两阶段设计干净利落，每个阶段独立可用且效果叠加
- 强化学习的奖励设计简洁有效，精度和效率的 trade-off 通过单一超参数 $\lambda_F$ 控制

## 局限与展望

- 仅在 FG-SBIR 任务上验证，是否适用于其他草图任务（如草图生成、草图分割）需要探索
- 画布选择器使用简单的 1 层 GRU，更强的向量编码器（如 Transformer）可能带来进一步提升
- 蒸馏依赖预训练好的大教师网络，如果教师本身精度有限则学生也受限
- Douglas-Peucker 算法截断序列长度可能丢失细节信息
- 学生网络固定为 MobileNetV2，其他轻量架构（如 ShuffleNet、GhostNet）未探索

## 相关工作与启发

- **MobileNet/EfficientNet**：照片高效网络的代表，直接用于草图效果不佳
- **知识蒸馏**：从 logit 蒸馏发展到关系蒸馏（RKD），本文针对跨模态检索场景适配
- **草图强化学习**：RL 在草图领域已有应用（stroke selector、abstraction modeling），本文扩展到分辨率选择
- 启发：利用数据模态的固有特性（稀疏性、抽象性、向量格式）设计高效方法，而非无差别地套用通用压缩方案

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4 | 首个草图高效推理研究，画布选择器设计新颖 |
| 实验充分度 | 4 | 多数据集多基线验证，包含详细pilot study |
| 写作质量 | 4 | 动机清晰，pilot study说服力强 |
| 实用价值 | 4 | 99%+FLOPs缩减，直接适用于商业部署 |

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Sketch to Adapt: Fine-Tunable Sketches for Efficient LLM Adaptation](../../ICML2025/model_compression/sketch_to_adapt_fine-tunable_sketches_for_efficient_llm_adaptation.md)
- [\[ICCV 2025\] Multi-Object Sketch Animation by Scene Decomposition and Motion Planning](../../ICCV2025/model_compression/multi-object_sketch_animation_by_scene_decomposition_and_motion_planning.md)
- [\[ICCV 2025\] VQ-SGen: A Vector Quantized Stroke Representation for Creative Sketch Generation](../../ICCV2025/model_compression/vq-sgen_a_vector_quantized_stroke_representation_for_creative_sketch_generation.md)
- [\[ICML 2025\] Lego Sketch: A Scalable Memory-augmented Neural Network for Sketching Data Streams](../../ICML2025/model_compression/lego_sketch_a_scalable_memory-augmented_neural_network_for_sketching_data_stream.md)
- [\[CVPR 2025\] Alternating Gradient Flow Utility: A Unified Metric for Structural Pruning and Dynamic Routing in Deep Networks](alternating_gradient_flow_utility_a_unified_metric_for_structural_pruning_and_dy.md)

<!-- RELATED:END -->
