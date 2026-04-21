---
title: >-
  [论文解读] DOS: Distilling Observable Softmaps of Zipfian Prototypes for Self-Supervised Point Representation
description: >-
  [AAAI 2026][模型压缩][自监督学习] 提出DOS框架，通过仅在可观测（未掩码）点上蒸馏语义软图（Softmap），结合Zipfian先验的Zipf-Sinkhorn正则化来处理3D语义的长尾分布，在六个3D基准上实现了自监督学习的SOTA，线性探测可达监督性能的95%。
tags:
  - AAAI 2026
  - 模型压缩
  - 自监督学习
  - 点云表征
  - 自蒸馏
  - 长尾分布
  - 语义软图
---

# DOS: Distilling Observable Softmaps of Zipfian Prototypes for Self-Supervised Point Representation

**会议**: AAAI 2026  
**arXiv**: [2512.11465](https://arxiv.org/abs/2512.11465)  
**代码**: 无（释放了预训练权重）  
**领域**: 模型压缩  
**关键词**: 自监督学习, 点云表征, 自蒸馏, 长尾分布, 语义软图

## 一句话总结
提出DOS框架，通过仅在可观测（未掩码）点上蒸馏语义软图（Softmap），结合Zipfian先验的Zipf-Sinkhorn正则化来处理3D语义的长尾分布，在六个3D基准上实现了自监督学习的SOTA，线性探测可达监督性能的95%。

## 研究背景与动机

3D点云的自监督学习（SSL）近年来取得了显著进展，但仍面临三个核心挑战：

**挑战一：位置信息泄露导致捷径学习。** 现有掩码自蒸馏方法（如Sonata、MSM）用掩码token的位置嵌入来推断特征，模型可以"走捷径"——不真正理解语义，仅利用位置信息恢复特征。这导致学到的是低级几何特征而非高级语义。

**挑战二：监督信号不够丰富。** 基于特征回归的方法逐点独立对齐，忽略了不同点对于给定语义概念的相对重要性。聚类方法假设均匀的原型使用，但这与3D场景中的语义不平衡相矛盾。

**挑战三：3D语义呈长尾分布。** 在真实场景中，部分语义类别（如"道路"）点数极多，而其他类别（如"行人"、"交通锥"）点数稀少。标准的Sinkhorn-Knopp约束均匀原型使用，与自然频率分布不匹配，导致频繁结构被过度分割。

**本文的核心idea**：通过三个创新设计同时解决以上三个挑战：

**可观测点蒸馏**：丢弃掩码token，仅在可见点上监督，从根本上消除位置泄露

**语义软图（Softmap）蒸馏**：将prototype的激活在点间归一化，促进点间竞争，提供比聚类更丰富的梯度

**Zipf-Sinkhorn正则化**：用幂律先验替代均匀先验，对齐真实世界3D语义的长尾分布

## 方法详解

### 整体框架

DOS采用标准的学生-教师掩码自蒸馏框架（如图2所示）：

1. 点云被裁剪生成两个视图，各自应用不同的增强和随机掩码
2. **学生**仅处理可见点 $\mathcal{P}_v$，**教师**处理完整点云 $\mathcal{P}$
3. 教师权重通过EMA更新
4. 计算与可学习原型 $\mathcal{Q}$ 的相似度，归一化为软图
5. 教师软图经Zipf-Sinkhorn正则化，学生学习匹配之

### 关键设计

#### 1. **可观测点蒸馏（Observable Point Distillation）**

标准掩码蒸馏在掩码点上监督学生，但恢复这些点往往依赖位置嵌入实现"捷径学习"。DOS的做法极其简洁：

- **彻底丢弃掩码token**，仅在可见点 $\mathcal{I}_v$ 上施加监督
- 教师看到完整输入并在可见子集上产生目标
- 学生仅看到部分输入，被隐式鼓励推理缺失区域

这个设计看似"反直觉"——不监督掩码点，还怎么学习？关键在于：教师的输出反映了完整上下文，而学生只看到部分输入，因此在可见点上匹配教师就已经隐含了对缺失区域的推理。

消融实验验证了其效果：从naive（54.7 mIoU）到observable（69.3 mIoU），**+14.6 mIoU的巨大提升**。

#### 2. **语义软图（Softmap）蒸馏**

与传统聚类方法在每个点独立归一化不同，Softmap在**点之间**进行归一化：

$$S_T(i,k) = \frac{s^T_{ik}}{\sum_{j \in \mathcal{I}_v} s^T_{jk}}, \quad S_S(i,k) = \frac{s^S_{ik}}{\sum_{j \in \mathcal{I}_v} s^S_{jk}}$$

其中 $s^T_{ik} = \exp(\cos(\phi_T(\mathcal{P})_i, q_k) / \tau_T)$ 是可见点与原型的余弦相似度。

**关键区别**（如图3所示）：
- **聚类**：对每个点，在所有prototype上归一化（点→prototype分配）
- **Softmap**：对每个prototype，在所有点上归一化（prototype的空间激活图）

这个转换带来的好处：
- 每个prototype诱导点间竞争，将点视为关于某个语义概念的软正/负样本
- 鼓励空间定位的表征学习
- 即使弱激活的prototype也能影响学习，避免信息丢失

损失函数为KL散度：

$$\mathcal{L}_\sigma(\mathcal{P}, \mathcal{P}_v, \mathcal{Q}) = \frac{1}{K}\sum_{k=1}^K \text{KL}(\tilde{S}_T(:,k) \| S_S(:,k))$$

同视图和跨视图蒸馏各占一半权重，总损失 $\mathcal{L}_{\text{total}} = \mathcal{L}_1 + \mathcal{L}_2$。

#### 3. **Zipf-Sinkhorn正则化**

Softmap虽然稳定，但仍可能导致语义原型崩溃或点级崩溃。标准Sinkhorn强制均匀原型使用，但3D语义天然遵循Zipf分布（如图4所示，ScanNet200类别频率呈幂律分布）。

**Zipf-Sinkhorn算法**（Algorithm 1）：

1. 计算Zipf先验：$w_k \propto 1/k^\alpha$，归一化
2. 初始化：归一化相似度矩阵 $F$
3. 迭代Sinkhorn（T轮）：
    - 行归一化：使每个点的总贡献相等
    - 列归一化到 $\mathbf{w}$：使prototype使用匹配Zipf分布
4. 列归一化得到最终软图 $\tilde{S}_T$

**Zipf指数 $\alpha$ 的作用**：
- $\alpha = 0$：退化为均匀先验（标准Sinkhorn）
- $\alpha \approx 1.3-1.6$：最优范围，平衡语义覆盖和特化
- $\alpha \geq 2$：过度集中，训练不稳定

### 损失函数 / 训练策略

- 使用PTv3作为默认编码器，2×A100训练约20小时
- 掩码率：语义分割70%，目标检测60%
- 掩码块大小：室内40cm，室外1m
- 体素大小：室内0.08m，室外0.2m
- 1024个原型
- 分割任务附加轻量PTv3解码器，检测任务用CenterPoint检测器

## 实验关键数据

### 主实验

**语义分割线性探测（LP）结果**：

| 方法 | nuScenes mIoU | Waymo mIoU | SemKITTI mIoU | ScanNet mIoU |
|------|-------------|-----------|--------------|-------------|
| PTv3 (监督) | 80.4 | 71.3 | 69.1 | 77.6 |
| NOMAE | 65.1 | 59.2 | - | - |
| Sonata* | 66.1 | 60.5 | 62.0 | 72.5 |
| **DOS*** | **74.8** | **67.0** | **68.3** | **73.9** |

DOS* LP达到了监督PTv3的**93-99%性能**，是SSL方法的新SOTA。

**微调（FT）结果**：

| 方法 | nuScenes mIoU | Waymo mIoU | SemKITTI mIoU | ScanNet mIoU |
|------|-------------|-----------|--------------|-------------|
| PTv3 (监督) | 80.4 | 71.3 | 69.1 | 77.6 |
| Sonata* | 81.7 | 72.9 | 72.6 | 79.4 |
| **DOS*** | **81.8** | **73.9** | **73.5** | **79.7** |

微调后DOS*在所有数据集上**超过监督基线**。

**目标检测（nuScenes）**：

| 方法 | NDS | mAP | 设置 |
|------|-----|-----|------|
| CenterPoint (无预训练) | 65.4 | 57.6 | 100%标签 |
| NOMAE | 60.9 | 54.4 | 20%标签 |
| **DOS** | **62.1** | **57.1** | 20%标签 |
| **CenterPoint + DOS** | **69.7** | **65.5** | 100%标签 |

DOS预训练为CenterPoint带来+4.3 NDS、+7.9 mAP的提升。

### 消融实验

**组件增量消融**（nuScenes LP）：

| 组件 | mIoU | 增量提升 |
|------|------|---------|
| Naive掩码自蒸馏 | 54.7 | 基线 |
| + Token jitter (Sonata) | 55.1 | +0.4 |
| + 可观测点蒸馏 | 69.3 | **+14.6** |
| + Softmap (替代聚类) | 72.3 | +3.0 |
| + 特征回归 (替代聚类) | 63.0 | -6.3（不如聚类） |
| + Zipf-Sinkhorn (α=1.3) | **74.1** | +1.8 |

**Zipf指数 $\alpha$ 消融**（ScanNet200）：

| α | ScanNet mIoU | ScanNet200 mIoU | Head(66) | Common(68) | Tail(66) |
|---|-------------|----------------|----------|-----------|----------|
| 0.0 | 71.9 | 27.9 | 50.4 | 20.5 | 10.6 |
| 1.3 | 72.8 | 29.1 | 50.8 | **23.5** | **13.2** |
| 3.0 | 69.8 | 27.7 | - | - | - |

Zipf先验主要提升了**中频和低频类别**（Common +3.0, Tail +2.6），而对高频类别影响微小。

### 关键发现

1. **可观测点蒸馏是最重要的贡献**：+14.6 mIoU，远超token jitter的+0.4
2. **Softmap优于聚类和特征回归**：72.3 vs 69.3 vs 63.0，点间竞争提供更丰富的梯度
3. **Zipf先验主要帮助中低频类别**：在ScanNet200上Tail类别提升+2.6 mIoU
4. **跨域迁移能力强**：仅在Waymo训练的DOS在nuScenes和SemKITTI上超过所有方法
5. **少样本能力出色**：仅5个标注场景在ParisLuco上即可达到85.9 mIoU
6. **数据效率高**：0.1%标注数据下表现远超监督训练

## 亮点与洞察

1. **可观测点蒸馏的设计简洁而有效**：不监督掩码点反而大幅提升性能，违反直觉但逻辑严密——消除了位置泄露这个根本问题
2. **Softmap的概念创新**：将归一化方向从"每个点在prototype上分配"转为"每个prototype在点上分布"，建立了与InfoNCE和聚类的统一视角（图3）
3. **Zipf先验理论基础扎实**：自然界中的频率分布普遍服从幂律，3D语义也不例外，用Zipf分布替代均匀分布是自然的推理
4. **通用预训练权重的发布**具有实际价值——跨数据集预训练的LiDAR骨干可直接用于新域
5. **实验覆盖极其全面**：6个基准、LP/FT/检测三种评估、跨域/少样本/数据效率等多种设置

## 局限与展望

1. 仅使用PTv3骨干，虽然在附录中有替代架构实验，但主实验缺乏多架构验证
2. Zipf指数 $\alpha$ 需要手动选择，自适应调整α可能进一步提升性能
3. 1024个原型是固定设计，原型数与语义粒度的最优关系可进一步探索
4. 室内场景上的LP提升不如室外场景显著（ScanNet200: 30.7 vs 监督35.3）
5. 未探索与2D-3D跨模态蒸馏（如D-DITR）的结合

## 相关工作与启发

- **Sonata**：最接近的前作，使用掩码自蒸馏+token jitter，DOS在其基础上解决位置泄露和长尾问题
- **SwAV** (Caron et al., 2020)：首次将Sinkhorn用于SSL原型平衡，DOS扩展为Zipf感知版本
- **NOMAE**：重建局部邻域避免泄露，但仍受限于几何快捷方案
- **PTv3**：Point Transformer V3，DOS的默认骨干架构
- 启发：在SSL中显式建模数据分布先验（如长尾）可能是一个被低估的研究方向；"少即是多"的原则（仅监督可见点）值得在其他掩码学习方法中探索

## 评分
- 新颖性: ⭐⭐⭐⭐⭐（三个创新组件各自独立且相互协同）
- 实验充分度: ⭐⭐⭐⭐⭐（6个基准、多种评估协议、详尽消融）
- 写作质量: ⭐⭐⭐⭐⭐（逻辑清晰，图示精美，公式推导规范）
- 价值: ⭐⭐⭐⭐⭐（3D SSL的新SOTA，释放预训练权重具有直接应用价值）

<!-- RELATED:START -->

## 相关论文

- [MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](../../CVPR2026/model_compression/mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)
- [AutoSSVH: Exploring Automated Frame Sampling for Efficient Self-Supervised Video Hashing](../../CVPR2025/model_compression/autossvh_exploring_automated_frame_sampling_for_efficient_self-supervised_video_h.md)
- [VESSA: Video-based objEct-centric Self-Supervised Adaptation for Visual Foundation Models](../../NeurIPS2025/model_compression/vessa_video-based_object-centric_self-supervised_adaptation_for_visual_foundatio.md)
- [Distilling Cross-Modal Knowledge via Feature Disentanglement](distilling_cross-modal_knowledge_via_feature_disentanglement.md)
- [Distilling Balanced Knowledge from a Biased Teacher](../../CVPR2026/model_compression/distilling_balanced_knowledge_from_a_biased_teacher.md)

<!-- RELATED:END -->
