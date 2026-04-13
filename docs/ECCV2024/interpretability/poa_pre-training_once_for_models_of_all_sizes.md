---
title: >-
  [论文解读] POA: Pre-training Once for Models of All Sizes
description: >-
  [ECCV 2024][自监督预训练] POA 提出在自监督自蒸馏框架中引入**弹性学生分支**，通过参数共享和随机子网络采样，**一次预训练即可同时产出上百个不同大小的预训练模型**（如从 ViT-L 直接提取 ViT-S/B），各子网络在 k-NN、线性探测和下游任务上均达到 SOTA 水平。
tags:
  - ECCV 2024
  - 自监督预训练
  - 弹性网络
  - 自蒸馏
  - 一次训练多尺寸模型
  - Once-for-All
---

# POA: Pre-training Once for Models of All Sizes

**会议**: ECCV 2024  
**arXiv**: [2408.01031](https://arxiv.org/abs/2408.01031)  
**代码**: https://github.com/Qichuzyy/POA (有)  
**领域**: 模型压缩 / 自监督学习  
**关键词**: 自监督预训练, 弹性网络, 自蒸馏, 一次训练多尺寸模型, Once-for-All

## 一句话总结

POA 提出在自监督自蒸馏框架中引入**弹性学生分支**，通过参数共享和随机子网络采样，**一次预训练即可同时产出上百个不同大小的预训练模型**（如从 ViT-L 直接提取 ViT-S/B），各子网络在 k-NN、线性探测和下游任务上均达到 SOTA 水平。

## 研究背景与动机

自监督学习（SSL）在大模型上取得了出色的视觉表征能力，但实际部署时需要一系列不同大小的模型来匹配不同算力/存储限制（如 Gemini Nano/Pro/Ultra）。目前的做法是先训练一个大模型，再通过剪枝、知识蒸馏或从头重训小模型来适配不同场景——这些方案开发成本高昂。

**核心矛盾**：现有 SSL 方法（DINO、iBOT 等）每次只训练一个固定大小的模型，要得到 N 个不同大小的模型就要训练 N 次，计算预算线性增长。

**切入角度**：现代网络结构（ViT、Swin、ResNet）天然具有"小模型是大模型子网络"的特性（宽度减少 = 减少注意力头，深度减少 = 跳过部分 block）。基于此观察，POA 在 teacher-student 自蒸馏框架中新增一个**弹性学生分支**，每步随机采样一个子网络参与训练，使得预训练结束后可直接从 teacher 中提取任意大小的高质量子网络。

**核心 idea**：用弹性子网络采样 + 同视图蒸馏，把 "预训练" 和 "多尺寸模型生成" 统一到一次训练中。

## 方法详解

### 整体框架

POA 是一个**三分支自蒸馏框架**：Teacher、Intact Student（完整学生）、Elastic Student（弹性学生）。

- 输入图像 $x$ 生成两个增强视图 $x_a, x_b$
- Teacher 处理 $x_a$，两个 Student 处理 $x_b$
- Intact Student 与 Teacher 做跨视图蒸馏（常规 SSL 表征学习）
- Elastic Student 同时接受 Teacher 的跨视图蒸馏 **和** Intact Student 的同视图蒸馏
- Teacher 通过 EMA 更新，融合 Intact Student 和 Elastic Student 的参数
- 每个训练步，Elastic Student 从完整学生中随机采样一个子网络

### 关键设计

1. **弹性学生（Elastic Student）**:
   - 做什么：在每个训练 iteration 随机采样 width 和 depth，构造一个子网络参与训练
   - 核心思路：通过参数切片实现弹性。对于 ViT，弹性宽度通过减少注意力头数实现，弹性深度通过等间距选取 block 实现。定义 $M+1$ 种弹性宽度：$D_i = (N_h - i) \cdot D_h$，以及 $N+1$ 种弹性深度：$L_i = L_{max} - i$。总共可生成 $(M+1) \times (N+1)$ 个子网络。
   - 参数提取方式：对于弹性 MSA，输入投影权重 $w_i^{a1} = w^{a1}[:, :D_i] \cdot \alpha_i$，其中缩放因子 $\alpha_i = D_{max}/D_i$ 补偿维度缩减带来的尺度变化。MLP 和 LN 类似处理。
   - 深度弹性：对深度 $L_i$，按等间距选取 block ID：$BID_j^{L_i} = \lfloor (L_{max}-1) \cdot j / (L_i - 1) \rfloor$
   - 设计动机：小模型天然是大模型的子网络，通过参数共享使子网络在训练过程中被充分优化；弹性分支还起到**模型集成**的效果（每步不同子网络参与 teacher 的 EMA）和**训练正则化**的作用（稳定训练、防止 loss 发散）

2. **同视图蒸馏（Same-view Distillation）**:
   - 做什么：让 Elastic Student 在相同视图上模仿 Intact Student 的输出
   - 核心思路：这是标准的知识蒸馏，将 Intact Student 已学到的更好表征传递给弹性子网络。损失为 $\mathcal{L}_{ES2}^g = -p_{b1} \log(p_{b2})$
   - 设计动机：消融实验表明，同视图蒸馏对子网络质量的贡献**远大于**跨视图蒸馏。因为跨视图蒸馏是表征学习，而同视图蒸馏是从已有的好表征直接蒸馏，对小网络更高效。去掉 $\mathcal{L}_{ES2}$ 导致 ViT-S 的 k-NN 下降 3.4%。

3. **多投影头（Multiple Projection Heads, MPH）**:
   - 做什么：在 backbone 后接多个结构相同但 prototype 数量不同的投影头
   - 核心思路：对每个投影头独立计算蒸馏损失，最终取平均：$\mathcal{L} = \frac{1}{H} \sum_{i=1}^H \mathcal{L}_{S_i}$
   - 设计动机：由于每步只随机选一个子网络，单个子网络的训练不够充分。MPH 引入不同语义空间（不同 prototype 数量），使 teacher 能从多个角度蒸馏知识到子网络中，对小网络的提升尤其明显。

### 损失函数 / 训练策略

总损失由 Intact Student 和 Elastic Student 的损失加权组合：

$$\mathcal{L}_S = \lambda \mathcal{L}_{IS} + (1-\lambda)(\mathcal{L}_{ES1} + \mathcal{L}_{ES2})$$

其中 $\mathcal{L}_{IS}$ 是完整学生的跨视图蒸馏，$\mathcal{L}_{ES1}$ 是弹性学生的跨视图蒸馏，$\mathcal{L}_{ES2}$ 是弹性学生的同视图蒸馏。同时采用 multi-crop 策略生成多个局部视图，增加 local-to-global 的对应学习。

训练设置：ImageNet-1K 无标签预训练，AdamW 优化器，batch size 1600（32×A100），学习率 $lr = 0.004 \times \sqrt{batch\_size / 1024}$，linear warmup 10 epochs + cosine decay。教师温度从 0.04 warmup 到 0.07。

## 实验关键数据

### 主实验

| 数据集 | Backbone | 指标 | POA | 之前 SOTA | 提升 |
|--------|----------|------|-----|-----------|------|
| ImageNet-1K | ViT-L/16 | k-NN | 82.3% | 82.0% (DINOv2) | +0.3% |
| ImageNet-1K | ViT-L/16 | LP | 83.6% | 83.3% (DINOv2) | +0.3% |
| ImageNet-1K | ViT-B/16 (提取) | k-NN | 80.9% | 77.1% (iBOT) | +3.8% |
| ImageNet-1K | ViT-S/16 (提取) | k-NN | 76.8% | 75.3% (ENT) | +1.5% |
| COCO | ViT-S/16 | AP^b | 50.6 | 49.4 (iBOT) | +1.2 |
| COCO | ViT-B/16 | AP^b | 52.4 | 51.2 (iBOT) | +1.2 |
| ADE20K | ViT-B/16 | mIoU(linear) | 40.3 | 38.3 (iBOT) | +2.0 |

**关键亮点**：ViT-S/16 和 ViT-B/16 是**零额外预训练**（直接从 ViT-L teacher 提取），有效训练 epoch 为 0，却超过了单独预训练 3200 epochs 的 DINO/iBOT。

### 消融实验

| 配置 | ViT-S k-NN | ViT-B k-NN | ViT-L k-NN | 说明 |
|------|-----------|-----------|-----------|------|
| MPH + $\mathcal{L}_{ES1}$ + $\mathcal{L}_{ES2}$ | 76.8 | 80.9 | 82.3 | 完整 POA |
| MPH + $\mathcal{L}_{ES1}$ (去掉同视图蒸馏) | 72.8 | 79.1 | 82.1 | ViT-S 下降 4.0% |
| MPH + $\mathcal{L}_{ES2}$ (去掉跨视图蒸馏) | 75.1 | 80.2 | 82.2 | 同视图蒸馏更重要 |
| 无 MPH + $\mathcal{L}_{ES1}$ + $\mathcal{L}_{ES2}$ | 76.2 | 80.7 | 82.2 | MPH 对小网络帮助大 |

### 关键发现

- **同视图蒸馏 $\mathcal{L}_{ES2}$ 是最关键组件**：对小网络贡献最大，去掉后 ViT-S k-NN 下降 4.0%
- **POA 大幅优于 DINOv2+SEED 两阶段方案**：ViT-S k-NN 76.8% vs 74.0%（同 epoch），且 POA 不需要额外蒸馏阶段
- **弹性分支双重作用**：(1) 稳定训练（ResNet 实验中，无弹性分支 loss 发散到 NaN）；(2) 作为模型集成增强 teacher 表征质量
- 三分支变体 POA-V1/V2（去掉 Intact Student）性能大幅下降，验证了完整学生分支的必要性

## 亮点与洞察

- **"预训练一次，部署多次"范式**：从一个 ViT-L 训练中同时获得 143 个不同大小的高质量模型，极大降低了部署成本
- **弹性参数提取 + 缩放因子 $\alpha_i$**：简洁优雅的设计，直接切片权重矩阵并乘以维度比例因子，无需额外参数
- **同视图蒸馏的洞见**：从已有的好表征蒸馏比直接做跨视图表征学习更有效，特别是对小模型
- **通用性强**：适用于 ViT、Swin Transformer、ResNet 三种主流架构

## 局限性 / 可改进方向

- 目前仅在 ImageNet-1K 上验证，未在更大数据集（ImageNet-22K、LAION）上测试
- 弹性设计主要覆盖宽度和深度两个维度，未考虑 patch size、分辨率等其他弹性维度
- 尚未扩展到多模态大语言模型（作者提到这是未来方向）
- POA-V3（增加弹性 teacher）效果略好但计算成本更高，说明还有进一步提升空间

## 相关工作与启发

- **vs DINO/iBOT/DINOv2**：这些方法每次只训练一个固定大小的模型，POA 一次生成多个，且子网络性能更好
- **vs SEED（自监督蒸馏）**：SEED 是两阶段方案（先训 teacher 再蒸馏），POA 将预训练和多尺寸生成统一到一阶段，相同 epoch 下 POA 子网络优于 SEED 蒸馏出的模型
- **vs NAS（AutoFormer 等）**：NAS 搜索空间巨大（>10^16），训练后需要搜索+重训。POA 搜索空间紧凑（143个），训练完直接可用
- **vs OFA（Once-for-All）**：OFA 是有监督的，POA 首次在自监督场景实现 Once-for-All

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将 Once-for-All 思想引入自监督学习，三分支设计简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ 三种 backbone、多个下游任务、详细消融、与 KD 方法对比
- 写作质量: ⭐⭐⭐⭐ 整体清晰，公式推导完整，但部分符号较重
- 价值: ⭐⭐⭐⭐⭐ 对实际部署有极大意义，一次训练多次部署大幅降低成本
