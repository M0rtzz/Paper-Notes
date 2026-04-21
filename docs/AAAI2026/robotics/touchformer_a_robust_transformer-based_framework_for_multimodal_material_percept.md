---
title: >-
  [论文解读] TouchFormer: A Robust Transformer-based Framework for Multimodal Material Perception
description: >-
  [AAAI 2026][机器人][多模态融合] 提出 TouchFormer，一个鲁棒的多模态融合框架，通过模态自适应门控（MAG）、模态内/模态间注意力机制和跨实例嵌入正则化（CER）三个互补模块，在视觉受损条件下实现可靠的材质感知，并在火灾场景机器人分拣实验中验证有效性。
tags:
  - AAAI 2026
  - 机器人
  - 多模态融合
  - 材质感知
  - 触觉传感
  - 自适应门控
  - Transformer
---

# TouchFormer: A Robust Transformer-based Framework for Multimodal Material Perception

**会议**: AAAI 2026  
**arXiv**: [2511.19509](https://arxiv.org/abs/2511.19509)  
**代码**: [https://touchformer.github.io/TouchFormer/](https://touchformer.github.io/TouchFormer/)  
**领域**: 机器人  
**关键词**: 多模态融合, 材质感知, 触觉传感, 自适应门控, Transformer

## 一句话总结

提出 TouchFormer，一个鲁棒的多模态融合框架，通过模态自适应门控（MAG）、模态内/模态间注意力机制和跨实例嵌入正则化（CER）三个互补模块，在视觉受损条件下实现可靠的材质感知，并在火灾场景机器人分拣实验中验证有效性。

## 研究背景与动机

材质感知是机器人与环境交互的关键能力。表面材质分类（SSMC——已知类别）和未知表面材质分类（USMC——未见过的类别）是两个代表性任务。

**视觉失效场景的挑战**：在火灾、浓雾、暗工厂等场景中，视觉方法性能严重退化。因此非视觉材质感知（基于触觉和听觉）变得特别重要。

**现有非视觉方法的三大问题**：

**天真融合策略**：MTCNN 等方法直接拼接不同模态特征，给所有模态分配相同权重。但识别不同材质时，关键模态是变化的——有些材质靠声音更好识别，有些靠摩擦力。等权重融合会削弱关键模态的优势。

**对噪声和缺失不鲁棒**：真实世界中传感器采样率不同导致时序不对齐，传感器可能故障或数据被噪声污染。现有模型在噪声环境下性能急剧下降。

**忽略细粒度子类别区分**：如区分软木和硬木——现有方法只做粗粒度分类（8 大类），未探索 193 种细粒度子类别。

**核心动机**：如何设计一个对噪声输入鲁棒、能自适应调节模态权重、且能处理细粒度分类的多模态融合框架？

## 方法详解

### 整体框架

TouchFormer 接收可能带噪或不完整的多模态序列（声音 S、法向力 N、摩擦力 F、加速度 A），经过三个互补模块处理：

1. **Modality-Adaptive Gating (MAG)**：在输入端动态评估模态质量并过滤噪声
2. **Intra- and Inter-modal Transformer Fusion**：在潜空间中自适应整合跨模态特征
3. **Cross-Instance Embedding Regularization (CER)**：优化原型空间增强类间可分性

### 关键设计

#### 1. **模态自适应门控（MAG）**

传统方法对所有模态等权处理，MAG 则动态评估每个模态的信息质量并分配权重：

**第一步**：中间特征计算
$$H_m = \text{ReLU}(W_1 X_m + b_1)$$

**第二步**：门控权重生成
$$g_m = \sigma(W_2 H_m + b_2), \quad g_m \in [0, 1]$$

**第三步**：低质量模态过滤——引入阈值 $gate_{th}$，低于阈值的模态被丢弃，防止污染融合表示。

**第四步**：softmax 归一化得到最终重要性权重
$$\alpha_m = \frac{\exp(g_m)}{\sum_k \exp(g_k)}, \quad m \in \{S, N, F, A\}$$

**第五步**：加权 + 位置编码
$$Z_m = \alpha_m \odot (X_m + PE(T, d))$$

设计动机：让模型在源头就过滤掉噪声模态，提升后续融合的可靠性。

#### 2. **模态间/模态内 Transformer 融合模块**

受 MulT 启发，但做了关键改进——在最终特征整合前再次应用 MAG 重加权，阻断负迁移。

**时序卷积 + 位置编码**：
$$\hat{X}_m = \text{Conv1D}(X_m, k_m), \quad Z_m^{[0]} = \hat{X}_m + PE(T_m, d)$$

**模态间 Transformer**（Cross-Modal Attention）：
$$\hat{Y}_{\alpha \leftarrow \beta} = \text{softmax}\left(\frac{Q_\alpha K_\beta^\top}{\sqrt{d_k}}\right) V_\beta$$

注意力输出被 MAG 的权重 $\alpha_\beta$ 调制：$Y_{\alpha \leftarrow \beta} = \alpha_\beta \hat{Y}_{\alpha \leftarrow \beta}$。可靠性高的源模态贡献更大。

**模态内 Transformer**（Self-Attention）：
$$\tilde{Z}_\alpha = Z_\alpha^{[0]} + Y_{\alpha \leftarrow \beta}$$
$$Z_\alpha^{\text{intra}} = \text{Transformer}(\tilde{Z}_\alpha, \tilde{Z}_\alpha, \tilde{Z}_\alpha)$$

**最终融合**：四个模态的内部表示经重加权后拼接：
$$Z = \text{Concat}[\alpha_S Z_S^{\text{intra}}, \alpha_N Z_N^{\text{intra}}, \alpha_F Z_F^{\text{intra}}, \alpha_A Z_A^{\text{intra}}]$$

关键创新：不需要显式的时序对齐——通过 cross-modal attention 隐式处理不同采样率的模态。

#### 3. **跨实例嵌入正则化（CER）**

基于对比学习原理，从跨实例视角在嵌入空间中施加结构约束：

$$\mathcal{L}_C = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\sum_{j \neq i} \mathbf{I}_{\{y_i = y_j\}} \exp(S_{ij}/\tau)}{\sum_{j \neq i} \exp(S_{ij}/\tau)}$$

其中 $S_{ij} = z_i^\top z_j$ 是 $\ell_2$ 归一化嵌入的相似度矩阵，$\tau$ 是温度超参数。

作用：拉近同类嵌入、推远异类嵌入，增强类间可分性和类内紧凑性。特别有利于细粒度子类别的区分（如区分不同种类的木材）。

### 损失函数 / 训练策略

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}}(y, \hat{y}) + \lambda \mathcal{L}_C$$

- 分类损失 $\mathcal{L}_{\text{cls}}$ + CER 正则化 $\mathcal{L}_C$ 联合训练
- MAG 模块参数与网络同步更新
- Adam 优化器，权重衰减 0.1，初始学习率 0.1，余弦退火
- Batch size 32，训练 50 epochs
- USMC 任务采用 5 折交叉验证
- 实验在 A100 GPU 上进行

## 实验关键数据

### 主实验

在 LMTHM 数据集上的表面材质分类性能：

| 任务 | 类别数 | 方法 | 准确率 (%) | G_mean |
|---|---|---|---|---|
| SSMC | 8 | DDQN | 92.71 | 0.93 |
| SSMC | 8 | MMUSR（含视觉） | 93.8 | 0.94 |
| SSMC | 8 | **TouchFormer (Ours, 无视觉)** | **95.19 (+2.48)** | **0.95** |
| USMC | 8 | LSTM | 74.23 | 0.73 |
| USMC | 8 | MTCNN | 87.55 | 0.88 |
| USMC | 8 | **TouchFormer (Ours)** | **94.38 (+6.83)** | **0.94** |
| 细粒度 | 193 | MTCNN | 80.21 | 0.80 |
| 细粒度 | 193 | **TouchFormer (Ours)** | **89.77 (+9.56)** | **0.90** |

最突出的结果：不使用视觉的 TouchFormer（95.19%）超过了使用视觉的 MMUSR（93.8%）；在 193 类细粒度分类上提升 9.56%。

### 消融实验

| 配置 | SSMC | USMC | 细粒度 | 说明 |
|---|---|---|---|---|
| Baseline（仅 Transformer 融合） | 91.32% | 90.17% | 83.53% | 无 MAG、无 CER |
| Baseline + MAG | 93.15% | 92.56% | 84.88% | +自适应门控 |
| **Baseline + MAG + CER** | **95.19%** | **94.38%** | **89.77%** | 完整框架 |

MAG 贡献：+1.83%/+2.39%/+1.35%（三个任务）
CER 贡献：+2.04%/+1.82%/+4.89%（三个任务）
CER 对细粒度分类提升最大（+4.89%），验证了对比学习在子类别区分上的有效性。

### 模态缺失鲁棒性

| 多模态输入 (S/N/F/A) | LSTM | MTCNN | TouchFormer |
|---|---|---|---|
| ✗/✓/✓/✓（无声音） | 73.78% | 83.43% | 87.92% |
| ✓/✗/✓/✓（无法向力） | 71.69% | 83.18% | 88.55% |
| ✓/✓/✗/✓（无摩擦力） | 69.09% | 84.99% | 89.12% |
| ✓/✓/✓/✗（无加速度） | 75.50% | 87.00% | 89.84% |
| ✓/✓/✓/✓（完整） | 74.23% | 87.55% | **94.38%** |

关键发现：即使缺失任意单个模态，TouchFormer 的性能（87.92%-89.84%）仍然超过其他方法使用全部模态时的性能。MAG 门控机制在模态缺失时自动调整权重分配。

### 关键发现

1. **无视觉 > 有视觉**：在视觉受损条件下，TouchFormer 不使用视觉仍超过需要视觉输入的 MMUSR
2. **CER 对细粒度分类至关重要**：193 类子分类上提升 4.89%，t-SNE 可视化显示嵌入聚类显著改善
3. **噪声鲁棒性**：在各种噪声比例（0.0-1.0）下，TouchFormer 始终优于基线
4. **真实机器人验证**：在模拟火灾场景中，RealMan RM65-B 机械臂利用 TouchFormer 成功完成无视觉的材质分拣任务

## 亮点与洞察

1. **MAG 门控机制的精巧设计**——不仅加权，还通过阈值过滤低质量模态，从输入源头保证信息质量
2. **不需要显式时序对齐**——通过 cross-modal attention 隐式处理不同采样率的模态，大幅简化了数据预处理
3. **CER + MAG 的互补性**：MAG 提升输入可靠性，CER 提升输出可区分性，两者从不同维度增强模型
4. **FISHM 数据集的工程价值**——专门为火灾场景收集的多模态触觉数据集，填补了极端环境下的数据空白
5. **从感知到操作的闭环验证**——不仅验证分类精度，还展示了在真实机器人上的分拣应用

## 局限与展望

1. **uSkin 传感器的特定性**：结果与硬件高度关联，换用其他触觉传感器时效果未知
2. **分类而非回归**：仅做离散类别分类，未探索材质属性的连续预测（如硬度、粗糙度）
3. **火灾场景实验仍是受控模拟**：物体位置固定、使用固定参数程序抓取，真实火灾环境更加不可预测
4. **仅在两个数据集上测试**：LMTHM 和 FISHM，泛化性验证不足
5. **MAG 的阈值需要手动调节**：$gate_{th}$ 的最优值可能随任务变化

## 相关工作与启发

- **与 MulT 的关系**：TouchFormer 在 MulT 的 cross-modal attention 基础上加了 MAG 重加权，是关键改进
- **与对比学习的结合**：CER 本质上是 SupCon loss 的应用，但在触觉材质感知中的应用是创新
- **应急机器人应用前景**：消防救援机器人在能见度为零时的材质感知能力，具有重要实际价值
- **传感器融合的通用性**：MAG + Transformer 融合模式可扩展到其他多传感器场景（如自动驾驶、医疗机器人）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — MAG 门控和 CER 正则化的组合设计有效，但各模块技术本身不算全新
- **实验充分度**: ⭐⭐⭐⭐⭐ — 消融、噪声鲁棒性、模态缺失、细粒度分类、真实机器人实验非常全面
- **写作质量**: ⭐⭐⭐⭐ — 问题动机清晰，方法描述详细
- **价值**: ⭐⭐⭐⭐ — 对极端环境下的机器人材质感知有重要工程价值

<!-- RELATED:START -->

## 相关论文

- [From Passive Perception to Active Memory: A Weakly Supervised Image Manipulation Localization Framework Driven by Coarse-Grained Annotations](from_passive_perception_to_active_memory_a_weakly_supervised_image_manipulation_.md)
- [Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning](attention_as_binding_a_vector-symbolic_perspective_on_transformer_reasoning.md)
- [SpatialActor: Exploring Disentangled Spatial Representations for Robust Robotic Manipulation](spatialactor_exploring_disentangled_spatial_representations_for_robust_robotic_m.md)
- [Robust Out-of-Order Retrieval for Grid-Based Storage at Maximum Capacity](robust_out-of-order_retrieval_for_grid-based_storage_at_maximum_capacity.md)
- [RF-MatID: Dataset and Benchmark for Radio Frequency Material Identification](../../ICLR2026/robotics/rf-matid_dataset_and_benchmark_for_radio_frequency_material_identification.md)

<!-- RELATED:END -->
