---
title: >-
  [论文解读] iDPA: Instance Decoupled Prompt Attention for Incremental Medical Object Detection
description: >-
  [ICML 2025][医学图像][增量目标检测] 提出 iDPA 框架，通过实例级 Prompt 生成（IPG）和解耦 Prompt 注意力（DPA）两大模块，在冻结的视觉-语言目标检测模型上实现增量医学目标检测（IMOD），仅训练 1.4% 的参数即在 13 个跨模态医学数据集上全面超越 SOTA。
tags:
  - ICML 2025
  - 医学图像
  - 增量目标检测
  - 持续学习
  - 提示学习
  - 多模态融合
  - 医学目标检测
---

# iDPA: Instance Decoupled Prompt Attention for Incremental Medical Object Detection

**会议**: ICML 2025  
**arXiv**: [2506.00406](https://arxiv.org/abs/2506.00406)  
**代码**: [https://github.com/HarveyYi/iDPA](https://github.com/HarveyYi/iDPA)  
**领域**: 医学图像  
**关键词**: 增量目标检测, 持续学习, prompt tuning, 多模态融合, 医学目标检测

## 一句话总结

提出 iDPA 框架，通过实例级 Prompt 生成（IPG）和解耦 Prompt 注意力（DPA）两大模块，在冻结的视觉-语言目标检测模型上实现增量医学目标检测（IMOD），仅训练 1.4% 的参数即在 13 个跨模态医学数据集上全面超越 SOTA。

## 研究背景与动机

**核心问题**：医学目标检测需要不断学习新的疾病类别（如新型病变、新器官），但为每个任务单独训练检测器效率低下，联合训练又不现实（无法预定义所有医学概念）。持续学习（Continual Learning）是理想方案，但面临**灾难性遗忘**问题。

**现有方法的不足**：

**全局 Prompt 的前景-背景耦合**：现有 Prompt-based CL 方法（L2P, DualPrompt, CODA-Prompt 等）为分类设计，使用全局 Prompt 编码知识，混合了前景和背景信息。但目标检测需要细粒度的实例级信息，全局 Prompt 中的过量背景信息会干扰定位和分类，尤其在医学影像中不同模态（CT、MRI、X-ray）相似度高时容易导致类别混淆。

**Prompt 与图像-文本 Token 的耦合注意力**：将 Prompt 直接拼接到图像/文本 Token 前，由于图像-文本 Token 长度远超 Prompt 长度，Prompt 信息被稀释，阻碍了任务特定知识的学习。此外，这种拼接在视觉-语言模型的跨模态交互中引入视觉与文本 Prompt 之间的额外干扰。

**知识注入位置不当**：现有方法仅在 backbone 层插入 Prompt，但检测所需的细粒度推理发生在 backbone 之后的融合层，限制了 Prompt 微调的效果。

## 方法详解

### 整体框架

iDPA 建立在冻结的预训练视觉-语言目标检测（VLOD）模型（如 GLIP）之上，包含两大核心模块：

- **IPG（Instance-level Prompt Generation）**：从训练图像中解耦实例级特征，生成聚焦于密集预测的 Prompt
- **DPA（Decoupled Prompt Attention）**：解耦原始 Prompt 注意力，在跨模态融合编码器中实现更高效的知识注入

整体工作流：对每个增量任务 $\mathcal{T}_i$，先用 IPG 从训练数据中提取实例级表示并生成 Prompt，然后通过 DPA 将该 Prompt 注入冻结模型的融合编码器中。训练结束后，生成的 Prompt 存入 Prompt Pool，推理时根据查询-键匹配选择对应 Prompt。

### 关键设计

#### 1. 实例级 Prompt 生成（IPG）

**Step 1: 解耦实例特征**

对每个任务 $\mathcal{T}_i$ 的每个类别 $c$，使用冻结模型提取图像特征，通过 RoI Pooling 和放大的边界框（缩放因子 $\gamma = 1.3^2$）提取 $M$ 个实例级特征表示：

$$v_c^{(j)} = \text{RoIPool}(\Phi(\text{Img}, \text{Text}), \gamma b)$$

其中 $M=1000$（全数据设定）或 $M=m$（few-shot 设定，$m$ 为每类可用样本数）。放大边界框是为了捕获额外的上下文信息。

**Step 2: 持续概念感知与知识集成（CCPKI）**

利用交叉注意力机制，从 $M$ 个实例表示中解耦出 $l$ 个概念（$l$ 为 Prompt 长度），生成任务特定 Prompt：

$$\dot{p_i} = \text{softmax}\left(\frac{p_i (\mathcal{W}_k \mathcal{I}_i)}{\sqrt{d}}\right)(\mathcal{W}_v \mathcal{I}_i)$$

$$\ddot{p_i} = p_i + \alpha \cdot \sigma(\tau \cdot \dot{p_i})$$

关键设计要素：
- **Query-Answer 框架**：初始 Prompt $p_i \in \mathbb{R}^{l \times d}$ 作为查询，实例特征 $\mathcal{I}_i$ 通过 $\mathcal{W}_k, \mathcal{W}_v$ 投影为键值对
- **可学习缩放因子** $\tau \in \mathbb{R}^{l \times 1}$：动态调节不同概念的权重，适应不同任务可能涉及的不同概念
- **非线性激活** $\sigma(\cdot)$（tanh）：过滤和增强有意义的概念成分
- **残差连接**：将激活后的概念通过 $\alpha \in \mathbb{R}^{1 \times d}$ 缩放后加到初始 Prompt 上
- **跨任务初始化**：第 $i$ 个任务的 CCPKI 参数从第 $i-1$ 个任务继承，实现知识传递

#### 2. 解耦 Prompt 注意力（DPA）

**动机推导**：作者从数学上分析了传统 Prompt Attention（PA）在多模态融合中的行为。将 Prompt $p_v, p_t$ 拼接到视觉/文本特征后，注意力输出可分解为四项：

- $\text{Attn}_{v \to t}(f_v, f_t)$：原始视觉-文本交互（**保留**）
- $\text{Attn}_{v \to t}(p_v, f_t)$：Prompt-to-文本知识注入（**保留**）
- $\text{Attn}_{v \to t}(f_v, p_t)$：对 Prompt token 的注意力输出（**丢弃**，会干扰文本特征）
- $\text{Attn}_{v \to t}(p_v, p_t)$：Prompt 间交互（**丢弃**，冗余且增加计算）

**DPA 的最终形式**：

$$\tilde{f_t} = f_t + \text{Attn}_{v \to t}(f_v, f_t) + \text{Attn}_{v \to t}(p_v, f_t)$$

$$\tilde{f_v} = f_v + \text{Attn}_{t \to v}(f_t, f_v) + \text{Attn}_{t \to v}(p_t, f_v)$$

即保留三个关键组件：
1. **V↔T**：原始视觉-语言相互增强
2. **$P_t \to V$**：文本 Prompt 向视觉的知识注入
3. **$P_v \to T$**：视觉 Prompt 向文本的知识注入

**DPA 的优势**：
- 分离 Prompt 与 Token 表示，加速知识注入
- 避免 Prompt 信息被长序列 Token 稀释
- 保留原始类别分布，缓解灾难性遗忘
- 降低计算复杂度和显存占用

#### 3. 融合编码器级知识注入

创新性地将知识注入位置从 backbone 级别扩展到跨模态融合编码器，因为检测所需的细粒度推理主要发生在融合阶段，这使得 Prompt 信息能更直接地影响定位和分类决策。

### 损失函数 / 训练策略

- **基础检测损失**：使用 VLOD 模型（如 GLIP）的标准检测损失（分类损失 + 定位损失）
- **冻结策略**：冻结预训练 VLOD 模型所有参数，仅训练 IPG 和 DPA 模块（共约 1.4% 参数）
- **增量训练**：每个新任务训练新的 Prompt，CCPKI 参数从上一个任务继承初始化
- **Prompt Pool 管理**：训练完成后将生成的 Prompt 存入 Pool，推理时通过余弦相似度匹配选取
- **无需样本回放**：无需存储旧任务样本（exemplar-free），降低隐私和存储开销

## 实验关键数据

### 数据集：ODinM-13

作者收集了 13 个临床、跨模态、多器官、多类别的医学数据集构成 ODinM-13 基准，涵盖：
- 多种成像模态：CT、MRI、X-ray、PET、皮肤镜等
- 多种器官和疾病：糖尿病足溃疡（DFUC）、胃肠道病变（Kvasir）、视神经（OpticN）、血细胞（BCCD）、细胞分裂（CPM-17）、乳腺癌（BreastC）、结核（TBX11K）、肾脏肿瘤（KidneyT）、肺结节（Luna16）、阿尔茨海默（ADNI）、脑膜瘤（Meneng）、乳腺肿瘤（BreastT）、甲状腺结节（TN3k）

### 主实验

| 方法 | FAP (%) ↑ | CAP (%) ↑ | FFP (%) ↓ | 类型 |
|------|-----------|-----------|-----------|------|
| Zero-shot | 3.12 | - | - | 基线 |
| Joint (Upper) | 54.67 | - | - | 上界 |
| Sequential | 4.40 | 15.87 | 57.81 | 非Prompt |
| ER | 39.91 | 48.73 | 19.25 | 非Prompt |
| ZiRa | 3.66 | 16.37 | 49.67 | 非Prompt |
| L2P | 39.88 | 46.04 | 8.24 | Prompt |
| DualPrompt | 28.89 | 42.24 | 20.57 | Prompt |
| S-Prompt | 41.02 | 46.70 | 8.87 | Prompt |
| CODA | 42.08 | 49.78 | 2.80 | Prompt |
| NoRGa | 44.84 | 49.90 | 4.92 | Prompt |
| **iDPA (Ours)** | **50.28** | **54.10** | **2.48** | Prompt |

**全数据设定下 iDPA 的 FAP 达到 50.28%，超越最佳对比方法 NoRGa 5.44 个百分点**，同时遗忘率 FFP 最低仅 2.48%。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|----------|------|
| 去掉 IPG，用随机初始化 Prompt | FAP 显著下降 | 实例级特征解耦对目标检测至关重要 |
| 去掉 DPA，用传统 PA | FAP 下降，FFP 上升 | 耦合注意力稀释 Prompt 信息 |
| 仅在 backbone 注入 Prompt | 性能弱于融合层注入 | 融合编码器是更优的注入位置 |
| 去掉 CCPKI 跨任务初始化 | 新任务学习效率下降 | 跨任务知识传递有效 |
| 去掉缩放因子 τ | 概念调节能力下降 | 动态概念权重对多任务适应重要 |

### 关键发现

1. **CPM-17 数据集的显著突破**：iDPA 在细胞分裂检测（CPM-17）上达到 36.54% AP，远超所有 Prompt-based 方法（最高仅 8.37%），说明实例级特征解耦在细粒度检测中优势显著
2. **Few-shot 场景优势更大**：在 10-shot 设定下 FAP 提升 12.88%，说明实例级知识在数据稀缺时价值更高
3. **极低遗忘率**：FFP 仅 2.48%，得益于 DPA 保留了原始类别分布
4. **仅 1.4% 参数**：高效的参数利用率使得方法可扩展到更多任务
5. **跨模态泛化**：在 CT、MRI、X-ray、PET、皮肤镜等多种模态上均表现稳定

## 亮点与洞察

1. **理论驱动的架构设计**：通过数学推导分析 Prompt Attention 等价形式，发现四项注意力中仅两项对检测有用，由此设计 DPA——这种"先分析后设计"的范式值得借鉴
2. **实例级 vs. 全局级 Prompt**：首次指出全局 Prompt 在目标检测任务中的根本缺陷，提出从实例特征中通过概念解耦生成 Prompt 的方案
3. **知识注入位置的洞察**：发现融合编码器比 backbone 更适合注入检测知识，因为融合阶段才是细粒度推理发生的位置
4. **ODinM-13 基准**：构建了首个大规模增量医学目标检测基准，覆盖 13 个跨模态数据集，填补了该领域评估标准的空白
5. **Exemplar-Free**：无需存储旧任务样本，避免了医学数据的隐私问题

## 局限与展望

1. **Prompt Pool 线性增长**：每个新任务增加一组 Prompt，随任务数增长 Pool 线性膨胀，可探索 Prompt 压缩或共享机制
2. **依赖预训练 VLOD 质量**：框架建立在 GLIP 等自然域预训练模型之上，其在医学域的初始表示质量直接影响上限
3. **RoI Pooling 需要标注框**：IPG 模块依赖训练集的边界框标注来提取实例特征，对标注质量敏感
4. **缩放因子 γ 固定**：$\gamma = 1.3^2$ 为固定值，不同数据集/器官可能需要不同的上下文范围
5. **仅评估类增量学习**：未探索域增量（如从 CT 到 MRI）或任务增量场景
6. **可扩展到 3D 检测**：当前仅处理 2D 切片，未涉及 3D 体积检测

## 相关工作与启发

- **GLIP / Grounding DINO**：VLOD 基础模型，提供视觉-语言融合的目标检测能力
- **L2P / DualPrompt / CODA-Prompt**：Prompt-based 持续学习的代表方法，但均为分类设计
- **ZiRa**：首个将预训练 VLOD 模型适配到持续目标检测的工作，但性能有限
- **MQ-Det**：多模态查询编码器的设计启发了 IPG 模块中的实例特征提取
- **Eclipse**：高效持续全景分割方法，同样使用 visual prompt tuning 避免重训练
- 启发方向：可将 iDPA 的解耦 Prompt 思想推广到持续分割、持续姿态估计等密集预测任务

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|------------|------|
| 新颖性 | 8 | IPG + DPA 的组合具有理论创新，DPA 的数学推导优雅 |
| 实用性 | 8 | 医学增量检测是真实临床需求，1.4% 参数高效可部署 |
| 实验充分度 | 9 | 13 个跨模态数据集 + 全数据/few-shot + 消融实验 |
| 写作质量 | 8 | 结构清晰，数学推导严谨，图表丰富 |
| 总分 | 8.5 | 高质量工作，解决了一个重要且被忽视的问题 |

<!-- RELATED:START -->

## 相关论文

- [Unleashing the Power of Prompt-driven Nucleus Instance Segmentation](../../ECCV2024/medical_imaging/unleashing_the_power_of_prompt-driven_nucleus_instance_segmentation.md)
- [Mitigating Object Hallucination in LVLMs via Attention Imbalance Rectification](../../CVPR2026/medical_imaging/mitigating_object_hallucinations_in_lvlms_via_attention_imbalance_rectification.md)
- [The Four Color Theorem for Cell Instance Segmentation](the_four_color_theorem_for_cell_instance_segmentation.md)
- [Enhancing Medical Dialogue Generation through Knowledge Refinement and Dynamic Prompt Adjustment](../../ACL2025/medical_imaging/enhancing_medical_dialogue_generation_through_knowledge_refinement_and_dynamic_p.md)
- [Pathology-knowledge Enhanced Multi-instance Prompt Learning for Few-shot Whole Slide Image Classification](../../ECCV2024/medical_imaging/pathologyknowledge_enhanced_multiinstance_prompt_learni.md)

<!-- RELATED:END -->
