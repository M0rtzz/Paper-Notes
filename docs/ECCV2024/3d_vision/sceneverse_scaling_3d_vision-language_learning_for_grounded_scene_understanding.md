---
title: >-
  [论文解读] SceneVerse: Scaling 3D Vision-Language Learning for Grounded Scene Understanding
description: >-
  [ECCV 2024][3D视觉][视觉语言] 提出首个百万级 3D 视觉-语言数据集 SceneVerse（68K 室内场景 + 2.5M 场景-语言对），结合多层级对比预训练框架 GPS，在 3D visual grounding 和 QA 任务上取得 SOTA，并展现零样本迁移能力。
tags:
  - ECCV 2024
  - 3D视觉
  - 视觉语言
  - data scaling
  - 场景理解
  - 对比学习
  - pre-training
---

# SceneVerse: Scaling 3D Vision-Language Learning for Grounded Scene Understanding

**会议**: ECCV 2024  
**arXiv**: [2401.09340](https://arxiv.org/abs/2401.09340)  
**代码**: https://scene-verse.github.io  
**领域**: 3D视觉 / 视觉-语言  
**关键词**: 3D vision-language, data scaling, grounded scene understanding, contrastive learning, pre-training

## 一句话总结
提出首个百万级 3D 视觉-语言数据集 SceneVerse（68K 室内场景 + 2.5M 场景-语言对），结合多层级对比预训练框架 GPS，在 3D visual grounding 和 QA 任务上取得 SOTA，并展现零样本迁移能力。

## 研究背景与动机
**领域现状**：2D 视觉-语言领域已通过大规模数据（如 CLIP 的亿级图文对）取得巨大成功，但 3D 场景的视觉-语言对齐仍处于初级阶段。

**现有痛点**：3D 数据采集依赖扫描设备，成本极高；现有 3D-VL 数据集规模仅数千场景，远远落后于 2D 数据集；3D 场景中对象配置复杂、属性丰富、关系多样，使得语言描述需求量巨大。

**核心矛盾**：数据规模不足 → 无法支撑有效的预训练对齐 → 现有模型高度依赖任务特定设计（复杂损失函数或模型结构），泛化能力差。

**本文目标**：如何系统性地扩展（scale up）3D 视觉-语言数据，并设计统一的预训练框架来利用这些数据。

**切入角度**：统一多个已有 3D 场景数据集 + 利用 3D 场景图和 LLM 自动生成大规模语言描述 + 多层级对比学习。

**核心 idea**：数据规模是 3D-VL 的瓶颈，通过场景图 + LLM 实现百万级数据扩展，配合多层级对比预训练即可取得 SOTA。

## 方法详解

### 整体框架
SceneVerse 由两部分组成：(1) 数据集构建——整合 7 个来源的 68K 3D 场景，通过人工标注和自动生成管线收集 2.5M 场景-语言对；(2) GPS（Grounded Pre-training for Scenes）——基于 Transformer 的预训练模型，通过物体级、场景级和引用-物体级三层对比对齐学习 3D 场景与文本的匹配。

### 关键设计

#### 1. **场景整合与标注 (Scene Curation & Annotation)**
   - 功能：统一来自 ScanNet、ARKitScenes、HM3D、3RScan、MultiScan 等真实场景和 Structured3D、ProcTHOR 等合成场景的数据
   - 核心思路：对每个场景进行房间分割、点云子采样、轴对齐和归一化。每个扫描表示为 $\mathrm{P} \in \mathbb{R}^{N \times 8}$（3D 坐标 + RGB + instance id + 语义标签）。共收集 68,406 个场景。
   - 人工标注 96,863 条 referring expression（AMT 标注 + 双人验证），重标注率仅 4.8%
   - 设计动机：充分利用已有数据源，避免重复采集

#### 2. **基于场景图的语言生成管线 (3D Scene Graph + LLM Generation)**
   - 功能：自动生成三种粒度的语言描述——物体描述（object caption）、物体引用（object referral）、场景描述（scene caption）
   - 核心思路：
     - 构建层次化场景图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$，每个节点 $v$ 由质心 $\boldsymbol{p}_i \in \mathbb{R}^3$ 和边界框大小 $\boldsymbol{b}_i \in \mathbb{R}^3$ 参数化，边 $\mathcal{E}$ 表示空间关系（垂直/水平邻近、多物体关系）
     - Object Caption：通过点云渲染定位对象在多视角图像中的出现 → BLIP2 生成初始描述 → CLIP 筛选 top-10 → LLM 精炼总结
     - Object Referral：从场景图提取空间关系三元组 $(v_i, v_j, e_{ij})$ → 模板生成 (target-object, spatial-relation, anchor-objects) → LLM 重述增加自然度
     - Scene Caption：随机采样场景图子集 + 物体计数 + 房间类型 → 提示 LLM 生成全局描述
   - 设计动机：模板保证覆盖度，LLM 重述增加多样性和自然度；人工验证 96.93% 通过率（高于 ReferIt3D 的 86.1%）

#### 3. **GPS：多层级对比预训练 (Grounded Pre-training for Scenes)**
   - 功能：在三个粒度上同时对齐 3D 场景和文本
   - **物体级对齐 $\mathcal{L}_{\text{obj}}$**：
     - 点云编码器提取物体特征 $\boldsymbol{f}^O_i$，冻结语言模型编码物体描述得到 $\boldsymbol{f}^T_i$
     - 双向对比损失：$\mathcal{L}_{\text{obj}} = -\frac{1}{2}\sum_{(p,q)} \left(\log\frac{\exp(D^{\text{obj}}(p,q))}{\sum_r \exp(D^{\text{obj}}(p,r))} + \log\frac{\exp(D^{\text{obj}}(p,q))}{\sum_r \exp(D^{\text{obj}}(r,q))}\right)$
     - 其中 $D^{\text{obj}}(p,q) = \boldsymbol{f}^O_p \boldsymbol{f}^T_q / \tau$，$\tau$ 为可学习温度参数
   - **场景级对齐 $\mathcal{L}_{\text{scene}}$**：
     - 空间 Transformer 编码物体特征 + 位置特征得到 $\boldsymbol{f}^S_i = \text{SpatialAttn}(\{\boldsymbol{f}^O_i\}, \{\boldsymbol{l}_i\})$
     - 投影 + max-pooling 得到场景特征 $\boldsymbol{g}^S$，与场景描述特征 $\boldsymbol{g}^T$ 做 inter-scene 对比
   - **引用-物体级对齐 $\mathcal{L}_{\text{ref}}$**：
     - 自注意力推理 Transformer 接收场景-物体特征和引用文本
     - **intra-scene 对比**：$\mathcal{L}_{\text{ref}} = -\log\frac{\exp(\bar{\boldsymbol{h}}^S \boldsymbol{h}^T / \tau)}{\sum_p \exp(\boldsymbol{h}^S_p \boldsymbol{h}^T / \tau)}$，正对在场景内选取，p 遍历同一场景的所有物体
     - 设计动机：模仿 2D-VL 中 intra-image 和 inter-image 对比的成功经验
   - 额外使用 MLM 损失 $\mathcal{L}_{\text{MLM}}$ 微调语言编码器
   - 总损失：$\mathcal{L} = \mathcal{L}_{\text{obj}} + \mathcal{L}_{\text{scene}} + \mathcal{L}_{\text{ref}} + \mathcal{L}_{\text{MLM}}$

### 训练策略
- 两阶段训练：先用物体级对齐训练点云编码器获得好的特征初始化，再联合训练场景级和引用级目标
- 无需复杂辅助损失或任务特定架构

## 实验关键数据

### 主实验：3D Visual Grounding

| 方法 | Nr3D Overall | Sr3D Overall | ScanRefer Acc@0.5 |
|------|-------------|-------------|-------------------|
| 3DVG-Trans | 40.8 | 51.4 | 34.7 |
| BUTD-DETR | 54.6 | 67.0 | 39.8 |
| ViL3DRel | 64.4 | 72.8 | 37.7 |
| 3D-VisTA (pre-train) | 64.2 | 76.4 | 45.8 |
| **GPS (scratch)** | 58.7 | 68.4 | 40.4 |
| **GPS (pre-train)** | 55.2 | 74.1 | **47.1** |
| **GPS (fine-tuned)** | **64.9** | **77.5** | **48.1** |

GPS 预训练后直接在 ScanRefer 上已超过所有方法（47.1），fine-tune 后进一步提升。

### 零样本迁移

| 方法 | Nr3D | Sr3D | ScanRefer@0.5 |
|------|------|------|---------------|
| 3D-VisTA (zero-shot) | 35.2 | 31.2 | 29.6 |
| 3D-VisTA (zero-shot text) | 43.1 | 36.1 | 36.4 |
| **GPS (zero-shot)** | 32.4 | 33.3 | **31.1** |
| **GPS (zero-shot text)** | 41.9 | **38.1** | 35.8 |

在 SceneVerse-val 上零样本迁移：GPS 达 59.2%（vs 3D-VisTA 52.9%），SceneVerse 数据大幅增强泛化能力。

### 3D QA

| 模型 | ScanQA val | SQA3D |
|------|-----------|-------|
| ScanQA | 20.3 | 46.6 |
| 3D-VisTA | 22.4 | 48.5 |
| **GPS** | **22.7** | **49.9** |

### 关键发现
- 从 scratch 训练时 GPS 不如使用复杂设计的模型，但数据扩展后仅用简单对比对齐即可大幅超越 → 证明数据规模才是核心瓶颈
- SceneVerse 的数据扩展效应不限于 GPS，对 RegionPLC 等其他模型在语义分割上也有显著提升
- 自动生成的语言数据质量高（96.93% 通过率），且 LLM 重述后的多样性接近人工标注

## 亮点与洞察
- **数据扩展为王**：首次在 3D-VL 领域验证了 2D 领域"数据规模决定性能"的规律，68K 场景 + 2.5M 语言对的规模相比之前最大数据集（~3K 场景）提升一个数量级
- **场景图 + LLM 管线设计精巧**：模板保证空间关系覆盖的完整性，LLM 重述保证自然度，两者互补实现高质量自动标注
- **多层级对比设计简洁有效**：物体级（inter-scene 对比）、场景级（inter-scene 对比）、引用级（intra-scene 对比）三层对齐无需辅助损失，优雅且高效

## 局限与展望
- 仅限室内场景，未涵盖户外3D环境
- 合成场景（Structured3D、ProcTHOR）与真实场景仍有域差异
- 描述主要基于静态空间关系，缺乏动态/交互性信息
- 语言生成质量依赖 LLM 能力，长尾场景可能覆盖不足

## 相关工作与启发
- **vs 3D-VisTA**：同为预训练方法，但 3D-VisTA 依赖分类目标（softmax），GPS 使用对比对齐，在零样本场景下 GPS 泛化更好
- **vs ScanScribe**：ScanScribe 仅 278K 语言对，SceneVerse 达 2.5M，规模提升近 10 倍
- **vs CLIP (2D)**：借鉴了 CLIP 的对比学习范式和可学习温度参数设计，将其理念成功迁移到 3D 场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个百万级3D-VL数据集 + 场景图-LLM生成管线
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖grounding/QA/零样本/语义分割多个任务，消融详尽
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，从数据到方法到实验一气呵成
- 价值: ⭐⭐⭐⭐⭐ 为3D-VL领域奠定数据基础，类似ImageNet/CLIP对各自领域的贡献

<!-- RELATED:START -->

## 相关论文

- [WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [Zero-Shot Multi-Object Scene Completion](zero-shot_multi-object_scene_completion.md)
- [SceneGraphLoc: Cross-Modal Coarse Visual Localization on 3D Scene Graphs](scenegraphloc_cross-modal_coarse_visual_localization_on_3d_scene_graphs.md)
- [Learning to Generate Conditional Tri-Plane for 3D-Aware Expression Controllable Portrait Animation](learning_to_generate_conditional_tri-plane_for_3d-aware_expression_controllable_.md)
- [Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)

<!-- RELATED:END -->
