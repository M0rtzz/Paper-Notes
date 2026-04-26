---
title: >-
  [论文解读] ExCap3D: Expressive 3D Scene Understanding via Object Captioning with Varying Detail
description: >-
  [ICCV 2025][3D视觉][3D场景理解] 提出 ExCap3D，一个在 3D 室内场景中对物体生成多粒度描述的方法，包含物体级和部件级两个描述层次，通过部件→物体的信息共享和语义/文本一致性损失确保描述的准确性与一致性，在新建的 190K 描述数据集上 CIDEr 评分比 SOTA 分别提升 17% 和 124%。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D场景理解
  - 稠密标注
  - 多粒度描述
  - 物体-部件联合生成
  - 3D高斯
---

# ExCap3D: Expressive 3D Scene Understanding via Object Captioning with Varying Detail

**会议**: ICCV 2025  
**arXiv**: [2503.17044](https://arxiv.org/abs/2503.17044)  
**代码**: 即将公开  
**领域**: 3d_vision  
**关键词**: 3D场景理解, 稠密标注, 多粒度描述, 物体-部件联合生成, 3D高斯

## 一句话总结

提出 ExCap3D，一个在 3D 室内场景中对物体生成多粒度描述的方法，包含物体级和部件级两个描述层次，通过部件→物体的信息共享和语义/文本一致性损失确保描述的准确性与一致性，在新建的 190K 描述数据集上 CIDEr 评分比 SOTA 分别提升 17% 和 124%。

## 研究背景与动机

3D 室内场景理解是 AR/VR/机器人应用的基础。自然语言描述能编码复杂信息并支持更自然的人-场景交互。但现有 3D 标注方法存在关键局限：

**单一粒度描述**：Scan2Cap、ScanQA 等仅在单一详细程度描述物体，侧重物体间空间关系

**缺乏部件级细节**：无法描述物体各部分的外观、材质、功能等高度局部化属性

**不同应用需要不同粒度**：机器人导航仅需"躺椅"，而辅助 AI 需要"带高靠垫、木质扶手、软座垫和可调脚凳的躺椅"

**核心贡献**：提出表达性 3D 标注（Expressive 3D Captioning）任务——对每个检测到的物体生成物体级描述（语义类别、外观）和部件级描述（各部分的材质、颜色、功能）。

## 方法详解

### 整体框架（Fig. 3）

1. **3D 实例分割**：使用 Mask3D 检测场景中物体
2. **联合描述生成**：两个独立标注头生成物体级和部件级描述
3. **一致性约束**：语义和文本一致性损失确保两级描述的一致性

### 3D 实例分割

输入 3D 场景网格 $M=(\mathcal{V}, \mathcal{F})$，体素化后送入 Mask3D：
- 3D 稀疏卷积 UNet 编码器提取稠密特征 $F \in \mathbb{R}^{N_{vox} \times D}$
- 并行掩码模块通过 transformer 编码器层迭代精炼查询向量 $Q \in \mathbb{R}^{N_q \times D_q}$
- 输出：精炼的实例感知查询 $Q_r$ 和实例掩码 $I \in \{0,1\}^{N_q \times N_{vox}}$

### 联合标注与信息共享

两个 transformer 语言模型 $\Psi_{obj}$ 和 $\Psi_{part}$ 进行自回归 token 预测，条件化于两个信息源：

**1. 标注感知查询**：将精炼查询线性投影为标注初始化 token：

$$Q_{c,o} = \Phi_{query}(Q_{r,o})$$

**2. 分段级上下文特征**：通过交叉注意力层关注对象对应的 3D 特征。为减少计算复杂度，在网格面上预计算的分段（segments）内聚合特征，得到 $S_o \in \mathbb{R}^{n_{s,o} \times D_{caption}}$。

**部件→物体信息共享**（关键设计）：先生成部件级描述，将 $\Psi_{part}$ 最后一层隐状态经线性投影后拼接到物体标注器的上下文特征中：

$$H_{part,o} = \Phi_{hidden}([h_{part,1} \ldots h_{part,i}])$$

物体级标注在 $Q_{c,o}$ 和 $[H_{part,o}; S_{obj,o}]$ 条件下生成。

### 语义和文本一致性损失

**语义一致性**：将两级隐状态投影到低维后分类为 $N_{sem}$ 个细粒度类别，用对称交叉熵约束：

$$\mathcal{L}_{semantic} = CE(sem_{obj}, SG(sem_{part})) + CE(sem_{part}, SG(sem_{obj}))$$

其中 $SG$ 为 stop-gradient 算子。

**文本一致性**：将隐状态序列聚合为单向量，最小化两级描述间的距离：

$$\mathcal{L}_{textual} = d(\bar{h}_{obj,text}, \bar{h}_{part,text})$$

### 总损失

$$\mathcal{L} = w_1 \mathcal{L}_{caption} + w_2 \mathcal{L}_{semantic} + w_3 \mathcal{L}_{textual}$$

其中 $w_1=1, w_2=w_3=0.1$。

## ExCap3D 数据集

### 构建方法

基于 ScanNet++ 的 947 个场景，34K 物体，使用自动化管线：

1. **物体级**：将 3D GT 实例标注投影到 DSLR 图像→裁剪物体区域→VLM（LLaVA 1.6-7B）生成多视角描述→LLM（Llama 3.1-8B）聚合
2. **部件级**：MaskClustering + SAM 生成部件伪掩码→VLM 描述各部件→LLM 聚合为单一部件级描述

### 数据集统计（Table 1）

| 数据集 | 描述数 | 类别数 | 物体数 | 粒度 |
|--------|--------|--------|--------|------|
| Scan2Cap | 46k | 265 | 9.9k | 场景+物体 |
| ScanQA | 35k | 370 | 9.5k | 场景+物体 |
| **ExCap3D** | **190k** | **2k** | **34.7k** | **物体+部件** |

描述数量是最大现有数据集的 4 倍以上，类别覆盖达 2000 个。

## 实验关键数据

### 主实验：与 SOTA 的比较（Table 2）

**物体级描述**：

| 方法 | CIDEr↑ | ROUGE↑ | METEOR↑ |
|------|--------|--------|---------|
| D3Net | 6.7 | 5.4 | 6.7 |
| Vote2Cap-DETR | 13.3 | 12.9 | 17.2 |
| PQ3D | 27.9 | 11.6 | 12.5 |
| **ExCap3D** | **32.7** | **16.6** | **17.9** |

**部件级描述**：

| 方法 | CIDEr↑ | ROUGE↑ | METEOR↑ |
|------|--------|--------|---------|
| D3Net | 10.5 | 7.9 | 7.9 |
| Vote2Cap-DETR | 13.3 | 20.7 | 22.7 |
| PQ3D | 14.4 | 16.3 | 15.6 |
| **ExCap3D** | **32.3** | **21.7** | **20.8** |

CIDEr 提升：物体级 +17%（vs PQ3D），部件级 +124%（vs PQ3D）。部件级的巨大提升表明现有方法根本无法处理细粒度描述。

### 消融实验（Table 3）

| 方法 | 物体级 CIDEr | 部件级 CIDEr |
|------|-------------|-------------|
| 独立模型（baseline） | 29.8 | 18.7 |
| + 语义一致性 | 30.2 | **24.8** |
| + 文本一致性 | **32.2** | 19.6 |
| + 部件→物体信息共享 | **34.8** | 25.4 |
| **完整模型** | 32.7 | **32.3** |

关键发现：
- 语义一致性主要改善部件级（18.7→24.8），文本一致性主要改善物体级（29.8→32.2）——互补效果
- 部件→物体信息共享对两级均有提升，特别是物体级（29.8→34.8）
- 所有组件组合后部件级提升最为显著（18.7→32.3，+73%）

### 信息共享方向对比（Table 4）

| 方向 | 物体级 CIDEr | 部件级 CIDEr |
|------|-------------|-------------|
| 物体→部件 | 32.8 | 15.6 |
| **部件→物体** | **32.7** | **32.3** |

"物体作为部件之和"的建模方式远优于"部件作为物体组成"——物体级描述不含细粒度部件信息，无法有效指导部件描述生成。

### 上下文特征消融（Table 5）

| 方法 | 物体级 CIDEr | 部件级 CIDEr |
|------|-------------|-------------|
| 无上下文特征 | 33.7 | 27.0 |
| 有上下文特征 | 32.7 | **32.3** |

分段级上下文对部件级至关重要（27.0→32.3），因为描述低级部件细节需要更精细的特征。

## 亮点与洞察

1. **"物体是部件之和"的建模哲学**：先描述部件再综合为物体描述的信息流方向优势显著，与人类认知中"自底向上"感知物体的模式一致
2. **一致性损失的互补性**：语义一致性保证两级描述指向同一语义对象，文本一致性保证描述内容的重叠——分别从不同维度约束一致性
3. **VLM 管线的可扩展性**：利用 VLM + LLM 自动生成 190K 高质量标注，避免人工标注瓶颈，方法可推广到其他 3D 数据集
4. **端到端学习优于分离管线**（Table 6）：即使使用相同 VLM，端到端训练的标注质量远超"检测+VLM 描述"的两阶段方案

## 局限性

1. 使用两个独立标注头通过交叉注意力共享信息，可能限制了信息传递的充分性
2. 稀疏卷积骨干网络的体素分辨率约 2cm，对小型或细薄物体的标注能力受限
3. 部件伪掩码来自 MaskClustering + SAM，质量不如人工标注，可能引入噪声
4. 数据集基于 ScanNet++，泛化到其他 3D 扫描格式需要验证

## 相关工作与启发

- 与 Cap3D（Luo et al., 2023）描述孤立 3D 物体不同，ExCap3D 在完整 3D 场景中联合检测和多粒度描述
- 多粒度描述思路可与 3D 视觉-语言对齐（如 3D-VISTA）结合，用于更精细的具身理解
- 部件→物体的信息共享范式可推广到其他层次化生成任务（如场景图生成）
- ExCap3D 数据集可作为基础训练数据支持 3D 具身 AI 的细粒度指令跟随

## 评分 ⭐⭐⭐⭐

创新性 ★★★★☆：多粒度标注任务的定义和信息共享机制有新意
实验 ★★★★☆：消融实验充分，各组件贡献清晰可见
写作 ★★★★☆：图表清晰，方法描述系统
实用性 ★★★★☆：190K 数据集价值高，但依赖 ScanNet++ 的高分辨率 DSLR 图像

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Open-Vocabulary Octree-Graph for 3D Scene Understanding](open-vocabulary_octree-graph_for_3d_scene_understanding.md)
- [\[ICCV 2025\] Articulate3D: Holistic Understanding of 3D Scenes as Universal Scene Description](articulate3d_holistic_understanding_of_3d_scenes_as_universa.md)
- [\[ICCV 2025\] 3DGraphLLM: Combining Semantic Graphs and Large Language Models for 3D Scene Understanding](3dgraphllm_combining_semantic_graphs_and_large_language_models_for_3d_scene_unde.md)
- [\[ICCV 2025\] HIS-GPT: Towards 3D Human-In-Scene Multimodal Understanding](his-gpt_towards_3d_human-in-scene_multimodal_understanding.md)
- [\[ICCV 2025\] VertexRegen: Mesh Generation with Continuous Level of Detail](vertexregen_mesh_generation_with_continuous_level_of_detail.md)

<!-- RELATED:END -->
