---
title: >-
  [论文解读] FedBPrompt: Federated Domain Generalization Person Re-Identification via Body Distribution Aware Visual Prompts
description: >-
  [CVPR 2026][自动驾驶][FedDG-ReID] 提出FedBPrompt框架，通过身体分布感知视觉提示机制(BAPM)将prompt分为Body Part Alignment Prompts和Holistic Full Body Prompts两组，配合Prompt-based Fine-Tuning Strategy(PFTS)冻结ViT backbone仅训练轻量prompt（通信量降至~1%），在FedDG-ReID任务上平均mAP提升3.3%、Rank-1提升4.9%。
tags:
  - CVPR 2026
  - 自动驾驶
  - FedDG-ReID
  - 视觉提示
  - 身体部位对齐
  - 参数高效微调
  - ViT
  - 联邦聚合
---

# FedBPrompt: Federated Domain Generalization Person Re-Identification via Body Distribution Aware Visual Prompts

**会议**: CVPR 2026  
**arXiv**: [2603.12912](https://arxiv.org/abs/2603.12912)  
**代码**: [leavlong/FedBPrompt](https://github.com/leavlong/FedBPrompt)  
**领域**: 行人重识别 / 联邦学习  
**关键词**: FedDG-ReID, 视觉提示, 身体部位对齐, 参数高效微调, ViT, 联邦聚合

## 一句话总结

提出FedBPrompt框架，通过身体分布感知视觉提示机制(BAPM)将prompt分为Body Part Alignment Prompts和Holistic Full Body Prompts两组，配合Prompt-based Fine-Tuning Strategy(PFTS)冻结ViT backbone仅训练轻量prompt（通信量降至~1%），在FedDG-ReID任务上平均mAP提升3.3%、Rank-1提升4.9%。

## 研究背景与动机

联邦域泛化行人重识别(FedDG-ReID)要求在隐私保护的联邦学习框架下，从多个去中心化摄像头域学习域不变表征，使模型能泛化到未见目标域。ViT凭借强表征能力成为主流backbone，但其**全局注意力机制**在FedDG-ReID中暴露两个核心缺陷：

**背景干扰失焦**：ViT的全局自注意力无差别地处理所有patch token，包括行人区域和背景区域。当不同客户端的背景分布差异大时（室内/室外、商场/街道），模型容易将注意力分散到高相似度的背景上，造成不同身份的行人因背景相似被错误匹配

**视角变化导致身体部位错位**：不同客户端的摄像头角度差异（俯视/平视、正面/侧面）使同一行人的身体部位在图像中的空间位置大幅变化。ViT的全局注意力无法感知这种空间结构差异，导致同一行人的跨视角特征相似度急剧下降

这两个问题在联邦场景下被**客户端间数据分布异质性**进一步放大——每个客户端只能看到自己场景的数据，无法通过集中训练来学习跨域不变性。

现有FedDG-ReID方法（如DACS的数据增强、FedReID的特征对齐）主要在数据层面增加多样性，但未直接从模型注意力机制层面解决背景失焦和部位错位问题。

## 方法详解

### 整体框架

FedBPrompt构建在ViT-B/16基础上，包含两个核心组件：

1. **BAPM (Body Distribution Aware Visual Prompts Mechanism)**：结构化视觉提示，引导注意力聚焦行人并对齐身体部位
2. **PFTS (Prompt-based Fine-Tuning Strategy)**：冻结backbone仅训练prompt参数，大幅降低联邦通信开销

### 模块1：BAPM — 身体分布感知视觉提示

**核心思路**：将一组可学习的prompt token $\mathbf{P}$（共50个，维度$d$）嵌入ViT的每一层，并将其划分为**两组四个子集**，分别承担不同功能：

**第一组：Body Part Alignment Prompts（15个，解决视角错位）**

- $\mathbf{P}^{\text{upper}}$（5个）：仅与图像上半部分的patch token交互 → 对应头部/肩部
- $\mathbf{P}^{\text{mid}}$（5个）：仅与图像中间区域的patch token交互 → 对应躯干
- $\mathbf{P}^{\text{lower}}$（5个）：仅与图像下半部分的patch token交互 → 对应腿部/脚部

空间区域的定义采用**重叠分区**策略（非刚性分割）：假设图像被划分为$H \times W$个patch，则：

$$I_{\text{upper}} = \{j \mid 1 \leq j \leq n/2\}, \quad I_{\text{mid}} = \{j \mid n/4+1 \leq j \leq 3n/4\}, \quad I_{\text{lower}} = \{j \mid n/2+1 \leq j \leq n\}$$

三个区域之间有25%的重叠，避免刚性切割丢失跨区域信息。

**第二组：Holistic Full Body Prompts（35个，解决背景失焦）**

- $\mathbf{P}^{\text{Full}}$：可以与**所有**图像patch token交互，没有空间约束
- 功能：捕获行人整体外观特征，抑制跨客户端背景噪声

**约束注意力机制**：通过结构化注意力掩码$M$实现空间约束：

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

$$M_{ij} = \begin{cases} -\infty & \text{if } (q_i, k_j) \in \mathcal{C}_{\text{mismatch}} \\ 0 & \text{otherwise} \end{cases}$$

其中$\mathcal{C}_{\text{mismatch}}$表示不匹配对：即某个Body Part Prompt与非对应区域patch的配对。

**关键设计：prompt间自由通信**：所有prompt token之间的注意力掩码恒为0，即$M_{ij} = 0, \forall q_i, k_j \in \mathbf{P}$。这意味着Body Part Prompts和Full Body Prompts可以自由交互——部位prompt提供结构化局部信息，全局prompt将其整合为连贯的全身表征。这种设计区别于PCB等刚性分块方法，保留了全局一致性。

每个ViT层拥有独立的prompt参数$\mathbf{P}_{i-1}$，层间更新规则为：

$$[\mathbf{x}_i, \_, \mathbf{E}_i] = L_i([\mathbf{x}_{i-1}, \mathbf{P}_{i-1}, \mathbf{E}_{i-1}])$$

### 模块2：PFTS — 基于Prompt的高效微调策略

**动机**：ViT-B/16全模型参数约86M，在联邦学习中每轮通信开销巨大，不适合资源受限的部署环境。

**方案**：
1. 服务器先在集中数据上预训练一个标准ReID模型（不含prompt）
2. 将预训练模型分发给所有客户端，**冻结backbone参数$\Theta_b$**
3. 每个客户端"植入"随机初始化的BAPM prompt参数$\Theta_p$（约0.46M）
4. 客户端仅训练prompt参数，目标函数为：

$$\mathcal{L}_k(\Theta_p) = \sum_{(x,y) \in D_k} \mathcal{L}_{\text{ReID}}(g(x; \Theta_b, \Theta_p), y)$$

5. 训练完成后仅上传prompt参数到服务器，按数据量加权聚合：

$$\Theta_p^{t+1} = \sum_{k=1}^{K} \frac{|D_k|}{\sum_{j=1}^{K}|D_j|} \Theta_{p,k}^{t+1}$$

**通信效率**：每轮仅通信0.46M参数（vs 全模型86M），降至约**0.5%**。几轮聚合即可获得显著性能提升。

### 训练策略

- 支持两种模式：**Full-Parameter训练**（整个模型+BAPM）和**PFTS训练**（仅prompt）
- 损失函数采用标准ReID损失（交叉熵+triplet loss）
- BAPM可作为即插即用模块集成到任意ViT-based FedDG-ReID框架中

## 实验关键数据

### 数据集与协议

- **数据集**：CUHK02 (C2)、CUHK03 (C3)、Market1501 (M)、MSMT17 (MS)
- **Protocol-1**：Leave-One-Out，3个域训练、1个域测试
- **Protocol-2**：源域性能评估

### Protocol-1 主实验（Table 1）

以最强基线SSCU（MM 2025）为例：

| 设置 | →M mAP | →M Rank-1 | →C3 mAP | →C3 Rank-1 | →MS mAP | →MS Rank-1 | Avg mAP | Avg Rank-1 |
|------|--------|-----------|---------|------------|---------|------------|---------|------------|
| SSCU原始 | 46.3 | 69.6 | 33.7 | 33.4 | 20.0 | 43.7 | 33.3 | 48.9 |
| +PFTS | 48.9(+2.6) | 72.4(+2.8) | 35.5(+1.8) | 35.8(+2.4) | 21.3(+1.3) | 46.0(+2.3) | 35.2(+1.9) | 51.4(+2.5) |
| +BAPM | **49.1**(+2.8) | **73.4**(+3.8) | **37.4**(+3.7) | **38.4**(+5.0) | **23.4**(+3.4) | **49.5**(+5.8) | **36.6**(+3.3) | **53.8**(+4.9) |

对弱基线提升更加显著——在FedProx上，BAPM带来平均mAP +10.0%、Rank-1 +13.5%的提升。

### 消融实验（Table 3）

以SSCU为基线，在"C2+C3+M→MS"设置下：

| 配置 | Holistic | Part Align | mAP | Rank-1 |
|------|----------|------------|-----|--------|
| Baseline | — | — | 20.0 | 43.7 |
| +Holistic Only | ✓ | — | 22.9 | 48.2 |
| +Part Align Only | — | ✓ | 22.7 | 48.5 |
| +BAPM (Full) | ✓ | ✓ | **23.4** | **49.5** |

两组prompt各自有效，组合后进一步提升。Part Alignment Prompts在视角变化大的场景（→MS）上贡献更突出。

### 注意力质量量化（Table 4）

| 方法 | Class Token Ins. AUC | RISE Ins. AUC |
|------|---------------------|---------------|
| SSCU | 0.6160 | 0.6516 |
| +VPs（普通prompt） | 0.7103 | 0.7494 |
| +BAPM | **0.7559** | **0.7737** |

Insertion AUC衡量注意力图的忠实度——BAPM显著优于普通（无结构化）visual prompt，证明其注意力确实更精准地聚焦在行人区域。

### Protocol-2 源域性能

BAPM在提升跨域泛化的同时，不损害源域性能，甚至在源域测试中也有明显提升（FedPav +BAPM在C2上mAP从66.5→74.3）。

## 亮点与洞察

- **结构化prompt的精妙设计**：不同于VPT等方法将所有prompt同质化处理，BAPM为每组prompt赋予明确的空间语义——部位prompt负责"在哪看"，全局prompt负责"看什么"。功能分工+自由通信的设计比刚性分块更灵活
- **重叠分区避免信息断裂**：upper/mid/lower区域有25%重叠，解决了硬性三等分中边界区域信息丢失的问题
- **PFTS的实用价值极高**：0.5%的通信量、几轮即收敛——这对实际联邦部署的可行性至关重要。且PFTS本身就能带来一致的正向增益
- **即插即用的通用性**：在6种不同基线方法上全部有效，平均提升稳定，说明BAPM捕捉到的是方法间共通的瓶颈
- **注意力可视化直观有力**：Figure 3中部位prompt精准锁定对应身体区域，全局prompt覆盖全身，Baseline则注意力散漫——视觉证据令人信服

## 局限与展望

1. **空间分区假设依赖行人直立**：upper/mid/lower的固定比例划分假设行人大致直立且居中。对弯腰、坐姿、严重遮挡等非标准姿态，固定分区可能失效。可考虑自适应分区或基于pose估计的动态分区
2. **仅验证了四个ReID数据集**：CUHK02/03、Market1501、MSMT17都是经典但相对"干净"的数据集。在更复杂的实际场景（低光照、极端天气、超高密度人群）中的表现需要进一步验证
3. **prompt数量的敏感性**：论文使用50个prompt（15+35分配），虽然附录有敏感性分析，但最优比例可能与数据集/域数量相关，缺乏自适应调整机制
4. **预训练模型的质量依赖**：PFTS模式依赖一个高质量的预训练模型作为起点。如果初始模型质量差，仅通过prompt微调能否弥补尚不明确
5. **跨模态/跨更多域的扩展**：仅在RGB可见光图像上验证，跨模态ReID（红外-可见光）中身体分布特征是否同样有效值得探索

## 相关工作与启发

- **VPT (Visual Prompt Tuning)**：开创性的视觉prompt方法 → FedBPrompt的关键创新在于给prompt赋予空间结构语义
- **PromptFL**：联邦学习中的prompt通信思路 → FedBPrompt将其从NLP迁移到视觉ReID并设计了任务定制的prompt结构
- **PCB/MGN等部件模型**：传统的行人部件对齐方法 → BAPM通过prompt+注意力掩码实现软性分区，比物理切割更灵活
- **SSCU (MM 2025)**：当前FedDG-ReID SOTA → BAPM在其基础上仍有3-5%的显著提升
- **DACS (AAAI 2024)**：数据增强路线 → BAPM从模型注意力机制层面互补
- **启发**：结构化prompt的思路可推广到其他需要空间对齐的视觉任务（如细粒度识别、医学图像分析）。PFTS的极低通信开销对边缘设备联邦学习场景有广泛适用性

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 3.5 | 将visual prompt与身体部位空间语义结合是有新意的设计，但整体框架基于VPT的自然延伸 |
| 实用性 | 4.5 | PFTS通信量降99%+即插即用特性，实际部署价值极高 |
| 实验充分度 | 4.0 | 6种基线、2种协议、消融完整、注意力可视化+量化，缺少计算开销详细对比 |
| 写作质量 | 4.0 | 问题定义清晰、方法叙述结构化、公式完整，结论部分略简短 |

<!-- RELATED:START -->

## 相关论文

- [Open-Vocabulary Domain Generalization in Urban-Scene Segmentation](open-vocabulary_domain_generalization_in_urban-scene_segmentation.md)
- [F3DGS: Federated 3D Gaussian Splatting for Decentralized Multi-Agent World Modeling](f3dgs_federated_3d_gaussian_splatting_for_decentralized_multi-agent_world_modeli.md)
- [ProOOD: Prototype-Guided Out-of-Distribution 3D Occupancy Prediction](proood_prototype-guided_out-of-distribution_3d_occupancy_prediction.md)
- [Generalizing Visual Geometry Priors to Sparse Gaussian Occupancy Prediction](generalizing_visual_geometry_priors_to_sparse_gaussian_occupancy_prediction.md)
- [CCF: Complementary Collaborative Fusion for Domain Generalized Multi-Modal 3D Object Detection](ccf_complementary_collaborative_fusion_for_domain_generalized_multi-modal_3d_obj.md)

<!-- RELATED:END -->
