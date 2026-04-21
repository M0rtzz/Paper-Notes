---
title: >-
  [论文解读] HouseMind: Tokenization Allows MLLMs to Understand, Generate and Edit Architectural Floor Plans
description: >-
  [CVPR 2026][多模态][建筑平面图] 提出 HouseMind，通过层次化 VQ-VAE 将建筑平面图的轮廓和房间实例分别离散化为空间 token，与文本 token 统一到同一词汇表中，使小规模 LLM（0.6B）就能在单一自回归框架下实现平面图的理解、生成和编辑三大任务，性能全面超越基于扩散模型和大规模 VLM 的方法。
tags:
  - CVPR 2026
  - 多模态
  - 建筑平面图
  - VQ-VAE
  - 多模态LLM
  - 空间推理
  - 层次化token
---

# HouseMind: Tokenization Allows MLLMs to Understand, Generate and Edit Architectural Floor Plans

**会议**: CVPR 2026  
**arXiv**: [2603.11640](https://arxiv.org/abs/2603.11640)  
**代码**: https://housemind.github.io/  
**领域**: 多模态VLM  
**关键词**: 建筑平面图, VQ-VAE, 多模态LLM, 空间推理, 统一生成

## 一句话总结
提出HouseMind框架，通过层次化VQ-VAE将建筑平面图离散化为轮廓token和房间实例token的结构化序列，结合三阶段多模态对齐和指令微调，以Qwen3-0.6B为backbone实现了平面图理解、生成、编辑三项任务的统一建模，几何有效性和可控性大幅超越现有方法。

## 研究背景与动机

**领域现状**：AI辅助建筑平面图设计已有GAN、图神经网络、扩散模型等多种方法。近期LLM/MLLM引入了新范式，如Tell2Design、ChatHouseDiffusion等尝试用语言驱动布局生成。

**现有痛点**：(1) 扩散和自回归模型将布局视为纯视觉过程，缺乏房间实例级的显式推理；(2) 大模型多为黑箱，空间可控性弱；(3) 理解、生成、编辑无法统一；(4) 计算开销大，难以本地部署。

**核心矛盾**：平面图设计需要同时处理几何和语义信息，现有方法要么擅几何但缺语义推理，要么有语言能力但缺空间精度。

**本文目标** 构建高效、可本地部署的统一框架，同时支持平面图的空间理解、条件生成和可控编辑。

**切入角度**：将关键突破定位在"表示"上——用VQ-VAE将平面图离散化为结构化token序列，使LLM可以像处理语言一样处理空间布局。

**核心 idea**：通过层次化VQ-VAE将平面图表示为包含轮廓token和房间实例token的离散序列，让MLLM以自回归方式统一三项设计任务。

## 方法详解

### 整体框架
平面图被分解为轮廓和N个房间。两个独立VQ-VAE分别编码为离散token序列，加上房间标签组成结构化序列 $Z = [\mathbf{z}_o, \ell_{r_1}, \mathbf{z}_{r_1}, \dots]$，与文本token交错输入Qwen3-0.6B，通过三阶段训练实现统一自回归建模。

### 关键设计

1. **层次化VQ-VAE（轮廓+条件房间离散化）**:

    - 功能：将全局轮廓和每个房间实例分别编码为离散token
    - 核心思路：轮廓用CNN encoder在codebook $\mathcal{Z}_o$ 中量化；房间以房间mask和轮廓为联合输入进行条件编码 $z_{i,j}^{(r)} = \arg\min_k \|E_r(x_{r_i}, x_o)_j - e_k^{(r)}\|_2$
    - 设计动机：条件编码让房间token同时捕获几何形状和相对于轮廓的空间位置

2. **三阶段多模态对齐训练**:

    - 功能：逐步建立空间token与语言token之间的对齐
    - 核心思路：Stage 1将VQ-VAE codebook嵌入LLM词表；Stage 2在大规模配对数据上自回归预训练；Stage 3在三任务指令数据上SFT
    - 设计动机：渐进式对齐避免直接在复杂任务上训练导致的优化不稳定

3. **统一序列建模（理解/生成/编辑）**:

    - 功能：将三项任务统一为条件自回归预测
    - 核心思路：生成 $p(Z|\mathbf{z}_o, s) = \prod_t p(Z_t|Z_{<t}, \mathbf{z}_o, s)$；理解输出文本描述；编辑给定原始序列+指令输出修改后序列
    - 设计动机：统一格式让同一模型在不同任务间共享知识

### 损失函数 / 训练策略
VQ-VAE用重建损失+codebook loss。LLM部分用交叉熵自回归损失。基于Qwen3-0.6B，RPLAN数据集78,738样本，单张RTX 3090推理。

## 实验关键数据

### 主实验

| 方法 | Micro IoU | FID↓ | Node F1 | Edge Overlap | 推理(s) |
|------|-----------|------|---------|-------------|---------|
| Tell2Design | 0.390 | 30.5 | 0.808 | 0.197 | ~15 |
| ChatHouseDiffusion | 0.589 | 11.3 | 0.985 | 0.710 | ~30 |
| **HouseMind-G** | **0.709** | **1.91** | **0.994** | **0.880** | ~2 |

### 消融实验（理解任务）

| 方法 | RMR | LocAcc | AreaDiff↓ | AdjAcc | RelAcc |
|------|-----|--------|-----------|--------|--------|
| Qwen3-VL-8B | 0.698 | 0.347 | 5.837 | 0.382 | 0.128 |
| InternVL3.5-8B | 0.847 | 0.546 | 12.234 | 0.469 | 0.157 |
| **HouseMind-U** | **0.998** | **0.969** | **0.549** | **0.990** | **0.808** |

### 关键发现
- FID从ChatHouseDiffusion的11.3降至1.91，生成质量大幅提升
- 理解任务上房间定位精度0.969比最佳VLM高40+个点，面积误差仅0.549m²
- 编辑任务ΔIoU达0.608，远超FLUX(0.053)和Qwen-Edit(0.088)
- 统一模型HouseMind-O在所有任务上接近单任务版本

## 亮点与洞察
- **房间级token化**：完美匹配建筑设计的认知过程（先定外形→再布局房间）
- **极致轻量化**：基于0.6B参数超越7-8B VLM，推理仅~2秒/样本
- **与GPT-5/Gemini Pro的对比**：领域特定token化在精确性上优于通用大模型

## 局限与展望
- 编辑限于房间增删，不支持复杂拓扑变换
- 未建模门窗家具等功能组件
- 仅基于RPLAN数据集

## 相关工作与启发
- **vs MaskPLAN**: MaskPLAN编码整体，HouseMind按房间分层编码保留结构
- **vs FloorPlanLLaMA**: VQ-VAE编码整体布局致边界不一致，HouseMind通过条件VQ-VAE保持几何精度

## 评分
- 新颖性: ⭐⭐⭐⭐ 层次token化理念新颖，三任务统一设计优雅
- 实验充分度: ⭐⭐⭐⭐ 三任务全面评测，但只用RPLAN一个数据集
- 写作质量: ⭐⭐⭐⭐ 结构清晰
- 价值: ⭐⭐⭐⭐ 对建筑设计AI有直接应用价值
---
title: "HouseMind: Tokenization Allows MLLMs to Understand, Generate and Edit Architectural Floor Plans"
description: "提出HouseMind框架，通过层次化VQ-VAE将建筑平面图离散化为房间级token，结合多模态大语言模型实现平面图的理解、生成和编辑的统一建模"
tags: ["建筑平面图", "VQ-VAE", "多模态LLM", "空间推理", "token化"]
---

# HouseMind: Tokenization Allows MLLMs to Understand, Generate and Edit Architectural Floor Plans

**会议**: CVPR 2026  
**arXiv**: [2603.11640](https://arxiv.org/abs/2603.11640)  
**代码**: https://housemind.github.io/ (有)  
**领域**: 多模态VLM  
**关键词**: 建筑平面图, VQ-VAE, 多模态LLM, 空间推理, 层次化token

## 一句话总结
提出 HouseMind，通过层次化 VQ-VAE 将建筑平面图的轮廓和房间实例分别离散化为空间 token，与文本 token 统一到同一词汇表中，使小规模 LLM（0.6B）就能在单一自回归框架下实现平面图的理解、生成和编辑三大任务，性能全面超越基于扩散模型和大规模 VLM 的方法。

## 研究背景与动机

**领域现状**：建筑平面图设计需要对几何、语义和空间层次的联合推理，是 AI 设计领域最具认知挑战性的任务之一。现有方法包括 GAN-based（如 HouseGAN）、Graph-based（如 Graph2Plan）和 Diffusion-based（如 ChatHouseDiffusion）等。

**现有痛点**：(1) 多数方法将布局生成视为纯视觉过程，缺乏房间实例级的显式推理，导致局部合理但全局不协调；(2) 大规模 VLM 方法是黑箱生成器，空间可控性差；(3) 现有框架难以在单一架构内统一理解、生成和编辑三项任务；(4) 计算开销大，本地部署困难。

**核心矛盾**：连续的几何布局与 LLM 的离散 token 序列建模之间存在根本性的表征隔阂——如何将空间几何信息有效编码为 LLM 可理解的离散符号？

**本文目标** 构建一个高效、可本地部署的统一多模态模型，在单一框架内实现平面图理解、生成和编辑的联合推理。

**切入角度**：用层次化 VQ-VAE 将几何信息离散化为 token，让 LLM 可以用同一套序列建模机制处理空间和语言信息。

**核心 idea**：通过房间级 token 化桥接连续几何布局与离散语言建模，实现统一的空间推理。

## 方法详解

### 整体框架
HouseMind 由两个核心组件构成：(1) 房间实例 Token 化——使用两套独立 VQ-VAE 将轮廓和各房间分别编码为离散 token 序列；(2) 多模态对齐与指令微调——三阶段渐进式训练将空间 token 与语言 token 对齐。最终序列格式为 $Z = [\boldsymbol{z}_o, \ell_{r_1}, \boldsymbol{z}_{r_1}, \ldots, \ell_{r_N}, \boldsymbol{z}_{r_N}]$，交织了几何 token 和语义标签 token。

### 关键设计

1. **层次化房间实例 Token 化**:

    - 功能：将平面图分解为轮廓 $x_o$ 和 $N$ 个房间实例 $\{x_{r_i}\}_{i=1}^N$，分别离散化
    - 核心思路：轮廓 VQ-VAE 将二值轮廓掩码编码为 $\boldsymbol{z}_o = E_o(x_o)$；条件房间 VQ-VAE 在轮廓上下文条件下编码每个房间 $\boldsymbol{z}_{r_i} = E_r(x_{r_i}, x_o)$，各通过最近邻量化映射到码本 $\mathcal{Z}_o, \mathcal{Z}_r$。CNN 编码器 + 转置 CNN 解码器
    - 设计动机：条件编码让房间表征感知空间上下文（邻接关系、在轮廓中的位置），而非孤立编码，这对保持全局一致性至关重要

2. **三阶段多模态对齐训练**:

    - 功能：渐进式地建立空间 token 与语言 token 的跨模态对应
    - 核心思路：Stage 1（嵌入初始化）——将 VQ-VAE 码本中的每个码字分配一个可训练的 token 嵌入，融入 LLM 词汇表；Stage 2（多模态预训练）——在大规模文本+空间 token 配对数据上用自回归目标训练，学习双向文本-空间对齐；Stage 3（指令微调 SFT）——在理解/生成/编辑三类指令数据上微调
    - 设计动机：直接端到端训练会因空间 token 和语言 token 的分布差异导致优化不稳定，渐进对齐确保稳定且高质量的跨模态理解

3. **统一任务建模**:

    - 功能：将理解、生成和编辑统一为条件序列生成问题
    - 核心思路：理解——给定 $Z$ 和文本提示，输出描述/泡泡图/JSON；生成——给定轮廓 $\boldsymbol{z}_o$ 和文本 $s$，自回归生成 $p(Z | \boldsymbol{z}_o, s) = \prod_t p(Z_t | Z_{<t}, \boldsymbol{z}_o, s)$；编辑——给定源布局 $Z^{\mathrm{src}}$ 和编辑指令 $s$，生成 $Z^{\mathrm{tgt}}$
    - 设计动机：统一建模让单一模型可以处理多种任务，避免为每个任务训练独立模型的冗余

### 损失函数 / 训练策略
VQ-VAE 使用标准重构损失 + commitment 损失 + 码本损失。LLM 阶段使用自回归交叉熵损失。基于 Qwen3-0.6B 构建，单卡 RTX 3090 可推理。

## 实验关键数据

### 主实验（生成任务）

| 方法 | Micro IoU | FID ↓ | Node F1 | Edge Overlap | 时间(s) |
|------|-----------|-------|---------|-------------|---------|
| Tell2Design | 0.390 | 30.5 | 0.808 | 0.197 | ~15 |
| ChatHouseDiffusion | 0.589 | 11.3 | 0.985 | 0.710 | ~30 |
| FloorPlanLLaMA | 0.607 | 49.3 | 0.922 | 0.574 | ~1 |
| **HouseMind-G** | **0.709** | **1.91** | **0.994** | **0.880** | **~2** |

### 消融实验（理解任务）

| 方法 | RMR | LocAcc | AreaDiff↓ | AdjAcc | RelAcc |
|------|-----|--------|-----------|--------|--------|
| LLaVA-v1.6-7B | 0.616 | 0.225 | 3.649 | 0.134 | 0.056 |
| Qwen3-VL-8B | 0.698 | 0.347 | 5.837 | 0.382 | 0.128 |
| InternVL3.5-8B | 0.847 | 0.546 | 12.234 | 0.469 | 0.157 |
| **HouseMind-U** | **0.998** | **0.969** | **0.549** | **0.990** | **0.808** |

### 关键发现
- HouseMind 在所有三个任务上全面超越现有方法，且推理速度极快（2-3秒/样本）
- FID 从竞争方法的 11.3-49.3 降到 1.91，说明生成质量接近真实分布
- 理解任务：AdjAcc 从最好的 0.597 提升到 0.990，面积误差从数平方米降到 0.549 m²
- 消融显示三阶段训练每一步都有贡献，去掉 Stage 1 优化不稳定、去掉 Stage 2 缺乏高层对应

## 亮点与洞察
- 房间级 token 化是核心创新——不是把整张平面图当图像处理，而是分解到实例级，让 LLM 可以进行结构化推理。这一范式可迁移到其他结构化设计任务（电路设计、UI 布局等）
- 0.6B 参数模型超越 7-8B VLM 的事实说明，正确的表征比模型规模更重要
- 编辑任务中 Node F1 达到 0.998，说明模型可以精确修改指定房间而不影响其他区域，实现了真正的局部可控编辑

## 局限与展望
- 编辑能力仅限于简单的房间增删，不支持复杂的拓扑变换
- 未建模门窗家具等细节构件，限制了在精细室内设计中的应用
- 与人类设计偏好和美学约束未对齐，生成结果在功能合理性上可能不满足专业标准
- 仅在 RPLAN 数据集上验证，更多样的建筑风格（如不规则形状、多楼层）有待探索

## 相关工作与启发
- **vs MaskPLAN**: MaskPLAN 也用 VQ-VAE 编码几何属性，但用 masked transformer autoencoding，本文更进一步引入 LLM 做多任务统一推理
- **vs ChatHouseDiffusion**: 扩散模型方法在简单布局上表现好但处理复杂配置时困难；HouseMind 通过离散推理保持全局一致性
- **vs Tell2Design**: Tell2Design 建立了文本-平面图的基准但泛化有限；HouseMind 的 token 化范式更具扩展性

## 评分
- 新颖性: ⭐⭐⭐⭐ 房间级 token 化 + LLM 统一建模的idea很有新意
- 实验充分度: ⭐⭐⭐⭐ 三任务全面评估，对比方法充分
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，但问题形式化偏重
- 价值: ⭐⭐⭐⭐ 为 AI 辅助建筑设计提供了实用的统一方案

<!-- RELATED:START -->

## 相关论文

- [Tokenization Allows Multimodal Large Language Models to Understand, Generate and Edit Architectural Floor Plans (HouseMind)](tokenization_allows_multimodal_large_language_models_to_understand_generate_and_.md)
- [Token Warping Helps MLLMs Look from Nearby Viewpoints](token_warping_helps_mllms_look_from_nearby_viewpoints.md)
- [EgoMind: Activating Spatial Cognition through Linguistic Reasoning in MLLMs](egomind_activating_spatial_cognition_through_linguistic_reasoning_in_mllms.md)
- [HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)

<!-- RELATED:END -->
