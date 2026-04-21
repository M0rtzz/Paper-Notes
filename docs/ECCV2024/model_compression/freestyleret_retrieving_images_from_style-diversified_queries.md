---
title: >-
  [论文解读] FreestyleRet: Retrieving Images from Style-Diversified Queries
description: >-
  [ECCV 2024][模型压缩][风格多样化检索] 提出首个风格多样化查询图像检索（Style-Diversified QBIR）任务及数据集DSR，设计了轻量即插即用的FreestyleRet框架，通过Gram矩阵提取查询的纹理/风格特征，构建风格空间并以此初始化prompt token，使冻结的视觉编码器能适配文本、草图、低分辨率、艺术画等多种查询风格的检索。
tags:
  - ECCV 2024
  - 模型压缩
  - 风格多样化检索
  - Gram矩阵
  - 提示学习
  - 跨风格图像检索
  - 即插即用框架
---

# FreestyleRet: Retrieving Images from Style-Diversified Queries

**会议**: ECCV 2024  
**arXiv**: [2312.02428](https://arxiv.org/abs/2312.02428)  
**代码**: https://github.com/CuriseJia/FreeStyleRet  
**领域**: 模型压缩（轻量化检索）  
**关键词**: 风格多样化检索, Gram矩阵, prompt tuning, 跨风格图像检索, 即插即用框架

## 一句话总结

提出首个风格多样化查询图像检索（Style-Diversified QBIR）任务及数据集DSR，设计了轻量即插即用的FreestyleRet框架，通过Gram矩阵提取查询的纹理/风格特征，构建风格空间并以此初始化prompt token，使冻结的视觉编码器能适配文本、草图、低分辨率、艺术画等多种查询风格的检索。

## 研究背景与动机

基于查询的图像检索（QBIR）是计算机视觉的基础任务，在图像搜索引擎、跨模态任务中有广泛应用。然而，**当前检索研究几乎完全聚焦于文本-图像检索**，忽视了用户在实际场景中可能使用多种查询方式表达检索意图。

**现有痛点**：

**查询风格单一**：主流检索模型（CLIP、BLIP等）主要针对文本查询设计，对于草图（sketch）、艺术风格画（art）、低分辨率截图（low-res）等非文本查询缺乏适配能力。

**用户意图鸿沟**：不同用户习惯不同的查询方式——有人愿意手绘草图描述物体形态，有人截取模糊图片，有人用文字描述——单一查询模态无法充分表达检索意图。

**多风格无法协同**：现有模型无法同时处理多种风格的查询输入，更无法利用不同风格间的互补信息提升检索性能。

**核心矛盾**：如何让一个检索框架**同时**理解并适配多种风格的查询输入，在计算开销极低的前提下实现风格感知的检索？

**本文切入角度**：借鉴图像风格迁移中的Gram矩阵来捕获不同查询风格的纹理特征，通过K-Means聚类构建风格空间，然后用风格特征初始化prompt token，以极低参数代价在冻结的预训练视觉编码器上实现风格感知能力。

## 方法详解

### 整体框架

FreestyleRet包含三个核心模块：(1) Gram矩阵风格提取模块，(2) 风格空间构建模块，(3) 风格初始化Prompt调优模块。训练时迭代数据集两次：第一次构建风格空间，第二次执行风格初始化的prompt tuning。推理时用构建好的风格空间直接进行风格感知检索。

### 关键设计

1. **Gram矩阵风格提取模块（Gram-based Style Extraction）**：对任意风格的查询输入 $q_i$，先用冻结的VGG网络提取第三卷积层特征（$112 \times 112 \times 128$），下采样后计算Gram矩阵：

$$g_i = (f_d(v_i))^\mathsf{T} f_d(v_i)$$

Gram矩阵 $g_i$ 捕获了查询的纹理特征和通道间空间关系，能有效区分草图的线条纹理、艺术画的笔触纹理、低分辨率的模糊纹理等。

**设计动机**：Gram矩阵在风格迁移领域被验证能有效表征图像的风格/纹理信息，将其引入检索任务可以自然地区分不同查询风格而无需手动设计风格标签。

2. **风格空间构建模块（Style Space Construction）**：将所有查询的Gram矩阵通过**K-Means算法**聚类为4个簇（对应4种查询风格），聚类中心 $\mu_1, ..., \mu_4$ 作为风格空间的基向量。对于新输入的查询 $q_i$，通过加权求和计算其风格特征：

$$w_j = \frac{e^{\cos(q_i, \mu_j)}}{\sum_{j=1}^{4} e^{\cos(q_i, \mu_j)}}, \quad s_i = \sum_{j=1}^{4} w_j \mu_j$$

**设计动机**：显式构建风格空间使模型能够自适应地理解任意查询的风格特性，而非将风格类别硬编码。软性加权组合允许处理介于两种风格之间的查询。

3. **风格初始化Prompt调优模块（Style-Init Prompt Tuning）**：在冻结的ViT视觉编码器的每一层插入4个可学习的prompt token。关键创新在于token的**初始化策略**：浅层用风格空间特征 $s_i$ 初始化，深层用Gram矩阵 $g_i$ 初始化（而非随机初始化），使编码器从一开始就具备风格感知能力：

$$[x_i, \_, E_i] = L_i(x_{i-1}, P_{i-1}, E_{i-1}), \quad i=1,...,n$$

**设计动机**：VPT证明了深度prompt优于浅层prompt，但随机初始化无法注入风格信息。分层差异化初始化——浅层注入全局风格信息、深层注入细粒度纹理信息——使编码器在不同抽象层次上都能感知风格。

### 损失函数 / 训练策略

使用Triplet Loss作为训练目标：

$$\mathcal{L} = \frac{1}{B}\sum_{i=1}^{B}\max(0, \text{dist}(F_i, P_i) - \text{dist}(F_i, N_i) + \alpha)$$

其中 $\text{dist}(x,y) = 1 - \cos(x,y)$，$\alpha=1.0$。正样本为ground-truth检索结果，负样本为同风格集合中随机选取的其他图像。学习率1e-5，20个epoch，batch size 24，A100 GPU训练。

## 实验关键数据

### 主实验：DSR数据集风格多样化检索

| 方法 | Text R@1 | Sketch R@1 | Art R@1 | Low-Res R@1 | 说明 |
|------|----------|------------|---------|-------------|------|
| CLIP (zero-shot) | 66.1 | 47.5 | 58.5 | 45.0 | 无风格适配 |
| CLIP* (prompt tuned) | 72.2 | 63.6 | 58.2 | 78.8 | 标准prompt tuning |
| BLIP* (prompt tuned) | 74.3 | 67.1 | 51.1 | 77.2 | 标准prompt tuning |
| VPT* | 69.9 | 73.3 | 66.7 | 81.4 | 视觉prompt tuning |
| ImageBind | 71.0 | 50.8 | 58.2 | 79.0 | 多模态模型 |
| LanguageBind | 79.7 | 63.6 | 67.5 | 78.6 | 多模态模型 |
| **FreestyleRet-CLIP** | 69.9 | **80.6** | **71.4** | **86.4** | 本文（CLIP编码器） |
| **FreestyleRet-BLIP** | **81.6** | **81.2** | **74.5** | **90.5** | 本文（BLIP编码器） |

FreestyleRet-BLIP在所有4种查询风格上均取得最优R@1，Sketch提升14.1%（vs BLIP*），Art提升23.4%。

### 消融实验：Prompt Token设计

| 配置 | Sketch R@1 | Art R@1 | Low-Res R@1 | 说明 |
|------|-----------|---------|-------------|------|
| 浅层Random + 深层Random | 68.1 | 63.5 | 78.8 | 随机初始化基线 |
| 浅层StyleSpace + 深层Random | 76.7 | 69.1 | 82.4 | 浅层注入风格 |
| 浅层Random + 深层Gram | 76.8 | 69.2 | 81.8 | 深层注入纹理 |
| 浅层Gram + 深层StyleSpace | 78.1 | 69.5 | 84.4 | 反向配置 |
| **浅层StyleSpace + 深层Gram** | **80.6** | **71.4** | **86.4** | 最佳配置 |
| 最佳配置，1 token | 68.2 | 64.7 | 79.1 | token数不足 |
| 最佳配置，2 tokens | 72.3 | 65.9 | 82.8 | 性能上升 |
| 最佳配置，8 tokens | 77.9 | 67.1 | 80.7 | 过多token反降 |

### 计算效率对比

| 方法 | 参数量 | 推理速度 |
|------|--------|----------|
| CLIP | 427M | 68ms |
| ImageBind | 1200M | 372ms |
| FreestyleRet-CLIP | 476M (+29M) | 96ms (+28ms) |
| FreestyleRet-BLIP | 940M (+29M) | 101ms (+39ms) |

仅增加29M参数和约30ms推理时间，即可支持多风格检索。

### 关键发现

- **多风格查询互相增益**：FreestyleRet中，sketch+text联合查询的Text R@1从69.9%提升到82.5%（+12.6%），而CLIP/BLIP中联合查询反而降低性能
- **5-10个epoch即可收敛**：FreestyleRet的收敛速度远快于基线模型（需50+epoch），每个epoch仅需4分钟
- **风格空间初始化优于随机初始化**：消融实验显示，最佳配置（浅StyleSpace + 深Gram）较随机初始化在Sketch上提升12.5%

## 亮点与洞察

- **新任务定义**：首次系统定义了风格多样化QBIR任务，提出DSR和ImageNet-X两个评估数据集，为该方向奠定基础
- **Gram矩阵的跨领域妙用**：将风格迁移中的纹理表征工具引入检索任务，自然且优雅
- **即插即用设计**：冻结编码器 + 仅训练prompt tokens的方式使框架可以无缝嫁接到CLIP、BLIP等任意ViT编码器上
- **多查询协同**：不同风格的查询可以互相增强而非干扰，这是现有多模态模型做不到的

## 局限与展望

- **查询风格仍然有限**：目前仅考虑text、sketch、art、low-res四种，未涵盖3D、视频、语音等更多模态
- **DSR数据集规模较小**：仅10000张图像，大规模场景下的性能有待验证
- **文本查询未充分利用**：文本分支直接用CLIP文本编码器，未像视觉分支一样进行风格感知适配
- **K-Means聚类的风格数固定为4**：当引入新查询风格时需要重新聚类，自适应性不足

## 相关工作与启发

- **VPT (Visual Prompt Tuning)**：首次将prompt tuning引入视觉模型，本文在其基础上创新了风格初始化策略
- **CoOP / CoCoOP**：文本prompt学习的经典工作，启发了可学习prompt的设计
- **Gram矩阵 (Gatys et al.)**：风格迁移的核心表征工具，本文创造性地将其用于风格空间构建
- **ImageBind / LanguageBind**：多模态统一模型，但缺乏对风格差异的细粒度建模
- 启发：prompt初始化策略可推广到其他需要条件适配的下游任务中

## 评分

- 新颖性: ⭐⭐⭐⭐ 新任务+新数据集+Gram矩阵初始化prompt的创意组合，有开创性
- 实验充分度: ⭐⭐⭐⭐ 两个数据集、多个基线对比、详细消融、定性分析完整
- 写作质量: ⭐⭐⭐⭐ 图示精美，问题动机清晰，三个模块的逻辑递进清楚
- 价值: ⭐⭐⭐⭐ 定义新任务且提供了强基线，对检索社区有长远影响

<!-- RELATED:START -->

## 相关论文

- [SpaceJAM: a Lightweight and Regularization-free Method for Fast Joint Alignment of Images](spacejam_a_lightweight_and_regularization-free_method_for_fast_joint_alignment_o.md)
- [Is Retain Set All You Need in Machine Unlearning? Restoring Performance of Unlearned Models with Out-Of-Distribution Images](is_retain_set_all_you_need_in_machine_unlearning_restoring_performance_of_unlear.md)
- [Style Quantization for Data-Efficient GAN Training](../../CVPR2025/model_compression/style_quantization_for_data-efficient_gan_training.md)
- [DuoLoRA: Cycle-Consistent and Rank-Disentangled Content-Style Personalization](../../ICCV2025/model_compression/duolora_cycle-consistent_and_rank-disentangled_content-style_personalization.md)
- [A Fano-Style Accuracy Upper Bound for LLM Single-Pass Reasoning in Multi-Hop QA](../../ICLR2026/model_compression/a_fano-style_accuracy_upper_bound_for_llm_single-pass_reasoning_in_multi-hop_qa.md)

<!-- RELATED:END -->
