---
title: >-
  [论文解读] ImageGem: In-the-wild Generative Image Interaction Dataset for Generative Model Personalization
description: >-
  [ICCV 2025][图像生成][生成模型个性化] 提出 **ImageGem**，首个大规模真实用户生成式交互数据集（57K用户 × 242K定制LoRA × 3M文本提示 × 5M生成图像），利用个体用户偏好标注实现三大应用：**聚合偏好对齐**超越 Pick-a-Pic、**个性化检索与生成式推荐**（VLM排序显著提升）、以及首次提出的**生成模型个性化**——在 LoRA 潜权重空间（W2W）中学习偏好编辑方向以定制扩散模型。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "生成模型个性化"
  - "用户偏好对齐"
  - "LoRA权重空间"
  - "扩散模型"
  - "推荐系统"
---

# ImageGem: In-the-wild Generative Image Interaction Dataset for Generative Model Personalization

**会议**: ICCV 2025  
**arXiv**: [2510.18433](https://arxiv.org/abs/2510.18433)  
**代码**: [项目页面](https://maps-research.github.io/imagegem-iccv2025/)  
**领域**: 图像生成  
**关键词**: 生成模型个性化, 用户偏好对齐, LoRA权重空间, DiffusionDPO, 推荐系统, 扩散模型

## 一句话总结

提出 **ImageGem**，首个大规模真实用户生成式交互数据集（57K用户 × 242K定制LoRA × 3M文本提示 × 5M生成图像），利用个体用户偏好标注实现三大应用：**聚合偏好对齐**超越 Pick-a-Pic、**个性化检索与生成式推荐**（VLM排序显著提升）、以及首次提出的**生成模型个性化**——在 LoRA 潜权重空间（W2W）中学习偏好编辑方向以定制扩散模型。

## 研究背景与动机

### 核心问题

> "一千个读者眼中有一千个哈姆雷特"

当用户输入"哈姆雷特肖像"时，每个人想象的版本不同。现有文本到图像模型只能生成**符合大众偏好**的图像，无法捕捉和产出**个体偏好**。

### 现有数据集的局限

**聚合偏好数据集**（如 Pick-a-Pic）：收集用户对图像对的评分，但反映的是**群体平均偏好**

**身份定制数据集**（如 DreamBooth）：支持特定人/物的概念注入，但不处理**偏好风格**

**缺少个体级偏好**：没有大规模记录单个用户与其定制模型交互的数据

**现有个性化方法**（如 ViPer）限于零样本，无法利用用户间相似性

### 数据来源

从 **Civitai**（最大 AIGC 平台）采集用户公开分享的定制模型和生成图像，捕获真实世界的交互数据。

## 方法详解

### 数据集构建

#### 核心统计（Table 1）

| 指标 | 原始数据 | 安全过滤后 |
|------|---------|-----------|
| 图像总数 | 5,658,107 | 4,916,134 |
| 唯一提示 | 2,975,943 | 2,895,364 |
| LoRA 模型 | 242,889 | 242,118 |
| 唯一模型标签 | 105,788 | 97,434 |
| 总用户数 | 57,245 | - |
| 模型上传者 | 19,003 | 18,889 |
| 每上传者平均图像 | 49 | 48 |
| 每上传者平均模型 | 12 | 13 |
| 每模型平均图像 | 62 | 54 |

#### 安全检查

- 利用 Civitai 的 NSFW 分类
- 用 **Detoxify** 检测提示文本毒性
- 过滤不安全概率 > 0.2 的提示对应图像
- 获得 IRB 伦理审批

#### 数据结构

三元关系：**图像 ↔ LoRA模型 ↔ 用户**

两类用户交互数据：
1. **个体级**：用户-模型交互记录（提示、生成配置等），含 1.74M 展示图 + 3.18M 历史图
2. **聚合级**：表情符号反馈（点赞、爱心、笑、哭）

### 应用 1：聚合偏好对齐

利用 DiffusionDPO 框架：

$$\max_{p_\theta} \mathbb{E}[r(\mathbf{c}, \mathbf{x}_0)] - \beta \mathbb{D}_{\text{KL}}[p_\theta(\mathbf{x}_{0:T}|\mathbf{c}) \| p_{\text{ref}}(\mathbf{x}_{0:T}|\mathbf{c})]$$

**偏好对构建方法**：
- 先对提示的 CLIP 嵌入做 HDBScan 聚类
- 在每个聚类内用 HPS v2 的 min-max 配对构建偏好对

### 应用 2：检索与生成式推荐

**候选检索**：
- 图像检索：FAISS + ViT 初始化嵌入 → Two-tower 模型最优
- 模型推荐：SASRec（自注意力序列推荐）捕获时序演化

**生成式推荐**（VLM 驱动）：
- 用 **Pixtral-12B** 进行两阶段流程：
  1. **描述阶段**：从用户历史图像中提取偏好文本描述 $q_i$
  2. **排序阶段**：比较偏好描述与候选项，输出评分+解释
- 随机化评分策略缓解 VLM 排序不稳定性

### 应用 3：生成模型个性化（核心创新）

#### W2W 权重空间构建

1. **SVD 标准化**：对每个 LoRA 权重矩阵做 SVD，保留 top-1 分量
2. **展平拼接**：所有层的压缩矩阵拼接为向量 $\theta_i \in \mathbb{R}^d$
3. **PCA 降维**：在 $D = \{\theta_1, ..., \theta_N\}$ 上做 PCA，保留 top-$m$ 主成分
4. 基向量 $\{w_1, ..., w_m\}$ 编码用户偏好方向

#### 偏好方向学习

- 用二分类标签（用户是否喜欢某模型）训练线性分类器
- 超平面的法向量 $v$ 即为偏好遍历方向
- 编辑公式：$\theta_{\text{edit}} = \theta + \alpha v$，$\alpha$ 控制编辑强度

#### 个体偏好学习流程

1. 对用户所有生成图像计算 CLIP 嵌入 → HDBScan 聚类 → 找代表性偏好簇
2. VLM 对簇中 top-9 图像生成风格描述
3. 用 CLIP 相似度标记 LoRA 模型的偏好标签
4. 学习个体超平面 → 多方向编辑

## 实验关键数据

### 聚合偏好对齐（Table 4）

| 数据子集 | Pick Score↑ | HPSv2↑ | CLIP Score↑ |
|---------|------------|--------|------------|
| 原始 SD1.5 | 0.1977 | 0.2637 | 0.3581 |
| Pick-a-pic Cars | 0.1993 | 0.2690 | 0.3607 |
| **ImageGem Cars Small** | **0.2004** | **0.2741** | **0.3745** |
| 原始 SD1.5 | 0.2010 | 0.2646 | 0.3560 |
| Pick-a-pic Dogs | 0.2058 | 0.2739 | 0.3617 |
| **ImageGem Dogs** | **0.2069** | **0.2789** | **0.3683** |
| 原始 SD1.5 | 0.1954 | 0.2640 | 0.3446 |
| Pick-a-pic Scenery | 0.1936 | 0.2676 | 0.3289 |
| **ImageGem Scenery Large** | **0.1961** | **0.2747** | **0.3427** |

**核心结果**：在所有三个主题上，ImageGem 训练的 DPO 模型在 Pick Score、HPSv2、CLIP Score 上均优于 Pick-a-Pic。

### 检索与推荐（Table 5-7）

| 任务 | 最佳方法 | Recall@10 / NDCG@10 |
|------|---------|---------------------|
| 图像检索 (1M) | Two-tower | Rec@100=0.2402 |
| 模型推荐 (200K) | SASRec | Rec@10=0.1839, NDCG@10=0.1239 |
| 排序（图像） | VLM | Rec@5=0.9500, NDCG@5=0.6745 |
| 排序（模型） | VLM | Rec@5=**0.7222**, NDCG@5=**0.4981** |

**关键发现**：VLM 在模型推荐排序上大幅超越传统方法（SASRec Rec@5 仅 0.50），且提供可解释的文本理由。

### 生成模型个性化消融

| 策略 | 动漫→写实 | 写实→动漫 | 评估 |
|------|----------|----------|------|
| SVD-based W2W | ✓ 双向有效 | ✓ 双向有效 | 最鲁棒 |
| attn_v layers | ✓ 单向有效 | ✗ 反向失败 | 部分有效 |
| FF layers | ✗ 双向差 | ✗ 双向差 | 不推荐 |

多用户个性化验证：对3个用户学习3个不同偏好方向，CLIP和VLM排序均确认编辑后模型生成更符合各用户偏好的图像。

## 亮点与洞察

1. **数据集的独特价值**：首个捕获真实用户与生成模型交互的大规模数据集，弥合了聚合偏好与个体偏好之间的鸿沟
2. **新范式——生成模型个性化**：不是个性化提示或后处理，而是**直接编辑模型权重**来对齐个体偏好，这是一个全新方向
3. **VLM 作为推荐引擎**：利用 VLM 的多模态理解能力进行可解释的排序，在推荐系统中开辟新可能
4. **偏好的数据驱动表征**：通过 W2W 空间中的线性方向表征偏好，使得模型编辑既高效又可控

## 局限性

1. **PCA 的限制**：依赖 PCA 约束模型选择为低秩（rank 8/16），限制了模型多样性
2. **偏好对构建**：使用 HPS 在聚类内配对，未充分利用用户隐式反馈（如交互日志）
3. **域的稀疏性**：当前在人物域效果最好，但风景等冷门域因模型数量少难以学习有效 W2W 空间
4. **安全与隐私**：虽有安全过滤和 IRB 审批，Civitai 生态中的 NSFW 内容仍需持续关注
5. **仅验证 SD 系列**：未在 Flux 等更新模型架构上验证

## 相关工作与启发

- **与 Pick-a-Pic 的区别**：Pick-a-Pic 依赖人工标注显式偏好对，ImageGem 使用自然观测数据的隐式偏好
- **与 Weights2Weights 的区别**：原始 W2W 使用自训练 rank-1 LoRA 在人脸身份空间编辑，ImageGem 扩展到用户偏好空间，处理混合 rank 的 Civitai LoRA
- **与 ViPer 的区别**：ViPer 是零样本偏好学习（推理时捕获），ImageGem 从历史交互数据学习，可利用用户间相似性
- **启发**：W2W 空间概念可推广到视频生成模型、3D 模型的个性化；VLM 驱动的推荐范式可扩展到更多创意工具

## 评分 ⭐⭐⭐⭐

**创新性**: ⭐⭐⭐⭐⭐ — 数据集 + 新范式（生成模型个性化）+ 多应用验证  
**实用性**: ⭐⭐⭐⭐ — 直接面向 AIGC 平台的实际需求  
**实验深度**: ⭐⭐⭐⭐ — 三大应用场景均有定量验证，含多种 baseline 对比  
**写作质量**: ⭐⭐⭐⭐ — 结构清晰，应用场景划分合理

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SummDiff: Generative Modeling of Video Summarization with Diffusion](summdiff_generative_modeling_of_video_summarization_with_diffusion.md)
- [\[ICCV 2025\] Understanding Flatness in Generative Models: Its Role and Benefits](understanding_flatness_in_generative_models_its_role_and_benefits.md)
- [\[ICCV 2025\] CaO2: Rectifying Inconsistencies in Diffusion-Based Dataset Distillation](cao2_rectifying_inconsistencies_in_diffusion-based_dataset_distillation.md)
- [\[NeurIPS 2025\] DeCaFlow: A Deconfounding Causal Generative Model](../../NeurIPS2025/image_generation/decaflow_a_deconfounding_causal_generative_model.md)
- [\[ICCV 2025\] Generative Modeling of Shape-Dependent Self-Contact Human Poses](generative_modeling_of_shape-dependent_self-contact_human_poses.md)

</div>

<!-- RELATED:END -->
