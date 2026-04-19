---
title: >-
  [论文解读] CAMILA: Context-Aware Masking for Image Editing with Language Alignment
description: >-
  [NeurIPS 2025][图像生成][上下文感知编辑] 提出 CAMILA，一种上下文感知的图像编辑方法，利用多模态大语言模型（MLLM）自动判断指令是否可在给定图像上执行，生成 [MASK]/[NEG] 专用 token 区分可编辑区域和应忽略区域，实现精准多指令编辑并有效过滤不可执行指令。
tags:
  - NeurIPS 2025
  - 图像生成
  - 上下文感知编辑
  - 多指令图像编辑
  - 指令可执行性判断
  - MLLM
  - 扩散模型编辑
---

# CAMILA: Context-Aware Masking for Image Editing with Language Alignment

**会议**: NeurIPS 2025  
**arXiv**: [2509.19731](https://arxiv.org/abs/2509.19731)  
**代码**: 无  
**领域**: 图像编辑  
**关键词**: 上下文感知编辑, 多指令图像编辑, 指令可执行性判断, MLLM, 扩散模型编辑

## 一句话总结

提出 CAMILA，一种上下文感知的图像编辑方法，利用多模态大语言模型（MLLM）自动判断指令是否可在给定图像上执行，生成 [MASK]/[NEG] 专用 token 区分可编辑区域和应忽略区域，实现精准多指令编辑并有效过滤不可执行指令。

## 研究背景与动机

文本引导图像编辑已成为内容创作的关键工具，但现有模型（如 InstructPix2Pix、MGIE、SmartEdit）存在一个根本缺陷：它们试图执行所有用户指令，即使指令对当前图像不可行或自相矛盾。例如，用户要求"移除盘子里的煎饼"但图像中根本没有煎饼，模型仍会尝试操作，产生不合理的输出。

现有方法的局限集中在三个层面：(1) CLIP 等简单文本编码器无法理解复杂多步指令的细粒度语义；(2) 基于 cross-attention 的区域定位（如 FoI）经常对齐错误——当修改涉及空间关系或非关键词关联区域时尤其严重；(3) 依赖 GPT 解析或重组指令会增加管线复杂度并传播中间错误。CAMILA 首次将"指令可执行性判断"显式纳入编辑流程，填补了这一研究空白。

## 方法详解

### 整体框架

CAMILA 由三个核心模块串联：(1) MLLM 联合处理图像和指令，输出 [MASK] 或 [NEG] token 序列；(2) Token Broadcaster 将 token 与扩散模型的文本嵌入对齐；(3) Token Decoder 将 [MASK] token 解码为二值掩码。最终掩码用于调制扩散模型的 cross-attention，实现精准区域编辑。

### 关键设计

1. **[MASK]/[NEG] 双 token 机制**: MLLM（LLaVA-7B）联合处理图像 $x_{\text{img}}$ 和文本指令 $x_{\text{txt}}$，输出 token 序列 $\mathcal{O} = \{\mathcal{O}_1, ..., \mathcal{O}_n\}$，每个 token 被分类为 [MASK]（标记需修改区域）或 [NEG]（标记不应修改区域/不可执行指令）。这是上下文感知的核心——模型不仅理解指令内容，还判断其在给定图像上的可行性。[NEG] token 直接映射为全黑掩码，完全抑制对应区域的编辑。

2. **Token Broadcaster 跨空间对齐**: MLLM 输出 token 和扩散模型的文本嵌入 $c_T$ 处于不同潜空间，需要对齐。通过可训练的投影矩阵 $W_O, W_T$ 将两者映射到共享空间，计算余弦相似度矩阵 $S_{i,j}$，然后对每个文本嵌入 $j$ 找到在 MLLM token 中最佳匹配的索引 $\alpha_j = \arg\max_i \text{softmax}(S_{i,j})$。这保证编辑掩码与扩散模型的文本条件精确匹配，是将 MLLM 理解能力传递到扩散编辑过程的关键桥梁。

3. **Cross-Attention 掩码调制**: 将所有 token 的二值掩码拼接为统一掩码 $\mathcal{M}$，用于调制 U-Net 中 16×16 分辨率的 cross-attention 层。编辑区域使用完整的文本+图像条件得分 $\mathcal{X}$，非编辑区域使用仅图像条件得分 $\mathcal{Y}$：$\mathcal{A}' = \text{softmax}(\mathcal{X} \odot \mathcal{M} + \mathcal{Y} \odot (1 - \mathcal{M}) / \sqrt{d})$。这确保非目标区域保持原始图像特征不被修改。

### 损失函数 / 训练策略

主训练阶段使用四项损失联合优化：

$$\mathcal{L}_{\text{main}} = \lambda_1 \mathcal{L}_{\text{CE}}^{\text{token}} + \lambda_2 \mathcal{L}_{\text{CE}}^{\text{broadcast}} + \lambda_3 \mathcal{L}_{\text{dice}} + \lambda_4 \mathcal{L}_{\text{BCE}}$$

- $\mathcal{L}_{\text{CE}}^{\text{token}}$: token 分类损失（[MASK]/[NEG] 分类）
- $\mathcal{L}_{\text{CE}}^{\text{broadcast}}$: 广播对齐损失（token-嵌入映射）
- $\mathcal{L}_{\text{dice}}$: 掩码重叠损失（空间精度）
- $\mathcal{L}_{\text{BCE}}$: 像素级二值交叉熵（掩码细粒度）

额外设计 Surrogate Module：一个单层 Transformer 近似 CLIP-T 分数，用于间接优化掩码质量。先训练 Surrogate 拟合真实 CLIP-T，再以其预测值作为信号微调 MLLM+Broadcaster+Decoder。$\lambda_1=\lambda_2=\lambda_3=\lambda_4=1$，$\lambda_5=10$。使用 LoRA 微调 MLLM，冻结视觉骨干和文本编码器。2×A100 80GB 训练约 3 天。

## 实验关键数据

### 主实验

多指令编辑和上下文感知编辑（MagicBrush 扩展数据集）：

| 方法 | L1↓ | L2↓ | CLIP-I↑ | DINO↑ | CLIP-T↑ |
|------|-----|-----|---------|-------|---------|
| IP2P | 0.1460 | 0.0514 | 0.7975 | 0.6429 | 0.2715 |
| MGIE | 0.1592 | 0.0750 | 0.8090 | 0.6519 | 0.2637 |
| SmartEdit | 0.1111 | 0.0495 | 0.8739 | 0.7726 | 0.2824 |
| FoI | 0.0891 | 0.0284 | 0.8895 | 0.8190 | 0.2888 |
| **CAMILA** | **0.0661** | **0.0222** | **0.9296** | **0.8932** | **0.3006** |

PickScore（人类偏好）上 CAMILA 超过 FoI 24%（上下文感知任务）。

### 消融实验

| 模块/变体 | L1↓ | CLIP-I↑ | DINO↑ | CLIP-T↑ |
|----------|-----|---------|-------|---------|
| 无 Surrogate（Multi） | 0.0957 | 0.8961 | 0.8329 | 0.2975 |
| 有 Surrogate（Multi） | 0.0945 | 0.8980 | 0.8392 | 0.2984 |
| 无 Surrogate（Context-Aware） | 0.0673 | 0.9284 | 0.8910 | 0.3002 |
| 有 Surrogate（Context-Aware） | 0.0661 | 0.9296 | 0.8932 | 0.3006 |

用 SAM 替代 Token Decoder 的实验表明，自训练的 Token Decoder 在 CLIP 和 DINO 分数上更优，因为它结合了编辑指令信息，而 SAM 缺乏编辑意图的感知。

### 关键发现

- Token 分类准确率达 90.21%，表明 MLLM 有效区分可执行/不可执行指令
- 掩码质量（IoU 0.3819, Dice 0.4986）虽非像素级精确，但作为高层引导足够——编辑重点是语义保真而非严格空间匹配
- 在 EMU 数据集上，CAMILA 的 CLIP-dir 在上下文感知任务中最高，证明对语义编辑方向的把控最为精准
- CAMILA MLLM 推理仅 0.7s，总推理 9.2s，与 FoI (9.1s) 相当

## 亮点与洞察

- 首次将"指令可执行性判断"显式建模为图像编辑的一等任务，开创"上下文感知图像编辑"新范式
- [MASK]/[NEG] 双 token 设计简洁优雅——用离散分类替代连续注意力权重，决策边界更清晰
- Surrogate Module 巧妙绕过扩散模型多步前向传播不可微的难题，用可学习近似器间接优化掩码质量
- 不依赖 GPT 做指令解析（FoI 需要），减少管线复杂度和错误传播

## 局限与展望

- Token Decoder 的掩码定位有时不精确——特别是添加物体时掩码区域小于预期，因为底层 IP2P 扩散模型未针对掩码编辑优化
- 基础扩散模型（Stable Diffusion）的编辑能力限制了最终效果的上限
- 训练数据依赖 ChatGPT-4V 生成不可执行指令，可能引入偏差
- 仅在 IP2P 框架上验证，尚未整合到更新的编辑框架（如 FLUX 编辑器）
- 上下文感知评估数据集规模有限（约 2600 样本）

## 相关工作与启发

- **FoI**: 最强竞争者，使用 cross-attention 做多指令编辑，但依赖 GPT 提取关键词且注意力图常对齐错误
- **SmartEdit**: 用 MLLM 增强指令理解，但缺乏上下文感知能力，所有指令平等处理
- **LISA / GSVA**: 参考分割 MLLM 中的 [SEG] token 设计，启发了 CAMILA 的 [MASK]/[NEG] 方案
- 核心启发：编辑模型不应盲目服从所有指令，"拒绝"不合理操作本身就是智能的体现

## 评分

⭐⭐⭐⭐ — 新任务定义（上下文感知编辑）有实际价值，[MASK]/[NEG] 设计简洁有效，在多指令场景显著超越 FoI。缺点是依赖较旧的 IP2P 基础框架，且掩码精度有待提升。

<!-- RELATED:START -->

## 相关论文

- [ICEdit: Enabling Instructional Image Editing with In-Context Generation in Large Scale Diffusion Transformer](in-context_edit_enabling_instructional_image_editing_with_in-context_generation_.md)
- [SAO-Instruct: Free-form Audio Editing using Natural Language Instructions](sao-instruct_free-form_audio_editing_using_natural_language_instructions.md)
- [Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking](beyond_masked_and_unmasked_discrete_diffusion_models_via_par.md)
- [A Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking](a_diffusion_model_for_regular_time_series_generation_from_irregular_data_with_co.md)
- [Describe, Don't Dictate: Semantic Image Editing with Natural Language Intent](../../ICCV2025/image_generation/describe_dont_dictate_semantic_image_editing_with_natural_language_intent.md)

<!-- RELATED:END -->
