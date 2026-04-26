---
title: >-
  [论文解读] MacPrompt: Maraconic-guided Jailbreak against Text-to-Image Models
description: >-
  [AAAI 2026][图像生成][文本到图像安全] 提出 MacPrompt，一种黑盒跨语言攻击方法，通过将有害词汇翻译为多语言候选并进行字符级重组构造"通心粉词（macaronic words）"作为对抗 prompt，能够同时绕过文本安全过滤器和概念移除防御，在色情内容上攻击成功率高达 92%，在暴力内容上达 90%。
tags:
  - AAAI 2026
  - 图像生成
  - 文本到图像安全
  - 越狱攻击
  - 跨语言对抗
  - 通心粉词
  - 概念移除
---

# MacPrompt: Maraconic-guided Jailbreak against Text-to-Image Models

**会议**: AAAI 2026  
**arXiv**: [2601.07141](https://arxiv.org/abs/2601.07141)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 文本到图像安全, 越狱攻击, 跨语言对抗, 通心粉词, 概念移除

## 一句话总结

提出 MacPrompt，一种黑盒跨语言攻击方法，通过将有害词汇翻译为多语言候选并进行字符级重组构造"通心粉词（macaronic words）"作为对抗 prompt，能够同时绕过文本安全过滤器和概念移除防御，在色情内容上攻击成功率高达 92%，在暴力内容上达 90%。

## 研究背景与动机

文本到图像（T2I）模型（如 Stable Diffusion、DALL·E、Midjourney）在创意设计中广泛应用，但由于训练数据来自互联网未经严格过滤，存在生成 NSFW（Not Safe for Work）内容的风险。

**现有防御体系**：

**安全过滤器（Safety Filters）**：
   - 文本过滤器：基于关键词黑名单匹配（text-match）或 BERT 分类器（text-classifier）
   - 图像过滤器：检测生成图像中的 NSFW 内容
   - 潜空间过滤器：如 LatentGuard

**概念移除（Concept Removal）**：直接修改模型权重，消除 NSFW 概念（如 ESD、SLD、FMN、SafeGen 等）

**现有攻击的局限**：
- 大多数攻击只能绕过一类防御（过滤器或概念移除），无法同时攻破两者
- 能同时绕过两者的攻击通常需要模型内部信息（白盒/灰盒），不实用
- 基于同义词替换的方法（如 DiffZOO）容易被语义匹配过滤器检测

**核心洞察**：T2I 模型虽然主要以英文训练，但其他语言的 prompt 也能触发相似的视觉语义。更重要的是，某些跨语言组合词能保留视觉语义同时在文本语义上显著偏离原始有害词，成功绕过安全过滤。

## 方法详解

### 整体框架

MacPrompt 的整体流程：

1. **敏感词识别**：在原始有害 prompt 中检测敏感词
2. **跨语言候选选择**：将敏感词翻译为 79 种语言，筛选最有效候选
3. **通心粉替代词构造**：通过参数化字符级重组，从多语言候选中构造对抗替代词
4. **零阶优化**：以 NSFW 检测分数为反馈信号，迭代优化构造参数
5. **对抗 prompt 生成**：用替代词替换原始敏感词

### 关键设计

#### 1. **敏感词识别**

采用两种策略：
- **黑名单匹配**：与预定义有害词列表对比
- **语义相似度打分**：计算每个词嵌入与有害概念嵌入的余弦相似度，超过阈值 $\tau$ 即标记为敏感：

$$\max_j \cos(\text{Embed}(w_i), e_{harm}^j) > \tau$$

#### 2. **跨语言候选选择**

对每个敏感词 $w_{\lambda_i}$：

1. 使用 LLM 翻译为 $L=79$ 种语言，生成候选池 $V^{(\lambda_i)}$
2. 将每个候选插入模板 prompt，生成 10 张图像
3. 用两个指标评估候选质量：
    - **有害性分数** $\mathcal{H}$：NSFW 检测器输出的目标类概率
    - **视觉语义相似度** CLIPSim：候选生成图像与安全 prompt 图像的 CLIP 分数
4. 按综合分数排名，保留 top-$k$（$k=10$）候选

#### 3. **通心粉替代词构造（核心创新）**

由于 T2I 模型的 tokenizer 对非英文语言的**不可逆性**（$\epsilon(\epsilon^{-1}(\epsilon(v))) \neq \epsilon(v)$），无法直接在 token 层面操控。因此提出**字符级操控**策略。

对每个敏感词 $w_{\lambda_i}$ 的 $k$ 个候选，定义三组参数：

- **边界参数** $\beta_1^{(\lambda_i)}, \beta_2^{(\lambda_i)} \in [0,1]^k$：控制从每个候选中提取子串的起止位置
- **排序参数** $\alpha^{(\lambda_i)} \in \mathbb{R}^k$：控制子串的拼接顺序

子串提取位置计算：

$$\mu_{1,j}^{(\lambda_i)} = \lfloor l_j \cdot \beta_{1,j}^{(\lambda_i)} \rfloor, \quad \mu_{2,j}^{(\lambda_i)} = \lfloor l_j \cdot \beta_{2,j}^{(\lambda_i)} \rfloor$$

从候选词中提取子串 $\bar{v}_j^{(\lambda_i)} = \hat{v}_j(\mu_{1,j}:\mu_{2,j})$，按 $\alpha$ 降序排列后拼接形成通心粉替代词。

例如："nudity"可能被替换为"nuditéudenakt"（融合法语、德语等字符片段）。

#### 4. **零阶优化（ZOO）**

以 NSFW 检测概率为目标函数：

$$\mathcal{L} = \|\mathcal{H}(p_{adv}) - \mathbf{1}\|_2$$

通过有限差分近似梯度：

$$\nabla_{\beta_r}\mathcal{L} \approx \frac{\mathcal{L}(\beta_r + \delta) - \mathcal{L}(\beta_r - \delta)}{2\delta}$$

学习率 0.1，迭代 100 次，扰动幅度初始值 $\delta_0 = 0.25$，支持早停。

### 损失函数 / 训练策略

本方法是推理时的对抗攻击方法，不涉及模型训练。优化过程完全在黑盒设定下进行，仅依赖生成图像的反馈。

## 实验关键数据

### 主实验

**NSFW 概念生成攻击性能**（色情类）：

| 方法 | 黑名单BPR | LatentGuard BPR | BERT BPR | SD ASR-5 | ESD ASR-5 | SLD.Ma ASR-5 | SafeGen ASR-5 | FMN ASR-5 |
|-----|----------|----------------|---------|---------|----------|-------------|-------------|----------|
| DACA | 94% | 98% | 72% | 40% | 36% | 34% | 34% | 36% |
| PGJ | 96% | 98% | 54% | 38% | 46% | 54% | 62% | 50% |
| SurPro | 100% | 94% | 76% | 52% | 60% | 68% | 68% | 48% |
| DiffZOO | 52% | 56% | 36% | 52% | 50% | 74% | 66% | 28% |
| **MacPrompt** | **100%** | 82% | 70% | **96%** | **74%** | **96%** | **88%** | **76%** |

**暴力类**：

| 方法 | 黑名单BPR | SD ASR-5 | ESD ASR-5 | SafeGen ASR-5 | FMN ASR-5 |
|-----|----------|---------|----------|-------------|----------|
| DACA | 78% | 85% | 72% | 80% | 80% |
| DiffZOO | 48% | 66% | 66% | 40% | 70% |
| **MacPrompt** | **100%** | **72%** | **74%** | **90%** | **74%** |

**语义一致性**（CLIPScore/BLIPScore）：

| 内容类型 | 原始↔对抗prompt CLIPScore | 原始↔对抗图像 CLIPScore | 原始prompt↔对抗图像 BLIPScore |
|---------|------------------------|----------------------|---------------------------|
| Sex | 0.8768 | 0.7893 | 0.5602 |
| Violence | 0.8618 | 0.8012 | 0.5893 |
| Dog | 0.9223 | 0.8597 | 0.9572 |
| Car | 0.9348 | 0.7335 | 0.5047 |

### 消融实验

**禁止对象生成攻击（Banned Objects）**：

| 对象 | 方法 | SD ASR-1/5 | ESD ASR-1/5 | FMN ASR-1/5 | EAP ASR-1/5 |
|-----|-----|-----------|------------|------------|------------|
| Dog | MMP-Attack | 66/90 | 78/88 | 60/90 | 52/94 |
| Dog | **MacPrompt** | **96/100** | 64/88 | **78/98** | 46/88 |
| Car | MMP-Attack | 76/84 | 52/86 | 70/88 | 62/92 |
| Car | **MacPrompt** | **92/100** | 50/94 | **86/98** | 60/96 |

### 关键发现

1. **跨语言字符重组能有效保留视觉语义同时规避文本检测**：通心粉词在文本嵌入空间中远离原始词，但在图像嵌入空间中生成图像高度一致
2. **对概念移除防御尤其有效**：MacPrompt 对被认为更强的概念移除模型攻击成功率极高（SD 上 ASR-5 达 96%）
3. **强迁移性**：单个对抗 prompt 可同时攻破多个防御模型
4. **对商用系统也有效**：在 DALL·E 3 上 ASR 65%，在 Doubao 上 ASR 96%
5. **平均 BLIPScore 0.6953**，显著优于 MMP-Attack 的 0.414，表明更好的语义一致性

## 亮点与洞察

- **新颖的攻击向量**：利用 T2I 模型隐含的多语言能力，发现了全新的安全漏洞
- **字符级操控**：巧妙解决了 tokenizer 不可逆性问题，实现比 token 级操控更细粒度的控制
- **实用性强**：完全黑盒，不需要模型内部信息，可应用于任何 T2I 系统
- **揭示系统性安全缺陷**：当前所有 T2I 安全机制（从简单关键词匹配到SOTA概念移除）都无法有效应对多语言对抗
- **通心粉词的可视化分析**很有说服力：文本空间分离但图像空间聚集

## 局限与展望

- 需要对 79 种语言候选进行大量图像生成，计算成本较高
- ZOO 优化的迭代次数固定为 100，对不同难度的敏感词可能需要自适应
- 防御方向：论文启发了对多语言鲁棒性的重新思考——未来的安全过滤器应考虑跨语言 token 共享和视觉语义级别的检测
- 伦理问题：虽然作者做了负责任的披露，但方法的公开可能被恶意利用

## 相关工作与启发

本文位于 T2I 安全攻防研究前沿：
- **防御侧**：从简单黑名单 → BERT 分类器 → LatentGuard → 概念移除（ESD/SLD/FMN/SafeGen/DUO/EAP/PromptGuard）的防御演进
- **攻击侧**：从白盒（Prompting4Debugging）→ 灰盒（P4D）→ 黑盒（DACA/DiffZOO/PGJ/SurrogatePrompt）→ 本文的跨语言黑盒攻击

核心启发：**T2I 安全机制不能局限于单语言假设**，需要在视觉语义层面而非文本层面构建防御。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐（跨语言字符级重组的攻击方式前所未见）
- 实验充分度: ⭐⭐⭐⭐⭐（横跨3种过滤器+9种概念移除+7种基线方法）
- 写作质量: ⭐⭐⭐⭐（方法描述清晰，公式化严谨）
- 价值: ⭐⭐⭐⭐⭐（揭示当前T2I安全体系的根本性漏洞）

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Multimodal Pragmatic Jailbreak on Text-to-image Models](../../ACL2025/image_generation/multimodal_pragmatic_jailbreak_on_text-to-image_models.md)
- [\[AAAI 2026\] Creating Blank Canvas Against AI-Enabled Image Forgery](creating_blank_canvas_against_ai-enabled_image_forgery.md)
- [\[AAAI 2026\] ReAlign: Text-to-Motion Generation via Step-Aware Reward-Guided Alignment](realign_text-to-motion_generation_via_step-aware_reward-guided_alignment.md)
- [\[AAAI 2026\] Melodia: Training-Free Music Editing Guided by Attention Probing in Diffusion Models](melodia_training-free_music_editing_guided_by_attention_probing_in_diffusion_mod.md)
- [\[AAAI 2026\] DogFit: Domain-guided Fine-tuning for Efficient Transfer Learning of Diffusion Models](dogfit_domain-guided_fine-tuning_for_efficient_transfer_learning_of_diffusion_mo.md)

<!-- RELATED:END -->
