---
title: >-
  [论文解读] SAO-Instruct: Free-form Audio Editing using Natural Language Instructions
description: >-
  [NeurIPS 2025][图像生成][音频编辑] 提出SAO-Instruct，首个支持完全自由格式自然语言指令的音频编辑模型，通过Prompt-to-Prompt、DDPM反演和手动编辑三条流水线构建编辑三元组训练数据，微调Stable Audio Open实现保持上下文一致的定向音频修改。 生成式音频模型在合成高保真…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "音频编辑"
  - "自然语言指令"
  - "Stable Audio Open"
  - "提示学习"
  - "扩散模型"
---

# SAO-Instruct: Free-form Audio Editing using Natural Language Instructions

**会议**: NeurIPS 2025  
**arXiv**: [2510.22795](https://arxiv.org/abs/2510.22795)  
**代码**: [GitHub](https://eth-disco.github.io/sao-instruct)  
**领域**: 图像生成  
**关键词**: 音频编辑, 自然语言指令, Stable Audio Open, Prompt-to-Prompt, 扩散模型

## 一句话总结

提出SAO-Instruct，首个支持完全自由格式自然语言指令的音频编辑模型，通过Prompt-to-Prompt、DDPM反演和手动编辑三条流水线构建编辑三元组训练数据，微调Stable Audio Open实现保持上下文一致的定向音频修改。

## 研究背景与动机

生成式音频模型在合成高保真音频方面已取得显著进步，但**音频编辑**仍是远未解决的挑战。现有方法存在几个关键瓶颈：

**零样本反演方法**（如ZETA）需要提供完整的目标音频描述，但用简洁文本准确描述音频的独特声学特征非常困难

**AUDIT等监督方法**只支持预定义的编辑操作集（添加、删除、替换、修复、超分辨率），无法处理灵活多样的用户指令

**编辑粒度问题**：用户可能说"让鸟叫声更响"、"加混响"或"去掉人声"，这些指令在范围和复杂度上差异极大，不适合硬编码分类

核心动机：**允许用户仅用一条自由格式的自然语言指令即可编辑音频**，系统自动理解编辑意图并精确执行，同时保留原始音频的背景上下文。

## 方法详解

### 整体框架

三阶段流水线：(1) 用LLM从输入描述生成（输入描述、编辑指令、输出描述）三元组；(2) 用三种互补方法生成对应的音频编辑样本对；(3) 微调Stable Audio Open为SAO-Instruct。推理时，模型接收输入音频+自由格式编辑指令，输出编辑后的音频。

### 关键设计

1. **Prompt-to-Prompt音频合成（全合成数据）**

   将图像域的Prompt-to-Prompt方法适配到音频域。核心思路：在用输出描述生成音频时，注入来自输入描述的注意力图（cross-attention map），从而实现局部编辑同时保持整体上下文一致。

   三个关键参数控制编辑强度：
    - $\lambda_{\text{frac}}^{\text{attn}}$：注意力注入比例（0=无影响，1=完全相同）
    - $\lambda_{\text{delay}}^{\text{attn}}$：注入延迟（跳过前N%的注意力图）
    - $\lambda_{\text{weight}}^{\text{attn}}$：对变化token的注意力权重增强

   由于不同编辑需要不同参数配置，使用**贝叶斯优化**（10次试验/样本）自动搜索最优配置，目标函数为：

    $\mathcal{L}_{\text{obj}} = \omega_1 \cdot M_{\text{CLAP}}^{\text{out}} + \omega_2 \cdot M_{\text{CLAP}}^{\text{dir}} + \omega_3 \cdot M_{\text{CLAP}}^{\text{sim}} - \omega_4 \cdot M_{\text{MEL}}^{\text{sim}}$

   权重通过小规模人类听测的ELO排名确定：$\omega_1=8, \omega_2=14, \omega_3=0.5, \omega_4=1.5$。

   此外设计了候选搜索流程：先用7种不同种子/CFG组合生成音频对，用Gemini 2.0 Flash做感知质量评估（阈值6分），再用CLAP相似度从通过筛选的候选中选最优。

2. **DDPM反演（半合成数据）**

   输入真实音频通过DDPM反演编码到潜空间，再用修改后的描述引导去噪生成编辑音频。关键参数 $T_{\text{start}}$ 控制反演深度（低值=高一致性/低编辑灵活度，高值反之）。同样用贝叶斯优化（7次试验/样本）自动调参。优势：输入音频是真实的，增加了数据多样性。

3. **手动编辑（全真实数据）**

   实现12种确定性音频编辑操作：ADD、REPLACE、DROP、SWAP、LOOP、PITCH、SPEED、LOW_PASS、HIGH_PASS、INPAINT、SUPER_RES、DENOISE。对每个操作用GPT-4.1 mini生成自然语言指令，并通过两阶段后处理（变体改写+精简压缩，各50%概率）增加指令多样性。

### 损失函数 / 训练策略

基于Stable Audio Open的扩散目标进行微调。模型使用三种条件：(1) 文本条件替换为自由格式编辑指令；(2) 时间条件设为输入音频长度；(3) 输入音频编码后拼接到模型输入通道。推理时对输入音频潜表示加入高斯噪声作为去噪初始点，100步去噪，CFG值5。

## 实验关键数据

### 消融实验 —— 不同数据源的贡献

| 训练数据 | 样本数 | FD(原始)↓ | FD(重生)↓ | IS↑ | CLAP↑ |
|----------|--------|-----------|-----------|-----|-------|
| Prompt-to-Prompt | 50k | 18.71 | 18.29 | 7.94 | 0.38 |
| DDPM反演 | 50k | 20.50 | 20.72 | 6.82 | 0.34 |
| 手动编辑 | 50k | **14.60** | 21.21 | 7.50 | 0.35 |
| 混合 | 50k | 19.11 | 19.24 | 7.69 | 0.38 |
| **混合-大** | **150k** | 18.38 | **18.97** | 7.59 | **0.38** |

### 与基线对比

| 模型 | 推理时间↓ | FD(原始)↓ | CLAP↑ | 质量MOS↑ | 相关性MOS↑ | 保真度MOS↑ |
|------|-----------|-----------|-------|----------|------------|------------|
| AudioEditor | 79.49s | 17.21 | **0.48** | 3.22 | 3.33 | 2.75 |
| ZETA ($T=50$) | 24.65s | **15.31** | 0.38 | 3.56 | 3.25 | 2.95 |
| ZETA ($T=75$) | 27.91s | 17.78 | 0.36 | 3.28 | 3.04 | 2.75 |
| **SAO-Instruct** | **9.94s** | 18.38 | 0.38 | 3.54 | **3.83** | **3.99** |

### 关键发现

1. **主观评测全面领先**：在编辑相关性（3.83 vs 3.33）和保真度（3.99 vs 2.95）上显著优于所有基线，且仅需自由格式编辑指令（基线需完整音频描述）
2. **推理效率最高**：9.94秒/样本，比AudioEditor快约8倍
3. **三种数据源互补**：Prompt-to-Prompt擅长编辑精确度（低重生FD），手动编辑保真度最高（低原始FD），混合使用平衡两者优势
4. **150k样本略优于50k**：扩大数据量在大多数指标上有边际提升，表明数据规模仍有扩展空间
5. **信息不对称下仍竞争**：SAO-Instruct只使用编辑指令，基线使用完整目标描述，但主观评测仍然更优

## 亮点与洞察

1. **首个全自由格式音频编辑**：突破了预定义操作集的限制，用户可以用任意自然语言描述编辑意图
2. **精巧的数据引擎设计**：全合成+半合成+全真实三条数据流水线互补，贝叶斯优化自动调参避免人工调优
3. **指令多样化策略**：三阶段指令生成（初始→变体→精简）有效模拟真实用户的多样化表达方式
4. **目标函数权重的人在回路设计**：通过小规模听测的ELO排名确定权重，比手动设计更可靠

## 局限与展望

- 数据生成流水线计算开销大（贝叶斯优化×多次去噪×多样本）
- 受底层Stable Audio Open生成质量限制，某些复杂场景可能失败
- 仅支持英语指令和通用音频，未扩展到音乐编辑
- 未支持多步编辑（一次只能执行一条指令）
- Prompt-to-Prompt注意力注入在复杂场景下保真度有限

## 相关工作与启发

- **音频生成**: Stable Audio Open, AudioLDM, AudioGen
- **音频编辑**: AUDIT（预定义操作）, ZETA（零样本反演）, AudioEditor
- **图像编辑启发**: InstructPix2Pix（指令微调）, Prompt-to-Prompt（注意力注入）

## 评分

- 新颖性: ⭐⭐⭐⭐☆ — 首个全自由格式音频编辑模型，但核心技术借鉴图像编辑
- 实验充分度: ⭐⭐⭐⭐⭐ — 消融+对比+主观听测，评估维度全面
- 写作质量: ⭐⭐⭐⭐☆ — 方法描述详尽，流水线清晰
- 价值: ⭐⭐⭐⭐☆ — 填补音频编辑的重要空白，开源模型可直接使用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Describe, Don't Dictate: Semantic Image Editing with Natural Language Intent](../../ICCV2025/image_generation/describe_dont_dictate_semantic_image_editing_with_natural_language_intent.md)
- [\[ICML 2025\] FlexiClip: Locality-Preserving Free-Form Character Animation](../../ICML2025/image_generation/flexiclip_locality-preserving_free-form_character_animation.md)
- [\[CVPR 2026\] Language-Free Generative Editing from One Visual Example](../../CVPR2026/image_generation/language-free_generative_editing_from_one_visual_example.md)
- [\[NeurIPS 2025\] CAMILA: Context-Aware Masking for Image Editing with Language Alignment](camila_contextaware_masking_for_image_editing_with_language.md)
- [\[NeurIPS 2025\] DiffEye: Diffusion-Based Continuous Eye-Tracking Data Generation Conditioned on Natural Images](diffeye_diffusion-based_continuous_eye-tracking_data_generation_conditioned_on_n.md)

</div>

<!-- RELATED:END -->
