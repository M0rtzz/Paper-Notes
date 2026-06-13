---
title: >-
  [论文解读] AutoPrompt: Automated Red-Teaming of Text-to-Image Models via LLM-Driven Adversarial Prompts
description: >-
  [ICCV 2025][图像生成][Red-Teaming] 本文提出APT（AutoPrompT），一种基于LLM的黑盒红队测试框架，通过"优化-微调"交替训练管线和双规避策略，自动生成可被人类阅读且不被内容过滤器拦截的对抗性后缀，有效突破T2I模型的安全机制，并具有零样本跨提示迁移能力。 文生图（T2I）扩散模型在大规模…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "Red-Teaming"
  - "text-to-image"
  - "提示学习"
  - "LLM"
  - "Safety Evaluation"
---

# AutoPrompt: Automated Red-Teaming of Text-to-Image Models via LLM-Driven Adversarial Prompts

**会议**: ICCV 2025  
**arXiv**: [2510.24034](https://arxiv.org/abs/2510.24034)  
**代码**: 无  
**领域**: 图像生成 / AI安全  
**关键词**: Red-Teaming, text-to-image, Adversarial Prompts, LLM, Safety Evaluation

## 一句话总结
本文提出APT（AutoPrompT），一种基于LLM的黑盒红队测试框架，通过"优化-微调"交替训练管线和双规避策略，自动生成可被人类阅读且不被内容过滤器拦截的对抗性后缀，有效突破T2I模型的安全机制，并具有零样本跨提示迁移能力。

## 研究背景与动机
文生图（T2I）扩散模型在大规模多模态学习中取得了空前的生成能力，但同时继承了不受控数据收集带来的安全风险——精心构造的对抗性提示可以诱导生成不安全内容（NSFW）。现有的安全机制包括训练数据过滤、NSFW安全检查器、推理引导和概念擦除微调等，但其有效性和鲁棒性缺乏标准化的自动化评估。

现有红队测试方法存在三大关键局限：

**白盒依赖**：大多数方法（Ring-A-Bell、P4D、UnlearnDiffAtk）需要目标模型的梯度信息，在实际场景中不可行

**语义不可读**：基于离散优化的方法生成的对抗提示是无意义的字符拼接（"gibberish"），容易被困惑度过滤器检测和拦截

**包含违禁词**：生成的对抗提示经常显式包含黑名单中的敏感词汇，被词过滤器直接拦截

核心创新点：利用LLM的自然语言生成能力，在纯黑盒设定下自动生成**人类可读**且**不被拦截**的对抗性后缀。

## 方法详解

### 整体框架
APT采用"优化-微调"交替训练策略。优化阶段冻结LLM，通过随机beam search逐token优化对抗后缀；微调阶段用优化得到的后缀作为目标微调LLM。双规避策略贯穿优化阶段，确保生成的对抗提示同时绕过困惑度过滤器和黑名单词过滤器。

### 关键设计

1. **对抗后缀优化**:

    - 功能：为给定的良性提示 $x$ 生成对抗后缀 $S_T = [s_1, \ldots, s_T]$，使拼接后的提示 $[x, S_T]$ 诱导T2I模型生成不安全内容
    - 核心思路：
        - **对齐约束**：$\ell_{align}(x, S_t) = \text{sim}(\mathcal{G}([x, S_t]), I) + \frac{1}{|c|} \sum_{w \in \mathcal{W}} \text{sim}(\mathcal{G}([x, S_t]), w)$，其中第一项将生成图像与不安全图像对齐，第二项与不安全文本概念对齐
        - **随机beam search**：每步从LLM预测分布中采样 $k=12$ 个候选token，保留目标函数最低的 $b=4$ 个beam，迭代至最大长度 $T=15$
        - **先验后缀**：在良性提示后附加先验后缀（如"and a beautiful girl's body with"），为LLM提供上下文引导
    - 设计动机：逐token优化允许精细控制每个生成步骤，结合LLM的语言先验保证生成质量

2. **双规避策略**:

    - 功能：使生成的对抗提示同时绕过困惑度过滤器和黑名单词过滤器
    - 核心思路：
        - **困惑度约束**：引入辅助预训练LLM $\mathcal{M}_\phi$ 计算困惑度：$\ell_{per}(S_t|x) = -\sum_{t=1}^T \log p_\phi(s_t | [x, S_{t-1}])$，整合为越狱约束：$\min_{S_T} \mathcal{L}_{jai} = -\ell_{align} + \lambda \ell_{per}$
        - **禁令token惩罚**：扫描tokenizer词表，识别与不安全词汇 $\mathcal{W}$ 语义相似度超过阈值的token，在预测时对其概率施加惩罚。额外检查多token组合可能拼成禁词的情况（取每个beam最后一个完整单词检查）
    - 设计动机：低困惑度确保可读性；禁令惩罚防止LLM走捷径直接生成敏感词

3. **后缀生成器微调**:

    - 功能：用优化得到的高质量后缀微调LLM，使其逐步学会直接生成有效后缀
    - 核心思路：将 $(x, S_T)$ 对存入回放缓冲区 $\mathcal{R}$，按成功越狱和最低 $\mathcal{L}_{jai}$ 确定优先级采样，使用交叉熵损失微调：$\mathcal{L}_{CE} = -\sum_{t=1}^T \log p_\theta(s_t | [x, S_{t-1}])$
    - 设计动机：优化阶段获得的后缀质量逐轮提升，微调使LLM内化越狱模式，最终实现零样本推理——对未见提示直接生成有效对抗后缀

### 实现细节
后缀生成器使用Llama-3.1-8B，辅助LLM也使用相同权重（冻结）。不安全图像集50张（经分类器验证），裸露相关禁词23个，暴力相关17个。良性提示截断至50个token。

## 实验关键数据

### 主实验（通过黑名单词过滤后的RSR红队成功率）

| 方法 | ESD↑ | SLD-MAX↑ | Receler↑ | AdvUnlearn↑ | 说明 |
|------|------|---------|---------|-------------|------|
| Ring-A-Bell | 2.00% | 2.50% | 1.00% | 0.50% | 白盒，几乎无效 |
| UnlearnDiffAtk | 18.50% | 52.00% | 16.50% | 3.00% | 白盒 |
| P4D-Union | 41.50% | 62.50% | 41.50% | 9.50% | 白盒，需梯度 |
| **APT (Ours)** | **61.50%** | **70.50%** | **36.50%** | **30.50%** | 黑盒，人类可读 |

### 消融实验（ESD模型，裸露类别）

| 配置 | RSR↑ | PPL_Avg↓ | BR↓ | 说明 |
|------|------|---------|-----|------|
| 无不安全图像对齐 | 38.5% | 0.175 | 1% | 缺乏视觉引导 |
| 无不安全词列表对齐 | 30.5% | 0.067 | 1% | 缺乏语义引导 |
| 无困惑度约束 | 35% | 0.198 | 1% | 可读性下降 |
| 无禁令token惩罚 | 9.5% | 0.171 | **87%** | 几乎全被拦截 |
| **完整APT** | **61.5%** | **0.167** | **2%** | 全部组件 |

### 关键发现
- APT的困惑度（PPL）仅为Ring-A-Bell的**1/70**（0.167 vs 11.646 ×10³），远优于所有基线
- APT的封锁率（BR）最低——裸露和暴力类别均约2%，而基线方法高达87%
- APT对AdvUnlearn的RSR达30.5%，**是P4D的3.2倍**（P4D仅9.5%），尤其在强防御下优势明显
- 跨模型迁移性强：为AdvUnlearn优化的提示在其他三个模型上均超40%成功率
- 可直接攻击SDXL、SD3.5、FLUX.1-dev等最新模型以及Leonardo.Ai等商业平台

## 亮点与洞察
- 黑盒 + 人类可读 + 不被拦截的三重约束同时满足，在实际部署中远比白盒方法有价值
- "优化-微调"交替策略使LLM逐步内化越狱模式，最终实现零样本泛化
- 回放缓冲区的优先级采样是训练稳定性的关键设计
- 禁令token惩罚的两层机制（单token级别 + 多token拼接检查）体现了工程上的完备性
- 对最新商业API的成功攻击揭示了现有安全措施的脆弱性

## 局限与展望
- 为维持低困惑度和规避过滤，可能牺牲一定的攻击强度——过于严格的禁令惩罚可能抑制语义关键token
- 先验后缀的选择目前是手工设定的（"and a beautiful girl's body with"），自动化选择可能进一步提升性能
- 每个安全T2I模型需要单独训练后缀生成器，跨防御方法的统一生成器尚未实现
- 论文聚焦裸露和暴力两类——对其他有害内容类型（仇恨、歧视等）的覆盖未探索
- 红队测试工具的发布需谨慎平衡研究价值与潜在滥用风险

## 相关工作与启发
- **vs Ring-A-Bell**: 基于遗传算法的离散优化，生成的提示困惑度极高（~11646），在过滤器下几乎完全失效
- **vs P4D**: 在连续空间优化并需要模型梯度，RSR较高但无法适配黑盒场景且提示不可读
- **vs AdvPromter**: 也是LLM驱动的方法但需要白盒梯度，APT完全黑盒化且引入了双规避策略
- **启发**：LLM的语言生成能力可以被"引导式微调"来生成特定对抗后缀，这一范式可能推广到其他安全评估任务

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 黑盒+可读+不可拦截的三重约束在T2I红队测试中首次同时实现
- 实验充分度: ⭐⭐⭐⭐ 四种安全T2I模型、最新架构和商业API、全面消融和迁移性分析
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，算法伪代码完整，对比分析到位
- 价值: ⭐⭐⭐⭐⭐ 揭示了现有T2I安全机制的根本脆弱性，为安全评估提供了实用工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] ANCHOR: LLM-driven Subject Conditioning for Text-to-Image Synthesis](../../ACL2026/image_generation/anchor_llm-driven_subject_conditioning_for_text-to-image_synthesis.md)
- [\[CVPR 2025\] FilmComposer: LLM-Driven Music Production for Silent Film Clips](../../CVPR2025/image_generation/filmcomposer_llm-driven_music_production_for_silent_film_clips.md)
- [\[ICCV 2025\] DIA: The Adversarial Exposure of Deterministic Inversion in Diffusion Models](dia_the_adversarial_exposure_of_deterministic_inversion_in_diffusion_models.md)
- [\[ICCV 2025\] PLA: Prompt Learning Attack against Text-to-Image Generative Models](pla_prompt_learning_attack_against_text-to-image_generative_models.md)
- [\[CVPR 2026\] LumiCtrl: Learning Illuminant Prompts for Lighting Control in Personalized Text-to-Image Models](../../CVPR2026/image_generation/lumictrl_learning_illuminant_prompts_for_lighting_control_in_personalized_text-t.md)

</div>

<!-- RELATED:END -->
