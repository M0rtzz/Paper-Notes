---
title: >-
  [论文解读] Evaluating and Rewarding LALMs for Expressive Role-Play TTS via Mean Continuation Log-Probability
description: >-
  [ICML 2026][音频/语音][Role-Play TTS] 本文把"预训练大音频语言模型对真值语音 token 的续写概率"包装成一个名为 MCLP 的客观风格一致性度量，再用 MCLP+CER 的门控混合奖励…
tags:
  - "ICML 2026"
  - "音频/语音"
  - "Role-Play TTS"
  - "LALM"
  - "Mean Continuation Log-Probability"
  - "GRPO"
  - "风格一致性"
---

# Evaluating and Rewarding LALMs for Expressive Role-Play TTS via Mean Continuation Log-Probability

**会议**: ICML 2026  
**arXiv**: [2601.22661](https://arxiv.org/abs/2601.22661)  
**代码**: https://github.com/y-ren16/MCLP  
**领域**: 音频/语音
**关键词**: Role-Play TTS, LALM, Mean Continuation Log-Probability, GRPO, 风格一致性  

## 一句话总结
本文把"预训练大音频语言模型对真值语音 token 的续写概率"包装成一个名为 MCLP 的客观风格一致性度量，再用 MCLP+CER 的门控混合奖励，通过 GRPO 在新构建的 WenetSpeech-RP-TTS 数据集上把角色扮演 TTS 的主观 MOS 从 1.86 推到 3.58。

## 研究背景与动机

**领域现状**：LLM 风格的 TTS（CosyVoice、VALL-E、Step-Audio 等）已经能做到很强的零样本音色克隆，最近的 Instruct-TTS 又允许用自然语言描述去控制风格；而 Speech Role-Playing Agent 类工作（OmniCharacter、SpeechRole、VoxRole 等）更进一步，希望模型在多轮对话里扮演特定人物。

**现有痛点**：在"角色扮演 TTS（RP-TTS）"这种**只控风格、不控音色**的场景下，现有方法两头不靠岸——Instruct-TTS 只能处理单句，把风格当作 utterance 级别的静态属性，无法跨多轮维持人设；Role-Playing Agent 把重心放在语义对齐而非声学风格，往往为了语义连贯牺牲表现力；想用 RL 拉齐风格又卡在**没有客观风格度量**这一根本瓶颈，只能退化到用情感分类器当代理奖励，结果只覆盖了情绪这一个维度。

**核心矛盾**：风格本身是连续、语境依赖、混杂韵律 / 情绪 / 副语言信息的"高维概念"，但现有评估和奖励都试图用离散标签（情绪类别、说话人 ID）去逼近它，必然丢信息；同时单一奖励（只看 CER 或只看相似度）极易触发 reward hacking——要么发出表现力极强但听不懂的"鬼话"，要么生成极清晰但平淡如开水的播报音。

**本文目标**：把这两件事一起解决——(1) 定义一个**可解释、连续、和人耳一致**的风格度量；(2) 把它接进 RL pipeline 的同时保住内容保真度。

**切入角度**：作者押了一个关键假设——在海量语音上预训练的 LALM **隐式学到了一个连续的语音风格隐空间**。给定一段转录文本和一段"候选语音"，如果候选的说话风格和真值一致，预训练 LALM 在用候选语音作上下文时，对真值语音 token 的续写概率就应该更高。这把"风格一致"翻译成了"续写似然"，刚好可以用一个数字量化。

**核心 idea**：把"预训练 LALM 对真值 audio token 的平均对数似然"直接当成风格度量 MCLP，既能离线评估，也能接进 GRPO 当 reward；再用 CER 做门控防止 reward hacking。

## 方法详解

### 整体框架

整个 pipeline 由三个阶段串成：(a) **MCLP 续写模型预训练**——以 Step-Audio-2 作初始化，在 300 万小时带转录语音上做"会话级"自回归训练，损失只算在 audio token 上，学到给定文本+前文语音去续写当前句子的能力；(b) **SFT 阶段**——基于 Step-Audio-2-mini-Base，在自建的 WenetSpeech-RP-TTS 上做多轮对话的有监督微调，让模型学会按场景描述 $\mathcal{S}$、角色档案 $\mathcal{P}$ 和历史 $\mathcal{H}_{<j}$ 生成第 $j$ 轮的 interleaved TA4 token 序列；(c) **GRPO 强化学习阶段**——只对最后一轮做 rollout，用一个 MCLP 风格奖励 + CER 内容惩罚 + 门控聚合的复合奖励驱动策略更新。

整个数据闭环建在 WenetSpeech 上：先按 "YouTube + drama" 标签筛出 17k 视频，下载到 8556 个，用 Demucs 去伴奏、pyannote 做说话人分离；再用 DeepSeek-R1 反推剧名/集数并基于全脚本生成角色档案 $\mathcal{P}$，用 Qwen-VL-7B 给每段场景生成环境描述 $\mathcal{S}$，5 秒静音切分场景、30 秒上限，最终留下 311k 场景 / 1435 小时，平均每场 7.3 句、涉及 2.33 位说话人；测试集严格做视频级 hold-out，留 200 视频 900 场景，按 2–10 轮分层抽样保证每个轮次 100 例。

### 关键设计

1. **MCLP 度量的"双 transcript + 反向续写"上下文设计**:

    - 功能：用预训练 LALM 的续写似然量化候选音频 $\mathbf{z}^{eval}$ 和真值音频 $\mathbf{z}^{gt}$ 之间的风格一致性。
    - 核心思路：把上下文构造成 $\mathcal{H}=[\mathbf{w},\mathbf{z}^{eval},\mathbf{w}]$——同一段转录文本 $\mathbf{w}$ 出现两次，中间夹着候选音频；然后让 LALM 在这个 $\mathcal{H}$ 之后续写 $\mathbf{z}^{gt}$，定义 $\text{MCLP}=\frac{1}{|\mathbf{z}^{gt}_A|}\sum_{k\in \mathbf{z}^{gt}_A}\log P_\theta(z_k^{gt}\mid \mathcal{H},z_{<k}^{gt})$，只在 audio token 子集上取平均。
    - 设计动机：重复 $\mathbf{w}$ 是为了在 teacher-forcing 下"钉住"文本内容，让续写似然的变化**只能来自风格信号**而非文本影响；选 Step-Audio-2 是因为它的 semantic speech tokenizer 主要保留语义和风格而非音色细节，这进一步把度量从"音色相似"偏置到"风格相似"；选用真值长度做归一化分母而不是反过来，是为了多个候选评估同一参考时长度对齐、公平可比；和情感分类奖励比，MCLP 给出一个**连续、密集、可解释**的标量，覆盖了情感、韵律、节奏等所有被预训练 LALM 编码的风格维度。

2. **门控 MCLP + CER 复合奖励防 Reward Hacking**:

    - 功能：在 GRPO 训练阶段同时优化表现力（MCLP）和清晰度（CER），用门控机制保证模型先学会"说人话"再去追求风格。
    - 核心思路：风格分支 $R_{style}=\text{MCLP}(\mathbf{z}^{roll},\mathbf{z}^{gt})+C$，加偏置 $C=15$ 把 MCLP 平移到正区间；内容分支 $R_{content}=\lambda\cdot\text{CER}(\hat{\mathbf{w}},\mathbf{w})$，$\lambda=10$，用 Step-Audio Token2Wav 把 rollout 解成波形再过 ASR 得到 $\hat{\mathbf{w}}$；最终奖励是 $R(\mathbf{z})=R_{style}-R_{content}$ 当 $\text{CER}\le\tau=0.2$，否则直接置 0。
    - 设计动机：作者通过消融观察到——只用 MCLP 奖励时 CER 会爆炸到 60%+（模型靠生成重复声学 pattern 刷分），只用 CER 时 MOS 跌到 2.33（平淡播报音）；门控等价于构造了一条**课程曲线**——必须先翻过清晰度门槛才能拿到风格分，自然避免了"表现力 gibberish"的退化路径，又让风格优化在清晰度满足之后还能持续生效。

3. **基于 GRPO 的最后一轮 RL 对齐**:

    - 功能：在 SFT 之后只对每个对话的最后一轮 token 做策略优化，利用相对优势归一化稳定训练。
    - 核心思路：对每个 query $\mathbf{q}=(\mathcal{S},\mathcal{P},\mathcal{H})$ 采样 $G=8$ 个 rollout，用 group 内的均值方差归一化得到 $\hat{A}_i=(R_i-\text{mean})/\text{std}$；目标函数是带 clip 的 importance ratio $\rho_{i,t}$ 与 $\hat{A}_i$ 的乘积加上 token 级 KL 约束 $\beta\mathbb{D}_{KL}$（$\beta=0.001$）；从 16,186 个高质量场景（2–6 轮、末句 >10 中文字符、非 Neutral 风格）里采样，迭代 1000 步，32 张 H800。
    - 设计动机：只对最后一轮做 RL 而历史保持真值，是为了避免误差累积污染上下文；GRPO 相比 PPO 不需要 critic 网络更省显存；KL 约束防止策略在风格奖励的诱惑下漂离 SFT 锚点；前面三条筛选规则共同保证 RL 批次里都是表现力提升空间大的样本，避免在情感平淡的句子上浪费 rollout 预算。

### 损失函数 / 训练策略

SFT 用 1 epoch、batch 64、学习率 $1\times 10^{-5}$ 余弦衰减、最大序列 16384、AdamW（$\beta_1=0.9, \beta_2=0.95$, weight decay 0.1, grad clip 1.0），目标是 $\theta^*=\arg\min_\theta\sum -\log P_\theta(\mathbf{y}\mid \mathcal{S},\mathcal{P},\mathcal{H},\mathcal{I})$。RL 用学习率 $1\times 10^{-6}$、global batch 128、$G=8$ rollout、temperature 1.0、max decode 1024。

## 实验关键数据

### 主实验

| 模型 | 设置 | CER↓ | CAM++↑ | Emo2Vec↑ | MCLP↑ | MOS↑ |
|------|------|------|--------|----------|-------|------|
| Ground Truth | — | — | — | — | — | 4.461 |
| GPT-Audio | w/ history | 11.97 | 0.636 | 0.875 | -4.849 | 1.915 |
| MiMo-Audio-7B | w/ history | 10.60 | 0.699 | 0.902 | -4.753 | 2.484 |
| Step-Audio-2-mini | w/ history | 3.28 | 0.629 | 0.864 | -4.829 | 1.856 |
| OV-InstructTTS | w/o history | 7.19 | 0.669 | 0.900 | -4.768 | 2.864 |
| **Ours** | w/ history | **1.13** | **0.724** | **0.917** | **-4.636** | **3.576** |
| **Ours** | w/o history | **1.63** | **0.704** | **0.910** | **-4.687** | **3.576** |

CER 几乎打到 baseline 的 1/10（GPT-Audio w/o history 44.7% → 本文 w/o history 1.63%），MOS 比最强 Instruct-TTS（OV-InstructTTS 2.864）高 0.71 分、比最强 LALM（MiMo 2.484）高 1.09 分。

### 消融实验

| 配置 | CER (w/ hist) | MCLP (w/ hist) | MOS |
|------|---------------|----------------|-----|
| Step-Audio-2-mini（baseline） | 3.28 | -4.829 | 1.856 |
| + SFT only | 3.33 | -4.725 | 3.178 |
| Full（SFT + RL with hybrid reward） | **1.13** | -4.636 | **3.576** |
| w/o CER Reward（只刷 MCLP） | 61.14 | **-4.590** | 1.145 |
| w/o MCLP Reward（只刷 CER） | **0.78** | -4.752 | 2.331 |

### 关键发现
- SFT 单独就能把 MOS 从 1.86 拉到 3.18（+1.32），证明数据集本身有用；RL 阶段再加 0.40，证明 MCLP 奖励确实带来了 SFT 之上的额外风格收益。
- 去掉 CER 奖励是最戏剧化的——MCLP 反而**最高**（-4.59），但 CER 飙到 61%、MOS 跌到 1.15，验证了作者关于"reward hacking 会生成表现力极强的胡话"的预判，门控机制是必要的而非可选。
- 只用 CER 时 CER 跌到 0.78%（比 full model 还低）但 MOS 只有 2.33，说明清晰度极限不等于体验极限；MCLP 提供的是**人耳真正在乎但 CER/ASR 看不到**的那部分质量。
- 人类配对实验显示 $\Delta\text{MCLP}>0.1$ 时 win rate 超过 0.8，说明 MCLP 作为偏好预测器在分辨力够大的情况下高度可靠。

## 亮点与洞察
- **"反向续写 + 双重 transcript"是巧妙的归一化技巧**：直觉上应该是"用 GT 当上下文预测 eval"，但作者反过来"用 eval 当上下文预测 GT"——理由是 GT 长度固定可以做公平归一化，且 teacher-forcing 文本后变化只能来自风格；这种"借 LALM 的语言先验把模糊概念变成似然"的范式可迁移到 image style、video motion 等任何有强生成先验的模态。
- **门控混合奖励是一个通用 RL-for-TTS 模板**：硬门控 + 平移到正区间 + 一个内容惩罚的组合，比简单加权和更难被 hack，可以直接套到 emotion-TTS、accent-TTS 等任何"风格+内容"双目标场景。
- **数据 pipeline 中"用 LLM 反推剧情、用 VLM 描述场景"的自动标注路径**展示了如何用现有大模型给传统 ASR 语料二次提质——一份纯语音语料经过 R1 + Qwen-VL 加工就长出了角色档案和场景描述两层结构化标注。

## 局限与展望
- 主实验只在中文 drama 上做，附录虽给了 English 初步结果但缺多语种系统验证，跨语言 MCLP 是否仍然有效需要进一步证据。
- 评估域局限在影视剧，audiobook（长篇叙事）、game NPC（短促反应）、virtual assistant（中性指令）这些差异更大的域没覆盖，门控阈值 $\tau$ 和奖励系数 $C, \lambda$ 是否需要 per-domain 重调未知。
- 场景与角色描述全部由 LLM/VLM 自动生成，存在标注噪声和偏置（例如 R1 可能臆造剧情或人设），下游 RL 的上限受这层噪声制约；可以考虑加入少量人工校验的 gold-standard 集做 self-distillation。
- MCLP 续写模型本身是一个 7B 量级的 LALM，做奖励时每个 rollout 都要前向一次，RL 计算成本会显著高于一个轻量情感分类器；后续可以蒸馏一个小尺寸 MCLP estimator。

## 相关工作与启发
- **vs CosyVoice3 / Higgs / OV-InstructTTS**：它们都靠 instruction prompt 控制单句风格，无法吃多轮 audio history；本文证明把多轮历史塞进 LALM context + 用 RL 拉齐风格能拿到比单句 Instruct-TTS 高 0.7+ MOS 的提升。
- **vs 情感分类奖励（Wang 2025, Gao 2025b）**：那条路用离散 emotion label 做 reward，本文用连续似然，能覆盖韵律节奏等情感之外的维度，且在 MCLP 这一项上 reward 信号天然密集不会塌陷。
- **vs Speech Role-Playing Agent（SpeechRole, VoxRole, OmniCharacter）**：它们重在端到端语义对齐，本文专注 content-specified 设定下的声学风格对齐，把"说什么"和"怎么说"解耦后，可作为 Agent 的 TTS 后端模块复用。

## 评分
- 新颖性: ⭐⭐⭐⭐ "用预训练 LALM 似然当连续风格度量"是 audio 领域少见但很自然的视角，反向续写 + 双 transcript 的设计有巧思。
- 实验充分度: ⭐⭐⭐⭐ 7 个强 baseline、客观+主观双评、消融能完整解释每个组件、人类配对实验验证度量本身，链条很扎实。
- 写作质量: ⭐⭐⭐⭐ 公式排版清晰、动机推导一气呵成，方法图和奖励门控的描述都很到位。
- 价值: ⭐⭐⭐⭐ MCLP 这个度量+RL pipeline 对 Role-Play TTS 是直接可用的工程方案，配套数据集对社区有持续贡献。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Computational Narrative Understanding for Expressive Text-to-Speech](../../ACL2026/audio_speech/computational_narrative_understanding_for_expressive_text-to-speech.md)
- [\[ICML 2026\] MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Evaluating LLM Safety](multibreak_a_scalable_and_diverse_multi-turn_jailbreak_benchmark_for_evaluating_.md)
- [\[ACL 2026\] DRInQ: Evaluating Conversational Implicature with Controlled Context Variation](../../ACL2026/audio_speech/drinq_evaluating_conversational_implicature_with_controlled_context_variation.md)
- [\[NeurIPS 2025\] Sound Logical Explanations for Mean Aggregation Graph Neural Networks](../../NeurIPS2025/audio_speech/sound_logical_explanations_for_mean_aggregation_graph_neural_networks.md)
- [\[ACL 2026\] ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis](../../ACL2026/audio_speech/restyle-tts_relative_and_continuous_style_control_for_zero-shot_speech_synthesis.md)

</div>

<!-- RELATED:END -->
