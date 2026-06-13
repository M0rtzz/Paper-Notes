---
title: >-
  [论文解读] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs
description: >-
  [ACL 2025][LLM/NLP][EMG-to-Text] 本文提出了一种基于可训练 EMG 适配器模块的方法，将无声肌电图（EMG）信号映射到大语言模型（LLM）的输入嵌入空间，在闭合词汇无声 EMG 转文本任务中实现了 0.49 的词错误率（WER），仅需 6 分钟训练数据即比专用模型提升约 20%…
tags:
  - "ACL 2025"
  - "LLM/NLP"
  - "EMG-to-Text"
  - "静默语音接口"
  - "多模态LLM"
  - "适配器网络"
  - "生物信号"
---

# Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs

**会议**: ACL 2025  
**arXiv**: [2506.00304](https://arxiv.org/abs/2506.00304)  
**代码**: [payalmohapatra/SilentSpeechLLM](https://github.com/payalmohapatra/SilentSpeechLLM)  
**领域**: LLM/NLP  
**关键词**: EMG-to-Text, 静默语音接口, 生物信号理解, 冻结 LLM, 个性化建模

## 研究背景与动机

这篇论文讨论的是一个非常具体但社会价值很高的问题：能否只依赖无声肌电信号，把用户想说的话还原成文本。
传统自动语音识别默认有声音或至少有声学表征，但对失声或无法发声的人来说，这个前提不存在。
无声表面肌电图 EMG 能记录口周和喉部肌肉活动，因此是静默语音接口的重要信号来源。
问题在于，过去许多 EMG-to-Text 方法虽然号称做“无声语音”，训练时却仍然依赖有声 EMG 或同步音频作为辅助监督。
这对真正不能发声的用户并不现实，因为他们根本没有可采集的有声配对数据。
作者把这个现实约束设为论文的出发点：只给模型无声 EMG，不给音频，不给 voiced EMG，看看 LLM 能不能直接学会解码。
另一个核心挑战是数据量极少。
公开单用户闭词汇数据集一共只有约 26 分钟、500 条样本，而实际可用训练量甚至可以低到 6 分钟。
如果沿用大数据驱动思路，模型极易过拟合或者根本学不到稳定映射。
EMG 还有极强的用户特异性，不同人的肌肉活动模式差异很大，论文中的 pilot 实验甚至能以 96% 准确率识别说话人身份。
这意味着统一的大模型未必能直接跨用户泛化，个性化轻量适配更关键。
作者因此没有去训练一个端到端超大 EMG 模型，而是反过来问：既然 LLM 已经内化了丰富语言先验，是否可以只训练一个很小的适配器，把 EMG 投到 LLM 的嵌入空间里，让冻结的 LLM 负责“语言那部分”的工作。
从研究动机上看，这篇论文有三层目标。
第一，验证 LLM 是否能“理解”一种此前几乎没接触过的语言模态，即无声生物信号。
第二，评估在极低资源条件下，冻结 LLM 加小适配器是否比专用模型更数据高效。
第三，探索适配器结构、输入特征和训练目标哪些最适合 EMG 这种噪声大、样本少、个体差异强的模态。

## 方法详解

整套方法的核心是一个 trainable EMG adaptor，加一个完全冻结的 LLaMA 模型。
作者没有直接把 EMG 当作离散 token，而是把它编码成一串连续嵌入，再拼接提示词送入 LLM 做自回归文本生成。
从系统设计看，这和很多 speech-to-LLM 或 vision-to-LLM 工作类似，但 EMG 信号更稀疏、更个体化、可解释性更差，因此适配器设计不能简单照搬音频方案。

输入信号记作 $\mathbf{X}^E \in \mathbb{R}^{T \times C}$，其中 $C$ 是 EMG 通道数，本文主数据集为 8 通道。
由于原始采样率超过 800Hz，序列非常长，直接输入 LLM 不现实。
作者首先用一个 stride 为 6 的一维卷积做第一次时间下采样，把长度降到原来的六分之一。
接着叠两层残差卷积块提取局部时序模式。
残差结构在这里的意义不是追求特别深的模型，而是帮助在小数据下稳定训练，同时保留原始局部肌电形态。
在残差块之后，作者加入一个 BiLSTM 建模时间依赖。
这是本文一个很有信息量的设计选择，因为很多人默认会用 Transformer，但作者实证发现 BiLSTM 明显更适合当前闭词汇、短序列、低资源场景。
随后模型再通过一个 stride 为 2 的一维卷积做第二次下采样。
综合下来，总时间压缩比约为 48 倍。
压缩后的特征再经过线性层投到 LLM 的词向量维度，LLaMA 2-7B 对应 4096 维，LLaMA 3.2-3B 对应 3072 维。
这一步生成 EMG embedding 序列，作为 LLM 的“伪输入 token”。

为了让冻结的 LLM 理解当前输入不是普通文本，作者设计了上下文化提示拼接。
在 EMG embedding 前加上文本标识 “Unvoiced EMG:”，在后面加上任务描述 “Prompt: Convert unvoiced EMG embeddings to text”。
前后提示会先经过 tokenizer 和词嵌入层变成普通文本 embedding，再与中间的 EMG embedding 串接起来。
这个设计非常像给模型建立一个任务上下文：前缀声明模态，后缀声明任务。
这样 LLM 在推理时不是被迫从奇怪的连续向量里盲猜，而是在一个熟悉的 prompt 框架下执行“转写”任务。

训练时只更新适配器参数，LLM 保持冻结。
损失函数采用带温度的交叉熵，温度 $\tau = 0.8$，优化器是 AdamW，学习率 $5 \times 10^{-5}$。
推理阶段使用 beam width 为 4 的自回归生成。
作者还尝试了 CTC，但效果不如交叉熵。
这说明一旦把 EMG 投进 LLM 的 embedding 空间，最好顺着 LLM 原本的自回归训练范式走，而不是强行回到传统语音识别的 CTC 思路。

本文另一条很有意思的线是输入特征选择。
作者不只测试原始 EMG，还测试了 112 维手工特征，包括时域和频域统计量。
对于传统专用模型，手工特征反而更差；但对 LLM 适配器方法，手工特征显著更好。
这说明冻结 LLM 的瓶颈不一定在语言层，而可能在前端适配器容量不足以从原始高噪声 EMG 中自己抽取最优表示。
换句话说，当可训练部分只有 600 万参数时，适度的领域先验特征工程依然非常重要。

作者还做了两类额外探索。
一类是比较 audio-to-LLM 与 EMG-to-LLM 的难度，发现即便用很简单的音频接入方式，LLM 处理音频也比处理 EMG 更轻松，说明 EMG 不是“换个模态就行”，而是本身就更难。
另一类是加入 voiced EMG 数据，结果专用模型收益更大，而 LLM 方法提升有限。
这意味着当前 LLM 适配方案还没有充分利用 voiced/unvoiced 对齐信号，后续如果引入显式跨模态对齐或 instruction tuning，可能还有空间。

| 模块 | 具体设计 | 作用 |
|------|----------|------|
| 时序下采样 1 | stride=6 的 1D 卷积 | 压缩高采样率原始 EMG |
| 局部特征提取 | 2 个残差卷积块 | 提取稳定局部时序模式 |
| 序列建模 | BiLSTM | 捕捉跨时间依赖，优于 Transformer |
| 时序下采样 2 | stride=2 的 1D 卷积 | 进一步降低序列长度 |
| 投影层 | 全连接 + GeLU | 对齐到 LLM embedding 维度 |
| 语言解码器 | 冻结 LLaMA | 利用已有语言先验完成文本生成 |

| 设计选择 | 作者结论 | 背后原因 |
|----------|----------|-----------|
| 冻结 LLM vs 直接微调 LLM | 冻结更稳 | 数据太少，直接调 LLM 容易过拟合 |
| BiLSTM vs Transformer | BiLSTM 更好 | 闭词汇短序列下需要更强局部时序偏置 |
| 手工特征 vs 原始 EMG | 对 LLM 方法手工特征更好 | 适配器容量有限，需要降噪后的输入 |
| CE vs CTC | CE 更好 | 更符合 decoder-only LLM 的训练方式 |

## 实验关键数据

实验主要基于 Gaddy 和 Klein 的单说话人 8 通道闭词汇数据集，共 67 个词、约 26 分钟无声 EMG 数据、500 条样本。
评估指标是词错误率 WER，使用三折验证，并在 8:1:1 的训练、验证、测试拆分下报告结果。
作者比较了两类基线：一类是 Gaddy 与 Klein 的专用 EMG-to-Text 模型，参数约 5400 万；另一类是作者提出的 EMG adaptor 加冻结 LLM，只训练约 600 万参数。

主结果非常直接。
在原始 EMG 输入下，最佳 LLM 方法是 EMG-Ad + Llama3-3B，WER 为 0.52，明显好于专用模型的 0.75。
在手工特征输入下，Llama2-7B 和 Llama3-3B 都能做到 0.49，而专用模型反而退化到 0.84。
这说明本文最核心的结论不是“LLM 稍微好一点”，而是“在极低数据下，冻结 LLM 能把语言先验真正转化为 EMG 解码收益”。

| 模型 | 输入特征 | 可训练参数 | WER |
|------|----------|------------|-----|
| App-Specific 基线 | Raw EMG | 54M | 0.75 ± 0.06 |
| EMG-Ad + Llama2-7B | Raw EMG | 6M | 0.65 ± 0.01 |
| EMG-Ad + Llama3-3B | Raw EMG | 6M | **0.52 ± 0.05** |
| EMG-Ad + Fine-tuned Llama3-3B | Raw EMG | 更高 | 0.62 ± 0.04 |
| App-Specific 基线 | 手工特征 | 54M | 0.84 ± 0.06 |
| EMG-Ad + Llama2-7B | 手工特征 | 6M | **0.49 ± 0.06** |
| EMG-Ad + Llama3-3B | 手工特征 | 6M | **0.49 ± 0.04** |
| EMG-Ad + Fine-tuned Llama3-3B | 手工特征 | 更高 | 0.55 ± 0.02 |

如果从相对提升看，最佳结果从 0.75 降到 0.49，绝对降低 0.26，已经是很明显的性能跨越。
更难得的是，这个收益是在只有几分钟训练数据的条件下获得的。
作者进一步做了训练数据量缩减实验，把训练量从约 26 分钟逐步降到 6 分钟。
虽然 WER 会随数据减少上升，但 LLM 方法在各数据量区间都平均优于专用模型约 26%。
这对于真实场景尤其关键，因为用户通常不可能贡献大量长时长标注数据。

消融实验也很有价值。
作者比较了只用全连接、残差块、残差块加 Transformer、残差块加 LSTM 等变体。
结果显示 ResBlock(2) + LSTM 在 Llama3-3B 下效果最好，WER 为 0.53；而加 Transformer 反而恶化到 0.79。
同样，在 Llama2-7B 上，把训练目标从 CE 换成 CTC，WER 从 0.65 退到 0.70。
这些结果共同说明，EMG 这种短、弱、噪声大的信号并不天然适合“更 Transformer 化”的前端。

| 消融设置 | 变体 | WER |
|----------|------|-----|
| 适配器结构 | Fully Connected | 0.70 |
| 适配器结构 | ResBlock(2) | 0.64 |
| 适配器结构 | ResBlock(2) + Transformer | 0.79 |
| 适配器结构 | ResBlock(2) + LSTM | **0.53** |
| 训练目标 | CE + Llama2-7B | **0.65** |
| 训练目标 | CTC + Llama2-7B | 0.70 |

作者还做了三个补充实验，分别揭示任务边界。
第一，person identification 实验能用 unvoiced EMG 达到 0.96 的身份识别准确率，说明信号中的个体特征极强，这从侧面解释了为什么个性化建模不可回避。
第二，时间偏移和 Hilbert phase 两种数据增强几乎无效，表明 EMG 的时序对齐极其敏感，不能指望通用增强技巧随便提升效果。
第三，对比 audio-to-LLM 与 EMG-to-LLM 后，作者发现音频任务更容易，说明 EMG 模态接入 LLM 还有大量表示学习问题尚未解决。

从实验总体质量看，这篇论文没有追求“大而全”的 benchmark，而是围绕一个极具体场景，把数据效率、前端结构、特征形式、损失函数和任务边界都做了扎实对照。
这比只汇报一个 SOTA 数字更有参考价值。

## 亮点与洞察

本文最大的亮点是问题设定非常干净：不依赖 voiced EMG、不依赖音频，只用无声 EMG 做转写，这让结论更接近真实辅助沟通场景。
第二个亮点是冻结 LLM 加小适配器在超低资源条件下居然确实成立，说明语言先验能跨模态迁移到生物信号任务，但前提是接入方式足够合适。
第三个亮点是手工特征对 LLM 方法有效、对专用模型无效，这个反常现象很有研究价值，它揭示了“前端特征工程”和“下游模型容量”之间存在强耦合。
第四个亮点是作者没有把 LLM 神化，反而通过音频对比和 voiced EMG 对比指出 EMG 模态更难，这让结论更可信。
对我而言，最有启发的一点是：在新模态低资源接入 LLM 时，冻结大模型并不意味着前端就可以随便设计，恰恰相反，适配器和输入特征会决定是否能真正借到 LLM 的先验。

## 局限与展望

第一，任务仍然是闭词汇设置，只有 67 个词，距离开放词汇、自然句子级输入还有明显距离。
第二，主实验只在单说话人数据上验证，多用户、多设备、多语言的泛化性还没有被证明。
第三，当前方法需要访问 LLM embedding 层，因此对只开放 API 的闭源模型不友好。
第四，作者尝试了简单数据增强但收益有限，说明在 EMG 场景下，数据稀缺问题还远没有被解决。
未来可以往三个方向扩展。
一是把闭词汇做成更大规模的“受控开放词汇”，逐步过渡到自然句子解码。
二是探索跨用户快速适配，比如 LoRA、元学习或 prototype-based personalization。
三是把 EMG、EEG、EOG 等多种生物信号统一接到同一个多模态 LLM 上，研究哪些语言先验能跨信号共享，哪些必须个体化学习。

## 相关工作与启发

和早期 silent speech 接口工作相比，本文最重要的区别是不再把 voiced 信号当必要中介。
和 Gaddy 与 Klein 的专用 EMG-to-Text 模型相比，本文不是在 EMG 模型内部继续堆结构，而是把“语言建模”外包给冻结 LLM，让前端只负责映射。
和 Benster 等把 LLM 当后处理纠错器的方法相比，本文进一步让 LLM 直接参与模态理解，而不是只在最后修正文句。
和语音或视频接入 LLM 的 adaptor 工作相比，这篇论文说明生物信号模态的难度更高，不能直接套成熟配方。
从方法迁移角度，这篇工作对脑机接口、可穿戴传感器文本化、神经肌肉疾病辅助沟通都很有启发。
如果以后做 EEG-to-Text 或 gesture-to-language，完全可以复用“轻量适配器 + 冻结语言模型 + 强约束任务 prompt”这条路线。
同时也要记住本文给出的反例：当信号太稀缺、太个体化时，适配器能力、特征工程和任务设定比单纯扩大语言模型参数更重要。

## 评分

- 新颖性: ⭐⭐⭐⭐☆  直接让 LLM 理解无声 EMG 而不是做后处理，问题设定和方法路线都很新。
- 实验充分度: ⭐⭐⭐⭐☆  主结果、数据量缩减、结构消融、损失消融、用户识别和增强实验都较完整，但数据集规模仍偏小。
- 写作质量: ⭐⭐⭐⭐☆  论文叙述清楚，实验逻辑连贯，尤其把“为什么冻结 LLM 仍能有用”解释得比较到位。
- 价值: ⭐⭐⭐⭐⭐  对辅助沟通和低资源多模态接入 LLM 都有很强现实意义。
- 综合评价: 8.8/10。它是一个很扎实的第一步，不夸张地证明了 LLM 可以接住无声 EMG，但距离开放场景应用还有不少系统性问题要补。---
title: >-
  [论文解读] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs
description: >-
  [ACL 2025][LLM/NLP][EMG-to-Text] 本文提出了一种基于可训练 EMG 适配器模块的方法，将无声肌电图（EMG）信号映射到大语言模型（LLM）的输入嵌入空间，在闭合词汇无声 EMG 转文本任务中实现了 0.49 的词错误率（WER），仅需 6 分钟训练数据即比专用模型提升约 20%。
tags:
  - ACL 2025
  - LLM/NLP
  - EMG-to-Text
  - 静默语音接口
  - 多模态LLM
  - 适配器网络
  - 生物信号
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] HiCUPID: Exploring the Potential of LLMs as Personalized Assistants](exploring_the_potential_of_llms_as.md)
- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [\[ACL 2025\] LLMs Can Be Easily Confused by Instructional Distractions](llms_can_be_easily_confused_by_instructional_distractions.md)
- [\[ACL 2025\] Can Large Language Models Understand Internet Buzzwords Through User-Generated Content](buzzword_understanding_ugc.md)
- [\[ACL 2025\] Biased LLMs Can Influence Political Decision-Making](biased_llms_can_influence_political_decision-making.md)

</div>

<!-- RELATED:END -->
