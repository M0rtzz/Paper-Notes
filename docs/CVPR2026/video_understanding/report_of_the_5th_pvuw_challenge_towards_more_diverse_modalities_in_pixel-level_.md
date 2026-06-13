---
title: >-
  [论文解读] Report of the 5th PVUW Challenge: Towards More Diverse Modalities in Pixel-Level Understanding
description: >-
  [CVPR 2026][视频理解][视频目标分割] 这是 CVPR 2026 第 5 届 PVUW（野外像素级视频理解）挑战赛的官方总结报告：今年设三条赛道——MOSE（复杂遮挡场景下的视频目标分割）、MeViS-Text（运动语言表达引导的分割）以及全新的 MeViS-Audio（用音频而非文字引导分割），报告系统梳理了各赛道前三名方案，发现冠军队普遍把 SAM3 当分割底座、把 MLLM 当"存在性验证"的智能体来压制幻觉掩码。
tags:
  - "CVPR 2026"
  - "视频理解"
  - "视频目标分割"
  - "Referring VOS"
  - "音频引导分割"
  - "SAM3"
  - "挑战赛报告"
---

# Report of the 5th PVUW Challenge: Towards More Diverse Modalities in Pixel-Level Understanding

**会议**: CVPR 2026  
**arXiv**: [2604.26031](https://arxiv.org/abs/2604.26031)  
**代码**: https://pvuw.github.io/ (挑战赛主页)  
**领域**: 视频理解 / 视频分割 / 多模态  
**关键词**: 视频目标分割, Referring VOS, 音频引导分割, SAM3, 挑战赛报告

## 一句话总结
这是 CVPR 2026 第 5 届 PVUW（野外像素级视频理解）挑战赛的官方总结报告：今年设三条赛道——MOSE（复杂遮挡场景下的视频目标分割）、MeViS-Text（运动语言表达引导的分割）以及全新的 MeViS-Audio（用音频而非文字引导分割），报告系统梳理了各赛道前三名方案，发现冠军队普遍把 SAM3 当分割底座、把 MLLM 当"存在性验证"的智能体来压制幻觉掩码。

## 研究背景与动机

**领域现状**：像素级视频理解（VOS / Referring VOS）是衡量模型能否在视频帧间稳定跟踪、识别目标的关键基准，直接关系到自动驾驶、机器人等需要在动态真实场景中感知的应用。PVUW workshop 已连续举办多届，逐步用更复杂的真实数据推动算法前沿。

**现有痛点**：图像级理解已相当成熟，但视频级理解要同时处理时序变化、目标运动、遮挡、小目标、恶劣天气等"野外"条件，现有方法在这些非约束环境下仍不够鲁棒；而且过去的赛道只覆盖视频本身（MOSE）和文字引导（MeViS-Text）两种模态，缺乏对更自然的引导信号（如语音）的探索。

**核心矛盾**：一方面分割基础模型（SAM 系列）在密集杂乱场景下容易跟丢、漂移；另一方面 Referring 类任务里"目标根本不存在"的 no-target 样本会诱发模型强行输出掩码（positive bias / 幻觉），既拉低指标又限制实用性。

**本文目标**：作为挑战赛报告，本文的"方法"是赛事本身的组织设计——通过三条赛道 + 新数据 + 统一评测协议，把上述痛点暴露给社区，并汇总参赛者给出的前沿解法。

**切入角度**：今年的关键动作是把模态从"视频 + 文本"扩展到"视频 + 文本 + 音频"，首次开设 MeViS-Audio 赛道，鼓励多感官输入的鲁棒视频理解。

**核心 idea**：用三条难度递进、模态各异的赛道（视频自跟踪 → 文字引导 → 音频引导）牵引社区，把 SAM3 + MLLM 智能体 + ASR 这套组合范式推到台前。

## 方法详解

### 整体框架
这是一篇挑战赛报告，"整体框架"指赛事的组织结构而非单一算法。PVUW 2026 在 CVPR 2026 举办，设三条赛道，统一在 CodaBench 平台自动评分、隐藏测试集标注以保证公平：

- **Track 1 — MOSE**：基于 MOSEv2 数据集的复杂视频目标分割（半监督 VOS，仅首帧给掩码），数据含低光、恶劣天气、多镜头切换、伪装目标、影子等非物理实体，703 段视频、1410 个标注目标、28 类、超 9.8 万张高质量掩码；76 队报名、38 队有效提交，冠军在隐藏测试集上拿到 $\mathcal{J}\&\mathcal{F}=88.45\%$。
- **Track 2 — MeViS-Text**：基于 MeViS 数据集的文字引导分割，表达式强调目标"如何运动"而非"长什么样"，比传统按静态外观 referring 更难。
- **Track 3（新）— MeViS-Audio**：与 MeViS-Text 共用视频底座，但把文字 query 换成专业录制的音频，要求模型先处理声学信号再与视觉对齐；评测指标与 Text 赛道一致，便于跨模态直接比较。

报告的主体是对每条赛道前三名"获奖配方"的拆解。下面的"关键设计"即按"三条赛道 + 各自冠军方案的核心做法"组织——它们共享一条清晰的范式主线：**SAM3 做分割底座 + MLLM/ASR 做语义与存在性把关**。

### 关键设计

**1. 统一评测协议：用 N-acc / T-acc 把"目标不存在"显式纳入打分**

报告所有 referring 赛道共享的痛点是 no-target 样本——表达式描述的目标在视频里根本不存在，模型却容易强行分割。MOSE 赛道沿用经典的区域相似度 $\mathcal{J}$、轮廓精度 $\mathcal{F}$ 及其均值 $\mathcal{J}\&\mathcal{F}$（以后者为最终排名依据）。两条 MeViS 赛道则在此之上增加 N-acc.（no-target accuracy，正确判定"无目标"的准确率）和 T-acc.（target accuracy，有目标样本上的判定准确率），最终成绩取 $\mathcal{J}\&\mathcal{F}$、N-acc.、T-acc. 三者的平均。这一协议直接把"该不该输出掩码"的决策能力计入分数，也正是后面几乎所有冠军方案都要单独做"存在性验证"模块的根本原因。

**2. MOSE 赛道获奖配方：给 SAM3 动态补"跟踪先验"以扛遮挡与漂移**

MOSE 的核心难点是密集杂乱、严重遮挡、小目标、消失—重现，SAM3 单独跑会跟丢。三支冠军队思路各异但都围绕"给 SAM3 注入额外锚点/提示"：

- **冠军 HITsz_Dragon（TEP, Tracking-Enhanced Prompt）**：三阶段串行。先用"掩码面积 + MLLM"把目标粗分为常规 / 微小 / 语义主导三类；对微小目标用 SUTrack（图像提示的跟踪器）生成中间帧 bbox，对语义主导目标用 Qwen3.5 先描述再逐帧检测出 bbox；最后在 Prompt Fusion 阶段计算 SAM3 预测掩码 bbox 与辅助 bbox 的 IoU，若低于阈值（疑似漂移）就触发提示切换——跟踪器 bbox 按置信度决定是否注入，MLLM bbox 则让 MLLM 比对两个候选裁剪图与首帧的语义保真度后择优。
- **亚军 tobedone（OAMVOS）**：在 SAM3 版 DAM4SAM 跟踪器上，把每帧分为 stable / ambiguous / recovery 三种模式；用外观、运动、几何、候选间隔四个可解释分数 + top IoU 判定可靠性，不确定时开"分支池"让多个假设（主掩码 / 强候选 / 目标缺席假设）各持独立推理状态并行演化，再让重新确认的分支回主路；同时维护 anchor bank（存归一化 object pointer）与 DRM 双长程记忆，并显式保留首个条件帧、扩大注意力预算来应对长间隔重现。
- **季军 HCVG（Re-Prompting via Object Retrieval）**：两阶段。先用首帧掩码当视觉提示，让 SAM3 检测分支在后续所有帧找同类候选；再用 DINOv3 对首帧目标（含翻转、旋转的变换视图构成特征池）与候选做 object-level 余弦相似度匹配，选 top 高相似度且时序去冗余的候选当额外锚点，连同首帧 GT 掩码一起重新喂回 SAM3 跟踪。

**3. MeViS-Text 赛道获奖配方：MLLM 做事件分解 + 存在性把关，SAM3 做接地与传播**

文字赛道难在"运动表达式"语义复杂、且含 no-target 样本。三支队伍共享"先用强 MLLM 理解 query，再交给 SAM3 出掩码，最后回头验证"的智能体范式：

- **冠军 HITsz_Dragon（Strong MLLMs Meet SAM3）**：三阶段。Stage 1 用 Gemini-3.1 Pro 做关键帧推理，把视频运动表达式分解成实例级 grounding 目标、为每个目标选最清晰帧并生成判别性描述，把难的视频问题转成多个图像 grounding 问题；Stage 2 用 SAM3-agent（带 planner 迭代选 SAM3 操作）在关键帧出种子掩码再双向传播；Stage 3 做自精修——空预测或高重叠预测说明描述不够判别，就用 Qwen3.5-Plus 重写描述重跑，对带方向/否定约束的表达还额外采样帧让 MLLM 判定跟踪结果是否真符合原事件，不符就回炉再 grounding。
- **亚军 SaSaSaSa2VA**：核心是"存在性感知验证"。为纠正 SaSaSa2VA 基模型的正偏置，让 Gemini 3-Flash 与 GPT-5.4 组成"双共识陪审团"，只有两模型**一致**判定目标缺席才标为 null-target 并直接返回空掩码 $\mathbf{M}=\mathbf{0}$，从而修复 N-acc.；推理上反而做减法，弃用多策略投票/多模型聚合，只用单模型 + Uniform+ 采样，对短于训练时长 $T$ 的视频在片段边界附近赋双 [SEG] token 取平均。
- **季军 AgentRVOS**：coarse-to-fine 五阶段，把语义接地、时序传播、决策精修解耦。Presence Agent 先判二值存在 $e=\Psi_{\mathrm{pres}}(V,q)$，$e=0$ 直接早停输出全零掩码；$e=1$ 才用 Sa2VA 出粗轨迹 $\tilde{\mathcal{M}}$，从中抽可靠锚帧、由掩码导出 bbox/点提示喂 SAM3 传播，最后用轻量 Planner 仅在歧义帧上比较 Sa2VA 原预测与 SAM3 传播预测、择优解决局部冲突（而非全局重搜）。

**4. MeViS-Audio 赛道获奖配方（新）：ASR 把音频转文本，再接文字赛道那套范式 + 存在性门控**

新赛道的增量难点是输入从文字变成语音，因此所有方案都先做语音识别，再复用文字赛道成熟管线，并强化"无目标"过滤：

- **冠军 APRVOS**：六阶段。VibeVoice-ASR 做长语音转写（保留说话人轮次、时序对齐等上下文）；Qwen3-VL 当视觉裁判判定描述实体能否在画面 grounding，结果存 `presence_info.target_exists`；存在则把转写转成 Sa2VA 期望的分割提示模板，由 Sa2VA 出粗掩码轨迹；再由 agentic verification（planner / scout / critic 分工，把 query 拆成正约束、负约束、时序提示）找可信锚帧；最后从锚帧导出 bbox/中心点提示初始化 SAM3 双向传播精修。
- **亚军 ASR-SaSaSa2VA**：三阶段模块化。Qwen3-ASR 转文本；SaSaSa2VA 做文字引导分割（它比 Sa2VA 增强了时序建模，弥补 Sa2VA 只采少量帧、单 [SEG] token 的长程缺陷）；并用 LoRA 微调 Qwen2.5-Omni 做二分类 $y=f_{\text{cls}}(\mathcal{A},\mathcal{V})$ 直接在"音频 + 视频"上判 no-target，$y=0$ 跳过分割。
- **季军 VIRST-Audio**：在 VIRST（CLIP 视频特征 + SAM2 编码特征经 ST-Fusion 用可学习 [ST] token 做 cross-attention，再当 SAM2 解码器提示）基础上接 ASR 把音频转文本当 VLM 语言输入；并加"存在性感知分割门控"：从 ST-Fusion 输出 $\mathbf{F}$ 经轻量模块得 $z=f_{\text{exist}}(\mathbf{F}),\,p=\sigma(z)$，$p<\tau$ 直接输出空掩码，训练用 BCE 监督（标签取 GT 掩码是否非空）。

### 损失函数 / 训练策略
报告本身不训练模型。可提炼的训练相关做法集中在存在性判定上：VIRST-Audio 用二元交叉熵（BCE）监督存在概率 $p$，标签按 GT 掩码是否非空定义；ASR-SaSaSa2VA 用参数高效的 LoRA 微调 Qwen2.5-Omni 做 no-target 二分类。其余冠军方案多为推理期组合大模型与跟踪器、无需端到端训练。

## 实验关键数据

### 主实验
报告以赛道参与规模与冠军成绩为主，未给出完整逐队排行榜表格。

| 赛道 | 数据集 | 报名队 | 有效/提交 | 关键指标 | 冠军成绩 |
|------|--------|--------|-----------|----------|----------|
| MOSE | MOSEv2 | 76 | 38 有效提交 | $\mathcal{J}\&\mathcal{F}$ | 88.45% |
| MeViS-Text | MeViS | 33 | 34 提交 | avg($\mathcal{J}\&\mathcal{F}$, N-acc., T-acc.) | 报告未给数值 |
| MeViS-Audio（新） | MeViS（音频版） | 26 | 51 提交结果 | 同 MeViS-Text | 报告未给数值 |

> MOSEv2 数据规模：703 段视频、1410 个目标、28 类、>98,000 张掩码。⚠️ MeViS 两赛道的具体排行榜数值原报告正文未列，以官网/CodaBench 排行榜为准。

### 消融实验
作为挑战赛报告，本文无标准消融表。可类比为"各冠军方案中不同模块的作用"：

| 模块 | 出现于 | 作用 |
|------|--------|------|
| 存在性验证 / 门控 | SaSaSaSa2VA、AgentRVOS、ASR-SaSaSa2VA、VIRST-Audio | 拦截 no-target 幻觉掩码，直接拉升 N-acc. |
| SAM3 重提示 / 锚点注入 | TEP、HCVG、AgentRVOS、APRVOS | 给 SAM3 补额外锚点，缓解遮挡/漂移/跟丢 |
| MLLM 事件分解 | Strong MLLMs Meet SAM3 | 把难的视频运动表达拆成多个图像 grounding 子问题 |
| ASR 转写 | 全部 Audio 赛道方案 | 把音频引导降维成已成熟的文字引导问题 |

### 关键发现
- **存在性验证是 referring 赛道的胜负手**：几乎所有文字/音频赛道冠军都单独做 no-target 判定（双模型共识、Presence Agent、LoRA 二分类、门控网络），因为评测把 N-acc. 计入平均分，压制幻觉直接提分。
- **SAM3 已成事实底座**：MOSE 三强全部围绕 SAM3/DAM4SAM 做"重提示"而非另起炉灶，说明 2026 年的范式是"基础模型 + 任务特定的提示/锚点工程"。
- **音频赛道 = ASR + 文字赛道范式**：新模态没有催生端到端音频分割模型，而是先 ASR 降维再复用成熟的文字管线——务实但也暗示纯声学-视觉对齐仍有空间。
- **MLLM 从"附属"变"核心智能体"**：Qwen / Gemini / GPT 不只做语义解释，更承担存在性验证、时空推理、冲突仲裁等决策角色。

## 亮点与洞察
- **"双共识陪审团"压幻觉**：SaSaSaSa2VA 让 Gemini + GPT 两个闭源大模型独立判定，只有一致认为目标缺席才输出空掩码——用集成的保守性专门修 N-acc.，思路可迁移到任何带 no-target 的 referring 任务。
- **分支池式不确定性处理**：OAMVOS 在帧不确定时不强行决策，而是让"主掩码 / 强候选 / 缺席假设"各持独立推理状态并行演化、跨帧累积证据再回主路，是对抗小目标消失—重现的优雅做法。
- **变换感知的特征池匹配**：HCVG 用 DINOv3 对目标的翻转/旋转视图建小特征池、取最大余弦相似度，缓解目标大幅形变后匹配失效，是简单可复用的 trick。
- **把视频问题降维成图像问题**：Strong MLLMs Meet SAM3 用 MLLM 选最清晰关键帧 + 判别性描述，把视频运动表达拆成多个图像 grounding，再传播——降低了直接做视频级 grounding 的难度。

## 局限性 / 可改进方向
- **报告缺定量细节**：除 MOSE 冠军 88.45% 外，正文几乎不给逐队数值与消融，难以横向定量比较各方案优劣（需查官网排行榜）。
- **音频赛道"伪多模态"**：所有方案都靠 ASR 先转文字，本质仍是文字引导，没有真正在声学特征层面与视觉对齐，未能体现音频独有信息（语调、非语言声音等）。
- **重提示工程依赖大模型推理成本**：多数冠军堆叠 SAM3 + 多个闭源大模型（Gemini/GPT/Qwen），推理开销大、可复现性与部署成本存疑。
- **改进方向**：作者计划持续刷新、扩展 MOSE 与 MeViS 评测集，引入更多感官模态推动野外像素级理解的前沿。

## 相关工作与启发
- **vs PVUW 2024 / 2025 报告**: 往届只有 MOSE + MeViS-Text 两赛道，本届首增 MeViS-Audio，把引导模态从文本扩展到语音，是模态多样性上的实质推进。
- **vs 单方法论文（如 SAM3、Sa2VA）**: 单方法论文给一个通用模型；本报告则展示在统一隐藏测试集上，社区如何围绕这些基础模型做任务特定的提示工程、存在性把关与智能体编排，反映真实"打榜级"工程取舍。
- **vs LSVOS 挑战赛**: 同属复杂/长视频分割方向，PVUW 更强调"野外非约束"条件与多模态引导，两者参赛队与方法高度重叠（如 SaSaSa2VA 系列）。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首开音频引导分割赛道、把存在性验证范式推到台前，但报告本身属于综述性整理而非原创方法。
- 实验充分度: ⭐⭐⭐ 覆盖三赛道前三名共九套方案、规模与冠军成绩清晰，但缺逐队定量榜与消融。
- 写作质量: ⭐⭐⭐⭐ 结构清晰、按赛道分层、趋势总结到位。
- 价值: ⭐⭐⭐⭐ 是 2026 年视频分割"打榜范式"的权威快照，对从业者了解 SAM3 + MLLM 智能体组合极有参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Understanding Identity Continuity in Thermal Video through Scene-Level Consistency](understanding_identity_continuity_in_thermal_video_through_scene-level_consisten.md)
- [\[CVPR 2026\] Less is More: Token-Efficient Video-QA via Adaptive Frame-Pruning and Semantic Graph Integration](less_is_more_token-efficient_video-qa_via_adaptive_frame-pruning_and_semantic_gr.md)
- [\[CVPR 2026\] CLCR: Cross-Level Semantic Collaborative Representation for Multimodal Learning](clcr_cross-level_semantic_collaborative_representation_for_multimodal_learning.md)
- [\[CVPR 2026\] OmniEgo-R2: A Routed Reasoning Framework for the 1st Cross-Domain EgoCross Challenge](omniego-r2_a_routed_reasoning_framework_for_the_1st_cross-domain_egocross_challe.md)
- [\[CVPR 2026\] TempRet: Temporal Enhancement and Two-Stage Reranking for CVPR 2026 EPIC-KITCHENS-100 Multi-Instance Retrieval Challenge](tempret_temporal_enhancement_and_two-stage_reranking_for_cvpr_2026_epic-kitchens.md)

</div>

<!-- RELATED:END -->
