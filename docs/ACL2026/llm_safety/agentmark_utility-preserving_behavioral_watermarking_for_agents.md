---
title: >-
  [论文解读] AgentMark: Utility-Preserving Behavioral Watermarking for Agents
description: >-
  [ACL 2026][LLM安全][agent watermarking] AgentMark 把 LLM agent 的「下一步选什么 tool / subgoal」建模为一个时间变化的离散信道…
tags:
  - "ACL 2026"
  - "LLM安全"
  - "agent watermarking"
  - "planning behavior"
  - "distribution-preserving sampling"
  - "erasure-resilient coding"
  - "provenance"
---

# AgentMark: Utility-Preserving Behavioral Watermarking for Agents

**会议**: ACL 2026  
**arXiv**: [2601.03294](https://arxiv.org/abs/2601.03294)  
**代码**: https://github.com/Tooooa/AgentMark (有)  
**领域**: LLM 安全 / 水印 / Agent 治理  
**关键词**: agent watermarking, planning behavior, distribution-preserving sampling, erasure-resilient coding, provenance

## 一句话总结
AgentMark 把 LLM agent 的「下一步选什么 tool / subgoal」建模为一个时间变化的离散信道，通过显式 elicit 行为分布 $P_t$ 并应用 FDPSS 式分布保持采样把多比特 ID 嵌入 planning 决策，配合 RLNC 编码使得即便 trace 被裁剪/删步也能从残余日志恢复水印；在 ALFWorld、ToolBench、OASIS 三类任务上既不掉准确率（保持任务 SR 与 baseline 差异 <0.7 pp），又能稳定提供 1.2-2.3 bps 的多比特容量，且与 SynthID-Text 的内容层水印正交可叠加。

## 研究背景与动机

**领域现状**：LLM 内容水印（KGW、SynthID-Text 等）已能可靠归因模型生成的文本，Google Gemini 已部署 SynthID。但 agent 真正可造成社会影响的是「行为决策序列」——选哪个 tool、走哪个 subgoal——而非最终文本。GUI 助手、金融工具调用、社交机器人都属于此类。

**现有痛点**：把内容水印直接搬到 agent 行为有三个失败模式：(1) 训练时水印需改模型权重，而 agent 多走闭源 API；(2) 推理时 token 级水印（KGW、SynthID）作用于 token 分布，但「行为」不是 token——一个「Alice bookmarked a post with the tag #TravelInspiration」会被编译成 tool call `bookmark()` + `tag(#TravelInspiration)`，水印信号在编译过程中被剥离；(3) 直接给行为概率加偏置（如 Agent Guide 的 RG 策略）会让分布漂移，错误在长程执行中复合放大，导致任务失败。

**核心矛盾**：水印要嵌入到 planning 层才能真正攻击「impersonation/IP 盗用/失控」等治理风险，但 planning 层的扰动会破坏 utility——这是矛盾的根源。

**本文目标**：实现「分布保持的行为水印」——同时满足 (a) 不改模型权重；(b) 在黑盒 API 下可用；(c) 水印嵌入后行为分布不动；(d) trace 被部分擦除/截断仍可恢复多比特 ID；(e) 与内容水印可正交叠加。

**切入角度**：把 planning 视为从隐式分布 $P_t^\star$ 采样的过程，让 agent 显式输出 $P_t \approx P_t^\star$，然后用 FDPSS 框架（差分重组 + 循环移位均匀编码）做分布保持的采样——边采样边嵌入比特，且边际分布不变。

**核心 idea**：「先把隐式策略 elicit 成显式概率列表 $P_t$，水印动作只发生在 $P_t$ 的采样过程，不动 $P_t$ 本身」。

## 方法详解

### 整体框架
AgentMark-F（具体实例化）每步流程：(1) **行为 elicit**：agent 不再黑盒选 $b_t$，而是显式输出候选集 $\mathcal{B}_t$ 上的概率列表 $P_t$；(2) **差分重组**：把 $P_t$ 按概率非增排序后用差分 $d_k=p_k-p_{k+1}$ 把分布分解为 $n$ 个均匀 bin，bin $k$ 含 top-$k$ 行为、权重 $q_k=k\cdot d_k$；(3) **分布保持采样**：用从共享密钥 $K_{\mathrm{sh}}$ 与步上下文 $C_t$ 派生的 PRG 采 bin index $K$，在选中 bin 内用 CyclicShift 嵌入 $\lfloor\log_2|T|\rfloor$ 或 $+1$ 个比特，输出水印行为 $\hat{b}_t$；(4) **RLNC 编码**：每步嵌入的比特视为对 payload $m\in\mathbb{F}_2^L$ 的一条线性方程 $y_{t,j}=\langle a_{t,j},m\rangle$，验证时把残余方程拼成矩阵 $A_{\mathcal{I}}m=y_{\mathcal{I}}$，rank 满则唯一解出 $m$。

### 关键设计

1. **行为级显式 elicit + 差分重组分布保持**:

    - 功能：让水印「只动采样过程，不动分布」，使长程执行不积累偏置。
    - 核心思路：差分重组把任意分布拆成 $n$ 个均匀 bin 的混合：先按 $p_1\ge\dots\ge p_n$ 排序，定义 $d_k=p_k-p_{k+1}$，每个 bin $T_k=\{b_{t,1},\dots,b_{t,k}\}$ 是均匀的、权重 $q_k=k\cdot d_k$。先按 $\mathrm{Cat}(q_1,\dots,q_n)$ 采 bin，再在 bin 内均匀采样。可证明边际分布 $\Pr[\hat{b}_t=b_{t,i}]=\sum_{k=i}^n q_k/k=\sum_{k=i}^n(p_k-p_{k+1})=p_i$，与原分布严格相同。
    - 设计动机：直接对 $P_t$ 加 bias（RG baseline）会引入逐步偏置，长程执行错误复合（ALFWorld-ID SR 从 89.5% 跌到 78.8%）；分布保持采样保证 marginal 不变，长程不漂移。差分重组的优雅之处在于「均匀 bin 内可用任意均匀编码方案嵌信息」，把分布保持问题转化为均匀分布上的隐写问题。

2. **CyclicShift 均匀编码 + 密钥派生 PRG 同步**:

    - 功能：在每个均匀 bin 内嵌入接近信息论上界的可变长度比特，并保证编解码同步。
    - 核心思路：对大小为 $n$ 的 bin，$k=\lfloor\log_2 n\rfloor$，$m=n-2^k$，bin 中 $2^k-m$ 个 index 对应 $k$ 位 codeword、$2m$ 个对应 $k+1$ 位，组成 prefix-free 码本；用 PRG 派生的循环移位 $R$ 把 payload 比特映射到 bin index $j$。期望容量 $\mathbb{E}[c\mid n]\in[\log_2 n-0.0861, \log_2 n]$，接近信息论上界。Encoder/Decoder 都用 $K_t=H(K_{\mathrm{sh}}\Vert C_t)$ 派生 PRG，确保跨方同步且对外伪随机。
    - 设计动机：均匀 bin 不是 2 的幂时直接用 $\lfloor\log_2 n\rfloor$ 位会浪费容量，CyclicShift 用变长码近最优。把 PRG seed 绑到 $C_t$（步索引、观察、历史）使得解码方无需额外通信即可重建随机性——这是黑盒 API 下能落地的关键。

3. **RLNC erasure-resilient 编码与截断鲁棒**:

    - 功能：把多比特水印分散到整条 trace 上，使得即便平台过滤、日志丢步或截断，只要观察到的总容量 $\ge L$ 即可恢复。
    - 核心思路：把每步嵌入的 $c_t$ 比特视为对 payload $m\in\mathbb{F}_2^L$ 的 $c_t$ 条线性方程，系数 $a_{t,j}=\mathrm{PRG}(K_t,j)\in\mathbb{F}_2^L$。验证时只取观察到的 $\mathcal{I}\subseteq\{1,\dots,T\}$ 上的方程，拼成 $A_{\mathcal{I}}m=y_{\mathcal{I}}$（$R=\sum_{t\in\mathcal{I}}c_t$ 行），Gaussian elimination 解 $m$。理论上当 $R=L+\Delta$ 时 rank 满概率 $\ge 1-2^{-\Delta}$，FPR 随 overhead $k$ 指数衰减。
    - 设计动机：repetition 编码在 erasure 高时会迅速崩盘；RLNC 与 Fountain code 思想类似——「rateless」让每一步都是独立线性测量，丢失任何子集都不破坏唯一恢复条件（只要总数够）。对长程 agent 这是最优的健壮性策略。

### 损失函数 / 训练策略
无训练；仅推理时改采样过程。关键超参：$\delta_{\mathrm{JSD}}$（差分量化精度 $\pi$，避免概率并列引起编解码不同步）；RG baseline $\gamma=0.5$、$\delta=2.0$（仅用于对比）。

## 实验关键数据

### 主实验
ALFWorld（DeepSeek-Chat）与 ToolBench（450 任务）上的 SR 与水印容量对比：

| 环境 | 方法 | SR (%) ↑ | Steps | bps ↑ | bpt ↑ | Δs/step ↓ |
|------|------|----------|-------|-------|-------|-----------|
| ALFWorld-ID | Baseline | 89.5 | 19.7 | – | – | – |
| ALFWorld-ID | RG (Agent Guide) | 78.8 (↓10.7) | 26.1 (↑6.4) | 0 bit | 0 bit | – |
| ALFWorld-ID | **AgentMark-F** | 89.3 (↓0.2) | 19.4 (↓0.3) | 1.19 | 25.5 | +0.10 |
| ALFWorld-OOD | Baseline | 96.8 | 15.9 | – | – | – |
| ALFWorld-OOD | RG | 94.5 (↓2.3) | 15.4 | 0 bit | 0 bit | – |
| ALFWorld-OOD | **AgentMark-F** | 97.5 (↑0.7) | 14.1 (↓1.8) | 1.34 | 28.4 | −0.18 |
| ToolBench | Baseline | 59.9 | 6.7 | – | – | – |
| ToolBench | **AgentMark-F** | 59.7 (↓0.2) | 7.2 | 0.49 | 4.93 | −1.27 |

### 消融实验
关键鲁棒性与对比指标：

| 实验维度 | 设置 | 关键结果 | 说明 |
|----------|------|---------|------|
| 步丢失鲁棒（RLNC vs repetition）| 丢步率 $p\in[0,0.7]$ | RLNC 明显更稳，全局聚合后接近 100% | rateless 性质 |
| 错钥/无水印 FPR | $k\in[0,16]$ overhead | $k\ge 8$ FPR <1%，$k\ge 14$ 零误报 | $\propto 2^{-k}$ 衰减 |
| 语义改写观察 | ALFWorld-OOD 134 任务 / 2326 步 | Behavior Match 49.45%，Bit Recovery 16.84% | 改写会破坏同步但聚合仍可用 |
| 与 SynthID-Text 叠加 | ToolBench | 行为解码 100%，内容检测 96.6% | 行为层 + 内容层正交 |
| OASIS 社交模拟 | Twitter/Reddit 100 trajectory | 5 项社交质量指标几乎不掉，水印检测率高 | 持续性验证 |

### 关键发现
- RG baseline 在长程任务上明显伤 utility（ALFWorld-ID SR −10.7 pp、步数 +6.4），是「为了水印牺牲性能」的典型反例；AgentMark-F 与 baseline 差异 <1 pp 证明「分布保持」承诺被严格兑现。
- Token/latency overhead 几乎可忽略（ALFWorld Δtok/step ±0.5%，ToolBench 平均 −6.25% 因为提前终止），说明 elicit 显式分布的额外提示开销在工程上可接受。
- 行为层水印与 SynthID-Text 内容水印「攻击不同失效模式」：行为层抗 trace 截断/丢步，内容层抗内容改写——两者叠加给出更鲁棒的 provenance。
- 跨模型验证（Gemini 2.0 Flash 上 SR 几乎不变，bps 1.96-2.28）说明方法是 model-agnostic。

## 亮点与洞察
- 「行为是新的 token」是这篇论文的核心 reframing——把内容水印的整套理论（KGW/SynthID/FDPSS）抬到 planning 决策层，是 LLM 安全 + 信息论的优雅 cross-over。
- 分布保持采样（差分重组 + CyclicShift）在长程 agent 中尤其重要：直接 bias 会复合错误，分布保持则保证 marginal 严格不变，提供了「安全/水印不必牺牲 utility」的存在性证明。
- RLNC 「rateless 测量」处理 trace 截断/丢步的健壮性，可迁移到任何「序列被部分观测」的可信场景，如分布式日志审计、长视频元数据嵌入。

## 局限与展望
- 需要 agent 暴露显式 planning 分布 $P_t$ 与候选集 $\mathcal{B}_t$，闭源 API 若不提供则需 prompt engineering 强制 elicit，可能损失保真度。
- 语义改写鲁棒性弱：观察被改写后 KL=3.227，bit recovery 仅 16.84%，是当前最大短板，需要语义层 reproducibility 才能强化。
- $P_t$ 高度峰化（如 only 1 个候选）时单步容量为 0，对短轨迹任务总容量受限；需要靠跨任务聚合补偿。
- 开源 LLM 可以直接从 logits 提取分布，但闭源依赖 elicit 提示，长期会希望厂商提供 native API 输出 planning 分布。

## 相关工作与启发
- **vs SynthID-Text (Nature 2024)**: SynthID 在 token 分布上嵌入零比特/低比特水印，主要防内容改写；AgentMark 在行为分布上嵌入多比特 ID，主要防 trace 截断/丢步，两者正交可叠加。
- **vs Agent Guide (Huang 2025, 即文中 RG)**: Agent Guide 是首个直接给行为概率加 bias 的方案，但会引入 distribution drift；AgentMark 用 FDPSS 严格保持分布，是工程上的关键修正。
- **vs Meteor/Discop (隐写经典)**: 这些是 token 序列上的分布保持隐写；AgentMark 把同样的范式应用到 agent 行为序列，并配 RLNC 解决 erasure 问题。

## 评分
- 新颖性: ⭐⭐⭐⭐ 第一次把分布保持隐写 + RLNC 系统化用到 agent planning 层，跨域整合优雅。
- 实验充分度: ⭐⭐⭐⭐ 3 类环境 × 2 模型 + 容量/鲁棒性/叠加性测试 + 理论 FPR 推导，少数任务上方差较大但整体充分。
- 写作质量: ⭐⭐⭐⭐ 形式化定义清晰，附录给出完整算法、证明和 1 步 worked example。
- 价值: ⭐⭐⭐⭐⭐ Agent 治理是即将到来的真实需求，方法可直接落地黑盒 API，且与内容水印兼容。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Why Agents Compromise Safety Under Pressure](why_agents_compromise_safety_under_pressure.md)
- [\[ACL 2026\] Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization](adaptive_text_anonymization_learning_privacy-utility_trade-offs_via_prompt_optim.md)
- [\[ACL 2026\] On Safety Risks in Experience-Driven Self-Evolving Agents](on_safety_risks_in_experience-driven_self-evolving_agents.md)
- [\[ACL 2026\] RISK: A Framework for GUI Agents in E-commerce Risk Management](risk_a_framework_for_gui_agents_in_e-commerce_risk_management.md)
- [\[ACL 2026\] SharedRequest: Privacy-Preserving Model-Agnostic Inference for Large Language Models](sharedrequest_privacy-preserving_model-agnostic_inference_for_large_language_mod.md)

</div>

<!-- RELATED:END -->
