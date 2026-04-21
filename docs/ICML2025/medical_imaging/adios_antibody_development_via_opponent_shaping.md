---
title: >-
  [论文解读] ADIOS: Antibody Development via Opponent Shaping
description: >-
  [ICML 2025][医学图像][对手塑形 (Opponent Shaping)] 将多智能体强化学习中的对手塑形（Opponent Shaping）引入抗体设计，提出 ADIOS 元学习框架：外层循环优化抗体，内层循环模拟病毒适应性逃逸，使设计出的"塑形抗体"（shapers）不仅能对抗当前病毒变种，还能主动引导病毒向更弱、更易被靶向的方向进化。
tags:
  - ICML 2025
  - 医学图像
  - 对手塑形 (Opponent Shaping)
  - 抗体设计
  - 病毒逃逸
  - 元学习
  - 博弈论
---

# ADIOS: Antibody Development via Opponent Shaping

**会议**: ICML 2025  
**arXiv**: [2409.10588](https://arxiv.org/abs/2409.10588)  
**代码**: [github.com/olakalisz/adios](https://github.com/olakalisz/adios)  
**领域**: 计算生物学 / 抗体设计  
**关键词**: 对手塑形 (Opponent Shaping), 抗体设计, 病毒逃逸, 元学习, 博弈论

## 一句话总结

将多智能体强化学习中的对手塑形（Opponent Shaping）引入抗体设计，提出 ADIOS 元学习框架：外层循环优化抗体，内层循环模拟病毒适应性逃逸，使设计出的"塑形抗体"（shapers）不仅能对抗当前病毒变种，还能主动引导病毒向更弱、更易被靶向的方向进化。

## 研究背景与动机

传统抗病毒疗法（包括疫苗和单克隆抗体）的设计只针对**当前**病毒株——这是一种**短视**（myopic）策略。虽然初始疗效可能很高，但治疗本身施加的选择压力会驱动新突变株出现，导致疗法失效。COVID-19 大流行就是典型案例：B.1.351 变异株显著降低了疫苗保护效力。

核心洞察：**我们的疗法不可避免地影响病毒进化，与其被动应对，不如主动利用这一影响力**。具体来说：
- 短视抗体：初始结合力强，但病毒很快进化出逃逸突变（variant B），疗法失效
- 塑形抗体（shapers）：考虑长期博弈，在保持结合力的同时，将病毒进化引导向更弱的变种（variant C）

现有抗体设计方法（能量优化、序列语言模型、扩散模型等）都未考虑疗法对病毒进化的反馈效应。EVEscape 等方法虽然能**预测**病毒逃逸，但不能**影响**逃逸方向。ADIOS 填补了这一空白。

## 方法详解

### 整体框架

ADIOS 将抗体与病毒的交互建模为**双人零和博弈**，并采用嵌套的元学习架构：

- **外层循环（Antibody Optimisation Loop）**：用遗传算法优化抗体序列，目标是最大化长期（考虑病毒逃逸后的）抗体适应度
- **内层循环（Simulated Viral Escape via Evolution）**：给定当前抗体，模拟病毒通过突变逐步逃逸的过程

三大核心组件：
1. **病毒-抗体博弈**：定义双方的动作空间和收益函数
2. **模拟病毒逃逸**：基于进化模拟病毒如何适应给定抗体
3. **抗体优化**：利用蒙特卡洛采样 + 遗传算法优化抗体

### 关键设计

#### 1. 病毒-抗体博弈（Virus-Antibody Game）

双方的动作为氨基酸序列：病毒序列 $v \in \mathbb{A}^{N_v}$，抗体序列 $a \in \mathbb{A}^{N_a}$（$\mathbb{A}$ 为 20 种氨基酸集合）。

抗体收益函数设计精妙，包含三项：

$$R_a(v, a) = B(v, a) - B(t_a^-, a) - B(v, t_v^+)$$

- $B(v, a)$：抗体与病毒的结合强度（越高越好）
- $B(t_a^-, a)$：抗体与人体蛋白（anti-target）的结合（惩罚项，防止抗体"太粘"攻击自身）
- $B(v, t_v^+)$：病毒与宿主细胞受体的结合（鼓励抗体间接阻断病毒感染能力）

病毒收益为 $R_v = -R_a$（零和博弈）。这一设计保证：
- 病毒不能通过变得完全惰性来逃避（否则丧失感染能力）
- 抗体不能通过万能粘合来取胜（否则会攻击人体蛋白）

#### 2. 模拟病毒逃逸（Inner Loop）

给定初始病毒 $\hat{v}$ 和固定抗体 $a$，模拟 $H$ 步进化逃逸：

**Algorithm 1 — 内层循环：**
1. 初始化 $\hat{v}_0 = \hat{v}$
2. 对每一代 $i = 0, ..., H-1$：
    - 生成种群：复制 $\hat{v}_i$ 共 $P=15$ 份，每份随机施加约 1 个氨基酸突变
    - 计算每个突变体的适应度 $R_v(v_k^i, a)$
    - 按 Boltzmann 分布采样下一代：$\mathbb{P}(\hat{v}_{i+1} = v_k^i) \propto \exp(\beta \cdot R_v(v_k^i, a))$
3. 输出逃逸轨迹 $\hat{\mathbf{v}} = [\hat{v}_0, \hat{v}_1, ..., \hat{v}_H]$

温度参数 $\beta$ 控制选择随机性：$\beta \to \infty$ 为确定性最优适应度选择。

#### 3. 抗体优化（Outer Loop）

定义抗体的**真实**目标函数——逃逸平均适应度：

$$F_{\hat{v}}^H(a) = \mathbb{E}_{\hat{\mathbf{v}} \sim \text{Ev}(\hat{v}, a)} \left[ \frac{1}{H+1} \sum_{i=0}^{H} R_a(\hat{v}_i, a) \right]$$

当 $H=0$ 时，退化为短视目标 $F_{\hat{v}}^0(a) = R_a(\hat{v}, a)$。

**Algorithm 2 — 外层循环：**
1. 从随机抗体 $\hat{a}_0$ 出发
2. 对每一步 $i = 0, ..., N-1$（$N=30$）：
    - 生成种群：$P_a = 40$ 个抗体（原始 + 39 个单点突变体）
    - 对每个候选抗体，用 $\eta = 5$ 条蒙特卡洛逃逸轨迹估计 $F_{\hat{v}}^H(a_k^i)$
    - 贪心选择：$\hat{a}_{i+1} = \arg\max_k \mathbb{E}[F_{\hat{v}}^H(a_k^i)]$
3. 输出最优抗体 $\hat{a}_N$

#### 4. GPU 加速绑定模拟器

原始 Absolut! 框架用 C++ 运行在 CPU 上，无法支撑大规模博弈模拟。作者的关键工程贡献：
- 用 JAX 重新实现核心绑定计算，支持 GPU 加速
- 利用 Miyazawa-Jernigan 能量矩阵计算结合能
- **Pose 剪枝**：登革病毒有约 150 万个可能 pose，但仅 1027 个曾作为最低能量 pose；保留 ≥18 对残基的 pose（约 37000 个），精度几乎不损失
- 提供高分辨率和低分辨率两种模拟器：训练用低分辨率，验证用高分辨率

### 损失函数 / 训练策略

- **目标函数**：最大化逃逸平均适应度 $F_{\hat{v}}^H(a)$，horizon $H$ 越大越接近真实长期目标
- **优化方式**：遗传算法（单点突变 + 贪心选择），非梯度方法
- **计算预算权衡**：每步内层循环需要 $O(H \cdot P)$ 次绑定查询，longer horizon 更准确但更昂贵
- **验证策略**：训练用低分辨率模拟器，报告结果用高分辨率模拟器（模拟从仿真到现实的迁移）

## 实验关键数据

### 主实验

实验在登革病毒（PDB: 2R29）上进行，抗体序列长度 $N_a = 11$（CDRH3 区域），病毒序列长度 $N_v = 97$。

| 指标 | Shaper (H=100) | Myopic (H=0) | 关键发现 |
|------|---------------|--------------|---------|
| 逃逸平均适应度 $F_v^{100}$ | 显著更高 | 较低 | Top 10% shapers 优于所有 myopic |
| 短视适应度 $R_a(v,a)$ | 略低 | 更高 | Shapers 牺牲短期换长期 |
| 逃逸后 10 步 | 略逊于 myopic | 初始更优 | 前 10 步 myopic 有优势 |
| 逃逸后 100 步 | 显著更优 | 大幅下降 | Shapers 长期优势明显 |

**跨病原体泛化性**（4 种额外病原体）：

| 病原体 | PDB | Shaping 效果 | 特殊发现 |
|--------|-----|-------------|---------|
| 西尼罗病毒 | 1ZTX | ✓ 有效 | H=20 在有限预算下表现更优 |
| 流感神经氨酸酶 | 4QNP | ✓ 有效 | 趋势与登革一致 |
| MERS-CoV | 5DO2 | ✓ 有效 | H=100 需要更多优化步才收敛 |
| 艰难梭菌（细菌） | 4NP4 | ✓✓ 特别强 | H=100 显著压倒其他所有配置 |

### 消融实验

| 配置 | 关键指标 $F_v^{100}$ | 说明 |
|------|---------------------|------|
| H=0 (myopic) | 基线 | 不考虑逃逸，初始绑定好但长期差 |
| H=5 | 优于 myopic | 短 horizon 已有改善 |
| H=10 | 进一步提升 | 中等 horizon |
| H=20 | 接近 H=100 | **最佳性价比**——计算归一化后几乎持平 |
| H=100 | 最优 | 步数归一化最佳，但计算代价高 |

JAX 加速效果：

| 实现 | 硬件 | 加速比 |
|------|------|--------|
| 原始 Absolut! (C++) | Apple M2 Max (CPU) | 1× |
| JAX 重实现 | Nvidia A40 (GPU) | ~10,000× |

### 关键发现

1. **"进攻即最佳防守"**：交叉评估实验表明，H=100 shapers 诱导的逃逸病毒 $v_{100}$ 对**所有**抗体（不仅是诱导它的 shaper）都更容易被靶向。这证明 shapers 确实在**塑形**病毒进化，而非仅仅变得更鲁棒。

2. **氨基酸分布差异**：shapers 的氨基酸分布更均匀，myopic 抗体倾向聚集于极端结合能的氨基酸。均匀分布使 shapers 对突变更鲁棒——病毒难以通过避开特定高结合氨基酸来逃逸。

3. **Pose 矩阵分析**：H=100 shapers 通过两种机制约束病毒逃逸——(a) 阻止病毒将抗体最低结合氨基酸（如 Lysine）纳入 pose；(b) 抑制病毒从 pose 中移除自身高结合氨基酸（如 Isoleucine, Methionine）。

4. **外部压力下的鲁棒性**：即使加入额外的短视抗体外部压力（模拟多种疫苗共存场景），shaping 效果虽有减弱但依然显著。

## 亮点与洞察

- **跨领域创新**：将多智能体 RL（LOLA, M-FOS）的对手塑形思想首次应用于计算生物学，是 AI for Science 的优秀范例
- **工程贡献突出**：JAX 重实现带来 10,000× 加速，使大规模博弈模拟成为可能
- **实用指导**：H=20 是成本效益最优的 horizon 选择，为计算预算有限的场景提供了实操建议
- **可解释性分析**：不仅展示 shapers 有效，还深入分析了**为什么**有效（氨基酸分布、pose 矩阵变化），使结果具有生物学可解读性
- **混合策略启示**：结合 shaping 抗体和 myopic 抗体的鸡尾酒疗法可能兼顾短期疗效和长期进化控制

## 局限与展望

1. **简化的绑定模型**：Absolut! 是离散化的简化模拟器，与真实蛋白质相互作用有较大差距；未来可集成 AlphaFold3 等更精确的模型
2. **固定结构假设**：假设病毒抗原结构在逃逸过程中不变，但实际突变可能导致构象变化
3. **单一抗体优化**：当前只优化单个抗体，未考虑多抗体组合的协同效应
4. **序列空间局限**：仅优化 CDRH3 区域（11 个氨基酸），未涉及抗体其他可变区域
5. **离 wet lab 验证尚远**：需要更精确的模拟器和实验验证才能走向临床
6. **进化模型简化**：假设每代平均 1 个突变，且仅考虑点突变，未涵盖重组、插入/缺失等

## 相关工作与启发

- **Opponent Shaping 谱系**：LOLA → M-FOS → ADIOS，从博弈论智能体扩展到生物分子博弈
- **病毒逃逸预测**：EVEscape、Han et al. 预测变异但不影响变异，ADIOS 同时预测并引导
- **抗体设计方法**：RAbD、AbDiffuser 等关注设计更好抗体，ADIOS 关注**如何考虑进化反馈来设计**
- **潜在扩展领域**：抗菌素耐药性、癌症单克隆抗体治疗（优化 mAb 同时塑形肿瘤细胞进化）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次将对手塑形引入抗体设计，跨领域创新性极强
- 实验充分度: ⭐⭐⭐⭐ — 5 种病原体验证 + 丰富消融，但缺乏 wet lab 验证
- 写作质量: ⭐⭐⭐⭐⭐ — 框架清晰，图示精美，可解释性分析到位
- 价值: ⭐⭐⭐⭐ — 概念验证阶段，但思路对未来抗病毒/抗癌治疗有深远启示

<!-- RELATED:START -->

## 相关论文

- [Fine-Tuning Diffusion Models via Intermediate Distribution Shaping](../../ICLR2026/medical_imaging/fine-tuning_diffusion_models_via_intermediate_distribution_shaping.md)
- [AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design](../../ICLR2026/medical_imaging/afd-instruction_a_comprehensive_antibody_instruction_dataset_with_functional_ann.md)
- [iDPA: Instance Decoupled Prompt Attention for Incremental Medical Object Detection](idpa_instance_decoupled_prompt_attention_for_incremental_medical_object_detectio.md)
- [CFP-Gen: Combinatorial Functional Protein Generation via Diffusion Language Models](cfp-gen_combinatorial_functional_protein_generation_via_diffusion_language_model.md)
- [Doubly Protected Estimation for Survival Outcomes Utilizing External Controls for Randomized Clinical Trials](doubly_protected_estimation_for_survival_outcomes_utilizing_external_controls_fo.md)

<!-- RELATED:END -->
