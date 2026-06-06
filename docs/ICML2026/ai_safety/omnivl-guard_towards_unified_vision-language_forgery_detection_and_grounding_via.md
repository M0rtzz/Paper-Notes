---
title: >-
  [论文解读] OmniVL-Guard: Towards Unified Vision-Language Forgery Detection and Grounding via Balanced RL
description: >-
  [ICML 2026][AI安全][伪造检测] 本文针对"图/文/视频混合伪造同时检测+定位"这一统一任务，提出 OmniVL-Guard，用 Self-Evolving CoT 合成高质量冷启动数据 + ARSPO（非线性奖励映射 + 动态任务权重）解决多任务 RL 中"简单的真假分类抢走梯度、细粒度定位学…
tags:
  - "ICML 2026"
  - "AI安全"
  - "伪造检测"
  - "Grounding"
  - "多任务RL"
  - "奖励整形"
  - "自演化CoT"
---

# OmniVL-Guard: Towards Unified Vision-Language Forgery Detection and Grounding via Balanced RL

**会议**: ICML 2026  
**arXiv**: [2602.10687](https://arxiv.org/abs/2602.10687)  
**代码**: https://github.com/shen8424/OmniVL-Guard (有)  
**领域**: AI安全 / 多模态伪造检测 / 强化学习 / VLM  
**关键词**: 伪造检测、Grounding、多任务RL、奖励整形、自演化CoT

## 一句话总结
本文针对"图/文/视频混合伪造同时检测+定位"这一统一任务，提出 OmniVL-Guard，用 Self-Evolving CoT 合成高质量冷启动数据 + ARSPO（非线性奖励映射 + 动态任务权重）解决多任务 RL 中"简单的真假分类抢走梯度、细粒度定位学不动"的难度偏置问题，在 In-Domain 上视频时序定位 tIoU +37.8、文本定位 F1 +22.9，并在四个 OOD benchmark 上做到零样本 SOTA。

## 研究背景与动机

**领域现状**：当前的伪造检测/篡改定位工作绝大多数是单模态（纯图、纯文、纯视频）或顶多双模态（图-文、视频-文），各自配一套专家模型；HAMMER、FKA-Owl、Fake-VLM、FakeSV-VLM 等代表性方法都只能"管一个口子"。

**现有痛点**：真实社交媒体上的虚假信息是图、文、视频高度交织的"全模态"内容，单/双模态检测器面对这种混合输入要么不能处理、要么无法同时给出"真假判定 + 篡改位置"，因而需要一个统一框架同时覆盖二分类与图/文/视频三种 grounding。但作者发现：直接用通用 MLLM（GPT-5/Gemini3/Seed1.6）零样本做这件事，二分类还能凑合到 73%，三个定位任务全部塌到 20-35（Table 1a）；直接 SFT 又因推理能力不足无法跨模态泛化。

**核心矛盾**：自然的选择是引入 RL（GRPO 一类）让 MLLM 自己探索推理路径，但作者通过实验+理论同时观察到一个"难度偏置"现象：二分类是判别题、奖励信号强且容易爬升；图/文/视频定位是回归/区间题、需要精细感知、奖励信号稀疏。GRPO 把所有任务平均一起优化时，二分类一路 +36%，图像定位反而退步 -0.1%——简单任务"绑架"了梯度更新方向。SAPO 等改进虽稍好但同样跑不动定位。

**本文目标**：拆成两个子问题——(1) 怎样为这种细粒度多模态推理任务造出高质量 CoT 冷启动数据；(2) 怎样设计一个新的 RL 目标，让简单任务不抢资源、难任务能持续受益。

**切入角度**：作者对 GRPO 类目标做了对参数 $\theta$ 的二阶展开（公式 4），把梯度变化率拆成"奖励映射敏感度 $g_k'(\cdot)$"和"任务难度敏感度 $H_k'(\theta,q,\tau)$"两项。难任务因停留在性能平台期，$H_k'$ 天然很小，所以即便归一化奖励到同尺度，简单任务仍主导"梯度加速度"。这条解析式直接给出补救方向：用一个**凸的、随性能上升斜率变陡的非线性奖励映射函数** $g_k(\cdot)$，把高分响应的梯度贡献放大，从而对冲难任务的 $H_k'$ 小所带来的衰减。

**核心 idea**：用"非线性奖励整形 + 动态任务权重"取代均匀加权 RL，让简单任务该收敛就收敛、难任务该被加权就被加权；并用自演化 CoT 合成出能真正解题（而非反推答案）的冷启动数据，避免 GT 注入式蒸馏带来的 hindsight bias。

## 方法详解

### 整体框架
输入是任意一段图/文/视频或它们的组合，模型要同时输出 (1) 真假二分类 和 (2) 对应模态上的篡改区域——图像空间 mask（IoU）、文本 token 跨度（F1）、视频时间区间（tIoU）。整个 pipeline 走"FSFR 数据集 → Qwen3VL-8B 上 SFT 冷启动 → ARSPO 多任务 RL → 在 In-Domain 与 OOD 测试"四步。其中 FSFR 由 Self-Evolving CoT Generation 离线构造，含 73k SFT 高质量 CoT 样本和 110k RL 训练样本；ARSPO 在 RL 阶段持续工作，按训练步动态调整四个任务（二分类 / 图像定位 / 文本定位 / 视频定位）的奖励曲线和任务权重。

### 关键设计

1. **Self-Evolving CoT Generation（FSFR 数据集构造）**：

    - 功能：为细粒度伪造检测+定位任务造一份"既能解题又不带答案泄漏"的高质量 CoT 冷启动数据集 $\text{FSFR}_{\text{sft}}$，配套留出 $\text{FSFR}_{\text{rl}}$ 给 RL 用。
    - 核心思路：作者把"造 CoT"形式化成一个 *Efficiency-Bias Dilemma*——闭源 MLLM 直接生成 CoT 质量太低，而注入 GT 又导致 hindsight bias、模型从答案反推过程，破坏 RL 探索。解法是四阶段自演化：(a) 从公开数据池（FakeNewsCorpus、ForgeryNet、GenVideo、DGM4 等）汇总并按比例划分 $D_s/D_r/D_t$；(b) 用 SOTA MLLM 集合 $\mathcal{M}=\{\text{Seed1.6-VL, Gemini3, ChatGPT5}\}$ 推理 $D_s$ 的小子集，经"GT 过滤 + 另一个 MLLM 一致性核验"得到 6.7k 的种子集 $D_s^0$，SFT+RL 得到 warm-up 策略 $\pi_0$；(c) 第 $k$ 轮用 $\pi_{k-1}$ 给剩余样本生成 CoT，再经 GT 过滤 + SOTA MLLM 校验，合并入 $D_s^k$，**每轮都从 base Qwen3VL-8B 重训**以避免分布漂移；(d) 对始终错的 hard 样本走 Multi-Agent Collaborative Hard-CoT Synthesis——一个 MLLM 借 GT 生成 CoT，第二个 MLLM 当 "Refiner" 把痕迹改造成"假装不知道答案的自然推理"，第三个 MLLM 评分过滤。三轮自演化后即达到饱和（Table 5: $D_s^4$ 比 $D_s^3$ 几乎无增益）。
    - 设计动机：直接蒸馏不够（通用 MLLM 不懂法证细节），直接给答案让模型反推又毁了 RL；自演化用"模型自己跑得通"作为质量代理，再用第三方 MLLM 当裁判，把"是否走对推理"和"是否得到正确答案"两个信号解耦，规避 hindsight bias。

2. **ARSPO – Task-Based Reward Mapping Function（TBRMF）**：

    - 功能：用"按任务定制的非线性奖励映射"显式调整每个任务的梯度贡献，让简单任务别抢资源、难任务能持续学。
    - 核心思路：基于 4.1 节对梯度的二阶展开（核心公式 $\frac{d}{d\theta}(W_{i,t}(\theta)\hat{A}_{i,k}) = W'_{i,t}(\theta)\hat{A}_{i,k} + W_{i,t}(\theta)\cdot \frac{g_k'(H_k)}{G\sigma}[(G-1)-\hat{A}_{i,k}^2]\,H_k'(\theta,q,\tau)$)，作者发现可以通过把奖励函数 $A_{i,k}=g_k(x_{i,k})$ 由原始性能指标 $x_{i,k}$ 经 $g_k$ 映射得到。对二分类（容易）选恒等映射 $g_k(x)=x$，避免无谓放大；对三个细粒度定位任务，选凸函数 $g_k(x)=e^{a_k x}$（取 $a=3$，Figure 4 网格扫出来最佳），这样在高性能区域斜率更陡，组内得分高的 response 梯度被显著放大——把"接近正确答案但还差一点"的样本变成最强的学习信号。
    - 设计动机：纯靠归一化（GRPO/SAPO）只能让奖励尺度可比，不能改变"任务难度敏感度 $H_k'$"。凸映射相当于在奖励侧把高分样本的边际收益拉高，等价于补偿 $H_k'$ 的衰减；这也解释了 5.3 节实验为什么单任务训练下指数映射依然显著优于线性映射：ARSPO 的本质不是"平衡"，而是"重塑梯度信号"。

3. **ARSPO – Dynamic Coefficient Adjustment（DCA，Algorithm 1）**：

    - 功能：在训练过程中周期性监测每个任务的相对学习状态，自适应调节四个任务的全局权重 $l_{k,s}$，避免"水桶效应"。
    - 核心思路：先有一个 warm-up 阶段（$s<T_{warm}$）记录每个任务的均值作为冻结 baseline $B_k$；之后每 $T$ 步评估两个量——总体相对增益 $\Delta_{\text{total},k}=(\mu_k-B_k)/B_k$（衡量"长期是否跟得上"）和近期变化 $\delta_{\text{recent}}=\mu_k-\mu_{\text{past}}$（衡量"短期趋势"）；按"momentum 保护（在上升期则不动） → regression rescue（明显回退则乘 $\alpha_{\text{boost}}$ 抢救） → high-performance decay（任务已达标则乘 $\alpha_{\text{decay}}$ 缓慢退场，但下限为 1） → laggard support（找出 $k_{\text{lag}}=\arg\min_k\Delta_{\text{total},k}$ 并放大权重，上限 4）"四档优先级调整 $l_{k,s}$，最后整体除以最小系数做 rescaling，再代入 $\nabla_\theta \mathcal{J}_{\text{arspo}}$ 更新参数。
    - 设计动机：TBRMF 是"静态的奖励曲线整形"，但训练动态会随时间变化——某个任务可能本来落后但后来追上、或追上后又退化；DCA 给系统补一个闭环控制器，按"最弱者优先"原则把权重资源持续倾斜到当前最卡壳的任务上，与 TBRMF 形成"静态形状 + 动态权重"双重保险。

### 损失函数 / 训练策略
RL 目标在 GRPO 框架上加入动态系数 $l_{k,s}$：
$\mathcal{J}_{\text{arspo}}(\theta)=\sum_{k=1}^{K}\frac{|\mathcal{D}_k|}{|\mathcal{D}|}\mathbb{E}_{q\sim\mathcal{D}_k,\{y_i\}\sim\pi_{\theta_{\text{old}}}}\left[\frac{l_{k,s}}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}f_{i,t}(r_{i,t}(\theta))\hat{A}_{i,k}\right]$
其中优势 $\hat{A}_{i,k}=(A_{i,k}-\mu)/\sigma$ 仍按组内归一，但 $A_{i,k}=g_k(x_{i,k})$ 经过任务定制的非线性映射。基座为 Qwen3VL-8B，先用 $\text{FSFR}_{\text{sft}}$ SFT 冷启动，再用 $\text{FSFR}_{\text{rl}}$ 跑 ARSPO；warm-up 期间所有 $l_{k,s}=1$ 用来采 baseline。

## 实验关键数据

### 主实验
In-Domain（自建 $D_t$ 测试集，约 700k 样本）：

| 数据集 / 任务 | 指标 | 本文 | 之前 SOTA | 提升 |
|--------|------|------|----------|------|
| Text 二分类 | ACC | 96.20 | 89.23 (Qwen3VL-235B) | +6.97 |
| Image 二分类 | ACC | 93.12 | 90.39 (Fake-VLM) | +2.73 |
| Video 二分类 | ACC | 98.58 | 98.81 (FakeSV-VLM) | -0.23 |
| Text-Image 二分类 | ACC | 75.52 | 72.08 (FKA-Owl) | +3.44 |
| Image 定位 | IoU | 54.26 | 48.53 (HAMMER) | +5.73 |
| Text 定位 | F1 | 63.78 | 40.86 (HAMMER) | +22.92 |
| Video 定位 | tIoU | 59.22 | 21.43 (Qwen3VL-235B) | +37.79 |

OOD 零样本（无任何二次微调）：ISOT 文本 93.69（vs 88.74）、CASIA2.0 图像 63.64（vs 60.88）、MMFakeBench 图文 79.38（vs 62.32）、FakeSV 文-视频 63.55（vs 61.22），四个 benchmark 全部领先。

### 消融实验

| 配置 | Img-Loc IoU | Text-Loc F1 | Vid-Loc tIoU | $\Delta$ AVG |
|------|---|---|---|---|
| SFT only | 51.08 | 44.67 | 33.08 | — |
| SFT + SAPO | 51.24 | 54.33 | 44.10 | +24.33 |
| + TBRMF | 53.21 | 61.37 | 49.38 | +26.42 |
| + DCA | 52.95 | 59.88 | 53.49 | +26.93 |
| Full（SFT+SAPO+TBRMF+DCA） | **54.26** | **63.78** | **59.22** | **+28.33** |

### 关键发现
- **TBRMF 的"重塑梯度"作用是本文最强信号**：在单任务设置下（没有任务间资源竞争），指数映射依然比线性映射在图像定位上提升 +4%、文本定位 +8%，说明 ARSPO 的价值不在于"平衡多任务"，而在于"放大高分样本的梯度贡献"——这也直接对应 4.1 节对 $g_k'(\cdot)$ 的理论分析。
- **奖励曲率超过 $a=3$ 反而退化**：Figure 4(a-b) 显示 $a$ 太大时性能下降，作者归因于"奖励过拟合"——过陡的映射把 aleatoric 噪声当成信号、模型陷入局部最优；$a=3$ 是信号放大与训练稳定的甜点。
- **自演化在第 3 轮即饱和**：Table 5 显示 $D_s^4$ vs $D_s^3$ 几乎所有指标变化都在 0.2 以内，论证了把自演化停在 $k=3$ 的合理性，也节省了大量 MLLM 推理算力。
- **GRPO/SAPO 的"难度偏置"被实验直接捕捉到**：Table 1(b) 中 SFT+GRPO 让二分类暴涨 +36%、Image-Loc 反而 -0.1%——这是一个非常直观的"简单任务抢梯度"案例，也是本文整套方法的最强动机。

## 亮点与洞察
- **从二阶梯度展开里读出"难度偏置"的根因**：作者没有停留在"实验观察到 GRPO 偏向简单任务"这层现象，而是把目标对 $\theta$ 求二阶，把梯度加速度精准拆成"奖励敏感 $g_k'$ × 难度敏感 $H_k'$"两个因子；这一步把"该怎么改 RL"的问题降维成"该怎么选 $g_k$"，使得后续 TBRMF 的指数映射有理论可循而非纯调参，这种推导路径可以迁移到任何多任务 GRPO 的难度不平衡场景。
- **Hindsight-Bias-Free 的 CoT 合成范式很值得复用**：在所有"用 LLM 蒸馏 CoT"的工作里，"把 GT 喂给模型然后让它推理"几乎是默认做法，但本文明确指出这会让模型学会"反推"而不是"演绎"，并用"Refiner MLLM 重写 CoT 隐藏答案痕迹 + 第三方 MLLM 当裁判"两步把这个 bias 消掉；这一思路对所有需要"过程监督"而非"结果监督"的任务（数学推理、定理证明、形式化验证）都直接可用。
- **DCA 的"水桶效应"控制器是个轻量好用的小工具**：仅需四档优先级 + 一组阈值就能动态调整任务权重、没有梯度回传开销；可作为任何多任务 RL 训练的即插即用模块。

## 局限与展望
- 作者承认的局限较少，主要集中在"OOD 仍有提升空间"和"模型规模 8B 不大"，但实际局限还包括：(1) 全套流程依赖三个闭源 SOTA MLLM 做"裁判 + 生成 + 改写"，复现成本极高，开源社区难以平替；(2) 自演化每轮都从 base Qwen3VL 重训以避免分布漂移，3 轮意味着 4 次完整 SFT+RL，总训练开销在论文里没明确给出；(3) TBRMF 对每个任务都需要手调 $a_k$，仅在图/文定位上扫了 $a=3$，对更细粒度的子任务（如不同篡改类型）是否需要进一步差异化没有讨论。
- 一个有意思的改进方向是把 DCA 的四档启发式替换为基于 bandit / meta-learning 的可学习控制器，让权重调整本身也成为可优化对象，可能比当前固定阈值更鲁棒；同时可以把 TBRMF 的 $a_k$ 做成在线自适应，按当前任务在 reward 直方图上的分位数动态调整曲率。

## 相关工作与启发
- **vs HAMMER / FKA-Owl / AMD（双模态图-文专家）**：他们专注于图-文双模态、用专门的检测头做定位；本文做的是图/文/视频三模态统一、且把所有定位任务都转化为 MLLM 的文本输出（坐标 / token 跨度 / 时间区间），优势在于零样本跨模态泛化、劣势在于推理延迟显著高于专家模型。
- **vs DeepSeek-R1 / GRPO / SAPO**：本文承接 DeepSeek-R1 的"SFT 冷启动 + RL 提升推理"范式，但指出 GRPO/SAPO 在多任务设定下的难度偏置问题，并通过 ARSPO 提供解法；可视为 GRPO 在"任务难度不均衡"场景下的一个重要补丁。
- **vs Fake-VLM / FakeSV-VLM**：单模态专家模型在自己任务上很强（Fake-VLM 图像 90.39、FakeSV-VLM 视频 98.81），本文在单模态上仅持平或微差（Video 二分类 -0.23），但跨模态、跨数据集时领先幅度显著，体现"统一模型"的边际价值。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把"难度偏置"用二阶梯度公式精确归因，并用 TBRMF + DCA 给出有理有据的解法；自演化 CoT 中"Refiner 去 hindsight bias"也很有意思。
- 实验充分度: ⭐⭐⭐⭐⭐ 主表覆盖 4 种模态 × 7 个指标 + 4 个 OOD benchmark；消融拆开 SAPO/TBRMF/DCA 四种组合；并在单任务设定下二次验证 TBRMF 本质作用。
- 写作质量: ⭐⭐⭐⭐ 动机—理论—算法链条清晰，公式推导完整；只是 ARSPO 节里的 Algorithm 1 在正文需要反复跳到 Appendix 查阈值，可读性略受影响。
- 价值: ⭐⭐⭐⭐ 给"多模态全模态伪造检测"这件实际有用的事拿出了一个可工程化的统一方案，且 ARSPO 的难度偏置解法可被任意多任务 GRPO 训练复用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] A Unified Perspective on Adversarial Membership Manipulation in Vision Models](../../CVPR2026/ai_safety/a_unified_perspective_on_adversarial_membership_manipulation_in_vision_models.md)
- [\[CVPR 2026\] Towards Highly Transferable Vision-Language Attack via Semantic-Augmented Dynamic Contrastive Interaction](../../CVPR2026/ai_safety/towards_highly_transferable_vision-language_attack_via_semantic-augmented_dynami.md)
- [\[ICML 2026\] From Out-of-Distribution Detection to Hallucination Detection: A Geometric View](from_out-of-distribution_detection_to_hallucination_detection_a_geometric_view.md)
- [\[CVPR 2025\] Towards General Visual-Linguistic Face Forgery Detection](../../CVPR2025/ai_safety/towards_general_visual-linguistic_face_forgery_detection.md)
- [\[AAAI 2026\] Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection](../../AAAI2026/ai_safety/fine-grained_dino_tuning_with_dual_supervision_for_face_forgery_detection.md)

</div>

<!-- RELATED:END -->
