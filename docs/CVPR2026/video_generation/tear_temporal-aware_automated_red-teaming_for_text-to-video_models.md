---
title: >-
  [论文解读] TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models
description: >-
  [CVPR 2026][视频理解][文本到视频安全] 提出 TEAR，首个针对 T2V 模型时序维度漏洞的自动化红队测试框架，通过两阶段优化的时序感知测试生成器和迭代精炼模型，生成文本上无害但能利用时序动态触发有害视频的提示，在开源和商业 T2V 模型上达到 80%+ 的攻击成功率。
tags:
  - CVPR 2026
  - 视频理解
  - 文本到视频安全
  - 自动化红队测试
  - 时序感知
  - 对抗提示生成
  - AI安全
---

# TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models

**会议**: CVPR 2026  
**arXiv**: [2511.21145](https://arxiv.org/abs/2511.21145)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 文本到视频安全, 自动化红队测试, 时序感知, 对抗提示生成, AI安全

## 一句话总结
提出 TEAR，首个针对 T2V 模型时序维度漏洞的自动化红队测试框架，通过两阶段优化的时序感知测试生成器和迭代精炼模型，生成文本上无害但能利用时序动态触发有害视频的提示，在开源和商业 T2V 模型上达到 80%+ 的攻击成功率。

## 研究背景与动机
文本到视频（T2V）模型（如 Veo、Hailuo、Wan）已能生成高质量、时序连贯的视频，但也可能被触发生成有害内容，安全评估至关重要。

**核心矛盾**：现有红队测试方法主要针对静态图像和文本生成，无法捕捉视频生成中特有的**时序动态安全风险**。视频的危害性可以不存在于任何单帧中，而是由帧序列的时序组合产生——例如，单独描述"一个人拿起刀"和"另一个人倒下"都是无害的，但它们的时序连接可构成暴力场景。

**现有方法的不足**：
1. LLM 红队方法（如 CuriDial、FLIRT）关注文本对抗，完全忽略视频时序维度
2. 图像红队方法（如 ART、Groot）将视频视为独立帧序列，无法评估时序组合产生的新风险
3. T2VSafetyBench 是首个 T2V 安全基准，但仅使用静态有害提示，攻击成功率有限
4. 引入时序信息大幅扩展了搜索空间，带来新的技术挑战

**核心 idea**：将红队提示生成建模为 MDP，分两阶段优化 LLM 生成器——先在构造数据上初始化，再通过结合提示安全奖励和视频时序一致性奖励的在线偏好学习进行精炼，配合迭代精炼模型不断提升攻击效力。

## 方法详解

### 整体框架
TEAR 包含三个组件：
1. **时序感知测试生成器** — 核心组件，基于 LLM 训练，从种子提示生成时序重构的对抗提示
2. **精炼模型** — 基于多模态 LLM（Qwen-3-VL），根据判断反馈迭代改进提示
3. **目标 T2V 模型** — 被测试的视频生成模型

红队目标：发现提示集 $\mathcal{P}_v^*$，满足 $\Phi_P(p) = 0$（文本判断为安全）且 $\Phi_V(\mathcal{M}(p)) = 1$（生成视频判断为有害）。

### 关键设计

1. **规则化数据集构造（Stage 1 数据准备）**:

    - 功能：从元有害提示构造训练数据
    - 为什么：需要高质量的"文本无害但视频有害"的 prompt pair 来初始化生成器
    - 怎么做：对每个有害种子提示 $p_s$，用 LLM 按三条规则进行时序重写：
        - **时序解构**：将有害指令分解为按时间排列的离散静态事件描述
        - **顺序强制**：插入时序连接词（"首先"、"两秒后"）确保严格的时间进展
        - **时序-空间合成**：确保危害性不存在于任何单独描述中，仅从时序组合中涌现
    - 数据选择：只保留满足 $\Phi_P(p_t)=0 \wedge \Phi_V(\mathcal{M}(p_t))=1$ 的样本

2. **初始生成器训练（Stage 2）**:

    - 功能：在基础 LLM 上用构造数据进行初始化训练
    - 为什么：让生成器学习数据集的粗略分布，能生成初始对抗提示
    - 怎么做：自回归 NLL 损失 $\mathcal{L}_{Ini} = -\mathbb{E}_{(p_s,p_t)\sim \mathbf{D}_p} \log p(p_t|p_s, I)$
    - 基于 Llama-3 + LoRA 微调

3. **时序感知在线偏好学习（Stage 3 核心）**:

    - 功能：进一步优化生成器，使其既生成文本安全的提示，又确保视频层面的有害语义
    - 为什么：初始训练只学到了粗略分布，需要与实际 T2V 模型交互来精调
    - **提示空间优化** — 奖励函数 $\mathbf{R}_{pmt}$：
        - 安全性项 $(1 - \mathbf{g}_t(p_t))$：鼓励提示通过仇恨言论检测器
        - 模式对齐项 $\frac{\mathbf{g}_r(p_t)+1}{2}$：鼓励提示与预构造的时序风格样本的嵌入原型对齐（余弦相似度）
        - $\mathbf{R}_{pmt} = \alpha_1 \cdot (1 - \mathbf{g}_t(p_t)) + \alpha_2 \cdot \frac{\mathbf{g}_r(p_t)+1}{2}$
    - **时序空间一致性** — 奖励函数 $\mathbf{R}_{con}$：
        - 将生成视频分解为帧序列，用视频编码器提取时序特征
        - 全局一致性 $\mathbf{g}_{gc}$：衡量种子提示的有害语义与生成视频的全局时序对齐
        - 内部一致性 $\mathbf{g}_{ic}$：衡量视频自身的时序连贯性（生成质量）
        - $\mathbf{R}_{con} = \min(\beta, \frac{\mathbf{g}_{gc} - \gamma_1}{\theta_1} + \frac{\mathbf{g}_{ic} - \gamma_2}{\theta_2})$
    - 采用 PPO 范式最大化总奖励，附加 KL 惩罚防止过优化：
    - $\zeta = \mathbb{E}[\mathbf{R}_{pmt}(p_t) + \mathbf{R}_{con}(p_s, p_t) - \lambda \log \frac{G_\delta(p_t|p_s)}{G_{initial}(p_t|p_s)}]$

4. **测试用例精炼（迭代）**:

    - 功能：对生成器输出的初始提示进行迭代优化
    - 为什么：生成器输出只是初始化，需结合反馈进一步提升隐蔽性和有效性
    - 怎么做：精炼模型（Qwen-3-VL）接收提示、生成视频、$\Phi_P$ 和 $\Phi_V$ 的反馈（含分数、解释、建议），输出修订后的提示 $p_{t+1}$，形成闭环迭代

### 损失函数 / 训练策略
- Stage 2：NLL 损失初始化，4000 步，batch 8，LR $1.0 \times 10^{-5}$
- Stage 3：PPO 在线 RL，AdamW，LR $1.0 \times 10^{-6}$，cosine scheduler
- 生成用 beam search，$b=16$，100 token 上限

## 实验关键数据

### 主实验 — 开源模型攻击成功率
| 方法 | Hunyuan-Video ASR | Wan 2.2 ASR | 提示安全通过率 |
|------|-------------------|-------------|----------------|
| Naive | 2.6% | 2.3% | ~98% |
| T2VSafetyBench | 40.8% | 37.2% | ~52% |
| UVD | 29.0% | 31.0% | ~90% |
| FLIRT | 57.2% | 56.4% | ~51% |
| ART | 52.6% | 49.7% | ~92% |
| **TEAR** | **82.3%** | **80.5%** | **~96%** |

### 商业模型攻击成功率
| 模型 | 大部分类别 ASR | API/NSFW 通过率 |
|------|---------------|----------------|
| Veo-3.1 | ≥85% | ~98% |
| Hailuo-2.3 | ≥85% | ~98% |
| Ray-2 | 略低 | ~98% |

### 消融与分析
| 分析维度 | 结果 |
|---------|------|
| 无种子生成（Seed-free） | Hunyuan 79.2%, Wan 76.9%（仍大幅领先 FLIRT ~55%） |
| 迭代精炼效果 | ASR 从 57-71%（直接生成）提升至 83-95%（8轮精炼） |
| 提示多样性 | 1-AvgSelfBLEU: 0.71-0.76, 1-Cossim: 0.69-0.73 |
| 跨模型迁移性 | 20 个源-目标组合平均 ASR 76.4%，大部分 >70% |

### 关键发现
- TEAR 的 ASR（82.3%）远超此前最佳 FLIRT（57.2%），**提升 25 个百分点**
- 商业 T2V 服务的安全过滤器对时序组合攻击几乎无效（通过率 ~98% 但 ASR ≥85%）
- 跨模型迁移性极强（平均 76.4%），表明各 T2V 模型共享基本的时序安全漏洞
- 迭代精炼是关键：前 3 轮 ASR 提升最快，8 轮后趋于收敛
- Pornography 类别攻击最难（ASR 最低），可能因为此类别的安全过滤更严格

## 亮点与洞察
- **首次揭示 T2V 模型的时序维度安全漏洞**——单帧无害但时序组合有害，这是一个被严重忽视的风险
- 三条时序重写规则（解构、顺序强制、合成）优雅地定义了时序攻击的语义空间
- 双重奖励设计（提示安全 + 视频时序一致性）巧妙地平衡了隐蔽性和有效性
- 商业模型的安全失败令人警醒：提示过滤器几乎全部放行，但生成视频高度危险

## 局限与展望
- 方法主要站在攻击者角度，对应的防御策略未深入讨论
- 视频判断系统 $\Phi_V$ 依赖 GPT-4o API，其评估可靠性本身也有不确定性
- 6 个有害类别的划分可能不够全面，如虚假信息、隐私侵犯等时序场景未覆盖
- 在线 RL 阶段需要大量调用目标 T2V 模型生成视频，计算成本高昂
- 论文含有害内容示例，虽有警告但伦理讨论可更充分

## 相关工作与启发
- ART 和 FLIRT 是 T2I 红队方法的代表，本文将其适配到 T2V 但证明它们忽略时序维度
- T2VSafetyBench 是首个 T2V 安全基准，但仅用静态有害提示
- 对 AI 安全的启发：多模态模型的安全评估必须考虑模态间的时序/组合效应，单模态安全不等于多模态安全

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统定义和攻击 T2V 模型的时序安全漏洞，问题定义本身就是重要贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个 T2V 模型（含 3 个商业）、4 个 baseline、6 类有害内容、多维度分析
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰、方法层次分明、实验全面
- 价值: ⭐⭐⭐⭐⭐ 对 T2V 安全领域有奠基性贡献，直接推动商业模型安全改进

<!-- RELATED:START -->

## 相关论文

- [Target-Aware Video Diffusion Models](../../ICLR2026/video_generation/target-aware_video_diffusion_models.md)
- [EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models](../../ICCV2025/video_generation/efficientmt_efficient_temporal_adaptation_for_motion_transfer_in_text-to-video_d.md)
- [SweetTok: Semantic-Aware Spatial-Temporal Tokenizer for Compact Video Discretization](../../ICCV2025/video_generation/sweettok_semantic-aware_spatial-temporal_tokenizer_for_compact_video_discretizat.md)
- [VideoCoF: Unified Video Editing with Temporal Reasoner](videocof_unified_video_editing_with_temporal_reasoner.md)
- [Compressed-Domain-Aware Online Video Super-Resolution](compressed-domain-aware_online_video_super-resolution.md)

<!-- RELATED:END -->
