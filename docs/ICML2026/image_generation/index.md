---
title: >-
  ICML2026 图像生成方向22篇论文解读
description: >-
  22篇ICML2026的图像生成方向论文解读，涵盖扩散模型、图像编辑、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "图像生成"
  - "论文解读"
  - "论文笔记"
  - "扩散模型"
  - "图像编辑"
  - "对齐/RLHF"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎨 图像生成

**🧪 ICML2026** · **22** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (2)](../../ACL2026/image_generation/index.md) · [📷 CVPR2026 (209)](../../CVPR2026/image_generation/index.md) · [🔬 ICLR2026 (149)](../../ICLR2026/image_generation/index.md) · [🤖 AAAI2026 (78)](../../AAAI2026/image_generation/index.md) · [🧠 NeurIPS2025 (244)](../../NeurIPS2025/image_generation/index.md) · [📹 ICCV2025 (213)](../../ICCV2025/image_generation/index.md)

🔥 **高频主题：** 扩散模型 ×4 · 图像编辑 ×2 · 对齐/RLHF ×2

**[Adversarial Flow Models](adversarial_flow_models.md)**

:   作者在 GAN 训练目标上加一个最优传输正则 $\|G(z)-z\|^2$，把 GAN 的"任意搬运图"约束成 Wasserstein-2 最优搬运图，让纯 transformer 上的对抗训练第一次能稳定收敛并端到端做单步生成，ImageNet-256 上 1NFE FID 刷到 2.38（XL/2）和 1.94（112 层）。

**[Anomaly-Preference Image Generation (APO)](anomaly-preference_image_generation.md)**

:   作者把"少样本异常图像生成"重写为"无人工标注的偏好优化问题"：真实异常作为正样本，参考模型在同一时刻的去噪偏差作为隐式负样本，通过 DPO 风格 loss 让扩散模型对齐异常分布；再用按时间步调节 LoRA rank 的 TACA 保住结构多样性、用分层 CFG 调节文本-异常对齐强度，在 MVTec 等 benchmark 上同时刷新真实度和多样性。

**[Caracal: Causal Architecture via Spectral Mixing](caracal_causal_architecture_via_spectral_mixing.md)**

:   Caracal 用 $\mathcal{O}(L \log L)$ 的多头傅立叶（MHF）模块替换 Transformer 的 $\mathcal{O}(L^2)$ 注意力，通过"pad-FFT-multiply-iFFT-truncate"实现频域内的严格因果掩码，并完全去掉位置编码，仅用标准 FFT 算子（不依赖 Mamba 那样的 CUDA kernel）就在 Tiny→Large 全尺度上与 Llama / Mamba / Mamba-2 / Jamba 性能相当。

**[CARD: Coarse-to-fine Autoregressive Modeling with Radix-based Decomposition for Transferable Free Energy Estimation](card_coarse-to-fine_autoregressive_modeling_with_radix-based_decomposition_for_t.md)**

:   CARD 用"基数 $r$ 分解"把分子 3D 坐标双射映射为先粗后细的离散-连续混合 token 序列，让一个跨系统通用的自回归 Transformer 作为"零自由能 proposal"通过 BAR 直接估算任意分子系统的绝对自由能，在 70 个新系统的溶剂化任务上达到经典 MFES 的精度且推理快约 40 倍。

**[CoCoEdit: Content-Consistent Image Editing via Region Regularized Reinforcement Learning](cocoedit_content-consistent_image_editing_via_region_regularized_reinforcement_l.md)**

:   本文针对"编辑模型常在不该改的区域乱改"这一痛点，构造 CoCoEdit-40K 局部编辑数据集 + 提出 pixel-level 相似度 reward 补充 MLLM reward + 设计区域正则化 RL 目标（高奖励样本约束非编辑区一致、低奖励样本强迫编辑区做出改变），把 FLUX.1 Kontext 和 Qwen-Image-Edit 同时在编辑得分和 PSNR/SSIM 上提升，打破现有"提编辑能力必伤一致性"的 trade-off。

**[Conditional Diffusion Sampling](conditional_diffusion_sampling.md)**

:   本文提出 Conditional Diffusion Sampling（CDS）：通过推导一类条件随机插值（conditional interpolants），得到一个对未归一化目标分布的**精确闭式 SDE**（不需要神经网络拟合），再用 Parallel Tempering 高效采样这个 SDE 的初始分布——把 PT 的全局探索能力和扩散过程的局部细化能力拼起来，在 8 个目标分布、4 类任务上以更少的密度评估次数同时击败传统 MCMC、训练自由 MCMC 和神经采样器。

**[Diagnosing and Correcting Concept Omission in Multimodal Diffusion Transformers](diagnosing_and_correcting_concept_omission_in_multimodal_diffusion_transformers.md)**

:   论文用线性探针发现 MM-DiT (FLUX / SD3.5) 在中间层的某些注意力头里、其 text token 的 key 向量天然编码了"目标概念是否会出现"的二元信号，并由此提出 Omission Signal Intervention (OSI)：在 inference 时把"omission 类 - existence 类"的均值差方向以 $\alpha\sigma\boldsymbol{\theta}$ 的强度注入 Top-K 头的 key 向量，激发模型对缺失概念的"自我感知"并补全生成；在 FLUX 上 GenEval 6-object 准确率从 0.18 → 0.40，且无需任何 fine-tune。

**[End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer](end-to-end_autoregressive_image_generation_with_1d_semantic_tokenizer.md)**

:   EOSTok 用单阶段端到端管线把 1D ViT tokenizer 和自回归模型一起训练，靠新提出的 APR（Autoregressive Prediction Reconstruction）loss 把「next-token 预测」的梯度真正传回 pixel space 防止码本崩塌，再用「隐式对齐」把 DINOv2 语义注入 1D 隐空间而不破坏 1D 自回归结构，最终在 ImageNet 256 上无 guidance 拿到 1.48 的 FID（SOTA）。

**[Exploring and Exploiting Stability in Latent Flow Matching](exploring_and_exploiting_stability_in_latent_flow_matching.md)**

:   本文系统刻画了 Latent Flow Matching（LFM）的"轨迹稳定性"——同一噪声种子下，剪掉 75% 数据、换大小架构、改训练种子都能产生几乎相同的图像；进而把这个性质转化成两个实用算法：(1) 用 balanced-clustering 剪枝可在 CelebA-HQ 上把 50% 数据剪掉而 FID 反而轻微提升、ImageNet 上 75% 数据可剪；(2) Coarse-to-Fine 两段式生成，把 DiT-XL/2 (675M) 和 DiT-S/2 (33M) 拼起来，推理快 2.15×。

**[Factored Classifier-Free Guidance](factored_classifier-free_guidance.md)**

:   本文识别出 CFG 在扩散模型反事实生成中存在「属性放大 (attribute amplification)」失效模式——单一全局 $\omega$ 会把本不该改变的属性一起放大，并提出 FCFG：按因果图分组、为每组属性分配独立 guidance 权重，从而在 CelebA-HQ / EMBED / MIMIC-CXR 上显著降低非目标属性漂移、改善反事实可逆性。

**[GenExam: A Multidisciplinary Text-to-Image Exam](genexam_a_multidisciplinary_text-to-image_exam.md)**

:   GenExam 把"画图考试"作为衡量 T2I 模型推理-理解-生成综合能力的金标准，给 10 个学科、1000 道题各配上 ground-truth 图 + 细粒度评分点，结果连最强闭源模型 Nano Banana Pro 也只有 70.2% strict 分，多数开源 T2I/统一 MLLM 不到 3%。

**[Implicit Preference Alignment for Human Image Animation](implicit_preference_alignment_for_human_image_animation.md)**

:   作者提出 Implicit Preference Alignment (IPA)：一种只需"好样本"、不需要构造好/坏配对的后训练方法，通过最大化与预训练参考模型 KL 间隔来等价地最大化隐式奖励，并配合一个把手部 mask 加权进损失的 HALO 模块，让大尺度视频 DiT 在仅 93 个挑选样本下显著改善人体动画的手部保真度。

**[Krause Synchronization Transformers](krause_synchronization_transformers.md)**

:   作者把 Krause 有界置信共识模型搬进 Transformer，用"距离-RBF+局部窗+top-k 稀疏"替代全局 softmax 相似度，从理论上证明它鼓励多簇同步而非全局塌缩，并在 ViT / 自回归图像生成 / LLM 上同时获得更优性能和 30%+ 算力节省。

**[Offline Preference Optimization for Rectified Flow with Noise-Tracked Pairs](offline_preference_optimization_for_rectified_flow_with_noise-tracked_pairs.md)**

:   本文针对 rectified flow（RF）类文生图模型，提出 PNAPO——一种把"生成时用的先验噪声"和"赢者/输者图片"一起保存为六元组的离线偏好优化框架，配合 RF 直线轨迹假设做轨迹估计和动态正则系数调度，相比 Diffusion-DPO 在 SD3-M/FLUX 上同时提点又把训练算力降到 1/12。

**[Riemannian Generative Decoder](riemannian_generative_decoder.md)**

:   本文针对 Riemannian VAE 必须为每种流形手工设计复杂概率密度的痛点，提出 Riemannian Generative Decoder (RGD)——彻底丢掉 encoder，把每个样本的 latent 当作自由参数用黎曼优化器 (RiemannianAdam) 直接训，同时引入"按局部度量逆缩放的输入噪声"作为几何正则，在合成分支扩散树、人类线粒体 DNA、细胞周期 scRNA-seq 三个真实生物数据上恢复出更忠实的几何，且在高维下数值稳定胜过 VAE 基线。

**[SpatialReward: Bridging the Perception Gap in Online RL for Image Editing via Explicit Spatial Reasoning](spatialreward_bridging_the_perception_gap_in_online_rl_for_image_editing_via_exp.md)**

:   作者指出 MLLM 类编辑奖励模型存在"注意力坍缩"问题——评分时不去比较原图与编辑后图、而是塌缩到 sink token 上做盲判，进而提出 SpatialReward：先让 8B 模型预测编辑区域的边界框、再以这些 box token 为锚做交错式跨图推理；配上一个 260K 样本的空间感知数据集和 GRPO 两阶段训练后，在三个 reward benchmark 上 SOTA，并把 OmniGen2 的 GEdit-Bench 分数拉升 +0.90（是 GPT-4.1 提升的两倍）。

**[Speculative Coupled Decoding for Training-Free Lossless Acceleration of Autoregressive Visual Generation](speculative_coupled_decoding_for_training-free_lossless_acceleration_of_autoregr.md)**

:   本文发现 Speculative Jacobi Decoding (SJD) 在自回归视觉生成中加速有限的根因是连续迭代之间 draft token 的独立采样导致 collision 概率几乎为零；只需把独立采样换成 Maximal/Gumbel Coupling（一行修改、零额外训练），就能把图像生成最高加速到 $4.2\times$、视频生成 $13.6\times$，并严格保持输出分布与原 AR 解码一致。

**[Structured Diffusion Bridges: Inductive Bias for Denoising Diffusion Bridges](structured_diffusion_bridges_inductive_bias_for_denoising_diffusion_bridges.md)**

:   SDB 把模态翻译重写为"在所有满足边缘约束的耦合集合 $\mathcal{P}$ 中挑一个"，在 LDDBM 之上叠加边缘匹配（WTA + 容量约束）+ 端点级 + 轨迹级双层 cycle consistency，把成对监督仅作为可选启发式之一，从而在零成对、半成对、全成对三种监督预算下都能跑，并且全成对时也比 paired-only 基线更好（FFHQ→CelebA-HQ PSNR 从 25.6 提到 25.9）。

**[The Coupling Within: Flow Matching via Distilled Normalizing Flows](the_coupling_within_flow_matching_via_distilled_normalizing_flows.md)**

:   本文提出 NFM（Normalized Flow Matching），用预训练 TarFlow 这种自回归归一化流（NF）产生的"准确定性 data→noise 双射"作为 Flow Matching 的噪声-数据配对，从而把 FM 收敛速度、少步数 FID 同时拉到新的水平，并反过来比当老师的 NF 推理快若干个数量级。

**[Threshold-Guided Optimization for Visual Generative Models](threshold-guided_optimization_for_visual_generative_models.md)**

:   作者把 DPO 的成对偏好假设拆掉，证明 KL 正则化最优策略本质上是把每个样本的 reward 与一个无法计算的实例相关基线 $\tau^*(x)=\beta\log Z(x)$ 比较，于是用从分数分位数估出的全局阈值 $\tau$ 替代它，再加一个与 $|s-\tau|$ 成正比的置信度权重，让扩散模型和 MaskGIT 在仅有标量打分（无成对偏好）时也能稳定对齐，并在五个 reward model 三个测试集上一致优于 Diffusion-DPO / KTO / DSPO。

**[Visual Implicit Autoregressive Modeling](visual_implicit_autoregressive_modeling.md)**

:   本文把 Deep Equilibrium（DEQ）隐式不动点层嵌进 VAR 的 next-scale 自回归框架，用 Jacobian-Free Backpropagation 实现常数显存训练，把 VAR-d30 的 20 亿参数压到 7.7 亿，同时在推理时把每个 scale 的迭代次数变成"可调旋钮"——在 ImageNet-256 上 FID 2.16/sFID 8.07 不变的同时，4090 单卡峰值显存从 19.24GB 降到 8.53GB、吞吐从 15.16 提到 32.08 img/s。

**[Watch Your Step: Information Injection in Diffusion Models via Shadow Timestep Embedding](watch_your_step_information_injection_in_diffusion_models_via_shadow_timestep_em.md)**

:   本文揭示扩散模型里一直被忽视的"时间步嵌入"其实是一条尚未被占用的信息侧信道——通过把训练时的 timestep 范围扩展到一个"影子区间"（shadow timestep）并把另一个数据分布绑定到该区间，可以在不改变 scheduler 接口的前提下，让同一个 diffusion 模型在显式区间生成正常图、在影子区间生成"隐藏"图，既可做隐蔽后门攻击也可做模型水印验证；同时给出基于正弦位置编码的互相干（mutual coherence）理论分析，解释为什么两个不相交区间能携带独立信息。
