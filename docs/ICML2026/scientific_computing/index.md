---
title: >-
  ICML2026 科学计算方向19篇论文解读
description: >-
  19篇ICML2026的科学计算方向论文解读，涵盖超分辨率、压缩/编码、扩散模型、布局/合成、时序预测等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "科学计算"
  - "论文解读"
  - "论文笔记"
  - "超分辨率"
  - "压缩/编码"
  - "扩散模型"
  - "布局/合成"
  - "时序预测"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧮 科学计算

**🧪 ICML2026** · **19** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (4)](../../CVPR2026/scientific_computing/index.md) · [🔬 ICLR2026 (10)](../../ICLR2026/scientific_computing/index.md) · [🤖 AAAI2026 (8)](../../AAAI2026/scientific_computing/index.md) · [🧠 NeurIPS2025 (23)](../../NeurIPS2025/scientific_computing/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/scientific_computing/index.md) · [🧪 ICML2025 (8)](../../ICML2025/scientific_computing/index.md)

🔥 **高频主题：** 超分辨率 ×3 · 压缩/编码 ×2 · 扩散模型 ×2 · 布局/合成 ×2 · 时序预测 ×2

**[A Call to Lagrangian Action: Learning Population Mechanics from Temporal Snapshots](a_call_to_lagrangian_action_learning_population_mechanics_from_temporal_snapshot.md)**

:   本文从最小作用原理出发，提出 Wasserstein 拉格朗日力学（WLM）框架，学习二阶人口动力学而非传统梯度流的一阶动力学，从而能够捕捉周期性、旋转等更丰富的群体现象，并可在不需要参考过程的情况下完成插值与未来预报。

**[ANTIC: Adaptive Neural Temporal In-situ Compressor](antic_adaptive_neural_temporal_in-situ_compressor.md)**

:   为了把 PB-EB 级别 PDE 仿真数据"边算边压"，本文提出 ANTIC：用 physics-aware 时间选择器只保留物理上重要的快照，再用神经场 + LoRA 持续微调编码相邻快照之间的残差，在 2D Kolmogorov 流上拿到 435× 压缩、在 4.2 TiB 的 3D 双黑洞合并模拟上拿到 6807× 时空联合压缩。

**[Discovering Ordinary Differential Equations with LLM-Based Qualitative and Quantitative Evaluation](discovering_ordinary_differential_equations_with_llm-based_qualitative_and_quant.md)**

:   DoLQ 在 LLM 符号回归的搜索循环里插入一个 "Scientist Agent"，对候选项同时做定性（物理合理性）+ 定量（消融式 MSE 贡献）评估，把 LLM-SR 那种 "低误差但项数臃肿、物理上荒谬" 的方程逼到既数值精确又结构紧凑。

**[Flow Sampling: Learning to Sample from Unnormalized Densities via Denoising Conditional Processes](flow_sampling_learning_to_sample_from_unnormalized_densities_via_denoising_condi.md)**

:   本文提出 Flow Sampling，把流匹配/扩散模型从"数据驱动"反转为"噪声驱动"——以源噪声样本为条件构造去噪扩散漂移，在 interpolant 上用 detached 模型采得 $X_1$ 的能量梯度做回归目标，从而学到无数据情况下的高效扩散采样器，并自然推广到常曲率黎曼流形。

**[Mesh Field Theory: Port–Hamiltonian Formulation of Mesh-Based Physics](mesh_field_theory_port-hamiltonian_formulation_of_mesh-based_physics.md)**

:   从「局部性 + 置换等变 + 朝向协变 + 能量守恒/耗散不等式」四条物理原理出发，证明任何满足这些公理的网格物理动力学在雅可比层面都可以局部约化为 port-Hamiltonian 形式——其中守恒互联结构 $J$ 完全由网格拓扑（符号关联矩阵 $D_k$）固定，度量与耗散通过可学的 $G, R$ 进入；据此设计的 MeshFT-Net 在长时间 rollout 上能量漂移近零、色散与动量正确，并大幅领先 MGN / HNN。

**[Meta-learning Structure-Preserving Dynamics](meta-learning_structure-preserving_dynamics.md)**

:   把 modulation-based 元学习（hyper-network 把 latent code $\bm{z}^{(k)}$ 映射成层级调制参数）系统性地引入 Hamiltonian / GENERIC 神经网络，提出两种新颖调制——latent multi-rank (MR) 与 latent SVD-like 调制，让一个共享网络在不知道系统参数 $\bm{\mu}$ 的情况下少样本适配整族新参数实例，同时严格保持能量守恒 / 耗散结构。

**[MOOSE-Star: Unlocking Tractable Training for Scientific Discovery by Breaking the Complexity Barrier](moose-star_unlocking_tractable_training_for_scientific_discovery_by_breaking_the.md)**

:   MOOSE-Star 把"训练一个能直接生成科学假设的 LLM"这个原本要在 $\mathcal{O}(N^k)$ 组合空间里搜索的问题拆成"灵感检索 + 假设合成"两个序列子任务，再叠上层级树检索 + bounded composition + motivation 规划，把最优复杂度从指数级压到 $\mathcal{O}(\log N)$，并放出 108,717 篇带分解标注的 TOMATO-Star 数据集。

**[Phy-CoSF: Physics-Guided Continuous Spectral Fields Reconstruction and Super-Resolution for Snapshot Compressive Imaging](phy-cosf_physics-guided_continuous_spectral_fields_reconstruction_and_super-reso.md)**

:   为单次曝光式压缩光谱成像 (CASSI) 设计一个 train-render 两阶段、按波长可任意查询的深度展开框架——在每个展开 stage 内塞入连续光谱场 (CoSF) 先验模块，由 Fourier-Mamba 驱动的三分支跨域特征混合器 + 随机频率编码 + 谱合成头组成，离散波长训练即可在推理时合成任意连续波长的高光谱图像，实现连续光谱重建与零样本光谱超分。

**[PODiff: Latent Diffusion in Proper Orthogonal Decomposition Space for Scientific Super-Resolution](podiff_latent_diffusion_in_proper_orthogonal_decomposition_space_for_scientific_.md)**

:   PODiff 把扩散模型从像素空间搬到固定的、按方差排序的 POD 系数空间里跑，用极小的 MLP 就能在 $640\times 480$ SST 降尺度任务上拿到与像素级扩散相当的精度，同时因为重构是线性的，集成方差可以通过 $\Sigma_u=\Phi\Sigma_a\Phi^\top$ 解析回传到物理空间，得到空间上可解释、且校准良好的不确定性。

**[Rethink the Role of Neural Decoders in Quantum Error Correction](rethink_the_role_of_neural_decoders_in_quantum_error_correction.md)**

:   本文在 $d\le9$ 的表面码上系统重做 MLP/3D-CNN/TCN/Transformer/GNN 五类神经解码器，并把"量化 + 剪枝 + FPGA 资源建模"作为一等公民放进训练流程，结论是：近期解码性能由数据量而非架构复杂度主导，且 INT4 + QAT 是实现微秒级实时解码的必要前提。

**[Saving Foundation Flow-Matching Priors for Inverse Problems](saving_foundation_flow-matching_priors_for_inverse_problems.md)**

:   针对 Stable Diffusion / Flux 这类基础流匹配模型在求解逆问题上明显逊于领域专用先验甚至未训练先验的现象，作者提出 FMPlug：用一个由近似样本指导、时间可学习的 warm-start 加上锐利高斯壳层约束，把基础 FM 的潜变量塞回它真正"懂"的薄壳上，从而显著恢复其作为逆问题先验的能力。

**[Semi-Supervised Neural Super-Resolution for Mesh-Based Simulations](semi-supervised_neural_super-resolution_for_mesh-based_simulations.md)**

:   SuperMeshNet 用两个互补 MPNN——主模型预测 LR→HR，辅助模型预测 LR-LR 对应的 HR-HR 差分——在无配对 HR 的样本上互相生成伪标签，并配合节点级 / 消息级 centering 两个轻量归纳偏置，使得 PDE mesh 超分仅用 10% HR 数据就能超过 100% HR 全监督基线，跨 6 种 MPNN 架构一致下降 RMSE。

**[Skipping the Zeros in Diffusion Models for Sparse Data Generation](skipping_the_zeros_in_diffusion_models_for_sparse_data_generation.md)**

:   SED 把扩散模型从"对所有维度做全密集去噪"改成"只在非零维度上跑扩散+自回归解码维度-值对"，让计算量从随维度线性增长变成几乎随非零数恒定，同时严格保留科学数据中"显式零"这一语义信息。

**[Smoothness Errors in Dynamics Models and How to Avoid Them](smoothness_errors_in_dynamics_models_and_how_to_avoid_them.md)**

:   作者从理论上指出 Kiani 等人的 "unitary GNN" 因为强行保持 Rayleigh 商而对热扩散这类"天然会变光滑"的物理系统过度约束，进而提出"松弛 unitary 卷积"（R-UniGraph / R-UniMesh）并把整套 Rayleigh 商-unitary 卷积框架从图扩展到三角网格，在 MeshPDE 与 WeatherBench22 上同时超越多类强基线。

**[(Sparse) Attention to the Details: Preserving Spectral Fidelity in ML-based Weather Forecasting Models](sparse_attention_to_the_details_preserving_spectral_fidelity_in_ml-based_weather.md)**

:   MOSAIC 用"概率扰动 + 在 HEALPix 球面网格上的 mesh-aligned 块稀疏注意力"同时解决了 ML 天气预报模型的两类频谱退化（确定性平均带来的谱衰减 + 粗化潜空间带来的高频走样），在 1.5° 分辨率上仅 214M 参数就匹敌甚至超过 6× 高分辨率的模型，单 H100 12 秒生成 24 成员 10 天预报。

**[Teaching Molecular Dynamics to a Non-Autoregressive Ionic Transport Predictor](teaching_molecular_dynamics_to_a_non-autoregressive_ionic_transport_predictor.md)**

:   本文把昂贵的原子轨迹当作训练时的「特权辅助模态」，用一个双模态训练器先吃轨迹学动力学，再通过闭式岭回归把它的隐藏表示蒸到一个只看平衡结构的非自回归预测器上，在锂离子均方位移预测上比自回归 SOTA 快 200× 且更准。

**[Topology-Preserving Neural Operator Learning via Hodge Decomposition](topology-preserving_neural_operator_learning_via_hodge_decomposition.md)**

:   本文提出 Hodge Spectral Duality (HSD) 神经算子，把流形 PDE 的解算子按 Hodge 正交分解拆成"低频拓扑分量（谱基底）+ 高频几何分量（FNO 辅助网格）"双分支，再用一个交换子修正项耦合二者，从而在复杂网格上同时获得高精度与守恒律保真。

**[Unbiased and Second-Order-Free Training for High-Dimensional PDEs](unbiased_and_second-order-free_training_for_high-dimensional_pdes.md)**

:   本文针对 EM-BSDE 训练 loss 的离散化偏置问题，提出 Un-EM-BSDE：把单步误差用两组独立的 Monte Carlo 子样本平均后做"乘积"形成无偏估计，既消除偏置又不需要 Hessian，在 HJB/BSB/AC 等基准 PDE 上达到 Heun-BSDE / FS-PINNs 的精度但训练时间仅 1.79× EM-BSDE（相比 Heun-BSDE 的 42.91× 与 FS-PINNs 的 32.07×）。

**[WeatherSyn: An Instruction Tuning MLLM For Weather Forecasting Report Generation](weathersyn_an_instruction_tuning_mllm_for_weather_forecasting_report_generation.md)**

:   WeatherSyn 把气象预报员的报告写作流程拆解成"看图→列要点→出稿"的多模态指令任务，先建了首个覆盖 31 个美国城市、8 类天气要素的 WSInstruct 数据集，再用 SFT→RFT→DPO 三段式微调 Qwen3-VL-8B，让一个 8B 开源模型在多种评测指标上稳定打过 GPT-5-Nano、Claude-3.7-Sonnet 等闭源大模型，并对未见城市有零样本泛化能力。
