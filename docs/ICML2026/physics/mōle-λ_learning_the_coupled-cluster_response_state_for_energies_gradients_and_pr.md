---
title: >-
  [论文解读] MōLe-Λ: Learning the Coupled-Cluster Response State for Energies, Gradients, and Properties
description: >-
  [ICML 2026][物理/科学计算][耦合簇理论] MōLe-Λ 把分子轨道学习从只预测耦合簇右态 $T$ 振幅扩展到同时预测左态 $\Lambda$ 振幅，用一套等变网络从局域化 Hartree–Fock 轨道直接读出 $(T_1,T_2,\Lambda_1,\Lambda_2)$…
tags:
  - "ICML 2026"
  - "物理/科学计算"
  - "耦合簇理论"
  - "CCSD"
  - "Λ 振幅"
  - "等变神经网络"
  - "分子轨道"
  - "响应性质"
---

# MōLe-Λ: Learning the Coupled-Cluster Response State for Energies, Gradients, and Properties

**会议**: ICML 2026  
**arXiv**: [2605.29622](https://arxiv.org/abs/2605.29622)  
**代码**: 无  
**领域**: 物理 / 量子化学 / 等变神经网络  
**关键词**: 耦合簇理论, CCSD, Λ 振幅, 等变神经网络, 分子轨道, 响应性质  

## 一句话总结
MōLe-Λ 把分子轨道学习从只预测耦合簇右态 $T$ 振幅扩展到同时预测左态 $\Lambda$ 振幅，用一套等变网络从局域化 Hartree–Fock 轨道直接读出 $(T_1,T_2,\Lambda_1,\Lambda_2)$，在 QM7 上能量/受力 MAE 仅 0.10 mHa / 0.12 mHa/Bohr，同时把偶极、四极、极化率、电子密度、对密度等响应性质都从同一个学到的"响应态"里解出，相对 CCSD+$\Lambda$ 求解器加速两个数量级以上。

## 研究背景与动机

**领域现状**：耦合簇理论（CCSD/CCSD(T)）被视为量子化学的"金标准"，但其形式标度为 $\mathcal{O}(N^6)$，分子稍微大一点就跑不动。机器学习已经从两条路缓解这个矛盾：一是直接拟合能量与受力的机器学习势（MLIP, Mace/eSEN 等）；二是学习密度、密度矩阵或 Fock 矩阵等一粒子量来加速自洽场或重建单粒子观测量。

**现有痛点**：MLIP 只能产出能量/受力，拿不到偶极、四极、极化率这些依赖关联电子态的量；学密度/Hamiltonian 又只能恢复单粒子层面的信息。任何依赖完整响应态的量（偶极/四极/极化率/电子密度/电子对密度）都必须经由耦合簇拉格朗日里的左态 $\Lambda$ 振幅，而 $\Lambda$ 方程自身的求解开销仍然是 $\mathcal{O}(N^6)$，没有任何加速。

**核心矛盾**：CC 理论并非对右态 $T$ 振幅本身变分，因此能量对外参 $\xi$ 的全导数 $dE/d\xi$ 含有额外的 $\partial t_\mu/\partial \xi$ 项；只有引入 $\Lambda$ 作为伴随变量，把目标改写为拉格朗日 $\mathcal{L}(T,\Lambda)$ 后才有 $dE/d\xi = \partial \mathcal{L}/\partial \xi$。先前的 MōLe（Thiede et al., 2026）只学了 $T_1, T_2$，能拿能量却拿不到松弛响应观测量。

**本文目标**：(i) 训练单一神经网络同时给出 $(T_1,T_2,\Lambda_1,\Lambda_2)$；(ii) 保持原 MōLe 的等变/局域/尺寸可扩展先验；(iii) 不为每个性质单独训练读出头，所有下游量都通过标准 CC 后处理从振幅算出；(iv) 跨分子大小与几何畸变都能稳定外推。

**切入角度**：观察 $\Lambda_1,\Lambda_2$ 的张量结构与 $T_1,T_2$ 完全对称——都是占据/虚轨道指标上的反对称张量，都满足同样的轨道相位翻转奇等变性、同样的非相互作用片段间应消失的局域性。因此只需复用 MōLe 的共享等变 backbone，再镜像出两个"奇读出头"即可，无需重设计架构。

**核心 idea**：把"学性质"换成"学态"——预测完整的 CCSD 响应态 $(T,\Lambda)$，让传统 CC 后处理（1-RDM、2-RDM、CPHF）从同一对象里解析地导出所有可观测量。

## 方法详解

### 整体框架
输入是分子几何 $\{\mathbf{R}_A\}$；先跑一次廉价的限制性 Hartree–Fock 拿到 MO 系数 $\mathbf{C}$；做占据/虚轨道分别局域化（Foster–Boys 类），把全分子非局域 canonical MO 转成可跨分子迁移的局域 MO；把每个局域 MO 当作一张图，原子上的 AO 系数向量做 padding 后嵌入到等变隐空间；共享 backbone 在 MO 内做消息传递、在 MO 间做注意力交互；最后由四个独立"奇读出头"分别给出 $T_1, T_2, \Lambda_1, \Lambda_2$。预测出的振幅交给标准 CCSD 后处理（拉格朗日、CPHF、1-/2-RDM 重建）生成能量、受力、偶极、四极、极化率、密度与对密度。

### 关键设计

**1. 共享等变 backbone + 四头镜像读出：一个网络一次产出 $T$ 与 $\Lambda$ 四个振幅。**

依赖完整响应态的偶极/四极/极化率/密度都必须经由左态 $\Lambda$ 振幅，而 $\Lambda_1,\Lambda_2$ 的张量结构与 $T_1,T_2$ 完全对称——都是占据/虚轨道指标上的反对称张量、满足同样的轨道相位翻转奇等变性和片段间局域性。所以作者无需重设计架构，直接复用 MōLe 的共享等变 backbone：每个局域 MO 嵌成等变隐表示，经 Odd-MACE 消息传递与 MO 间注意力得到不变特征 $\mathbf{y}_{ia},\mathbf{y}_{ijab}$，再镜像出四个"奇读出头"——$t_i^a=\mathrm{OddReadout}_{T_1}(\mathbf{y}_{ia})$、$\lambda_a^i=\mathrm{OddReadout}_{\Lambda_1}(\mathbf{y}_{ia})$、$t_{ij}^{ab}=\mathrm{OddReadout}_{T_2}(\mathbf{y}_{ijab})$、$\lambda_{ab}^{ij}=\mathrm{OddReadout}_{\Lambda_2}(\mathbf{y}_{ijab})$（"奇读出"指对轨道相位翻转 sign-equivariant，保证振幅在 MO 相位规范下行为正确）。共享 backbone 既省参数，又把四个张量绑进同一隐空间，保留下游 CC 后处理所需的代数一致性，同时让 $\Lambda$ 的归纳偏置与 $T$ 完全对齐。

**2. MP2 残差目标：把"学整张量"换成"学相对 MP2 的修正"。**

CCSD 标签极贵（QM7 重算一次都要专门的 GPU 集群），低数据下网络很难把全部相关结构学出来。作者在闭壳实振幅情形下用 MP2 当零阶基线——$t_{ij,\mathrm{MP2}}^{ab}=\langle ij||ab\rangle/(\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b)$，且 $T_1^{\mathrm{MP2}}=0$、$\Lambda_2^{\mathrm{MP2}}=T_2^{\mathrm{MP2}}$、$\Lambda_1^{\mathrm{MP2}}=0$——把 canonical 基下的 MP2 振幅经同一组局域化矩阵转到局域规范后，让网络只拟合残差 $\Delta t_{ij}^{ab}=t_{ij,\mathrm{CCSD}}^{ab}-t_{ij,\mathrm{MP2}}^{ab}$ 与 $\Delta\lambda_{ab}^{ij}=\lambda_{ab,\mathrm{CCSD}}^{ij}-t_{ij,\mathrm{MP2}}^{ab}$。等于把已知的主阶动力学相关搬掉、NN 只补物理上小但化学上关键的高阶差值，相当于把物理先验注入目标本身，显著降低样本复杂度——消融显示残差模式在低数据区 MAE 明显更低。

**3. 振幅重建损失而非性质损失：监督"态"而不是"性质"。**

直接监督某个具体性质会让模型只在该性质上准、其他失真。MōLe-Λ 的损失里所有性质都不出现，只监督四个振幅张量

$$\mathcal{J}_{\mathrm{amp}}=\frac{1}{B}\sum_{b}\sum_{X\in\{T_1,T_2,\Lambda_1,\Lambda_2\}}w_X\sum_{n=1}^{N_X^{(b)}}\big(\hat X_{b,n}-X_{b,n}^{\mathrm{ref}}\big)^2$$

（本文 $w_X$ 全设 1）。下游量全部从这组振幅经标准 CC 后处理解析导出：能量 $E_{\mathrm{corr}}=\sum_{ijab}(\tfrac14 t_{ij}^{ab}+\tfrac12 t_i^a t_j^b)\langle ij||ab\rangle$，受力 $\mathbf{F}_A=-\partial\mathcal{L}(T,\Lambda)/\partial\mathbf{R}_A$（含 CPHF 轨道响应），单/二粒子观测量经 1-RDM $\gamma_{pq}=\langle\Phi_0|(1+\hat\Lambda)e^{-\hat T}a_p^\dagger a_q e^{\hat T}|\Phi_0\rangle$ 与 2-RDM 重建。这样所有性质共享同一组振幅的代数一致性，不会顾此失彼，还自然支持还没被监督过的新观测量（高阶多极矩、电子相关熵等）。

## 实验关键数据

### 主实验

QM7 训练（5732 分子）/测试（1433 分子）+ 三组泛化集（18 氨基酸、100 PubChem 14 重原子分子、三种几何畸变扫描），CCSD/def2-SVP 标签。能量与受力 MAE（单位 mHa, mHa/Bohr）：

| 方法 | QM7 E | QM7 F | 氨基酸 E | 氨基酸 F | PubChem E | PubChem F | Diels-Alder E | Diels-Alder F |
|------|-------|-------|----------|----------|-----------|-----------|---------------|---------------|
| MP2 | 57.32 | 1.50 | 60.49 | 1.33 | 82.55 | 1.32 | 69.33 | 1.18 |
| Mace（直拟 CCSD）| 0.79 | 1.20 | 9.03 | 9.99 | 19.45 | 9.44 | 11.25 | 7.99 |
| Mace+MP2（Δ-learning）| 0.16 | 0.23 | 0.51 | 1.90 | 2.07 | 2.49 | 1.61 | 1.43 |
| eSEN+MP2 | 0.15 | 0.17 | 3.20 | 0.69 | 8.12 | 1.81 | 1.81 | 1.94 |
| **MōLe-Λ**（本文）| **0.10** | **0.12** | **0.37** | 0.27 | **0.63** | **0.26** | **1.09** | **0.24** |

QM7 上振幅 MAE：$T_1, \Lambda_1$ 约 $2.6\text{-}2.7\times 10^{-5}$，$T_2, \Lambda_2$ 约 $5.3\times 10^{-7}$。响应性质（偶极、四极、极化率）相对 HF、MP2、仅右态的 MōLe-XCCSD 在 QM7 测试集上多极 MAE 显著下降（见原文 Fig. 3）。

### 消融与对比

| 配置 / 评估维度 | 关键观察 | 含义 |
|---|---|---|
| 直接模式 vs MP2 残差模式 | 残差模式在低数据区 MAE 显著更低；数据充足后两者接近 | 物理先验在样本稀缺时价值最大 |
| 仅右态 MōLe（XCCSD 重建）| 偶极/电子密度/对密度误差明显大于 MōLe-Λ | $\Lambda$ 是松弛响应观测量的必需信息 |
| 跨分子大小（QM7 → 氨基酸/PubChem）| 几何 MLIP 误差放大 10×+，MōLe-Λ 仅 3-6× | 局域轨道振幅是真正尺寸可迁移的表示 |
| 出离平衡扫描（butane 二面角 / methanol C–O 拉伸 / 椅式-船式）| Mace 给出"飘忽"预测，MōLe-Λ 稳定低误差 | 学态比学性质外推更稳健 |
| 计算开销（H100 / C17H36）| 常规 CCSD 显存爆掉；MōLe-Λ 可推到 C21 以上，$(T,\Lambda)$ 预测比 CCSD+Λ 求解快 100× 以上 | 实测标度远低于 $\mathcal{O}(N^3)$ |

### 关键发现
- **学态 > 学性质**：仅监督四个振幅张量，下游能量/受力/偶极/四极/极化率/电子密度/对密度全面更优，且未出现常见的"某一性质优化、其他性质退化"现象。
- **$\Lambda$ 是关键边际增益**：仅有 $T$ 时电子密度残差弥散到整个分子体积；加上 $\Lambda$ 后键附近误差大幅压低，对密度差异图（2-RDM）上原本宽阔的 MP2 误差瓣几乎被消除。
- **物理先验显著降数据需求**：MP2 残差化把"主阶相关"交给微扰论，网络只学差值；在低数据区数据效率提升尤其明显。
- **MLIP 在几何畸变上是脆弱的**：Mace 直拟 CCSD 在 butane 二面角扫描上误差远超 MōLe-Λ，说明几何特征空间无法承担电子重排的预测压力，必须经轨道。

## 亮点与洞察
- **"学态"范式重写监督粒度**：长期以来 ML for chemistry 都在选择"学哪个性质"，本文把目标提升到电子结构对象本身，所有性质成为副产品，自然规避多任务冲突。
- **架构镜像而非堆叠**：$\Lambda$ 头与 $T$ 头共享 backbone 且对称镜像，意味着新增 $\Lambda$ 几乎不增加参数和训练复杂度，却把可恢复观测量集合扩大了一个数量级——这种"几乎零成本扩展性质集合"的设计哲学值得在其他物理 ML 任务中借鉴。
- **局域 MO 是真正的可迁移表示**：geometry-based 几何 MLIP 在尺寸外推上垮得很快，而局域轨道振幅天然满足尺寸可扩展性（非相互作用片段间振幅应消失），这暗示分子级 ML 的"正确归纳偏置"未必是欧氏空间。
- **可迁移到其他需要伴随变量的物理**：CCSD 拉格朗日的"右态+左态"结构在弹性力学、最优控制、变分推断里都有同构形式，本文"学伴随"的思想可被迁移到其他需要响应导数的科学计算问题。

## 局限与展望
- **基组与元素受限**：仅在 def2-SVP 基组、C/N/O/S/H 五种元素的 QM7 上训练；更大基组（aug-cc-pVTZ）、过渡金属、开壳层尚未覆盖。
- **稠密 $T_2,\Lambda_2$ 输出仍是潜在瓶颈**：当前对几十重原子分子才显现，更大体系上需要稀疏/局域/压缩的 doubles 表示。
- **HF、局域化、MP2 预处理未优化**：CC 后处理在 CPU 上跑，与 GPU 上的 forward pass 节奏不匹配，是工程层面的下一步。
- **没有处理三激发**：CCSD(T) 才是真正的"金标准"，本文只学到 CCSD 级，三激发的 $T_3$ 修正如何融入仍是开放问题。

## 相关工作与启发
- **vs MōLe (Thiede et al., 2026)**：MōLe 只预测 $T$ 振幅、用 XCCSD 重建能量与一粒子密度；MōLe-Λ 把响应态闭环，可恢复偶极/四极/极化率/对密度等仅 $\Lambda$ 可达的观测量，是范式上的纵向延伸。
- **vs Mace / eSEN 等 MLIP**：MLIP 只能给能量与受力，且几何特征空间在尺寸外推与畸变扫描上脆弱；MōLe-Λ 通过轨道表征同时拿到电子结构对象，性能与可迁移性双赢。
- **vs Δ-learning（Mace+MP2, eSEN+MP2）**：Δ-learning 也用 MP2 作为基线，但还是在性质层面；MōLe-Λ 把残差化推到振幅层面，物理先验与监督目标同时一致化。
- **vs 学密度/Hamiltonian (Brockherde et al., 2017; Yu et al., 2023)**：那条路只能恢复一粒子观测量；本文经由 2-RDM 把二粒子量也纳入，覆盖范围更广。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次把 ML 监督目标从"性质"提升到完整的 CCSD 响应态 $(T,\Lambda)$，是范式级而非增量改进。
- 实验充分度: ⭐⭐⭐⭐ QM7 训练 + 三组泛化集 + 出离平衡扫描 + 多种观测量 + 计算标度对比，覆盖维度齐全；缺更大基组与重元素验证。
- 写作质量: ⭐⭐⭐⭐ 拉格朗日动机与四头镜像的因果链清楚；部分 RDM 重建细节挤在附录略可惜。
- 价值: ⭐⭐⭐⭐⭐ 把 CC 级响应性质从"O(N^6) 不可及"变成"前向一次性可达"，对催化、材料、分子设计的实际意义巨大。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](../../ICLR2026/physics/drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)
- [\[CVPR 2025\] Learning Phase Distortion with Selective State Space Models for Video Turbulence Mitigation](../../CVPR2025/physics/learning_phase_distortion_with_selective_state_space_models_for_video_turbulence.md)
- [\[ICML 2026\] A Call to Lagrangian Action: Learning Population Mechanics from Temporal Snapshots](a_call_to_lagrangian_action_learning_population_mechanics_from_temporal_snapshot.md)
- [\[ICLR 2026\] HyperKKL: Enabling Non-Autonomous State Estimation through Dynamic Weight Conditioning](../../ICLR2026/physics/hyperkkl_enabling_non-autonomous_state_estimation_through_dynamic_weight_conditi.md)
- [\[ICML 2026\] Topology-Preserving Neural Operator Learning via Hodge Decomposition](topology-preserving_neural_operator_learning_via_hodge_decomposition.md)

</div>

<!-- RELATED:END -->
