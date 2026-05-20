---
title: >-
  [论文解读] The Coupling Within: Flow Matching via Distilled Normalizing Flows
description: >-
  [ICML 2026][图像生成][Flow Matching] 本文提出 NFM（Normalized Flow Matching），用预训练 TarFlow 这种自回归归一化流（NF）产生的"准确定性 data→noise 双射"作为 Flow Matching 的噪声-数据配对…
tags:
  - "ICML 2026"
  - "图像生成"
  - "Flow Matching"
  - "Normalizing Flow"
  - "coupling"
  - "蒸馏"
  - "TarFlow"
---

# The Coupling Within: Flow Matching via Distilled Normalizing Flows

**会议**: ICML 2026  
**arXiv**: [2603.09014](https://arxiv.org/abs/2603.09014)  
**代码**: https://github.com/apple/ml-nfm  
**领域**: 扩散模型 / 生成模型 / 流匹配  
**关键词**: Flow Matching、Normalizing Flow、coupling、蒸馏、TarFlow

## 一句话总结
本文提出 NFM（Normalized Flow Matching），用预训练 TarFlow 这种自回归归一化流（NF）产生的"准确定性 data→noise 双射"作为 Flow Matching 的噪声-数据配对，从而把 FM 收敛速度、少步数 FID 同时拉到新的水平，并反过来比当老师的 NF 推理快若干个数量级。

## 研究背景与动机

**领域现状**：Flow Matching（FM）已经成为大规模生成模型的主流训练范式 —— 用 $x_t=(1-t)x+t\epsilon$ 把 data 与 noise 做线性插值，再回归速度 $v_t=\epsilon-x$，推理时用 ODE solver 从 $x_1\sim\mathcal{N}(0,I)$ 反演到 $x_0$。决定性能的一个关键设计就是"噪声-数据配对（coupling）"，独立耦合最简单但训练慢、推理曲率大，于是 OT-based coupling（如 SD-FM）成为主流改良方向。

**现有痛点**：OT 类方法本质上仍是基于几何距离的、模型无关的预处理，它没有真正利用"模型可学到的 data 表示"。同时另一条线 —— Normalizing Flow（NF），尤其是新近的 TarFlow，能直接学一个 data ↔ Gaussian 的双射，理论上配对一旦确定速度回归就是零误差、采样一步即可，但 NF 自回归采样慢得离谱。

**核心矛盾**：FM 推理快但配对粗糙；NF 配对完美但采样慢。两边都有结构性短板，过去鲜有工作把它们组合起来。

**本文目标**：(i) 用 NF 学到的（接近双射的）配对替换 FM 的随机 / OT 配对，给 FM 学生提供更"对齐"的训练对；(ii) 验证这种"蒸馏 NF 到 FM"既能加速 FM 收敛、又能反过来在 FID 上击败 NF 老师。

**切入角度**：作者把 NF 看作"学到的、模型相关的最优传输近似"。即便 NF 的映射在似然意义上不是 OT 最优的（它受限于网络容量），但只要它能把每个 data 点稳定映到一个特定的 Gaussian 表示，就足以充当一个高质量的 coupling。

**核心 idea**：训练 FM 学生时，把噪声端 $\epsilon$ 换成预训练 NF 老师对 data 的编码 $z_{\epsilon'}=f_{\text{NF}}(x+\eta\epsilon',c)/\sigma_f$，速度目标改为 $v_t=z_{\epsilon'}-x$，其他 FM 流程不变。

## 方法详解

### 整体框架
NFM 是一个两阶段蒸馏：第一阶段训练一个 TarFlow 老师 $f_{\text{NF}}$，按最大似然学一个 $x\mapsto z$ 的可逆映射；第二阶段冻结老师，训练一个 FM 学生 $g$（任意架构，不必可逆）。学生看到的训练对是 $(x, z_{\epsilon'})$ 而不是 $(x, \epsilon)$，其中 $z_{\epsilon'}=f_{\text{NF}}(x+\eta\epsilon',c)/\sigma_f$，$\eta$ 是老师训练时用的输入扰动幅度，$\sigma_f$ 是用来把老师输出归一化为单位方差的标量。学生用普通 FM loss $\mathcal{L}_{\text{FM}}=\|g((1-t)x+tz_{\epsilon'},c,t)-(z_{\epsilon'}-x)\|_2^2$ 训练，推理与标准 FM 完全一致。

### 关键设计

1. **用 NF 老师产生的 $z$ 取代随机噪声**:

    - 功能：把每个 data $x$ 与一个由 NF 老师确定的、近 Gaussian 的 $z_{\epsilon'}$ 绑定成训练对，让 FM 学生学习的不再是"任意 noise 到任意 data"的多对多映射，而是"特定 $z$ 到特定 $x$"的近确定映射。
    - 核心思路：训练时按 $z_{\epsilon'}=f_{\text{NF}}(x+\eta\epsilon',c)/\sigma_f$ 采样，其中 $\sigma_f^2=\mathbb{E}[f_{\text{NF}}(x+\eta\epsilon',c)^2]$ 保证 $z$ 的方差大致为 1，因此整体分布仍近似 $\mathcal{N}(0,I)$，FM 学生采样时直接从标准高斯起步就可用。值得注意的是，把变分爆炸记号转到方差守恒记号下，TarFlow 的扰动幅度 $\eta=0.05$ 对应 FM 的最大噪声水平 $t=\eta/(1+\eta)\approx0.0476$，远小于标准 FM 的 $t=1$。
    - 设计动机：FM 的核心难点是 $x_t$ 在大 $t$ 处条件方差很大，速度目标 $v_t=\epsilon-x$ 几乎只有"端点差"的方差信息可用；用 NF 给的 $z_{\epsilon'}$ 替换 $\epsilon$ 后，$\text{Var}(v_t|x_t,t)$ 显著降低，梯度更稳、轨迹更直，能直接转化为更少 NFE 下的 FID 提升。

2. **TarFlow 老师 + 输入扰动 $\eta$**:

    - 功能：选 TarFlow 作为老师是因为它在图像生成上已能与扩散模型抗衡；老师训练时给 $x$ 加小量 $\eta\epsilon'$ 扰动是为了让映射对 data 邻域有平滑性。
    - 核心思路：TarFlow 是用 Transformer 实现的 auto-regressive flow，每个 meta-block 内自回归生成 patch，因此采样慢但可逆性强；它在训练时用 $x'=x+\eta\epsilon'$ 输入网络，再以 NLL 最小化学习。NFM 完整保留这个 $\eta$，因为它既保证了 $z$ 在小邻域内的平滑性，又自然把 FM 的有效噪声水平压低到 $\sim\eta/(1+\eta)$。
    - 设计动机：老师的扰动 $\eta$ 在 NFM 中起两个作用：让 $z$ 不至于退化成纯确定映射（保留一定 stochasticity），并隐式控制 FM 学生看到的最大噪声水平 —— 实验中 $\eta$ 取得越大，最佳 FID 出现在更高 NFE 处，反之 $\eta$ 小则少步采样下表现更好。

3. **学生架构自由 + 与 FM 等价的训练目标**:

    - 功能：学生 $g$ 只需是普通 ViT/CNN，不必可逆，因此可以做得比 TarFlow 小、推理可调步数。
    - 核心思路：把 $\epsilon$ 换成 $z_{\epsilon'}$ 后，FM 的"沿线性插值回归速度"的形式保持不变，时间权重也不变，所以可以无缝接入任何 FM 训练 pipeline（实验用 SiT-XL）；推理用 Euler（NFE ≤ 5）或 Heun（NFE ≥ 5），步长按 $t^2=\{1, (1-\delta t)^2,\ldots\}$ 平方调度。
    - 设计动机：保留 FM 训练形式意味着 NFM 不引入新超参、不破坏现有代码栈；学生不必可逆又解锁了任意架构选择，从而让推理速度比可逆的 TarFlow 老师快若干个量级。

### 损失函数 / 训练策略
学生用 $\mathcal{L}_{\text{FM}}=\|g((1-t)x+tz_{\epsilon'},c,t)-(z_{\epsilon'}-x)\|_2^2$ 训练，类标签按概率 $p=0.1$ 随机置空以支持 classifier-free guidance；时间 $t$ 服从 $\text{lognorm}(-0.2,1)$。老师训 512 MiB 样本（约 420 epoch），学生只训 256 MiB（约 210 epoch）。

## 实验关键数据

### 主实验

| 数据集 / 教师 / NFE | FM | SD-FM | NFM (256 MiB) | NFM vs FM |
|----------------------|----|-------|---------------|-----------|
| ImageNet64, SiT-XL/4, 31 | 2.57 | 2.68 | 1.78 | -0.79 |
| ImageNet64, 15 | 4.80 | 3.15 | 2.15 | -2.65 |
| ImageNet64, 7 | 13.01 | 6.41 | 3.23 | -9.78 |
| ImageNet64, Euler-5 | 21.05 | 12.18 | 3.92 | -17.13 |
| ImageNet256, SiT-XL/2, 31 | 2.30 | – | 2.29 | -0.01 |
| ImageNet256, 7 | 12.41 | – | 3.43 | -8.98 |

学生 FID 在 ImageNet64 上达到 1.78，比同等参数的 TarFlow 老师（FID=1.98）还要好；ImageNet256 上 NFM (FID 2.29) 与老师 (FID 3.96) 拉开明显差距。

### 消融实验

| 配置 / 现象 | 结果 | 说明 |
|------------|------|------|
| Heun(t²), 31 NFE | FM 2.57 → NFM 1.78 | 大 NFE 下 NFM 优势小但仍稳定 |
| Euler(t), 128 NFE 曲率 $\kappa$ | FM 0.0386 / SD-FM 0.0289 / NFM 0.0181 | NFM 路径明显更直，能用更少 NFE |
| Heun, 5 NFE | 17.56 → 9.29 → 4.01 | NFM 在极少步采样下相对收益最大 |
| 大 $\eta$ vs 小 $\eta$（z-space 结构） | $\eta$ 越大不同图像 $z$ 越靠近 | 给 FM 提供更弱"准确定性"，少步采样 FID 反而变差 |

### 关键发现
- 学生 FID 反超老师：作者把它归因于"配对几乎确定 + 学生架构灵活 + EMA"三者的复合 —— 老师受困于可逆约束容量低，学生无此限制反而能拟合得更好。
- $z$ 空间结构出人意料：同一张图片在不同 $\eta\epsilon'$ 下的 $z$ 投影并不互为最近邻，反而是不同图片在相同 noise 下更近。这说明 NF 把"图像身份"和"噪声向量"在 $z$ 空间纠缠得很深，但即便如此 NFM 仍能 work，说明 FM 学生学的不是邻域结构，而是端点对应。
- 收益分布：在 NFE = 7 这一典型部署区间，NFM 比 FM 把 FID 砍到 1/4，比 SD-FM 仍降 50%。这恰好是 SD-FM 的 OT 配对覆盖不到的区域。

## 亮点与洞察
- 把 NF 当作"模型学到的 OT 近似"是一个很有解释力的视角：OT 的核心价值是"配对",而不是"几何最优",NFM 证明只要配对足够确定，OT 几何最优性并不必要。
- "FM 用 NF 蒸馏"的反方向兼有蒸馏（NF→FM）和混合（NF+FM）两种解读，且学生采样比老师快几个数量级，是少见的"学生比老师更快也更好"的例子。
- 整套方法没有改 FM 的训练公式、采样器、guidance、时间调度，工程接入成本极低 —— 任何现有 FM 代码只要把 $\epsilon$ 替换成 $z_{\epsilon'}$ 即可。

## 局限与展望
- 仍然需要先训练一个高质量 NF 老师，老师本身就贵；当老师 FID 较差时蒸馏出来的学生也会被拖累，本文未给出"老师有多差仍能 work"的边界。
- 实验只覆盖 ImageNet64 / 256 + class-conditional，文本到图像、视频、高分辨率（1024+）场景没验证，TarFlow 在这些设置下能否稳定训练仍是开放问题。
- $z$ 空间反直觉的结构（同图像 $z$ 不互为最近邻）作者也承认尚未完全理解，可能对蒸馏稳定性形成长期隐患。
- NFM 与 SD-FM 等 OT 类配对方法并不冲突，论文也建议未来组合使用，但目前没给出实际组合实验。

## 相关工作与启发
- **vs SD-FM / OT-CFM**：同样是"改 coupling"，但 OT 类方法是离散最优传输的近似，本文用 NF 学到的 deterministic map 取代之，效果在 ImageNet256 上显著更好，尤其在低 NFE 区间。
- **vs DiffFlow / DiNof**：这些工作把 NF 与 diffusion 联合训练；NFM 走的是"老师冻结、学生蒸馏"的更简洁路线，工程上更易复现。
- **vs Consistency Models / Rectified Flow**：CM 与 RF 试图缩短 ODE 路径以减少 NFE，NFM 走的是"换更好的端点"，本质上是从源头改善了路径曲率（实验中 $\kappa$ 减半）。
- **vs 一般蒸馏方法**（Progressive Distillation 等）：传统蒸馏让学生模仿老师采样轨迹，NFM 让学生学的是新目标分布上的 FM，因此学生不需要逐 NFE 对齐，泛化更好。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把 NF 当成 FM 的 coupling 来源是个清新而自然的视角，跨家族蒸馏的实例化做得干净。
- 实验充分度: ⭐⭐⭐⭐ 多 NFE / 多 solver / 不同 $\eta$ / 曲率分析全套都跑了，对比对象覆盖 FM、SD-FM、TarFlow 自身。
- 写作质量: ⭐⭐⭐⭐ 思路清晰，附带的 $z$ 空间结构分析诚实地汇报了反直觉现象。
- 价值: ⭐⭐⭐⭐ 直接给现有 FM 模型提供一个低成本的 FID 改善方案，特别是在少步采样（实用部署区间）效果显著。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Amortized Sampling with Transferable Normalizing Flows](../../NeurIPS2025/image_generation/amortized_sampling_with_transferable_normalizing_flows.md)
- [\[AAAI 2026\] Flowing Backwards: Improving Normalizing Flows via Reverse Representation Alignment](../../AAAI2026/image_generation/flowing_backwards_improving_normalizing_flows_via_reverse_representation_alignme.md)
- [\[ICML 2025\] Normalizing Flows are Capable Generative Models](../../ICML2025/image_generation/normalizing_flows_are_capable_generative_models.md)
- [\[ICML 2026\] Exploring and Exploiting Stability in Latent Flow Matching](exploring_and_exploiting_stability_in_latent_flow_matching.md)
- [\[ICML 2026\] Adversarial Flow Models](adversarial_flow_models.md)

</div>

<!-- RELATED:END -->
