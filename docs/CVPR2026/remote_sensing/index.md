---
title: >-
  CVPR2026 遥感方向 17篇论文解读
description: >-
  17篇CVPR2026 遥感方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛰️ 遥感

**📷 CVPR2026** · 共 **17** 篇

**[Acpv-Net All-Class Polygonal Vectorization For Seamless Vector Map Generation Fr](acpv-net_all-class_polygonal_vectorization_for_seamless_vector_map_generation_fr.md)**

:   提出 ACPV-Net，首个从航空影像一次性生成拓扑一致的全类别多边形矢量地图的框架，通过语义监督条件化扩散模型生成顶点热图，并借助命题驱动的 PSLG 重建确保零间隙/零重叠。

**[Asking Like Socrates Socrates Helps Vlms Understand Remote Sensing Images](asking_like_socrates_socrates_helps_vlms_understand_remote_sensing_images.md)**

:   揭示遥感VLM中的"伪推理"现象（显式推理链反而导致性能下降），归因于"一瞥效应"（单次粗浅感知不足），提出RS-EoT(Evidence-of-Thought)迭代证据搜索范式，通过SocraticAgent自博弈合成推理轨迹做SFT冷启动，再用两阶段渐进RL（grounding→VQA）增强和泛化，RS-EoT-7B在多个遥感VQA和grounding基准上达SOTA。

**[Avion Aerial Vision-Language Instruction From Offline Teacher To Prompt-Tuned Ne](avion_aerial_vision-language_instruction_from_offline_teacher_to_prompt-tuned_ne.md)**

:   提出 AVION 知识蒸馏框架，通过 LLM 生成语义丰富的文本原型和视觉-文本双侧提示调优，解决遥感 VLM 适配中的语义贫乏和视觉刚性问题，在少样本分类、基类到新类泛化和跨模态检索上全面超越 SOTA。

**[Avion Aerial Visionlanguage Instruction From Offli](avion_aerial_visionlanguage_instruction_from_offli.md)**

:   提出 AVION 蒸馏框架，通过 LLM 生成并视觉验证的文本原型解决遥感 VLM 的"语义贫乏"，通过双模态 Prompt Tuning 解决"视觉刚性"，在 6 个遥感基准上实现少样本和 base-to-novel 同时提升。

**[Cross-Modal Fuzzy Alignment Network For Text-Aerial Person Retrieval And A Large](cross-modal_fuzzy_alignment_network_for_text-aerial_person_retrieval_and_a_large.md)**

:   提出跨模态模糊对齐网络 CFAN，利用模糊逻辑量化 token 级可靠性实现精细对齐，并引入地面视图作为桥接代理缓解航拍图像与文本的语义鸿沟，同时构建了大规模文本-航拍行人检索基准 AERI-PEDES。

**[Exploring Spatiotemporal Feature Propagation For Video-Level Compressive Spectra](exploring_spatiotemporal_feature_propagation_for_video-level_compressive_spectra.md)**

:   首次将光谱压缩成像 (SCI) 从图像级推进到视频级重建，构建了首个高质量动态高光谱数据集 DynaSpec，提出 PG-SVRT Transformer 通过时空特征传播实现高光谱视频高质量、时间一致的重建。

**[Geoflow Real-Time Fine-Grained Cross-View Geolocalization](geoflow_real-time_fine-grained_cross-view_geolocalization.md)**

:   提出 GeoFlow，一种受流匹配启发的轻量级跨视图精细地理定位框架，通过学习概率位移场结合迭代精化采样（IRS）算法，在连续空间内实现从地面图像到卫星图像的精确 2-DoF 定位，以 29 FPS 的实时速度达到了与 SOTA 可比的精度。

**[Geoflow Real-Time Fine-Grained Cross-View Geolocalization Via Iterative Flow Pre](geoflow_real-time_fine-grained_cross-view_geolocalization_via_iterative_flow_pre.md)**

:   提出GeoFlow，将精细跨视图地理定位(FG-CVG)重新表述为概率位移回归——模型学习从任意假设位置到真实位置的位移场(距离+方向的概率分布)，配合迭代精化采样(IRS)算法让多个随机假设从不同起点"流向"共识位置，以7.8×更少参数和4×更少计算量实现29FPS实时推理+竞争性定位精度。

**[Joint And Streamwise Distributed Mimo Satellite Co](joint_and_streamwise_distributed_mimo_satellite_co.md)**

:   针对多LEO卫星协同服务多天线地面用户的下行链路，基于统计CSI提出联合非相干传输（WMMSE迭代预编码，支持一般凸功率约束）和流式分布传输（每流由单颗卫星发送，通过匈牙利算法做特征模式-卫星关联），在UE侧信道正交时流式传输几乎无损，非正交时呈现性能-开销权衡。

**[Joint And Streamwise Distributed Mimo Satellite Communications With Multi-Antenn](joint_and_streamwise_distributed_mimo_satellite_communications_with_multi-antenn.md)**

:   提出面向多天线地面用户的分布式LEO卫星下行链路两种传输方案（联合传输 & 流式传输），通过基于统计CSI的WMMSE预编码设计和基于匈牙利算法的流-卫星关联策略，在无需卫星间相位同步的前提下实现了高频谱效率与低前传开销的灵活折中。

**[Lumosaic Hyperspectral Video Via Active Illumination And Coded-Exposure Pixels](lumosaic_hyperspectral_video_via_active_illumination_and_coded-exposure_pixels.md)**

:   Lumosaic 是紧凑的主动高光谱视频系统，将窄带 LED 阵列与编码曝光像素 (CEP) 相机结合，在每帧视频内联合编码空间-时间-光谱信息，实现 30fps VGA 31 通道（400-700nm）运动鲁棒高光谱视频。

**[Metaspectra A Compact Broadband Metasurface Camera](metaspectra_a_compact_broadband_metasurface_camera.md)**

:   提出MetaSpectra+，一种基于双层超表面-折射光学混合设计的紧凑型相机，可在单次快照中同时获取高光谱数据立方体和HDR/偏振图像，工作带宽达250nm覆盖几乎整个可见光谱，在基准数据集上实现了最高的高光谱重建精度和最短的系统总光程长度。

**[Metaspectra A Compact Broadband Metasurface Camera For Snapshot Hyperspectral Im](metaspectra_a_compact_broadband_metasurface_camera_for_snapshot_hyperspectral_im.md)**

:   MetaSpectra+ 利用新型超表面-折射透镜混合光学系统，将入射光分为 4 个独立可控色散/曝光/偏振的通道，实现最紧凑且精度最高的快照式高光谱+HDR/偏振多功能成像。

**[No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors](no_labels_no_look-ahead_unsupervised_online_video_stabilization_with_classical_p.md)**

:   无监督在线视频稳定框架，经典三阶段管线+多线程缓冲，引入UAV多模态航拍测试集

**[Olbedo: An Albedo and Shading Aerial Dataset for Large-Scale Outdoor Environments](olbedo_an_albedo_and_shading_aerial_dataset_for_large-scale_outdoor_environments.md)**

:   首个大规模真实航拍反照率-着色分解数据集（5664张UAV图像），通过物理逆渲染管线生成真值

**[Rho Robust Holistic Osm-Based Metric Cross-View Geo-Localization](rho_robust_holistic_osm-based_metric_cross-view_geo-localization.md)**

:   提出首个面向恶劣天气和传感器噪声的OSM-based度量级跨视角定位基准CV-RHO（270万+ 图像），并设计双分支Pin-Pan架构RHO模型，结合全景去畸变（SUM）和位置-朝向融合（POF）机制，在多种退化条件下将定位性能提升高达20%。

**[Sdfnet Structureaware Disentangled Feature Learnin](sdfnet_structureaware_disentangled_feature_learnin.md)**

:   提出SDF-Net——物理引导的结构感知解耦特征学习网络，通过中间层梯度能量提取几何结构一致性(SCL)和终端层共享/模态专用特征解耦(DFL)+无参数加法融合，在HOSS-ReID上mAP达60.9%(+3.5% vs SOTA TransOSS)。
