# Efficient and Brain-Inspired Alternatives to Transformers for Local (On-Device) Language Models

Research date: 2026-11-23. Scope: architectures that could power a private, offline AI companion on a phone or laptop, compared with similar-size transformers.

How these notes were gathered: arxiv.org, huggingface.co and liquid.ai pages could not be fetched from this research session (network allow-list), so most numbers come from search-engine extracts of the primary papers, model cards and vendor blogs, plus GitHub READMEs fetched directly. Items tagged **[recalled]** come from the paper abstracts as known before this session and were not re-fetched; spot-check them before quoting. Unless marked "measured", vendor speed and energy numbers are self-reported. "Estimated" energy means arithmetic estimates (for example, picojoules per operation), not wall-power measurements.

## 1. Spiking language models (SpikeGPT, SpikeBERT, SpikeLM, SpikingBrain and others)

### Takeaway
Spiking LMs went from about 0.2B parameters (SpikeGPT, 2023) to 7B linear and 76B hybrid MoE (SpikingBrain, Sept 2025) and a 5B v2 (April 2026). They get to that scale by converting existing transformers (Qwen2.5/Qwen3) with a small amount of continued pre-training, and they recover about 90% of base-model accuracy. Nearly all of the energy savings are **estimated or simulated**: the big models run as "pseudo-spikes" on ordinary GPUs. The only measured neuromorphic-hardware LM numbers found are for a small MatMul-free model on Intel Loihi 2.

### Cited Findings
**Early spiking LMs (2023-2024, older results)**
- SpikeGPT (Feb 2023, RWKV-based) was the largest backprop-trained SNN when released, at **216M parameters** (about 3x the previous largest SNN). A 46M variant was reported competitive with similar-size transformers. — [SpikeGPT arXiv 2302.13939](https://arxiv.org/html/2302.13939v5); [ICLR listing](https://iclr.cc/virtual/2025/poster/31473)
- SpikeLM (June 2024) is described as the first fully spike-driven model for general language tasks, replacing matrix multiplications with accumulates. A later comparison table lists it at 139M (BART-style) and 680M (mBART-style). The 680M figure comes from a secondary table and was not checked against SpikeLM itself. — [SpikeLM arXiv 2406.03287](https://arxiv.org/html/2406.03287v1); [BiSpikCLM comparison, arXiv 2508.20122](https://www.arxiv.org/pdf/2508.20122)
- SpikeBERT is distillation-based, about 109M parameters, encoder-only (no generation). It is reported to drop to 59.7% on GLUE, well below BERT-class transformers. — [secondary comparison, arXiv 2508.20122](https://www.arxiv.org/pdf/2508.20122)

**SpikingBrain 1.0 (BICLab, Institute of Automation, Chinese Academy of Sciences; arXiv 5 Sept 2025, v3 Dec 2025, accepted to TMLR March 2026)**
- Two models: **SpikingBrain-7B** (linear-attention LLM) and **SpikingBrain-76B** (hybrid-linear MoE). The paper reports performance "comparable to open-source Transformer baselines" using only about **150B tokens of continual pre-training**. — [arXiv 2509.05276](https://arxiv.org/abs/2509.05276)
- Secondary summaries say the 7B keeps **about 90% of its base transformer's (Qwen2.5-7B) accuracy** on MMLU, CMMLU, ARC-C, HellaSwag and C-Eval, using about 2% of the base model's pre-training data. Exact MMLU numbers were not retrieved. — [Emergent Mind summary](https://www.emergentmind.com/topics/spikingbrain-models); [36Kr](https://eu.36kr.com/en/p/3460547193886339)
- Speed (vendor): **over 100x TTFT speedup at 4M tokens** (7B). One summary gives **26.5x at 1M tokens** vs Qwen2.5-7B, with the 100x figure extrapolated to 4M. — [arXiv 2509.05276](https://arxiv.org/abs/2509.05276); [ArXivIQ summary](https://arxiviq.substack.com/p/spikingbrain-technical-report-spiking)
- On-device-relevant result: a compressed **1B SpikingBrain** on a **CPU-based mobile inference framework** decoded **4.04x / 7.52x / 15.39x faster than Llama-3.2-1B at 64k / 128k / 256k context**. The exact phone/CPU hardware was not given in the extracts. — [36Kr](https://eu.36kr.com/en/p/3458000417642121); [arXiv 2509.05276v3](https://arxiv.org/html/2509.05276v3)
- Spiking sparsity is **69.15%**. Energy: a secondary summary gives an effective **0.034 pJ per MAC, about 97.7% lower than FP16 (about 1.5 pJ)**. This is an **estimate from per-operation energy, not a hardware measurement**. — [arXiv abstract](https://arxiv.org/abs/2509.05276); [Emergent Mind](https://www.emergentmind.com/topics/spikingbrain-models)
- Trained on **MetaX C550 GPUs** (non-NVIDIA, Chinese), at 23.4% MFU for the 7B. — [arXiv 2509.05276](https://arxiv.org/abs/2509.05276)
- The official repo says spiking is **"pseudo-spiking" at the tensor level**, and true event-driven spiking "requires asynchronous hardware", which the repo does not cover. It describes **no CPU, mobile or neuromorphic deployment** and **no energy measurements**. Weights (7B base, 7B SFT reasoning, 7B-VL, 7B W8ASpike quantized) are on **ModelScope**, MIT license. — [GitHub BICLab/SpikingBrain-7B](https://github.com/BICLab/SpikingBrain-7B)

**SpikingBrain 2.0 (arXiv 2604.22575, 24 April 2026)**
- 5B hybrid spiking models (SpB2.0-5B, SpB2.0-VL-5B) using "Dual-Space Sparse Attention", an inter-layer hybrid of MoBA sparse softmax attention and sparse linear attention. They recover most of **Qwen3-4B's** capability with **under 7k A100 GPU-hours**. Weights are reportedly open-sourced. — [arXiv 2604.22575](https://arxiv.org/html/2604.22575v1); [BAAI hub](https://hub.baai.ac.cn/view/54432)
- Secondary figures: **10.13x lower first-token latency at 4M context** and 2.52x speedup at 250k (FP8). Neuromorphic hardware results (**70.6% area and 46.5% power reduction at 500 MHz**) are **simulations**. — [besthub summary](https://besthub.dev/articles/spikingbrain-2-0-breaks-long-sequence-and-low-power-bottlenecks-in-brain-inspired-llms-ff81b73812bf)

**Neuromorphic hardware: the only measured numbers found**
- MatMul-free LM (Zhu et al., 2024, arXiv 2406.02528, later versions) on **Intel Loihi 2**: **59.4 tokens/s at 70.8 mJ/token** in generation, and about 8x human reading speed at **4.2 W**. The authors claim about 4x throughput and 10x energy efficiency vs comparable transformers on embedded GPUs. — [arXiv 2406.02528v7](https://arxiv.org/html/2406.02528v7)
- The follow-up workshop paper (Abreu et al., March 2025, ICLR 2025 workshop) says its Loihi 2 energy and throughput "were **estimated** based on a preliminary implementation". It claims up to 3x throughput and 2x less energy vs transformer LLMs on an edge GPU. — [arXiv 2503.18002](https://arxiv.org/html/2503.18002v2)
- **[recalled]** The original MatMul-free paper (June 2024) scaled ternary MatMul-free models to 2.7B parameters and ran billion-parameter models on a custom **FPGA at about 13 W**. — [arXiv 2406.02528](https://arxiv.org/abs/2406.02528)

### Inferences
- "Spiking LLM" at scale today means a linearized or hybrid-attention transformer with spike-coded activations, run as dense tensors on GPUs/CPUs. On phone or laptop hardware the practical gains come from the **linear/sparse attention** (constant memory, long-context speed), not from spikes. Spike-level energy savings need neuromorphic or asynchronous chips, and none ship in consumer phones or laptops.
- The roughly 10% accuracy gap vs the source transformer (SpikingBrain-7B vs Qwen2.5-7B) is a real cost for a companion's conversation quality. A plain 4-bit transformer or hybrid of the same size will likely beat it on quality.
- The 1B-on-mobile-CPU result matters most at very long contexts (64k-256k). At typical chat lengths (2-8k) the advantage will be much smaller.

### Gaps
- No measured (wall-power) joules/token for SpikingBrain on any device. Exact MMLU/CMMLU tables were not retrieved (arxiv.org blocked).
- SpikingBrain-76B: active-parameter count and open-weight status not confirmed.
- No llama.cpp/MLX support for SpikingBrain was found (repo targets vLLM and HF on GPUs).
- A 2026 preprint reportedly describes "NeuronSpark", a 0.9B SNN LM trained more natively; its URL and numbers could not be confirmed.

## 2. State-space models (S4, Mamba, Mamba-2, Mamba-3) and SSM-attention hybrids (Jamba, Zamba, Samba, Nemotron-H, Granite 4.0, Falcon-H1)

### Takeaway
Pure SSMs match transformers on many language benchmarks at 1-8B scale but consistently trail on **copying, in-context retrieval and few-shot (5-shot MMLU)** tasks, because their state has a fixed size. **Hybrids that keep a small fraction of attention layers (roughly 7-10% of layers, or 1 attention per 3-9 linear layers) match or beat same-size transformers** while cutting KV-cache memory 70-80%+ and raising long-context throughput 2-3x+. By 2025-2026 this is the most widely deployed transformer alternative (IBM, NVIDIA, TII, Alibaba, Moonshot, AI21, Zyphra).

### Cited Findings
**Pure SSMs**
- **[recalled]** Mamba (Gu & Dao, Dec 2023): selective SSM, reported 5x higher inference throughput than transformers. Mamba-3B outperformed same-size transformers and matched transformers twice its size (older result, pre-2025). — [arXiv 2312.00752](https://arxiv.org/abs/2312.00752)
- **Mamba-3** (Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu; arXiv 16 March 2026; ICLR 2026 oral) adds a more expressive discretization-derived recurrence, a **complex-valued state** (solves state-tracking tasks Mamba-2 cannot) and a **MIMO** formulation. At 1.5B scale, MIMO beats Gated DeltaNet by +1.8 points average downstream accuracy (SISO +0.6). It **matches Mamba-2 perplexity with half the state size**. Kernels are in state-spaces/mamba; described as "inference-first". — [arXiv 2603.15569](https://arxiv.org/abs/2603.15569); [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10010352); [Tri Dao blog](https://tridao.me/blog/2026/mamba3-part1/)
- NVIDIA's controlled study (June 2024, older): 8B Mamba, Mamba-2 and transformer trained on up to 3.5T tokens. Pure SSMs match or beat transformers on many tasks but **lag on tasks needing strong copying or in-context learning, including 5-shot MMLU**. An **8B Mamba-2-Hybrid (43% Mamba-2, 7% attention, 50% MLP) beat the transformer on all 12 short-context benchmarks (+3.5 on 5-shot MMLU)** and on long-context tasks. — [arXiv 2406.07887](https://arxiv.org/abs/2406.07887); [NVIDIA research page](https://research.nvidia.com/publication/2024-06_empirical-study-mamba-based-language-models)
- "Repeat After Me" (Jelassi et al., Feb 2024): a 2-layer transformer can copy exponentially long strings, while **any SSM fails to copy unless its state grows linearly with sequence length**. In experiments, Mamba needed about 100x more data to learn copying, and **Pythia transformers beat same-size Mamba on copying and context retrieval**. Later work (mimetic initialization, tuned single-layer recall) narrows but does not erase the gap. — [arXiv 2402.01032](https://arxiv.org/abs/2402.01032); [Kempner Institute](https://kempnerinstitute.harvard.edu/research/deeper-learning/repeat-after-me-transformers-are-better-than-state-space-models-at-copying); [Mamba COPY/CoT limits, arXiv 2410.03810](https://arxiv.org/pdf/2410.03810)
- Falcon Mamba 7B (Oct 2024): attention-free 7B with **constant generation throughput and no rise in peak CUDA memory** as generated tokens grow, vs Llama 3.1 8B / Mistral 7B. — [arXiv 2410.05355](https://arxiv.org/pdf/2410.05355)

**Hybrids**
- **NVIDIA Nemotron-H** (April 2025): 8B and 56B (plus 47B compressed). About **8% of layers are attention** (4 of 52 in 8B, 10 of 118 in 56B), the rest Mamba-2 and FFN. **Up to 3x inference throughput** vs similar-size transformers at 65,536 input / 1,024 output on H100. The 56B beats Llama-3.1-70B on 16 of 17 tasks. The 47B reasoning variant has about 4x throughput on 32K traces. — [arXiv 2504.03624](https://arxiv.org/abs/2504.03624v1); [NVIDIA ADLR page](https://research.nvidia.com/labs/adlr/nemotronh/); [NVIDIA blog](https://developer.nvidia.com/blog/nemotron-h-reasoning-enabling-throughput-gains-with-no-compromises)
- **NVIDIA Nemotron Nano 2** (Aug 21, 2025): 9B hybrid Mamba-Transformer reasoning model, built on Nemotron-H with most attention replaced by Mamba-2 (12B base pruned/distilled). — [arXiv 2508.14444](https://arxiv.org/abs/2508.14444v2); [tech report PDF](https://research.nvidia.com/labs/adlr/files/NVIDIA-Nemotron-Nano-2-Technical-Report.pdf)
- **IBM Granite 4.0** (Oct 2025): hybrid Mamba-2 + transformer (reported **9:1** Mamba:attention). IBM claims **over 70% lower RAM** for long inputs and concurrent batches vs conventional transformers (70-80% in another IBM article). This is vendor-reported, not independently verified. Lineup: H-Small 32B (9B active), H-Tiny 7B (1B active), H-Micro 3B, plus a transformer-only Micro. — [IBM announcement](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models); [IBM architecture explainer](https://www.ibm.com/think/news/hybrid-thinking-inside-architecture-granite-4-0); [InfoQ, Nov 2025](https://www.infoq.com/news/2025/11/ibm-granite-mamba2-enterprise)
- **Granite 4.0 Nano** (Oct 28, 2025): hybrid H-1B (about 1.5B) and H-350M, plus transformer-only 1B/350M variants for runtimes without optimized hybrid support. — [Hugging Face blog](https://huggingface.co/blog/ibm-granite/granite-4-nano); [It's FOSS](https://itsfoss.com/news/ibm-granite-4-nano/)
- **TII Falcon-H1** (July 30, 2025): **parallel** hybrid (attention and SSM heads side by side in each layer), 0.5B / 1.5B / 1.5B-Deep / 3B / 7B / 34B. Vendor claims: **0.5B is comparable to typical 2024 7B models**, and 1.5B-Deep rivals 7-10B models. On MBPP, Falcon-H1-0.5B scores 52.12 vs Qwen3-0.6B 51.06. — [arXiv 2507.22448](https://arxiv.org/abs/2507.22448); [HF model card 0.5B](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Base); [GitHub](https://github.com/tiiuae/Falcon-H1)
- **Zyphra Zamba2** (Nov 2024): 1.2B / 2.7B / 7.4B Mamba2 backbones with **shared** attention layers. Vendor: 2.7B has **2x faster TTFT, 27% less memory, 1.29x lower generation latency vs Phi-3-3.8B**. The report states 30-50% TTFT reduction and **6x less KV-cache memory** vs comparable transformers. Zyphra positions it for on-device use. — [arXiv 2411.15242](https://arxiv.org/html/2411.15242v1); [Zyphra blog](https://www.zyphra.com/post/zamba2-small)
- **[recalled]** AI21 **Jamba** (March 2024, older): Mamba + attention (1:7 attention:Mamba) + MoE, 52B total / 12B active, about 140K context on a single 80 GB GPU. — [arXiv 2403.19887](https://arxiv.org/abs/2403.19887)
- **[recalled]** Microsoft **Samba** (June 2024, older): Mamba + sliding-window attention at 3.8B, with large throughput gains over full-attention baselines at 128K prompts. — [arXiv 2406.07522](https://arxiv.org/abs/2406.07522)
- Distilled SSMs: **Llamba** (Feb 2025) distills Llama-3.x into Mamba-style models (Llamba-8B MMLU 60.0), ships quantized models and an **MLX implementation for iPhones/MacBooks with near-constant memory**. — [arXiv 2502.14458](https://arxiv.org/pdf/2502.14458)
- A systematic study (July 2025) recommends hybrids with **linear:full attention ratios of 3:1 to 6:1** (e.g., HGRN-2 or Gated DeltaNet) to reach transformer-level recall efficiently. — [arXiv 2507.06457](https://arxiv.org/html/2507.06457v1)

### Inferences
- For a companion that must recall names, dates and earlier conversation verbatim, **pure SSMs are risky** (copying/retrieval weakness). **Hybrids keep enough attention to fix recall** while giving most of the memory savings.
- KV-cache savings matter most on phones with 8-12 GB RAM shared with the OS. At 32k+ context a hybrid's memory stays nearly flat, while a transformer's KV cache grows linearly.
- The vendor throughput gains (Nemotron-H 3x, Granite 70% RAM) were measured on datacenter GPUs at long contexts. Gains on a laptop CPU or phone NPU at short chat contexts will be smaller and depend on how optimized the runtime kernels are.

### Gaps
- No independent on-phone benchmark comparing a hybrid (Granite-H, Falcon-H1, Zamba2) against a same-size transformer under the same runtime, quantization and hardware was found.
- Jamba and Samba numbers are recalled, not re-verified this session. No on-device data for them was found.
- Mamba-3 has no large open pretrained chat models confirmed yet (release is architecture plus kernels).

## 3. RWKV (v6/v7 "Goose"), RetNet, linear attention (Gated DeltaNet, Kimi Linear), xLSTM, Liquid AI LFM2

### Takeaway
These recurrent/linear designs all decode with **constant memory per token**. RWKV-7 (2.9B) reached 3B-class SoTA on multilingual tasks, and xLSTM 7B matches similar-size LLMs while being the fastest 7B its authors tested. **Liquid AI's LFM2/LFM2.5 is the most deployment-ready for phones**: a convolution + few-attention hybrid with vendor-measured ~2x CPU speedups on a Galaxy S24 Ultra and AMD laptop, plus native llama.cpp/MLX support. At large scale, linear attention survives mainly as hybrids (Qwen3-Next, Kimi Linear 3:1).

### Cited Findings
**RWKV**
- **RWKV-7 "Goose"** (Bo Peng et al., March 2025): generalized delta rule with vector-valued gating and in-context learning rates. The authors claim it can do state tracking and recognize all regular languages (beyond transformers' TC0 under standard conjectures), with **constant memory and constant time per token**. Its **2.9B model set a new 3B SoTA on multilingual tasks and matched 3B SoTA on English**. Apache 2.0. — [arXiv 2503.14456](https://www.arxiv.org/abs/2503.14456); [EleutherAI blog](https://www.eleuther.ai/papers-blog/rwkv-7-goose-with-expressive-dynamic-state-evolution)
- Informal community report (not peer-reviewed): RWKV-7 models trained at 4k context keep near-perfect memory to about 8-16k, then **may drastically forget**. — [HF paper page comments](https://huggingface.co/papers/2503.14456)
- RWKV-7 "G1" checkpoints continued into 2026 (e.g., 2.9B 20260420, 7.2B 20260414, 8k context). An Ollama mirror lists **1.5B / 2.9B / 7.2B / 13.3B** "G1l" base models and notes they are not safety-aligned assistants. — [HF BlinkDL/rwkv7-g1](https://huggingface.co/BlinkDL/rwkv7-g1/tree/main); [Ollama rwkv7-g1l](https://ollama.com/shoumenchougou/rwkv7-g1l)
- "State tuning" (2025) adapts RWKV-7's state matrix without changing weights, a cheap personalization route. — [arXiv 2504.05097](https://arxiv.org/pdf/2504.05097)

**RetNet (Microsoft, July 2023, older)**
- At 7B and 8k sequence length, RetNet decodes **8.4x faster and uses 70% less memory** than a KV-cached transformer (A100). Its recurrent form gives O(1) memory/compute per token, and training shows 25-50% memory savings and 7x acceleration. A 2026 ACL Findings survey describes its quality as "comparable to Transformers". — [arXiv 2307.08621](https://arxiv.org/abs/2307.08621); [Microsoft Research](https://microsoft.com/en-us/research/?p=958491); [ACL 2026 Findings survey](https://aclanthology.org/2026.findings-acl.256.pdf)

**Linear attention at scale (2025)**
- **Qwen3-Next-80B-A3B** (Sept 2025): 48 layers in a 3:1 pattern of **Gated DeltaNet : gated attention**, plus sparse MoE. At 128k input it is **3x faster than Qwen3-30B-A3B and 10x faster than Qwen3-32B** (Alibaba). **Independent testers placed it mid-pack on quality.** — [DeepLearning.AI The Batch](https://www.deeplearning.ai/the-batch/alibabas-new-model-uses-hybrid-attention-layers-and-a-sparse-moe-architecture-for-speed-and-performance); [Raschka hybrid-attention gallery](https://sebastianraschka.com/llm-architecture-gallery/hybrid-attention/)
- **Kimi Linear** (Moonshot, Oct 2025): Kimi Delta Attention (finer-gated Gated DeltaNet) hybridized with MLA, **48B total / 3B active**, about 1.4T tokens. The authors claim it outperforms full attention under fair comparisons. — [arXiv 2510.26692](https://arxiv.org/abs/2510.26692)

**xLSTM (Hochreiter / NXAI)**
- **[recalled]** Original xLSTM (Beck et al., May 2024; NeurIPS 2024): sLSTM (scalar memory, exponential gating, memory mixing) and **mLSTM (matrix memory with covariance update, fully parallelizable)**. These are direct extensions of the classic LSTM cell. — [arXiv 2405.04517](https://arxiv.org/abs/2405.04517)
- **xLSTM 7B** (March 2025; ICML 2025): downstream performance **comparable to similar-size LLMs**, with **significantly faster inference** than Llama- and Mamba-based LLMs. The poster calls it the fastest and most efficient 7B tested. Generation throughput stays **constant with prefill length** while Llama 2/3 degrade. A third-party summary estimates about 50% faster than Mamba. Weights: **NX-AI/xLSTM-7b** on HF, with open training code (xlstm, xlstm-jax). — [arXiv 2503.13427](https://arxiv.org/abs/2503.13427); [ICML 2025 poster](https://icml.cc/virtual/2025/poster/45587); [PMLR](https://proceedings.mlr.press/v267/beck25b.html); [GitHub NX-AI/xlstm](https://github.com/NX-AI/xlstm); [alphaXiv overview](https://alphaxiv.org/overview/2503.13427v1)

**Liquid AI LFM2 / LFM2.5**
- **LFM2** (July 2025 launch): vendor claims **2x faster decode and prefill than Qwen3 on CPU**. LFM2-700M is faster than Qwen3-0.6B in ExecuTorch and llama.cpp despite being 16% larger. Measured on a **Samsung Galaxy S24 Ultra (Snapdragon)** and **AMD Ryzen HX370**. — [Liquid AI blog](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models)
- LFM2 Technical Report (arXiv 2511.23404, Nov 2025): a hardware-in-the-loop architecture search converged on **gated short convolutions + a small number of GQA blocks, with no SSM or linear-attention operators**. Dense checkpoints deliver **up to ~2x prefill and decode speedups** vs similar-size baselines at maintained or better accuracy. — [arXiv 2511.23404](https://arxiv.org/abs/2511.23404)
- **LFM2-8B-A1B** (on-device MoE): 8.3B total / 1.5B active, **18 conv + 6 GQA blocks**. Now deprecated in favor of LFM2.5-8B-A1B. — [HF model card](https://huggingface.co/LiquidAI/LFM2-8B-A1B); [Liquid blog](https://www.liquid.ai/blog/lfm2-8b-a1b-an-efficient-on-device-mixture-of-experts)
- **LFM2.5-350M** (vendor-measured): llama.cpp Q4 on AMD Ryzen AI Max 395+ CPU gives **2.9K prefill / 313 decode tok/s, 434 MB peak memory**. Cactus Engine int8 rows show about **88 decode tok/s on iPhone 13 Mini, 42 on Pixel 6a, 30 on Raspberry Pi 5**. Model attribution for the Cactus rows was ambiguous in the extract. — [Liquid LFM2.5-350M blog](https://www.liquid.ai/en/blog/lfm2-5-350m-no-size-left-behind)

### Inferences
- xLSTM is the natural bridge for an LSTM practitioner. mLSTM is effectively a gated linear-attention/matrix-memory RNN in the same family as RWKV-7, Gated DeltaNet and Mamba-2 (all "fast-weight" / delta-rule recurrences). Expect the same recall weakness as pure SSMs unless hybridized.
- For a phone companion today, LFM2/LFM2.5 has the strongest evidence of real phone speedups, though only from the vendor. RWKV-7 is attractive for constant memory and community GGUF builds, but G1 models are base/lightly-tuned and not safety-aligned.
- RetNet's 2023 claims have largely been absorbed into later gated linear-attention designs. No major open RetNet chat model was found.

### Gaps
- No independent reproduction of LFM2's phone speedups. LFM2 benchmark tables (MMLU/IFEval vs Qwen3) not retrieved because liquid.ai was blocked.
- xLSTM 7B: exact MMLU and tokens/s values not extractable. llama.cpp/MLX support for xLSTM not confirmed.
- No head-to-head energy (J/token) data for RWKV, xLSTM or LFM2 on phones.

## 4. Mixture of Experts (sparse activation) for on-device use

### Takeaway
MoE cuts **compute per token** (active parameters) but not **storage**: every expert must sit in RAM or be streamed from flash. On phones, a 20-30B MoE is bottlenecked by memory capacity and flash I/O. Offloading experts to storage brings RAM down to near active size but costs **4-20x latency**. **Small "on-device MoEs" (Granite-4.0-H-Tiny 7B/1B active, LFM2-8B-A1B 8.3B/1.5B active) and laptop-class gpt-oss-20b (21B/3.6B active in about 16 GB)** are the realistic options.

### Cited Findings
- **gpt-oss-20b** (OpenAI, Aug 2025): **21B total, 3.6B active** across 32 experts. MXFP4 lets it run in **about 16 GB** (vs about 48 GB without). — [vLLM recipe](https://recipes.vllm.ai/openai/gpt-oss-20b); [Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/openai-intros-two-lightweight-open-model-language-models-that-can-run-on-consumer-gpus-optimized-to-run-on-devices-with-just-16gb-of-memory)
- Independent test (Micro Center): the 16 GB really needs to be **dedicated** GPU memory. A system with 16 GB total struggles, and a **16 GB Mac mini was slowest** of the machines tested. — [Micro Center](https://www.microcenter.com/site/mc-news/article/can-your-pc-run-gpt-oss-ai.aspx)
- AMD published (2026) a technical article on running gpt-oss-20b MoE inference on **Ryzen AI NPUs (Strix / Strix Halo)**. — [AMD](https://www.amd.com/en/developer/resources/technical-articles/2026/accelerating-gpt-oss-20b-on-amd-ryzen-ai-npus.html)
- **EdgeMoE** keeps non-expert weights in RAM and fetches experts from storage on activation. **SwapMoE** keeps a small, changing set of "virtual experts" in RAM under a tunable budget. **CoSMoEs** offloads unused experts so RAM matches the active size, at **4-20x increased latency**. **FlashMoE** argues RAM offloading becomes impractical as MoEs reach hundreds of GB. On **Apple Silicon unified memory**, large MoEs can stay fully resident. — [EdgeMoE arXiv 2308.14352](https://arxiv.org/pdf/2308.14352.pdf); [SwapMoE arXiv 2308.15030](https://arxiv.org/html/2308.15030v4); [CoSMoEs arXiv 2503.00245](https://export.arxiv.org/pdf/2503.00245); [MobileMoE, NeurIPS 2026](https://neurips.cc/virtual/2026/poster/150777)
- Small on-device MoEs: **Granite-4.0-H-Tiny 7B / 1B active** (hybrid Mamba-2), **Granite-4.0-H-Small 32B / 9B active** — [IBM](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models). **LFM2-8B-A1B 8.3B / 1.5B active** — [HF](https://huggingface.co/LiquidAI/LFM2-8B-A1B). **Qwen3-Next 80B / 3B active**, **Kimi Linear 48B / 3B active** (laptop/workstation class, see Q3).
- Related non-MoE trick: **Gemma 3n** (Google, 2025) stores about 5B (E2B) / 8B (E4B) raw parameters but runs in **about 2 GB / 3 GB**. It caches Per-Layer Embeddings on CPU or fast local storage and uses MatFormer nesting (E2B sub-model inside E4B), so only about 1.91B parameters need to be in accelerator memory for E2B. — [Google developer blog](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/); [Gemma 3n docs](https://ai.google.dev/gemma/docs/3n)

### Inferences
- On a phone with 8-12 GB RAM, keep the total parameter footprint (at 4-bit) under about 4-5 GB, which means roughly 8B total for an MoE. 1-1.5B active parameters then gives fast decode. LFM2-8B-A1B / LFM2.5-8B-A1B and Granite-4.0-H-Tiny fit this profile.
- On a 16-32 GB laptop (especially Apple Silicon unified memory), gpt-oss-20b or a ~30B-A3B MoE is feasible and gives much higher quality per FLOP than a dense 3-4B model.
- MoE and hybrid recurrence combine well. MoE reduces compute, the SSM/linear layers reduce KV-cache growth, and both help battery life.

### Gaps
- No measured joules/token comparison of a small MoE vs a same-active-size dense model on the same phone was found.
- Qwen3-30B-A3B specifics (total/active and laptop tokens/s) were not verified in this session.

## 5. Low-bit weights: BitNet b1.58 (ternary), BitNet b1.58 2B4T, bitnet.cpp

### Takeaway
BitNet b1.58 2B4T (April 2025) is the only **natively trained** ternary LLM at production quality. At 2B parameters and 4T tokens it roughly matches Qwen2.5-1.5B on average (54.19 vs 55.23) and beats Llama-3.2-1B. It uses **0.4 GB** of non-embedding weights (vs 2.6 GB bf16 / 0.7 GB INT4 for Qwen2.5-1.5B) and has **29 ms/token CPU decode vs 65 ms**. Energy figures are **estimated**, not measured, and the gains need **bitnet.cpp**; standard Transformers gives none. The ecosystem is thin: one official 2B model, with no larger official chat models.

### Cited Findings
- **[recalled]** "The Era of 1-bit LLMs" (Feb 2024, older): BitNet b1.58 with weights in {-1, 0, +1} matches FP16 LLaMA perplexity and end-task performance from 3B scale, with lower latency, memory and energy (estimated GPU figures). — [arXiv 2402.17764](https://arxiv.org/abs/2402.17764)
- **BitNet b1.58 2B4T** (released 14 April 2025; arXiv 2504.12285): open weights, about 2.4B parameters, trained on 4T tokens. Microsoft reports it is "on par with similar-sized open-weight full-precision LLMs" with much lower memory, energy and decoding latency. — [arXiv 2504.12285](https://arxiv.org/abs/2504.12285); [GitHub microsoft/BitNet](https://github.com/microsoft/BitNet)
- Comparison table (model card). Columns: LLaMA 3.2 1B / Gemma-3 1B / Qwen2.5 1.5B / SmolLM2 1.7B / MiniCPM 2B / **BitNet 2B**.
  - **CPU decoding latency:** 48 / 41 / 65 / 67 / 124 / **29 ms**
  - **Energy (estimated):** 0.258 / 0.186 / 0.347 / 0.425 / 0.649 / **0.028 J**
  - Column mapping is cross-checked by Tom's Hardware (Qwen 65 ms, BitNet 29 ms).
  - — [Azure AI catalog card](https://ai.azure.com/catalog/models/microsoft-bitnet-b1.58-2b-4t); [Tom's Hardware](https://tomshardware.com/tech-industry/artificial-intelligence/microsoft-researchers-build-1-bit-ai-llm-with-2b-parameters-model-small-enough-to-run-on-some-cpus)
- **Non-embedding memory:** BitNet 2B **0.4 GB**, Qwen2.5-1.5B bf16 2.6 GB, GPTQ-int4 0.7 GB, AWQ-int4 0.7 GB, Llama-3.2-1B 2 GB. On the report's subset, **BitNet averages 55.01 vs bf16 Qwen2.5-1.5B 55.72** and beats both INT4 Qwen variants. — [arXiv 2504.12285 Table 2](https://arxiv.org/html/2504.12285v1)
- **Quality:**
  - 11-benchmark average: BitNet **54.19**, Qwen2.5-1.5B 55.23, Llama-3.2-1B 44.90.
  - MMLU: **53.17**, 60.25, 45.58.
  - GSM8K: **58.38**, 56.79, 38.21.
  - All from Microsoft's own evaluation. — [HF model card](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T); [Tom's Hardware](https://tomshardware.com/tech-industry/artificial-intelligence/microsoft-researchers-build-1-bit-ai-llm-with-2b-parameters-model-small-enough-to-run-on-some-cpus)
- Runtime caveat: "Please do NOT expect performance efficiency gains (speed, latency, or energy) when using this model with the standard transformers library". You MUST use **bitnet.cpp**. — [HF Transformers BitNet docs](https://huggingface.co/docs/transformers/v5.13.1/model_doc/bitnet); [Azure card](https://ai.azure.com/catalog/models/microsoft-bitnet-b1.58-2b-4t)
- **bitnet.cpp** (v1.0, 17 Oct 2024; arXiv 2410.16144):
  - Speedups: **1.37x-5.07x on ARM**, **2.37x-6.17x on x86**.
  - Energy reductions: **55.4-70.0% on ARM**, **71.9-82.2% on x86** (vs full-precision). The README does not document the energy method or hardware.
  - Claims a **100B BitNet b1.58 model on a single CPU at 5-7 tok/s** (no hardware given).
  - Jan 15, 2026 CPU kernel update: additional **1.15x-2.1x**.
  - GPU kernel added May 20, 2025. BitNet embedding models (0.6B/270M) added July 2026.
  - Supported models: official 2B4T, plus community bitnet_b1_58-large (0.7B), 3B, Llama3-8B-1.58, and the Falcon3 / Falcon-E 1.58-bit families.
  - — [GitHub microsoft/BitNet README](https://github.com/microsoft/BitNet)
- **[recalled]** The 2B4T card footnote attributes the CPU latency to an Intel Core i7-13800H laptop CPU and energy to an operation-count-based model. Verify on the card. — [HF model card](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)

### Inferences
- For a phone companion, ternary weights are the biggest single memory lever: about 0.4 GB for a 2B model, so a ternary 7-8B would be about 1.5-2 GB. But the only strong native ternary model is 2B, with a 4k context window **[recalled]**, and ternary fine-tuning tooling is immature.
- Against a standard Q4 GGUF of Qwen-class 1.5-3B in llama.cpp, BitNet's advantage is roughly 1.7x memory and about 2x CPU decode speed, at some quality cost on knowledge-heavy tasks (MMLU -7 vs Qwen2.5-1.5B).
- The headline energy numbers (0.028 J vs 0.347 J) are **estimates**. Treat 55-82% reductions as plausible upper bounds until wall-power measurements exist.

### Gaps
- No independent wall-power measurement of BitNet 2B4T energy/token on a phone or laptop was found.
- No official BitNet model larger than 2B. Falcon-E and Llama3-8B-1.58 quality vs FP16 not checked.
- Whether BitNet runs on phone NPUs (vs CPU only) was not found.

## 6. Associative / test-time memory: modern Hopfield networks, Titans, ATLAS, TTT layers, Infini-attention, memory layers, and the hippocampus analogy

### Takeaway
There is a coherent research line: **attention = modern Hopfield associative retrieval** (2020). **Linear RNNs/SSMs = fast-weight memories**. **TTT/Titans/ATLAS = memories updated by gradient descent at test time**, driven by "surprise". These map loosely onto hippocampal fast episodic storage alongside slow neocortical weights (complementary learning systems). Results are strong in papers at ≤1B scale (Titans past 2M-token context; ATLAS +80% on BABILong at 10M). **No large open-weight Titans/ATLAS model exists**, and independent reproductions of Infini-attention were shaky. For a companion today, "hippocampal" memory is best built as **retrieval (RAG) plus a hybrid model**, not as a trainable in-weights memory.

### Cited Findings
- **Modern Hopfield networks** (Ramsauer et al., arXiv July 2020; ICLR 2021): the update rule of continuous-state modern Hopfield networks **is transformer attention**. Storage capacity is **exponential in dimension** (~c^((d-1)/4)), retrieval typically takes one update with exponentially small error, vs about 0.138d patterns for classical Hopfield nets. Caveat: later work notes softmax updates can fall into metastable states and the capacity result uses a relaxed retrieval notion. — [arXiv 2008.02217](https://arxiv.org/abs/2008.02217); [ICLR 2021](https://iclr.cc/virtual/2021/poster/2657); [Sparse & Structured Hopfield, arXiv 2402.13725](https://arxiv.org/pdf/2402.13725)
- **TTT layers** (Sun et al., July 2024): the hidden state is itself a model (TTT-Linear, TTT-MLP), updated by a self-supervised gradient step per token. They **keep reducing perplexity with more context, while Mamba plateaus after 16k**, and TTT-Linear is faster than a transformer at 8k context. Later versions note linear hidden states were developed further in Mamba-2 and Gated DeltaNet. — [arXiv 2407.04620](https://arxiv.org/abs/2407.04620)
- **Titans** (Behrouz, Zhong, Mirrokni; Dec 2024 preprint; NeurIPS 2025):
  - Attention works as accurate **short-term** memory, and a deep **neural long-term memory learns to memorize at test time**. The authors explicitly cite neuropsychology (short-term, working and long-term memory as a "confederation of systems").
  - Reported to beat transformers and modern linear RNNs on LM, commonsense reasoning, genomics and time series, and to **scale past 2M-token context** with better needle-in-haystack accuracy.
  - Google's Dec 4, 2025 blog (Titans + MIRAS) describes **"surprise"-driven** memorization without offline retraining.
  - — [arXiv 2501.00663](https://arxiv.org/abs/2501.00663); [Google Research pub](https://research.google/pubs/titans-learning-to-memorize-at-test-time/); [Google blog Dec 2025](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/); [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/119639)
- **ATLAS** (Behrouz et al., May 2025): a higher-capacity long-term memory that optimizes over current **and past** tokens (not just the latest). Reported **+80% accuracy at 10M context on BABILong**. — [arXiv 2505.23735](https://arxiv.org/abs/2505.23735)
- **Infini-attention** (Munkhdalai, Faruqui, Gopal, Google, April 2024): adds a fixed-size compressive memory to each attention layer. A 1B model solved **1M-token passkey** after fine-tuning on 5K-length samples, and an 8B model set SoTA on **500K book summarization**. — [arXiv 2404.07143](https://arxiv.org/abs/2404.07143)
  - **Independent reproduction failed:** Hugging Face's Aug 2024 "A failed experiment: Infini-Attention" found balance factors failed to converge and **performance dropped as the number of memory compressions grew**. — [HF blog](https://huggingface.co/blog/infini-attention)
  - A Dec 2025 small-scale study (300M LLaMA-style) found training sensitive to learning rate (vanishing gradients). It still saw up to **31% higher accuracy than baseline** at 16k despite degradation. — [arXiv 2512.23862](https://arxiv.org/abs/2512.23862v1)
- **Memory Layers at Scale** (Meta FAIR, Dec 2024; ICML 2025): trainable key-value lookup (product keys) that **adds parameters without adding FLOPs**, scaled to millions of keys, replacing some dense FFNs. It improves factual QA. The authors evaluated only short-form QA, and gains were larger early in training. **[recalled]** The abstract reports up to 128B memory parameters, ~1T tokens, beating dense models with over 2x the compute and matched MoEs. — [arXiv 2412.09764](https://arxiv.org/abs/2412.09764); [Meta AI](https://ai.meta.com/research/publications/memory-layers-at-scale/); [PMLR](https://proceedings.mlr.press/v267/berges25a.html); [Graphcore analysis](https://graphcore-research.github.io/memory-layers)
- **Explicit hippocampal design:** "A Hippocampus for Linear Attention" (HOLA, 2026) is inspired by **Complementary Learning Systems**. A compressive linear-attention state plays the neocortex role, and a bounded **exact episodic KV cache that preferentially stores surprising tokens** plays the hippocampus role. — [HF papers 2607.02303](https://huggingface.co/papers/2607.02303.md)

### Inferences
- Brain-analogy map:
  - **Hopfield/attention** is CA3-like autoassociative pattern completion over stored episodes.
  - **Linear-RNN/SSM state** is a lossy, fast-decaying working memory.
  - **Titans/TTT** gradient-updated memory is rapid synaptic plasticity ("one-shot" encoding gated by surprise).
  - **Pretrained weights** are slow neocortical consolidation.
  - This mapping is our interpretation (common in the field); the papers themselves use looser neuropsychology language.
- For a personal companion, the practical hippocampus is (a) a persistent on-device vector store / episodic log retrieved into an attention-capable model, plus (b) optional periodic LoRA or "state tuning" consolidation (RWKV-7 state tuning, Q3). Test-time memory models are promising but not yet available as runnable open weights.
- Memory layers resemble MoE for on-device use: huge sparse parameter tables cost storage but little compute, and could be paged from flash like Gemma 3n's PLE.

### Gaps
- No official open-weight Titans or ATLAS checkpoints, and no independent large-scale reproduction of Titans, were found. Model sizes in the Titans experiments (≤~760M **[recalled]**) were not re-verified.
- No on-device (phone/laptop) evaluation of any test-time-memory architecture was found.
- Modern Hopfield networks have no standalone LM at scale; they live on as the theory of attention (and in Hochreiter-lab follow-ups like xLSTM).

## 7. JEPA / world models (LeCun; V-JEPA 2, 2025) as a brain-inspired alternative to next-token prediction

### Takeaway
JEPA predicts in **latent/embedding space** instead of reconstructing tokens or pixels, which LeCun argues is closer to how brains build world models. V-JEPA 2 (June 2025, 1.2B) is a strong **video** world model that enables zero-shot robot planning. For **language**, JEPA so far is an **auxiliary fine-tuning loss** (LLM-JEPA, ICLR 2026), not a replacement for next-token LMs. It is not currently a basis for a chat companion's core LM.

### Cited Findings
- **V-JEPA 2** (Meta, released June 11, 2025): **1.2B-parameter** world model built on JEPA (first shared 2022). Results:
  - Motion understanding: **77.3 top-1 on Something-Something v2**.
  - Action anticipation: **39.7 R@5 on Epic-Kitchens-100**.
  - Video QA after aligning with an LLM: **84.0 PerceptionTest, 76.9 TempCompass**.
  - Robotics: post-trained (V-JEPA 2-AC) on **under 62 hours of unlabeled DROID robot video**, then deployed zero-shot on Franka arms in two labs for reaching, grasping and pick-and-place with image goals.
  - — [Meta AI blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/); [Meta V-JEPA page](https://ai.meta.com/vjepa/); [arXiv 2506.09985](https://arxiv.org/abs/2506.09985); [Meta publication](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/)
- **[recalled]** Meta reported pick-and-place success of roughly 65-80% in new environments, and pretraining on over 1M hours of video. — [Meta AI blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)
- **LLM-JEPA** (Huang, LeCun, Balestriero; arXiv Sept 2025; ICLR 2026): adds a JEPA embedding-prediction loss between two "views" (e.g., natural-language description and code) **on top of** the standard next-token loss. Reported to beat standard objectives "by a significant margin" and resist overfitting on NL-RX, GSM8K, Spider and RottenTomatoes across Llama3, OpenELM, Gemma2 and OLMo. Mostly fine-tuning, with some pretraining experiments. — [arXiv 2509.14252](https://www.arxiv.org/abs/2509.14252); [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ac818ec2da1c976b267de32f59709c69-Abstract-Conference.html)
- The search found no independent replication of V-JEPA 2's robot-planning numbers. — [W&B ML news summary](https://wandb.ai/byyoung3/ml-news/reports/Meta-Unveils-V-JEPA-2-A-World-Model-for-Physical-Understanding-Prediction-and-Robot-Planning--VmlldzoxMzE4Mjc2Mw)

### Inferences
- For a personal companion, JEPA is relevant (a) as a training trick to improve a small LM's representations (LLM-JEPA), and (b) if the companion later needs camera-based understanding (V-JEPA 2 encoder, 1.2B). It is not a drop-in generative chat architecture.
- A 1.2B video encoder is laptop-feasible, but no on-device benchmarks were found.

### Gaps
- No JEPA-based model that generates text competitively with same-size LMs was found. "VL-JEPA" and similar 2025-26 vision-language JEPA work was not checked.
- No llama.cpp/MLX ports of V-JEPA 2 were confirmed.

## 8. What is runnable today as open weights on a laptop or phone (llama.cpp, MLX, Ollama, ExecuTorch, etc.)

### Takeaway
Runnable today with mainstream local runtimes:
- **Hybrid SSM/conv models:** LFM2/LFM2.5, Granite 4.0-H (incl. Nano), Falcon-H1, Nemotron-H / Nemotron Nano 2.
- **Pure SSMs:** Mamba / Mamba-2 / Falcon Mamba.
- **RWKV:** RWKV-6, and RWKV-7 via community GGUF in Ollama.
- **MoEs:** gpt-oss-20b, Granite-H-Tiny, LFM2-8B-A1B.
- **Ternary:** BitNet 2B4T, via the **separate bitnet.cpp** fork.

Not practically runnable on consumer devices yet: spiking LMs (GPU/vLLM-oriented, ModelScope weights), xLSTM 7B (HF/NX-AI kernels; llama.cpp unconfirmed), Titans/ATLAS (no official weights), and JEPA LMs.

### Cited Findings
- **llama.cpp:**
  - The architecture table includes `mamba2`, `falcon-h1`, `nemotron_h` and (from search) `rwkv6`. Granite 4.0 hybrid support merged as "model : Granite Four (#13550)".
  - IBM's GGUF compatibility table lists Granite-4.0-H variants (e.g., H-Tiny GraniteMoeHybrid, 7B / 1B active) as working, tested against build b8742.
  - LFM2 hybrid support merged July 2025 (#14620). An older issue reporting "unknown architecture" for LFM2 conflicts with this and is likely stale.
  - — [IBM/gguf](https://github.com/IBM/gguf); [llama.cpp commit Granite Four](https://app.semanticdiff.com/gh/ggml-org/llama.cpp/commit/0aedae00e6fb48680324a5ac5da9cba0e35de6b5); [llama-arch.cpp copy](https://huggingface.co/kaisser/LLM-Maroc/blob/main/llama.cpp/src/llama-arch.cpp); [LFM2 issue #22287](https://github.com/ggml-org/llama.cpp/issues/22287)
- llama.cpp offers Android build docs and Apple XCFramework builds (iOS/macOS packaging). — [GitHub ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- **RWKV-7** G1l 1.5B / 2.9B / 7.2B / 13.3B are published on **Ollama** (community upload). Ollama runs llama.cpp-format models, which implies working RWKV-7 GGUF inference, but this is not confirmed in llama.cpp docs. — [Ollama rwkv7-g1l](https://ollama.com/shoumenchougou/rwkv7-g1l)
- Granite 4.0 Nano also ships **transformer-only** 1B/350M variants for runtimes lacking optimized hybrid support. — [HF blog](https://huggingface.co/blog/ibm-granite/granite-4-nano)
- **MLX:**
  - mlx-lm supports "thousands" of HF LLMs, maps Falcon-Mamba to its Mamba module, and LFM2 is officially documented for MLX by Liquid AI (mlx-community hosts quantized LFM2).
  - mlx-swift-examples added Falcon-H1, a Granite hybrid MoE and LFM2-MoE ports. Some newer hybrids still fail with "not supported".
  - Llamba ships an MLX implementation for iPhones/MacBooks.
  - — [mlx-lm GitHub](https://github.com/ml-explore/mlx-lm); [Liquid MLX docs](https://docs.liquid.ai/docs/inference/mlx); [mlx-swift-examples releases](https://github.com/ml-explore/mlx-swift-examples/releases); [DeepWiki mlx-lm architectures](https://deepwiki.com/ml-explore/mlx-lm/5-model-architectures); [mlx-lm issue #1233](https://github.com/ml-explore/mlx-lm/issues/1233); [Llamba arXiv 2502.14458](https://arxiv.org/pdf/2502.14458)
- **LFM2 runtimes:** ExecuTorch and llama.cpp were used for the vendor's phone benchmarks (Galaxy S24 Ultra). An ONNX export of LFM2-8B-A1B exists. — [Liquid blog](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models); [onnx-community/LFM2-8B-A1B-ONNX](https://huggingface.co/onnx-community/LFM2-8B-A1B-ONNX)
- **BitNet 2B4T:** official GGUF (`microsoft/bitnet-b1.58-2B-4T-gguf`) runs via **bitnet.cpp** on x86 and ARM CPUs. A demo shows a 3B BitNet on Apple M2. — [HF GGUF](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf); [GitHub microsoft/BitNet](https://github.com/microsoft/BitNet)
- **xLSTM 7B:** weights `NX-AI/xLSTM-7b` on HF, with code in NX-AI/xlstm (PyTorch/Triton kernels) and xlstm-jax. — [GitHub NX-AI/xlstm](https://github.com/NX-AI/xlstm)
- **SpikingBrain:** weights on ModelScope (MIT), inference via HF/vLLM plugin on GPUs. No CPU/mobile release in the repo. — [GitHub BICLab/SpikingBrain-7B](https://github.com/BICLab/SpikingBrain-7B)
- **Gemma 3n** (E2B/E4B) is designed for phones (about 2-3 GB). — [Gemma 3n docs](https://ai.google.dev/gemma/docs/3n)
- **Measured phone energy (context for any choice):**
  - PowerInfer-2 (2024) reported **0.257 J/token** vs 0.373 (Qualcomm QNN) and **0.672 J/token (llama.cpp)** on a smartphone. — [arXiv 2406.06282](https://arxiv.org/pdf/2406.06282)
  - A 2026 study ran 8 models (0.5B-9B) with llama.cpp on an unrooted **Galaxy S25 Ultra**, measuring J/token, latency and TTFT. — [arXiv 2603.26603](https://arxiv.org/pdf/2603.26603)
  - A 2026 battery study found on-device inference on average **3x less energy-efficient than batched server inference**, and that the quantization-energy relationship is non-monotonic. — [arXiv 2609.11940](https://arxiv.org/pdf/2609.11940)
  - Decode dominates on-device energy. — [MNN-AECS arXiv 2506.19884](https://arxiv.org/html/2506.19884v1)

### Inferences
Shortlist for an offline companion (as of late 2026):
- **Phone (8-12 GB):** LFM2.5 (350M-1.2B dense or 8B-A1B MoE), Granite-4.0-H-Micro/Nano, Falcon-H1-1.5B-Deep, or Gemma 3n E2B/E4B, at Q4 in llama.cpp / ExecuTorch / MLX (iOS).
- **Laptop (16-32 GB):** gpt-oss-20b, Granite-4.0-H-Tiny/Small, Nemotron Nano 2 9B, Falcon-H1-7B, or Qwen3-Next-class MoE on 32 GB+ Apple Silicon.
- **Experimental lanes:** BitNet 2B4T (lowest memory/CPU energy, limited model choice); RWKV-7 G1 (constant memory, base-model quality); xLSTM 7B (LSTM lineage, weak runtime support).

Other points:
- Hybrid models in llama.cpp may run slower than their datacenter numbers suggest (IBM shipping transformer-only Nano variants hints that hybrid kernels are less optimized in some runtimes). Benchmark on the target device.
- Energy on device is dominated by memory traffic during decode. Smaller active weights (low-bit, MoE with few active parameters) and no growing KV cache (hybrid/SSM) are the levers that matter. Spiking does not help without neuromorphic hardware.

### Gaps
- No single authoritative, current list of llama.cpp/MLX architecture support could be fetched (README model list not captured). RWKV-7, xLSTM, Kimi Linear and Qwen3-Next llama.cpp status are unconfirmed.
- No independent, same-device benchmark set comparing hybrid vs transformer vs BitNet vs RWKV on tokens/s and J/token on a phone was found. This would be the most decision-relevant missing data; the user may need to run it themselves (e.g., llama-bench plus a USB power meter or Android battery stats).
