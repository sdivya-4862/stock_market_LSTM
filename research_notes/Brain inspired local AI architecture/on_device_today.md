# Private, On-Device Personal AI Assistants: What a Small Team Can Build Today (state as of late 2025 to Nov 2026)

Method note for the report writer: these notes were compiled on 2026-11-23. In this session, WebFetch could not reach arxiv.org, huggingface.co, developers.google.com, security.apple.com or github.io; the session's network settings would need to allow those hosts. Because of that, every number below comes from search-engine summaries of the cited pages, not from reading the full pages. Primary sources (tech reports, model cards, official blogs) are preferred and labelled. Vendor self-reported numbers are marked "(vendor)". Anything older than 2025 is marked "(older)". When a search summary did not make clear which URL a number came from, that is flagged.

---

## 1. Best small open-weight models for on-device use (0.5B-14B), quality vs frontier, and licenses

### Takeaway
As of late 2026 the strongest candidates for a phone or laptop are Qwen3.5 small (0.8B/2B/4B/9B, March 2026), Google Gemma 4 E2B/E4B (April 2026, now Apache 2.0), Liquid AI LFM2/LFM2.5 (built for fast CPU inference), Microsoft Phi-4-mini (3.8B, MIT), and Apple's roughly 3B on-device model, which is reachable only through Apple's Foundation Models framework and was rebuilt for WWDC26. On reasoning benchmarks a 4B-9B model can now match 2024-25 cloud models. They still hallucinate badly on factual recall (80-82% hallucination rate for Qwen3.5 4B/9B on AA-Omniscience), and in agentic personal-assistant stacks they trail frontier models by 25-39 percentage points unless the scaffolding is tuned for them.

### Cited Findings
**Qwen3.5 small series (Alibaba, released 1 Mar 2026)**
- Released in four sizes (0.8B, 2B, 4B, 9B). The 2B reportedly runs offline on a recent iPhone and handles text and images — [mean.ceo blog, 2026](https://blog.mean.ceo/qwen-3-5-small-model-series-release/)
- Qwen3.5-9B scores 81.7 on GPQA Diamond, above gpt-oss-120b's 80.1, and 70.1 on MMMU-Pro, above Gemini 2.5 Flash-Lite at 59.7 — [VentureBeat, 2026](https://venturebeat.com/technology/alibabas-small-open-source-qwen3-5-9b-beats-openais-gpt-oss-120b-and-can-run). Artificial Analysis measured lower MMMU-Pro scores: 69.2% for 9B and 65.4% for 4B — [Artificial Analysis, 2026](https://artificialanalysis.ai/articles/qwen3-5-small-models)
- Artificial Analysis Intelligence Index (reasoning mode): 9B = 32, 4B = 27, 2B = 16, 0.8B = 9. At 4-bit, the 9B needs about 6 GB, the 4B about 3 GB, and the 2B and 0.8B under 2 GB — [Artificial Analysis, 2026](https://artificialanalysis.ai/articles/qwen3-5-small-models)
- **Hallucination:** Artificial Analysis reports hallucination rates of 80-82% on its AA-Omniscience factual test for the 4B and 9B — [Artificial Analysis, 2026](https://artificialanalysis.ai/articles/qwen3-5-small-models)
- Community tests found Qwen3.5-0.8B weak at code: 67% zero-shot on code fixing, falling to 33% with one example — [mean.ceo blog, 2026](https://blog.mean.ceo/qwen-3-5-small-model-series-release/) (secondary, low authority)
- Video-MME (with subtitles): 9B 84.5, 4B 83.5 — [VentureBeat, 2026](https://venturebeat.com/technology/alibabas-small-open-source-qwen3-5-9b-beats-openais-gpt-oss-120b-and-can-run)

**Qwen3 (2025) small models**
- Qwen3-4B: 32K context, about 2.6 GB at Q4, native tool calling, 83.7 on MMLU-Redux; Alibaba recommends Qwen-Agent for reliable function calling — [digitalapplied on-device agents guide, June 2026](https://www.digitalapplied.com/blog/small-language-models-on-device-agents-2026-guide) (cites model cards and Qwen3 tech report [arXiv 2505.09388, May 2025](https://arxiv.org/pdf/2505.09388))

**Google Gemma 4 (released 2 Apr 2026; one guide gives 31 Mar 2026 as the announcement date)**
- License changed to Apache 2.0; Gemma 1-3 had shipped under custom source-available terms — [Google blog, Apr 2026](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/); [HF model card google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B); [VentureBeat, Apr 2026](https://venturebeat.com/technology/google-releases-gemma-4-under-apache-2-0-and-that-license-change-may-matter); [Wikipedia: Gemma](https://en.wikipedia.org/wiki/Gemma_(language_model))
- E2B has 2.3B effective parameters (about 5.1B including embeddings) and E4B has 4.5B effective (about 8B total). Both have a 128K context and accept text, image and audio input, and both target phones and laptops — [wavespeed.ai](https://wavespeed.ai/blog/posts/what-is-google-gemma-4/); [aurigait guide](https://aurigait.com/blog/gemma-4-features-benchmarks-guide/); [codersera guide](https://codersera.com/blog/gemma-4-complete-guide-2026/) (secondary)
- Instruction-tuned MMLU-Pro reported as 69.4% (E4B) and 60.0% (E2B); AIME 2026 42.5%/37.5%; LiveCodeBench 52.0%/44.0%. These are third-party reports of the model card and could not be confirmed against the official card — [aurigait](https://aurigait.com/blog/gemma-4-features-benchmarks-guide/); [codersera](https://codersera.com/blog/gemma-4-complete-guide-2026/)
- One guide claims a 12B Gemma 4 variant has a custom license and a 700M MAU cap. This conflicts with the other sources and needs checking — [codersera](https://codersera.com/blog/gemma-4-complete-guide-2026/)

**Google Gemma 3n (May-June 2025; predecessor of Gemma 4 E-models)**
- E2B and E4B have raw parameter counts of 5B and 8B but run in about 2 GB and 3 GB of memory. Per-Layer Embeddings (PLE) keep much of the model on CPU, so only about 2B or 4B of core weights need to sit in accelerator memory — [Google Developers Blog: Gemma 3n developer guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/)
- Gemma 3 and earlier used Google's custom terms with a prohibited-use policy, not an OSI license — [braindetox license guide](https://braindetox.kr/en/posts/open_weight_model_license_guide.html); [Wikipedia](https://en.wikipedia.org/wiki/Gemma_(language_model))

**Apple on-device foundation model (2025 tech report, July 2025)**
- About 3B parameters, optimised for Apple silicon, using KV-cache sharing and 2-bit quantization-aware training — [arXiv 2507.13575](https://arxiv.org/abs/2507.13575); [Apple ML Research, 2025](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025)
- The model is split into two blocks. Block 2 drops its key/value projections and reuses Block 1's KV cache, cutting KV-cache memory by 37.5% — [arXiv 2507.13575v3](https://arxiv.org/html/2507.13575v3)
- Apple says it "performs favorably" against Qwen-2.5-3B in all languages and is "competitive" with the larger Qwen-3-4B and Gemma-3-4B in English. Commentary notes it falls short of Qwen-3-4B on MMLU, MMMLU and MGSM — [Apple ML Research update, 2025](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates); [AppleLeaker on X, Jul 2025](https://x.com/LeakerApple/status/1946000652750221530) (secondary). In human evaluations, Apple's on-device and server models still lag well behind GPT-4o — [Neowin, 2025](https://www.neowin.net/news/apples-ai-models-still-trail-behind-openais-gpt-4o-despite-latest-update/); [The Decoder, 2025](https://the-decoder.com/apples-new-ai-benchmarks-show-its-models-still-lag-behind-leaders-like-openai-and-google/)
- WWDC26 (June 2026) brought "a new on-device model, rebuilt from the ground up" with better reasoning and tool calling, vision input, and new APIs for context size and token counting — [WWDC26 session 241](https://developer.apple.com/videos/play/wwdc2026/241/). Third-party blogs claim an 8,192-token on-device context and a "20B sparse" on-device model; neither appears on Apple's session pages and both are unverified — [callstack, 2026](https://www.callstack.com/blog/on-device-ai-after-wwdc-2026-whats-new); [ofox.ai, 2026](https://ofox.ai/blog/apple-foundation-models-3-wwdc-2026-developer-read/)

**Microsoft Phi-4-mini (Jan 2025)**
- 3.8B dense, MIT license, 128K context, about 3 GB at Q4; MMLU 67.3% (5-shot) and GSM8K 88.6% per the Microsoft model card, as cited by a 2026 comparison guide. A math-specialist Phi-4-mini-reasoning variant also exists — [tinyweights.dev, 2026](https://tinyweights.dev/posts/best-small-language-models-2026/); MIT license confirmed in [dev.to license guide 2026](https://dev.to/jovan_chan_9500711396d4e6/open-source-llm-license-guide-2026-mit-apache-gpl-llama-205c)

**Liquid AI LFM2 / LFM2.5 (LFM2 July 2025; tech report Nov 2025; LFM2.5 later)**
- A hybrid architecture that mixes convolution and attention layers. Liquid claims about 2x CPU decode and prefill speed over Qwen3 (vendor) — [Liquid AI LFM2 blog, Jul 2025](https://www.liquid.ai/en/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models); [LFM2 tech report arXiv 2511.23404](https://arxiv.org/html/2511.23404v1)
- LFM2-700M beats Qwen3-0.6B on both decode and prefill in ExecuTorch and llama.cpp despite being 16% larger. Tested on a Samsung Galaxy S24 Ultra and an AMD Ryzen HX370 (vendor) — [Liquid AI blog](https://www.liquid.ai/en/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models). LFM2-700M reportedly outperforms Gemma 3 1B IT on quality — [MarkTechPost, Jul 2025](https://www.marktechpost.com/2025/07/13/liquid-ai-open-sources-lfm2-a-new-generation-of-edge-llms/)
- Later variants: LFM2-8B-A1B, an on-device MoE with about 1B active parameters — [Liquid AI](https://www.liquid.ai/blog/lfm2-8b-a1b-an-efficient-on-device-mixture-of-experts); LFM2.5-1.2B-Thinking, a reasoning model under 1 GB — [Liquid AI](https://www.liquid.ai/blog/lfm2-5-1-2b-thinking-on-device-reasoning-under-1gb); LFM2.5-2.6B, pitched for local agents — [HF blog](https://huggingface.co/blog/LiquidAI/lfm2-5-2-6b). LFM2.5 benchmark tables use llama.cpp Q4_0 with 1K prefill and 100 decode tokens — [Liquid AI LFM2.5](https://www.liquid.ai/en/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai)
- License: an "open license based on Apache 2.0". Commercial terms were not confirmed in these sources — [VentureBeat](https://venturebeat.com/ai/mit-offshoot-liquid-ai-releases-blueprint-for-enterprise-grade-small-model)

**Meta Llama 3.2 1B/3B (Sept-Oct 2024, older)**
- Llama community license: 700M MAU threshold. For Llama 4, multimodal rights are not granted to EU-based companies; sources differ on how broad the EU limit is — [dev.to license guide 2026](https://dev.to/jovan_chan_9500711396d4e6/open-source-llm-license-guide-2026-mit-apache-gpl-llama-205c); [developersdigest licenses](https://www.developersdigest.tech/local-models/licenses)
- From Qwen3 onward, Alibaba releases open-weight models under Apache 2.0, though licenses should be checked per repository — [braindetox license guide](https://braindetox.kr/en/posts/open_weight_model_license_guide.html)

**Gap to frontier in personal-assistant tasks (most decision-relevant number)**
- OpenJarvis (Stanford Scaling Intelligence + Lambda, arXiv May 2026): swapping Claude Opus 4.6 for Qwen3.5-9B in existing personal-AI stacks drops accuracy by 25-39 percentage points on tasks such as PinchBench and GAIA. With OpenJarvis's local-optimised spec the drop shrinks to 5.6-16.5 pp. The best local model overall (Qwen3.5-122B, a workstation-class model, not a phone one) reaches 80.3% average and comes within 3.2 pp of the best cloud model, matching or beating cloud on 4 of 8 benchmarks. Local execution cuts dollar cost about 800x and per-query energy about 1-20x — [arXiv 2605.17172](https://arxiv.org/abs/2605.17172); [Stanford PDF](https://scalingintelligence.stanford.edu/pubs/openjarvis.pdf); [project site](https://open-jarvis.github.io/OpenJarvis/). The method, "LLM-guided spec search", has frontier models propose scaffold edits offline; it closes 13-32 pp of the gap and makes no cloud calls at inference. Not yet independently replicated — [Pith review](https://pith.science/paper/2605.17172)

### Inferences
- **For a laptop (16-32 GB Apple silicon or a comparable PC):** Qwen3.5-9B (about 6 GB at 4-bit) is the most capable small dense model with confirmed sources. Gemma 4 E4B is the multimodal (image and audio) alternative with a clean Apache 2.0 license.
- **For a phone:** Gemma 4 E2B/E4B (via LiteRT-LM) or Qwen3.5-2B/4B. On iOS, Apple's built-in model costs no download and no RAM budget of its own, but it is closed and Apple-only.
- The license picture improved in 2026. Qwen (Apache 2.0), Gemma 4 (Apache 2.0) and Phi-4-mini (MIT) are all usable commercially without MAU caps. Llama's community license is the more restrictive one.
- The 80%+ hallucination rate means a personal companion should never answer factual "world knowledge" questions from model weights alone. Answers should be grounded in the user's own memory store or local documents, and the model should be allowed to say it doesn't know.
- The OpenJarvis result implies the scaffold (prompts, tool specs, retrieval) must be designed and tuned for the small local model. Porting a cloud-agent design unchanged loses about 25-39 pp.

### Gaps
- Exact Apple on-device benchmark tables (MMLU numbers) and exact Gemma 4 model-card numbers: the primary pages could not be fetched (arxiv.org and huggingface.co blocked in this session).
- No sourced 2025-2026 data was gathered for Mistral's small models (Ministral 3B/8B, Mistral Small 3.x) or HF SmolLM3, so they are not covered here.
- Llama 3.2-specific license text (vs Llama 4) was not confirmed; LFM license commercial thresholds were not confirmed.
- No independent (non-vendor) three-way comparison of LFM2 vs Qwen3/3.5 vs Gemma was found.

---

## 2. Runtimes and tooling: measured tokens/sec and RAM on named phones and laptops

### Takeaway
The working runtimes are llama.cpp (GGUF, cross-platform), MLX (Apple silicon; Ollama has reportedly moved onto it on Mac), ExecuTorch (Meta, mobile), LiteRT-LM (Google, Gemma on Android/iOS/desktop), MLC LLM, Apple's Foundation Models framework, Apple's new Core AI framework (WWDC26), and Android's ML Kit GenAI Prompt API (Gemini Nano). Measured numbers: a 3B 4-bit model decodes at about 20-30 tok/s on a 2025 iPhone before throttling, about 10-22 tok/s on 2024-2026 Android flagships, and 8B 4-bit models at about 50-90 tok/s on an M4 Max.

### Cited Findings
**Phones (named devices)**
- **iPhone Air (A19):** Llama 3.2 3B ran at 27.2 tok/s on llama.cpp with no throttling, and at 24.3 and 30.0 tok/s in throttled runs on MLX. This is one tester's single-device measurement — [pocket-ai-lab iPhone benchmarks](https://ananasdda.github.io/pocket-ai-lab/benchmarks/) (date not shown in search; iPhone Air shipped Sept 2025)
- Estimates (not controlled measurements): 3B Q4 at roughly 22-28 tok/s on iPhone 16 Pro and 18-25 on iPhone 15 Pro (llama.cpp + Metal); iPhone 16 Pro at about 15-23 tok/s before thermal throttling. Throttling can cut sustained speed by roughly a third or more — [pocketllm.app, 2026](https://pocketllm.app/blog/iphone-llm-speed-benchmarks/); [ertas.ai](https://www.ertas.ai/blog/llm-run-on-iphone-benchmarks)
- **Samsung S26 Ultra, Gemma 4 E4B on LiteRT-LM (Google's table):** about 18 tok/s decode on CPU and about 22 on GPU; peak memory about 3.3 GB on CPU and about 710 MB on GPU — [Google LiteRT-LM Gemma 4 page](https://developers.google.com/edge/litert-lm/models/gemma-4) (via search summary)
- **Gemma 3n on LiteRT-LM:** Google's table lists MacBook Pro M3 CPU at 27.6 tok/s decode and notes phone performance varies — [LiteRT-LM via search](https://developers.google.com/edge/litert-lm/models/gemma-4). Anecdotes: on a Galaxy Z Fold 4, E4B ran at about 7 tok/s decode on CPU and about 8 on GPU, with prefill rising from about 36 to 133 tok/s on GPU; on a Pixel 4a, E2B ran at 0.33 to about 5.7 tok/s — [Hacker News thread, May 2025](https://news.ycombinator.com/item?id=44044199)
- **OnePlus 12 (16 GB), ExecuTorch, Llama 3.2 1B (Oct 2024, older):** BF16 decode 19.2 tok/s, SpinQuant 4-bit 50.2 tok/s (2.6x), QAT+LoRA 45.8 tok/s (2.4x). Prefill went from 60.3 to 260.5 tok/s. RSS memory fell from 3,185 MiB to 1,921 MiB (-39.7%, SpinQuant) and file size from 2,358 MiB to 1,083 MiB — [ExecuTorch Llama example](https://android.googlesource.com/platform/external/executorch/+/HEAD/examples/models/llama); [ExecuTorch PR 6434](https://github.com/pytorch/executorch/pull/6434/files)
- **Llama 3.2 3B SpinQuant on OnePlus 12 (older):** decode 7.6 to 19.7 tok/s, prefill 21.2 to 89.7 tok/s, RSS 7,419 MiB to 3,726 MiB — [ExecuTorch PR 6359](https://github.com/pytorch/executorch/pull/6359/files). Meta's averages: 2-4x speedup, 56% smaller models, 41% less memory (OnePlus 12; similar on S24+ and S22) — [Meta AI blog, Oct 2024](https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/)
- **Samsung S24+ with Arm KleidiAI + ExecuTorch:** over 350 tok/s prefill for quantized Llama 3.2 1B — [Arm community blog](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/llm-inference-llama-quantized-models-executorch-kleidiai)
- Android via Termux+Ollama is estimated at about 10-20% behind iPhone, and a Snapdragon 8 Elite phone runs Llama 3.2 3B at about 10 tok/s (runtime not stated). Both are weakly sourced estimates — [PromptQuorum](https://www.promptquorum.com/local-llms/mobile-local-llms); [buildmvpfast](https://www.buildmvpfast.com/blog/on-device-llm-mobile-llama-ios-android-2026)

**Laptops/desktops (Apple silicon)**
- **M4 Max, 8B 4-bit:** MLX at 68-88 tok/s vs raw llama.cpp at 52-68 tok/s (blog measurements) — [yage.ai, 31 Mar 2026](https://yage.ai/share/mlx-apple-silicon-en-20260331.html) / [llms.blog](https://www.llms.blog/posts/local-llm-inference-on-apple-silicon-architecture-unified-memory-and-serving-benchmarks-for-mlx-llama-cpp-and-ollama) (the search summary did not say which of the two)
- **Qwen3-8B, academic comparison:** MLX-LM at 79.9 vs llama.cpp at 76.9 tok/s — from either [arXiv 2601.19139 (Jan 2026)](https://arxiv.org/pdf/2601.19139) or [arXiv 2511.05502 (Nov 2025, MLX vs MLC-LLM vs Ollama vs llama.cpp vs PyTorch MPS)](https://arxiv.org/pdf/2511.05502) (the search summary did not say which)
- **Benchmark pitfalls:** on a quiet machine MLX won decode by 2.4x; on a memory-starved machine llama.cpp won by 1.9x. The claim that "MLX uses half the RAM" turned out to be an artifact of llama.cpp pre-allocating a 32K KV cache — [hoodini/apple-silicon-arena on GitHub](https://github.com/hoodini/apple-silicon-arena). Several sources say MLX can lag llama.cpp on time-to-first-token, while one found MLX ahead from pp512 onward — [vettedconsumer](https://vettedconsumer.com/mlx-vs-llama-cpp-on-a-mac-which-is-faster-and-when-the-answer-flips/)
- A 2026 article's title says Ollama switched to MLX on Apple silicon and mentions the M5 Neural Accelerators — [yage.ai, 31 Mar 2026](https://yage.ai/share/mlx-apple-silicon-en-20260331.html) (title-level claim; details not read)

**Platform frameworks**
- **Apple Foundation Models framework (2025):** a Swift API to the on-device model with guided generation (structured output), constrained tool calling, and LoRA adapter fine-tuning — [LinkedIn summary of the Apple tech report](https://tr.linkedin.com/in/ozan-mut-04a8737b) (secondary); [Apple tech report](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025)
- **WWDC26 (June 2026):** the model abstraction layer is opened so "nearly any language model" can be used. Session 339 lists three new model options (Private Cloud Compute, Core AI and MLX) plus upcoming Anthropic and Google partner integrations. Other additions: a Python SDK, an `fm` command-line tool for on-device and PCC models, and Core AI, a new built-in framework for running your own local models. The framework is to become open source "later this summer" — a commitment, not a shipped release — [WWDC26 session 241](https://developer.apple.com/videos/play/wwdc2026/241/); [session 339](https://developer.apple.com/videos/play/wwdc2026/339/); [Platforms State of the Union, session 102](https://developer.apple.com/videos/play/wwdc2026/102/)
- **Android ML Kit GenAI Prompt API:** runs Gemini Nano on-device through AICore; alpha released Oct 2025. Supported devices are split into nano-v2 lists (e.g., Galaxy Z Fold7, Xiaomi models) and nano-v3 lists (e.g., Galaxy S26, Sony Xperia 1 VIII, vivo X300). Feature-specific APIs (summarize, proofread, rewrite, image description) have different device lists, including Pixel 9 and 10. Google suggests prototyping with Gemma 3n if you lack a supported device — [Android Developers Blog, Oct 2025](https://android-developers.googleblog.com/2025/10/ml-kit-genai-prompt-api-alpha-release.html); [ML Kit GenAI overview](https://developers.google.com/ml-kit/genai); [Gemini Nano page](https://developer.android.com/ml/gemini-nano)

### Inferences
- A phone can sustain a 2-4B model at conversational speed (15-30 tok/s, faster than reading speed), but throttling during long sessions is real. Design for short bursts: stream output, keep answers short, run heavy work while charging.
- For a cross-platform app, llama.cpp (GGUF) is the most portable single runtime. MLX is likely faster on Macs. LiteRT-LM is the natural path for Gemma on Android. Apple's framework (plus Core AI and MLX hooks since WWDC26) is the lowest-friction path on iOS.
- Both Apple's and Google's OS-level models are available only on recent flagship devices, so a product that relies on them excludes much of the installed base. Bundling your own GGUF/LiteRT model avoids that, at the cost of a 1.5-6 GB download and RAM pressure.

### Gaps
- No measured numbers were found for ONNX Runtime GenAI, LM Studio or MLC LLM on named 2025-2026 devices.
- No controlled energy-per-token or battery-drain measurements on named phones were found (only affiliate-blog claims; see section 7).
- Apple's 2024 report figures commonly quoted from memory (about 0.6 ms per prompt token time-to-first-token and about 30 tok/s on iPhone 15 Pro) could NOT be verified in this session and should not be used without checking [Apple ML Research](https://machinelearning.apple.com/).
- Throughput for Qwen3.5-9B / Gemma 4 E4B on named Windows/Snapdragon X laptops was not found.

---

## 3. Quantization: 4-bit formats and measured quality loss

### Takeaway
4-bit weight quantization (GGUF Q4_K_M, or SpinQuant/QAT for mobile) is the standard. On 8B models it costs little: about +0.15-0.18 perplexity and benchmark averages within about 0.3 points of 16-bit. Q5_K_M removes roughly two-thirds of that loss, and Q8_0 is near-lossless. Very small models (1-3B) are more sensitive, which is why Meta and Apple use quantization-aware training (Apple goes to 2-bit with QAT).

### Cited Findings
- **llama.cpp official perplexity README (Llama 3 8B scoreboard):** Q4_K_M adds about 0.175 perplexity with KL divergence about 0.031 (0.151-0.175 depending on importance matrix use); Q5_K_M adds about 0.057 (KLD about 0.011); Q8_0 adds about 0.0027 (KLD about 0.00136) — [llama.cpp perplexity README on GitHub](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)
- Q8_0 files are about 74% larger than Q4_K_M (7.96 GiB vs 4.58 GiB for Llama 3 8B). Q5_K_M's mean KLD is under a third of Q4_K_M's on Llama 3.1 8B — [agenticwire.news, 2026](https://www.agenticwire.news/article/q4-k-m-vs-q5-k-m) (note: some of its KLD figures are measured against Q8_0, not F16)
- **Llama 3.1 8B Instruct at Q4_K_M:** benchmark averages within about 0.3 points of 16-bit; perplexity 7.32 to 7.56 — [llamabuilds.ai, 2026](https://www.llamabuilds.ai/guides/llm-quantization-explained)
- KL divergence is the more sensitive metric because average benchmark scores can hide individual answer flips — [llama.cpp perplexity README](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md); [agenticwire.news](https://www.agenticwire.news/article/q4-k-m-vs-q5-k-m)
- The ExecuTorch team notes 1B/3B models are "sensitive to accuracy loss when regular PTQ quantization is applied", so they used SpinQuant 4-bit or QAT+LoRA — [ExecuTorch Llama example](https://android.googlesource.com/platform/external/executorch/+/HEAD/examples/models/llama) (Oct 2024, older)
- Apple's on-device model uses 2-bit quantization-aware training — [arXiv 2507.13575](https://arxiv.org/abs/2507.13575) (July 2025)
- Gemma 4 E4B ran at roughly 38 tok/s on an M2 Ultra via MLX int4 (unverified anecdote) — [VentureBeat/aggregators via search](https://venturebeat.com/technology/google-releases-gemma-4-under-apache-2-0-and-that-license-change-may-matter)

### Inferences
- Default to Q4_K_M (or the vendor's QAT checkpoint where one exists) for 4-9B models. Use Q5_K_M or Q6_K on laptops with spare RAM. Use vendor QAT/SpinQuant builds for 1-3B phone models rather than generic post-training quantization.
- KV-cache memory, not weights, becomes the binding constraint for long personal contexts on phones. Apple's KV-sharing (37.5% KV reduction) and Gemma's PLE show vendors working on this; a small team should cap context length and rely on retrieval instead.

### Gaps
- No AWQ/GPTQ vs GGUF head-to-head on the 2026 models (Qwen3.5, Gemma 4) was found.
- No quality-loss numbers specifically for Apple's 2-bit QAT relative to its 16-bit baseline were retrieved (the tech report table was not fetchable).

---

## 4. Long-term personal memory: local RAG, vector DBs, memory frameworks, consolidation, and benchmarks

### Takeaway
Memory is the core of a personal companion, and the field has converged on a few patterns: extract salient facts from conversations (Mem0); keep bi-temporal facts that are invalidated rather than deleted (Zep/Graphiti); let the agent self-manage tiered memory blocks with background "sleep-time" consolidation (Letta/MemGPT); and link atomic notes Zettelkasten-style (A-MEM). The benchmarks are LongMemEval (ICLR 2025) and LoCoMo. Vendor scores (90-95%) are not comparable to one another or to independent re-runs (as low as 32% for an open-source Mem0 setup). All of these can be run on top of SQLite plus a vector extension.

### Cited Findings
**Benchmarks**
- **LongMemEval (Wu et al., arXiv Oct 2024, ICLR 2025):** 500 manually created questions covering five abilities: information extraction, multi-session reasoning, temporal reasoning, knowledge updates and abstention. The S setting has about 115K tokens per question; the M setting has 500 sessions (about 1.5M tokens). Long-context LLMs drop 30-60% on LongMemEval-S. ChatGPT memory (GPT-4o) drops 37% and Coze drops 64% relative to reading the history directly, and commercial systems reached only 30-70% accuracy in a simpler setting — [arXiv 2410.10813](https://arxiv.org/abs/2410.10813); [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/d813d324dbf0598bbdc9c8e79740ed01-Paper-Conference.pdf)
- **Mem0 paper (Chhikara et al., arXiv Apr 2025):** on LoCoMo, Mem0 reports 91% lower p95 latency and over 90% token-cost savings vs full-context, and a 26% relative improvement over OpenAI memory on LLM-as-judge. Mem0g (graph variant) scores about 2% higher than base Mem0 — [arXiv 2504.19413](https://arxiv.org/abs/2504.19413). A secondary write-up gives about 26K to 1.8K tokens and 17.12 s to 1.44 s p95, and notes that the token saving counts retrieval only, not extraction overhead (unverified against the paper) — [dev.to/crabtalk](https://dev.to/crabtalk/mem0-what-three-memory-scopes-actually-cost-1kpc)
- **Vendor 2026 claims (not comparable to each other):** Mem0 reports about 92.5 on LoCoMo and 94.4 on LongMemEval, later revised to 91.6/93.4 at about 6,900 tokens per query (vendor) — [Mem0 blog: AI memory benchmarks 2026](https://mem0.ai/blog/ai-memory-benchmarks-in-2026); [Mem0 State of Agent Memory 2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026). Zep self-reports 90.2% on LongMemEval and 94.7% on LoCoMo; other sources cite 71.2% (GPT-4o judge) and 75.1% (third party). Letta: 74.0% on LoCoMo with GPT-4o-mini (filesystem setup), no published LongMemEval — [atlan.com, 2026](https://atlan.com/know/mem0-alternatives/)
- **Independent checks diverge:** one practitioner measured 32.4% on LongMemEval for the open-source Mem0 edition, against the managed platform's self-reported 93.4% (single unverified source) — [dev.to/everest_an, 2026](https://dev.to/everest_an/-i-benchmarked-ai-agent-memory-in-2026-and-the-numbers-tell-a-different-story-than-the-marketing-2ae4). In one independent LoCoMo table with a GPT-4o reader, A-Mem scored 44.43 F1 on single-hop questions vs Mem0's 39.12, the reverse of the vendor ordering — [atlan.com](https://atlan.com/know/mem0-alternatives/)
- **Evaluation itself is noisy:** a Sept 2026 preprint titled "Auditing Long-Term Memory Evaluation: Repeated Judging, Reader Variation, and Negative Controls" addresses judge and reader variance (title only) — [arXiv 2609.38021](https://arxiv.org/pdf/2609.38021)
- **Local reader evidence:** an Oct 2026 preprint is titled "MemStrata: 95% and 90.91% Source-Aware Accuracy on LongMemEval-500 and LoCoMo-1540 with a Local Qwen 3.8 27B Q4_K_M Reader". Its title claims near-ceiling memory scores with a locally run, 4-bit, 27B reader (title-level claim only; the 27B size is laptop/workstation-class) — [arXiv 2610.05343](https://arxiv.org/pdf/2610.05343)

**Memory architectures (how consolidation works)**
- **Letta (formerly MemGPT):** tiered memory that the agent edits itself through tool calls, stored as agent-state "memory blocks". "Sleep-time compute" (Apr 2025, [arXiv 2504.13171](https://arxiv.org/abs/2504.13171)) runs a second asynchronous agent that reorganises and consolidates the primary agent's core memory while it is idle, possibly using a larger model off the latency path. One article claims +18% accuracy and 2.5x lower cost per query (unverified) — [dataaspirant, 2026](https://dataaspirant.com/blog/mem0-vs-letta-vs-zep/); [Medium comparison of 5 memory systems](https://medium.com/@wasowski.jarek/i-compared-5-ai-agent-memory-systems-across-6-dimensions-none-wins-6a658335ed0a)
- **A-MEM (Feb 2025, NeurIPS 2025):** each interaction becomes an atomic note with a contextual description, keywords, tags, timestamp and a dense embedding. New notes link to related ones and can trigger updates to the context of existing notes ("memory evolution") — [atlan.com best memory frameworks 2026](https://atlan.com/know/best-ai-agent-memory-frameworks-2026/); [digitalapplied comparison](https://www.digitalapplied.com/blog/open-source-agent-memory-mem0-letta-zep-compared)
- **Zep / Graphiti (open-source engine):** a temporal knowledge graph. Each edge carries valid_at/invalid_at (world time) and created_at/expired_at (system time). Contradicted facts are invalidated, not deleted. Retrieval combines semantic similarity, full-text search and graph traversal. Trade-off: you must run a graph database — [Medium comparison](https://medium.com/@wasowski.jarek/i-compared-5-ai-agent-memory-systems-across-6-dimensions-none-wins-6a658335ed0a); [dev.to/plur9](https://dev.to/plur9/mem0-vs-letta-vs-zep-which-should-you-use-for-agent-memory-1n8m)
- **Integration cost:** "Mem0 asks least of your application, Letta asks you to adopt its runtime, and Zep gives you auditable history at the cost of running a graph database" — [dev.to/plur9](https://dev.to/plur9/mem0-vs-letta-vs-zep-which-should-you-use-for-agent-memory-1n8m)

**Local vector stores**
- In a self-published benchmark by the competing SQLRite project (desktop, 5K-record filtered cosine search), sqlite-vec exact search reached about 3,163 QPS at recall 1.0 vs LanceDB IVF_FLAT at about 1,331 QPS and recall 0.951 — [SQLRite performance docs](https://docs.rs/crate/sqlrite/1.0.2/source/docs/performance.md)
- SQLite AI's `sqlite-vector` extension (a different project from Alex Garcia's sqlite-vec) claims to run on iOS, Android, Windows, Linux and macOS using about 30 MB of memory by default, depending on configuration — [github.com/sqliteai/sqlite-vector](https://github.com/sqliteai/sqlite-vector)
- LanceDB is in-process but best suited to multimodal and object-storage-backed workloads, a weaker fit for phones — [Actian embedded vector DB comparison, 2026](https://www.actian.com/blog/developer/comparing-embedded-vector-databases/)
- MoRAGBench (Paderborn thesis, Mar 2026) evaluates mobile RAG pipelines on retrieval accuracy, latency, throughput and memory, and concludes approximate search methods matter on mobile — [Paderborn thesis PDF](https://digital.ub.uni-paderborn.de/hs/download/pdf/8256643)
- A practitioner reports sqlite-vec running on Android smart glasses (Snapdragon AR1) in brute-force, quantized and HNSW modes from 1K to 1M vectors (no numbers in snippet) — [LinkedIn post](https://co.linkedin.com/in/fabianvarelabonett)

### Inferences
- A practical local memory stack is a single encrypted SQLite file holding: (a) raw episodic logs; (b) extracted facts with bi-temporal validity columns (the Graphiti idea without a graph-DB server); (c) sqlite-vec embeddings plus FTS5 full-text for hybrid retrieval; and (d) a small "core profile" block always in the prompt (the Letta idea).
- Consolidation should be a "sleep-time" job that runs while the device is charging and idle. It would merge duplicates, invalidate contradicted facts, summarise old episodes and refresh the profile. This hides the latency and battery cost of using the small model for extraction.
- Benchmark claims of 90%+ come from vendors using GPT-4-class readers and judges. With a 3-9B local reader, expect substantially lower accuracy, and run your own LongMemEval-S / LoCoMo evaluation with a fixed judge. The MemStrata title suggests a 27B local reader can get close on a laptop or workstation, but not on a phone.
- LongMemEval's five abilities (extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention) are a good acceptance checklist for a personal companion. Knowledge updates and temporal reasoning argue for bi-temporal facts.

### Gaps
- No measured LongMemEval/LoCoMo results using a phone-class (1-4B) local model as both extractor and reader were found.
- No Chroma on-device data were found; no independent phone benchmark of sqlite-vec vs LanceDB vs ObjectBox was found.
- The sleep-time compute "+18% / 2.5x" claim and the Mem0 secondary latency figures were not verified against the primary papers (arxiv.org unreachable).

---

## 5. On-device personalization (LoRA) and on-device speech (ASR/TTS) for a voice companion

### Takeaway
LoRA fine-tuning of a 3B model now fits in under 2 GB of peak memory on a 2025 flagship phone, but the published phone results are feasibility studies, and wall-clock and energy costs are mostly projections (about 1-2 hours per 1,000-example epoch). On a Mac, MLX LoRA/QLoRA is routine. For voice, Moonshine (streaming ASR, 2024-2026) and Kokoro-82M (TTS, about 80 MB at INT8) make a fully local voice loop feasible on recent iPhones and Macs.

### Cited Findings
**LoRA on phones/laptops**
- On a 2025 flagship phone with 12 GB RAM, combining INT4 quantization, checkpointing, a softmax approximation and logits masking brought Llama-3.2-3B LoRA peak memory at 4K tokens to 1.59 GB, vs 40.41 GB for an FP32 LoRA baseline; a longer-context setting fit in 6.95 GB. The authors frame this as feasibility evidence, not a reproducible cross-device benchmark — [arXiv 2606.19528, June 2026](https://arxiv.org/html/2606.19528v1)
- MeBP (memory-efficient backprop) keeps fine-tuning of all tested LLMs under 1 GB, a practical target for background training on a phone — [arXiv 2510.03425, Oct 2025](https://arxiv.org/html/2510.03425)
- MobileFineTuner names RAM and energy as the two main barriers and adds energy-aware scheduling — [arXiv 2512.08211, Dec 2025](https://arxiv.org/html/2512.08211)
- AQLoRA projects about 1.2-1.3 hours per 1,000-example epoch on a phone, scaled by DRAM bandwidth (an upper-bound projection, not a measurement). A separate report gives about 1 h 40 min per epoch for a 1.7B model on a Qualcomm-class phone — [arXiv 2608.23816, Aug 2026](https://arxiv.org/pdf/2608.23816)
- LoRA can cut Llama-3.2-3B peak memory by up to 26x; evidence for zeroth-order and backprop-free methods is thin — [MDPI systematic review, 2025](https://www.mdpi.com/2674-0729/5/4/43)
- Apple's mlx-lm supports LoRA and QLoRA (QLoRA is automatic with a quantized checkpoint). A secondhand report says Apple's LORA.md documents about 250 tok/s training throughput on an M1 Max with 32 GB (WikiSQL). About 8-9 GB of unified memory is an estimated minimum for 7-8B 4-bit LoRA — [digitalapplied MLX guide 2026](https://www.digitalapplied.com/blog/apple-mlx-framework-local-ai-developers-2026-guide) (estimates, unverified)
- Apple's Foundation Models framework supports training custom LoRA adapters for its on-device model — [Apple tech report 2025](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025); [LinkedIn summary](https://tr.linkedin.com/in/ozan-mut-04a8737b) (secondary)

**Speech recognition**
- Moonshine (Useful Sensors, arXiv Oct 2024) was benchmarked against Whisper tiny.en/base.en with up to 3x lower latency (scaled to audio length) at roughly similar WER for the tiny size. Sources disagree on the exact tiny WERs (about 12.7-12.8%) — [arXiv 2410.15608](https://arxiv.org/abs/2410.15608); [MarkTechPost, Oct 2024](https://www.marktechpost.com/2024/10/23/moonshine-a-fast-accurate-and-lightweight-speech-to-text-models-for-transcription-and-voice-command-processing-on-edge-devices/)
- Moonshine Voice streaming models (about Feb 2026, vendor table):
  - Moonshine Medium Streaming: 6.65% WER, 245M params, 107 ms latency on a MacBook Pro, vs Whisper Large v3 at 7.44% WER, 1.5B params, 11,286 ms
  - Moonshine Tiny Streaming: 12.00% WER at 34 ms, vs Whisper Tiny at 12.81% and 277 ms
  - Caveat: latency is measured from the moment a VAD segment closes, which favours streaming designs — [moonshine GitHub README (fork)](https://github.com/grisuno/moonshine); [GIGAZINE, 25 Feb 2026](https://gigazine.net/gsc_news/en/20260225-moonshine-voice/)
- An offline Android port reports Moonshine v2 as 5-40x faster than Whisper at 6.65% WER — [Jackfood2/MoonshineSpeechToText on GitHub](https://github.com/Jackfood2/MoonshineSpeechToText). "Flavors of Moonshine" provides tiny specialised non-English ASR models for edge devices — [arXiv 2509.02523, Sept 2025](https://arxiv.org/pdf/2509.02523)

**Text-to-speech**
- Kokoro-82M Core ML port: 30 s of speech in 379 ms on a Mac Studio. On iPhone 15 Pro Max, 3 s clips take about 0.7-0.9 s and 7 s clips about 1.5 s. The iPhone Neural Engine compiler rejects the full-ANE plan that M-series Macs run, so iPhone uses a staged setup — [HF mattmireles/kokoro-coreml](https://huggingface.co/mattmireles/kokoro-coreml)
- iPhone 13 Pro FP32: about 0.6 real-time factor after weights load; cold start is slower — [HF hexgrad/Kokoro-82M discussion #18](https://huggingface.co/hexgrad/Kokoro-82M/discussions/18)
- About 80 MB at INT8; runs on the Neural Engine via Core ML — [soniqo.audio Kokoro guide](https://soniqo.audio/guides/kokoro)
- An older third-party benchmark on Galaxy S10 and iPad Pro found Kokoro slower than real time (exact source uncertain within search results) — [voiceping offline TTS eval](https://voiceping-landingpage-prod.pages.dev/en/blog/research-offline-tts-eval/); [React Native ExecuTorch inference-time docs](https://docs.swmansion.com/react-native-executorch/docs/0.3.x/benchmarks/inference-time)

### Inferences
- Phone-side LoRA works but is not yet a product feature: hours per epoch, thermal limits and no measured energy data. A better near-term design is to run personalization LoRA on the user's own laptop or desktop overnight (MLX/llama.cpp tooling) and sync the adapter (tens of MB) to the phone over the local network. On iOS, the Foundation Models adapter path is an alternative.
- Retrieval-based personalization (memory plus a profile block) should come first because it is immediate, inspectable and reversible. LoRA is better suited to style and tone than to facts, which matters given the memorization findings in section 6.
- A full local voice loop (Moonshine for streaming ASR, a 2-4B LLM, Kokoro TTS) fits in roughly 3-4 GB on a recent iPhone or flagship Android. Expect sub-second ASR and TTS on 2023+ Apple devices; older Android devices may struggle.

### Gaps
- No measured battery or energy cost (Wh or % per epoch) for on-device LoRA on a named phone.
- No measured end-to-end voice-loop latency (speech in to speech out) for a fully local stack on a named phone.
- Whisper-turbo / whisper.cpp numbers on 2025-2026 phones and Android Kokoro numbers were not gathered.

---

## 6. Privacy architectures and risks

### Takeaway
"Never shares data" is best achieved by a local-first design: data and models live on the device, encrypted at rest by hardware-backed OS protections, with no server in the loop. Apple's Private Cloud Compute is the reference design if any cloud fallback is ever allowed (stateless, attested, transparency-logged, non-targetable). The main risks left for a local assistant are device theft or unlock, memory poisoning and prompt injection through ingested personal content (an OWASP agentic top-10 item since Dec 2025), memorization in personal LoRA adapters, and vendor or acquisition risk (Limitless/Rewind).

### Cited Findings
**Apple Private Cloud Compute (June 2024 design; later expansions)**
- Core requirements are stateless computation on personal data (used only to fulfil the request), enforceable guarantees with no reliance on external components, no privileged runtime access, non-targetability, and verifiable transparency — [Apple Security Research: PCC, June 2024](https://security.apple.com/blog/private-cloud-compute/)
- Apple publishes measurements of all production PCC software in an append-only, cryptographically tamper-proof transparency log. The device wraps its request key only to the public keys of PCC nodes whose attested measurements match a logged release — [Apple Security Research: PCC](https://security.apple.com/blog/private-cloud-compute/)
- Apple released a Virtual Research Environment with a virtual Secure Enclave Processor, a Security Guide and some source code, and extended its bug bounty to PCC — [Apple: Security research on PCC, Oct 2024](https://security.apple.com/blog/pcc-security-research/); [SecurityWeek](https://www.securityweek.com/apple-opens-private-cloud-compute-for-public-security-inspection/)
- "Expanding Private Cloud Compute" describes running PCC on Google Cloud with "core PCC requirements remain exactly the same" (date not shown in search) — [Apple Security Research: Expanding PCC](https://security.apple.com/blog/expanding-pcc)
- An independent arXiv study (May 2026) measured PCC behaviour against Apple's public claims and open-sourced its framework. It notes the claims still need outside verification (findings not read in detail) — [arXiv 2605.24239](https://arxiv.org/abs/2605.24239)

**Encryption at rest / device theft**
- On Apple devices the Secure Enclave generates and stores the keys used to encrypt data at rest; Apple silicon underpins Data Protection, secure boot and biometrics. The guide documents per-file Data Protection classes and keybags tied to the passcode — [Apple Platform Security Guide (PDF)](https://help.apple.com/pdf/security/en_US/apple-platform-security-guide.pdf)

**Local-first software (2019, older but foundational)**
- Ink & Switch (Kleppmann, Wiggins, van Hardenberg, McGranaghan, 2019) set out seven ideals: fast with no spinners, multi-device, works offline, collaboration, longevity, privacy, and user ownership. The essay names CRDTs as a foundational technology for this kind of software — [Ink & Switch: Local-first software](https://www.inkandswitch.com/local-first/). A 2024 retrospective argues that software which stops working when its vendor shuts down servers is not local-first — [PowerSync blog, Jul 2024](https://powersync.com/blog/local-first-software-origins-and-evolution)

**Memory poisoning and prompt injection via personal data**
- MINJA (arXiv Mar 2025) injects malicious records into an agent's memory bank only through ordinary queries and observed outputs, with no direct memory access. It reports over 95% injection success and 70% attack success under idealized conditions — [alphaxiv 2503.03704](https://www.alphaxiv.org/abs/2503.03704); [vectorize.io explainer](https://vectorize.io/articles/ai-memory-poisoning)
- AgentPoison is a backdoor-style attack in which triggers cause poisoned demonstrations to be retrieved from memory. It assumes privileged write access to the memory bank — [arXiv 2601.05504, Jan 2026](https://arxiv.org/abs/2601.05504); [promptfoo LM security DB](https://www.promptfoo.dev/lm-security-db/vuln/agent-persistent-memory-poisoning-7e5fb607/)
- Poisoned memory persists across session restarts and context resets. In an EHR-agent scenario it led to lookups on the wrong patient and bypassed LLM-as-a-judge memory sanitization — [promptfoo](https://www.promptfoo.dev/lm-security-db/vuln/agent-persistent-memory-poisoning-7e5fb607/)
- A systematic study (June 2026) found that aggressive memory write and retrieval policies expand the attack surface and that existing prompt-injection defenses give incomplete coverage — [arXiv 2606.04329](https://arxiv.org/pdf/2606.04329). MemoryGraft shows persistent, trigger-free poisoning through retrieval of poisoned "experience" — [arXiv 2512.16962, Dec 2025](https://arxiv.org/pdf/2512.16962)
- OWASP lists "ASI06 Memory and Context Poisoning" in its Top 10 for Agentic Applications (Dec 2025) — [ssdnodes explainer](https://www.ssdnodes.com/learn/agent-memory-poisoning); [vectorize.io](https://vectorize.io/articles/ai-memory-poisoning) (secondary; OWASP page not read directly)

**Personal-data leakage from fine-tuned adapters**
- LoRA fine-tuning gave near-zero plagiarism-style memorization while keeping task performance, much less than full fine-tuning — [arXiv 2506.20856, June 2025](https://arxiv.org/abs/2506.20856). In federated settings, LoRA cut memorization by up to 10x — [arXiv 2502.05087](https://arxiv.org/html/2502.05087v2)
- Higher LoRA ranks raise memorization, with diminishing returns (medical PHEE data; membership-inference and prefix-prompting tests) — [arXiv 2507.21009, July 2025](https://arxiv.org/html/2507.21009v1). A 2026 paper measures LoRA adapter memorization capacity "in bits" to help judge the risk of sharing adapters trained on sensitive data — [arXiv 2607.21351](https://arxiv.org/pdf/2607.21351)

**Vendor/acquisition risk**
- Meta acquired Limitless (formerly Rewind), the AI pendant and memory app maker, in Dec 2025 — [CNBC, 5 Dec 2025](https://www.cnbc.com/2025/12/05/meta-limitless-ai-wearable.html). The Rewind Mac app was shut down (reported 19 Dec 2025), pendant sales stopped, and service was discontinued in the EU and UK; commentators say the local-first privacy promise did not survive the transition — [hedy.ai](https://www.hedy.ai/post/meta-acquires-limitless-ai-privacy/); [p4sc4l substack case study](https://p4sc4l.substack.com/p/limitlessai-a-case-study-in-how-small)

### Inferences
- Recommended baseline:
  - All inference and memory on-device, with no telemetry containing content.
  - The memory database encrypted with a key held in the Secure Enclave or Android StrongBox/Keystore, available only after first unlock (or Apple's "complete" protection class).
  - Optional end-to-end-encrypted device-to-device sync (CRDT-based), so a laptop can serve as the "heavy" node.
- Treat everything ingested (emails, web pages, messages from others) as untrusted:
  - Tag every memory write with provenance (user-stated vs. inferred vs. from third-party content).
  - Never let third-party content write "instructions" or profile facts without user confirmation.
  - Gate tool actions (send, pay, delete) behind explicit user approval. This directly addresses MINJA, AgentPoison and MemoryGraft-style persistence.
- Keep personalization adapters low-rank and treat them as sensitive data (encrypted, never uploaded).
- Making the company's promise durable requires open formats and export, and ideally open-source client code, so users keep their data if the company is acquired. The Limitless case shows the failure mode.
- If a cloud fallback is ever offered, a PCC-style design (attested, stateless, transparency-logged) is the bar. It should be an explicit opt-in per request, because any cloud path contradicts a strict "never shares" promise.

### Gaps
- No sources were found on model-inversion or extraction attacks specifically against on-device personal models or memory embeddings (e.g., reconstructing text from stored embeddings).
- Android-side equivalents (StrongBox/Keystore, Private Compute Core) were not researched.
- SQLCipher or other database-level encryption performance on mobile was not sourced.
- The date and details of Apple's "Expanding PCC" post and the findings of arXiv 2605.24239 were not read.

---

## 7. Existing products/projects and user complaints

### Takeaway
Shipping "personal AI on device" products fall into three groups: OS-level assistants (Apple Intelligence, Gemini Nano on Android), hobbyist runners for local models (PocketPal, MLC Chat, Layla, Jan, Ollama, LM Studio), and memory/"lifelogging" tools (Rewind/Limitless, now absorbed by Meta; Screenpipe as a local-first alternative). Recurring complaints are factual errors from small models (Apple paused news notification summaries in Jan 2025), heat and battery drain in long sessions, background-execution limits, and RAM limits on model quality. Open research frameworks such as Stanford's OpenJarvis (2026) are the closest thing to a reference architecture for a local personal AI.

### Cited Findings
- **Apple Intelligence quality failure:** Apple paused AI notification summaries for news and entertainment apps in the iOS 18.3 betas (mid-Jan 2025) after false headlines. A BBC alert was rewritten to claim Luigi Mangione had shot himself, and there were false claims about Rafael Nadal. Apple added italics, an "in development" warning and per-app off switches. The feature condenses notifications using the on-device model, but the sources do not attribute the errors specifically to it — [TechCrunch, 16 Jan 2025](https://techcrunch.com/2025/01/16/apple-pauses-ai-notification-summaries-for-news-after-generating-false-alerts)
- **PocketPal (open-source local LLM app, iOS/Android):**
  - A Play Store user says the Local API server stops responding when the phone locks — [Google Play listing](https://play.google.com/store/apps/details?id=com.pocketpalai)
  - Reviewers report the phone heats up with denser models, battery drain and heat during long chats, and that phone hardware limits model size and quality — [meetaitools review, 2026](https://meetaitools.com/pocketpal-ai-review/); [aibirdview, 2026](https://aibirdview.com/tools/pocketpal-ai)
  - Typical RAM guidance: about 6 GB+ for smaller models and 8 GB+ for "the good stuff" — [yaps.ai, 2026](https://www.yaps.ai/blog/best-offline-ai-apps-2026)
  - Battery claims (e.g., about 18%/hour for MLC Chat vs 14% for PocketPal on a Galaxy S24) come from low-authority affiliate sites and are unverified — [meetaitools PocketPal vs MLC Chat](https://meetaitools.com/pocketpal-ai-vs-mlc-chat-android/)
- **Companion-specific apps:** a 2026 roundup rates PocketPal 3.6/5 as a companion ("real on-device inference, no companion layer on top") and favours Layla as the stronger offline companion — [aicompanionguides, 2026](https://aicompanionguides.com/blog/best-offline-ai-companions-2026/). Jan is listed as the best desktop chat experience in an offline roundup — [meetaitools rural offline roundup](https://meetaitools.com/ai-tools-that-work-in-rural-areas-offline-free/)
- **Lifelogging/memory:** Limitless/Rewind was acquired by Meta (Dec 2025; see section 6). Screenpipe positions itself as a source-available, local-first Rewind alternative with local AI search and audio transcription on Windows and Linux (vendor page) — [Screenpipe blog, 2026](https://screenpipe.com/blog/rewind-ai-alternative-2026)
- **OS platforms:** Gemini Nano via the ML Kit Prompt API (alpha Oct 2025) on a limited set of 2025-2026 flagships — [Android Developers Blog](https://android-developers.googleblog.com/2025/10/ml-kit-genai-prompt-api-alpha-release.html). Apple's Foundation Models framework opened to third-party and local models at WWDC26 — [WWDC26 session 339](https://developer.apple.com/videos/play/wwdc2026/339/)
- **Research reference:** OpenJarvis (open source, Stanford/Lambda, May 2026) is a framework for personal AI on personal devices that optimizes local-only specs (see section 1) — [OpenJarvis site](https://open-jarvis.github.io/OpenJarvis/); [arXiv 2605.17172](https://arxiv.org/abs/2605.17172)

### Inferences
- The market gap is a companion that combines durable, inspectable personal memory, a voice loop and local-only operation. Current local apps are model runners without a memory or companion layer. OS assistants are local-first but either cloud-escalating (Apple PCC, ChatGPT/Gemini integrations) or limited to specific devices.
- Product risks to design against: heat and battery in long chats (keep answers short and push heavy work to charging time), iOS/Android background limits (consolidation must be resumable and scheduled), and confident-but-wrong summaries (always show sources from memory and let the user correct facts in one tap).

### Gaps
- No rigorous, non-affiliate user-review study of local LLM apps was found.
- Khoj, Reor, Ollama and LM Studio user complaints and current status were not covered by search results.
- Whether Apple re-enabled news summaries, and any later Apple Intelligence quality data, was not checked.

---

## 8. Hard limits: where small local models still fail vs frontier cloud models

### Takeaway
Small local models (1-9B) close much of the gap on short-form reasoning and math benchmarks. They remain clearly weaker on factual recall and hallucination (80%+ on hard factual tests), on agentic multi-step tool use (25-39 pp behind frontier in personal-assistant tasks unless the scaffold is tuned, and about two-thirds of frontier on BFCL at best), and on real long-context use (effective context often far shorter than advertised; 1B/3B models nearly fail at 128K).

### Cited Findings
- **Agentic/personal tasks:** swapping Claude Opus 4.6 for Qwen3.5-9B in existing stacks costs 25-39 pp on PinchBench and GAIA. A local-optimized spec reduces the gap to 5.6-16.5 pp — [OpenJarvis, arXiv 2605.17172, May 2026](https://arxiv.org/abs/2605.17172)
- **Tool calling (BFCL v4, secondary source):**
  - A purpose-built 1B model (xLAM-2-1b-fc-r, 30.44) scores about 3x a general 1B model (Llama-3.2-1B-Instruct, 10.82)
  - A Liquid AI 8B model rose from 25.52 to 48.50 across two generations
  - The best small model reaches about two-thirds of frontier tool-calling performance
  - Small-model "non-hallucination" (not inventing functions) improved from 7.46 to 63.47
  - All figures from [Medium: 5 tiny LMs for tool calling](https://medium.com/@minhle_0210/5-tiny-language-models-for-tool-calling-part-3-ebcda32c2518); verify against the BFCL leaderboard
- **Hallucination/factuality:** Qwen3.5-4B and 9B hallucinate 80-82% of the time on AA-Omniscience — [Artificial Analysis, 2026](https://artificialanalysis.ai/articles/qwen3-5-small-models). Apple's ~3B on-device model and server model trail GPT-4o in human evaluations — [The Decoder, 2025](https://the-decoder.com/apples-new-ai-benchmarks-show-its-models-still-lag-behind-leaders-like-openai-and-google/). The on-device summarization errors that forced Apple's Jan 2025 pause are a real-world example — [TechCrunch](https://techcrunch.com/2025/01/16/apple-pauses-ai-notification-summaries-for-news-after-generating-false-alerts)
- **Long context:**
  - RULER (Apr 2024, older): only about half of the tested models held up at 32K, though nearly all claimed 32K or more; larger models degrade less — [arXiv 2404.06654](https://arxiv.org/abs/2404.06654)
  - NoLiMa (Feb 2025), which removes literal overlap between question and "needle": sharp drops already at 2K-8K, and at 32K, 11 models fall below 50% of their short-context baselines — [arXiv 2502.05167](https://arxiv.org/html/2502.05167v3)
  - xGen-small report (May 2025, vendor-produced comparison): 4B-class RULER averages about 72-79 across 4K-128K (Llama-3.2-3B 78.98, xGen-small-4B 76.39) and 7-9B about 78-86; Llama-3.1-8B loses more than 10 points from 64K to 128K — [arXiv 2505.06496](https://arxiv.org/pdf/2505.06496)
  - Official Llama-3 1B and 3B models "nearly fail entirely at 128k" (attribution to this specific paper is from the search summary) — [arXiv 2412.18860](https://arxiv.org/pdf/2412.18860)
- **Memory over long histories:** even GPT-4o-based commercial memory (ChatGPT) loses 37% vs direct reading on LongMemEval, and long-context LLMs lose 30-60% on LongMemEval-S — [arXiv 2410.10813](https://arxiv.org/abs/2410.10813). Small local readers would be expected to do worse (see the gaps for this section).
- **Code/small-size limits:** Qwen3.5-0.8B scores 67% zero-shot on code fixing, falling to 33% with one example (community test) — [mean.ceo](https://blog.mean.ceo/qwen-3-5-small-model-series-release/)
- **Where small models are now strong (for balance):** on GPQA Diamond, Qwen3.5-9B scores 81.7 vs gpt-oss-120b's 80.1 — [VentureBeat](https://venturebeat.com/technology/alibabas-small-open-source-qwen3-5-9b-beats-openais-gpt-oss-120b-and-can-run); Gemma 4 E4B is reported at AIME 2026 42.5% and MMLU-Pro 69.4% (unconfirmed) — [aurigait](https://aurigait.com/blog/gemma-4-features-benchmarks-guide/)

### Inferences
- Design rules for the companion:
  - Never rely on the model's parametric knowledge for facts about the world or the user; answer from retrieved memory or documents with citations, and train or prompt the model to abstain.
  - Keep the working context short (4-16K) and rely on structured memory retrieval rather than a 128K raw transcript.
  - Limit autonomous tool use to a small, well-typed tool set with confirmation steps. Consider a function-calling-specialised model or adapter, since purpose-built 1B models score about 3x a general 1B on BFCL.
  - Tune the scaffold for the local model rather than porting a cloud-agent design.
- **Hard limit for "never shares":** open-ended world-knowledge questions, current events (no internet), complex multi-step agentic tasks and long-document reasoning will be noticeably worse than ChatGPT/Claude. The honest product position is "a private memory-centric companion", not "a private ChatGPT replacement".
- **Two-tier local architecture:** a phone-class model (2-4B) handles voice, quick replies and memory lookups. A laptop/desktop-class model (9B on a 16-32 GB laptop, or a 27B+ workstation model) on the user's own hardware handles consolidation, hard reasoning and personalization training, synced end-to-end encrypted. This keeps data within the user's own devices while recovering much of the quality gap (OpenJarvis's best local results used large local models).

### Gaps
- No small-model SimpleQA or tau-bench numbers vs frontier were found.
- No NoLiMa effective-context numbers specifically for 2026 small models (Qwen3.5, Gemma 4) were found.
- The OpenJarvis results are a single preprint, not independently replicated.
