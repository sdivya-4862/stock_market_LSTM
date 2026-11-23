# Data Efficiency: Closing the Human–LLM Data Gap (child-inspired learning, small models, personal adaptation)

> **Method note for the report writer (read first).** Research date: 2026-11-23. In this session, page fetching was blocked for `aclanthology.org`, `arxiv.org` and `technologyreview.com` (DNS refused by the sandbox network policy). So **every finding below comes from search-engine result snippets and abstracts, not from reading the full papers.** Numbers are given as the snippets reported them. Where a snippet looked garbled or two snippets disagreed, I say so. A few well-known facts I could not re-check this session are marked **[not re-verified this session]**. They are included only because they are standard, widely cited figures. Treat them as needing a check before publication.
>
> **Peer-review legend:** **[PR]** = peer-reviewed journal or main conference. **[WS]** = workshop or shared-task proceedings (organizer-written findings papers are lightly reviewed). **[TR]** = industry technical report or blog (not peer-reviewed). **[PP]** = arXiv preprint (not peer-reviewed as far as I could tell). arXiv IDs encode the month the paper was posted (YYMM). For example, 2606.xxxxx = June 2026.

---

## 1. How big is the gap? Words children hear vs. LLM training tokens, and what innate structure and evolution contribute

### Takeaway
Children reach adult-like grammar from tens of millions of words, and fewer than ~100M words by early adolescence. Frontier LLMs train on ~10^13 tokens, which is roughly 4–5 orders of magnitude more. The size of the gap is well established. What causes it is disputed. Explanations include an innate or evolved prior, grounded multimodal experience, social interaction, and a better learning algorithm. A 2024 EMNLP study suggests the difference lies in the child's *learning algorithm*, not in child-directed speech being special data.

### Cited Findings
**Child input estimates**
- Hart & Risley (1995, book *Meaningful Differences*; **older**) produced the famous "30-million-word gap" by age 3 between professional-class and welfare families. A LENA-based replication sample covered 329 English-speaking families in Denver with children aged 2–48 months. — [Words Alive summary (2019)](https://www.wordsalive.org/blog/2019/8/12/the-30-million-word-gap); [NPR/WLRN, 2018-06-01, "Let's Stop Talking About The '30 Million Word Gap'"](https://www.wlrn.org/npr-breaking-news/2018-06-01/lets-stop-talking-about-the-30-million-word-gap)
- Gilkerson et al.'s LENA automated recordings found a much smaller SES gap: about **4 million words by age 4**, not 30M by age 3. A gap close to Hart & Risley's appeared only when comparing the most-talkative 2% of families with the least-talkative 2%. — [NPR/WLRN, 2018-06-01](https://www.wlrn.org/npr-breaking-news/2018-06-01/lets-stop-talking-about-the-30-million-word-gap)
- The LENA "Power of Talk" technical report (Gilkerson & Richards; **[TR], older, ~2009**) lists average adult words per 14-hour day of roughly **30,142 / 17,514 / 8,624** across three groups (the snippet did not show which group is which). It puts Hart & Risley's professional-class parents at an estimated **30,137** words/day. — [LENA Power of Talk PDF](https://www.lena.org/wp-content/uploads/2016/07/LTR-01-2_PowerOfTalk.pdf)
- Romeo et al. (2018, *Psychological Science*, **[PR], older**) argue that *conversational turns* (back-and-forth exchanges), not raw word counts, are associated with language-related brain function. — [PMC5945324, "Beyond the 30-Million-Word Gap"](https://pmc.ncbi.nlm.nih.gov/articles/PMC5945324/)
- An Australian LENA study (BMC Pediatrics, 2020, **[PR], older**) found adult word counts at 6 months were almost the same as Gilkerson's 2008 US data (1,041 vs 1,052; the snippet did not state the unit, likely per hour). The authors draw implications for the 30M-gap hypothesis. — [BMC Pediatrics 2020](https://link.springer.com/article/10.1186/s12887-020-1946-0)
- Measurement caveats: LENA's speech identification is about 82% accurate for adult speech and 76% for child speech. Hart & Risley and Gilkerson counted only speech *directed to* the child, not overheard speech. — [Words Alive](https://www.wordsalive.org/blog/2019/8/12/the-30-million-word-gap); [NPR/WLRN 2018](https://www.wlrn.org/npr-breaking-news/2018-06-01/lets-stop-talking-about-the-30-million-word-gap)

**The gap framing**
- Michael C. Frank, "Bridging the data gap between children and large language models," *Trends in Cognitive Sciences* 27(11):990–991, Nov 2023, DOI 10.1016/j.tics.2023.08.007 (**[PR] Forum/opinion piece; older**). He argues that LLMs get far more language input than children, and that the gap is even larger compared with literate adults. Snippets disagree on the size: one says **"three to five orders of magnitude,"** another **"four or five."** Candidate explanations: (a) innate knowledge or evolutionary history, (b) pre-existing conceptual knowledge, (c) multimodal grounding, (d) social and interactive input. He calls for synchronized model–child evaluations that rule out memorization, and warns that LLMs are tested on complex reasoning while children are tested on simple, scaffolded tasks. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1364661323002036); [PDF mirror](https://sites.socsci.uci.edu/~lpearl/colareadinggroup/readings/Frank2023_ChildrenLLMs.pdf); [PDF mirror 2](https://danielwharris.com/teaching/268/readings/Frank.pdf)
- Frank gave an updated talk on the same theme at the Simons Institute workshop "LLMs, Cognitive Science, Linguistics and Neuroscience" (Feb 2025). — [Simons slides, 2025-02](https://simons.berkeley.edu/sites/default/files/2025-02/Michael%20Frank%20LLM25-1%20Slides.pdf)
- BabyLM organizers frame it this way: children acquire language from **<100M words**, while LLMs "typically require 3 or 4 orders of magnitude more data." — [BabyLM 2023 findings, arXiv 2504.08165](https://arxiv.org/html/2504.08165v1)
- LLM side: Meta stated that Llama 3 was pretrained on **"over 15T tokens"** (April 2024) **[not re-verified this session]**. The search in this session did not return the figure; the Llama 3 paper is arXiv 2407.21783. — [Meta Llama 3 blog, 2024-04-18](https://ai.meta.com/blog/meta-llama-3/); [The Llama 3 Herd of Models, arXiv 2407.21783 (Jul 2024)](https://arxiv.org/abs/2407.21783)

**Innateness, evolution and counterarguments**
- Feng, Goodman & Frank, "Is Child-Directed Speech Effective Training Data for Language Models?" (EMNLP 2024, **[PR]**). Child-directed speech is *not* uniquely valuable training data. Synthetic child-directed-style data (their "TinyDialogues") beat natural CDS. They conclude the child's advantage is more likely a more data-efficient *learning algorithm* than better data. — [ACL Anthology EMNLP 2024](https://aclanthology.org/2024.emnlp-main.1231.pdf); [arXiv 2408.03617 (Aug 2024)](https://arxiv.org/html/2408.03617)
- "The potential — and the pitfalls — of using pre-trained language models as cognitive science theories" (arXiv 2501.12651, Jan 2025, **[PP]**). Several studies show ANNs trained on child-scale data can handle various linguistic tasks, which some take as a challenge to innate linguistic priors. The same paper reports that adding an "innate" inductive prior to an ANN *improved* ANN–human alignment. — [arXiv 2501.12651](https://arxiv.org/pdf/2501.12651)
- "Beyond the data gap: Children create languages, violate their input statistics, and exhibit critical periods" (2025; venue and review status unclear). It argues children differ from LMs in three ways: they go beyond their input (e.g., creating languages), they follow a different learning trajectory, and they show a critical period. Until LMs reproduce these, innate language-specific mechanisms remain possible. — [ResearchGate](https://www.researchgate.net/publication/408310232_Beyond_the_data_gap_Children_create_languages_violate_their_input_statistics_and_exhibit_critical_periods)

### Inferences
- Rough arithmetic from the figures above: ~8.6k–30k adult words/day × 365 ≈ **3–11M words/year**, so about **15–55M by age 5** and **~40–140M by age 13**. This fits the BabyLM 10M/100M budgets. Llama 3's ~15T tokens (~11T words at ~0.75 words/token) is about **10^5×** a 100M-word child. That matches the "4–5 orders of magnitude" framing.
- The comparison is not like-for-like. (a) Children get hundreds to thousands of hours of synchronized vision, touch and social feedback. (b) Evolution provides a prior (brain architecture, learning biases) that can be viewed as "pretraining" done over evolutionary time. (c) Children learn *one* language to fluency. LLMs learn world knowledge, code and many languages. So the useful design target for a companion is to **inherit a strong prior (a pretrained small model) and make the personal learning phase child-efficient**, not to train from scratch on child-scale data.
- Romeo et al. (conversational turns) and Feng et al. (algorithm over data) both suggest that interactive, feedback-driven learning may matter more than raw volume. This is relevant to a companion that learns through dialogue.

### Gaps
- I could not open Gilkerson et al. 2017 (*AJSLP*) for clean per-age cumulative counts. The 30,142/17,514/8,624 figures come from a snippet whose column labels were not visible.
- I did not find a source that quantifies the evolutionary "prior" in bits or tokens. The evolution argument remains qualitative in what I found.
- An MIT Technology Review feature on kids vs. machines in language learning (2026-08-24) exists, but I could not fetch it. — [MIT Tech Review URL](https://www.technologyreview.com/2026/08/24/1141740/kids-machines-language-learning/)

---

## 2. BabyLM Challenge 2023, 2024 and 2025: budgets, winners, and how close they get to large-data models

### Takeaway
Over three rounds (2023 at CoNLL, 2024 at CoNLL, 2025 as the First BabyLM Workshop at EMNLP), the most reliable gains came from **architecture and objective changes**: LTG-BERT/ELC-BERT, then the hybrid causal+masked GPT-BERT, then adaptive masking. Curriculum learning and multimodal training mostly did **not** beat baselines. By 2025, the best 100M-word models were reported as **comparable to Llama-70B on BLiMP grammaticality judgments**, though not on knowledge or reasoning. The new 2025 Interaction track (learning from a teacher LLM's feedback) beat its baselines but generally scored below the Strict track.

### Cited Findings
**2023 (first round; older)**
- Findings: Warstadt et al., BabyLM shared task at CoNLL 2023 (**[WS]**; also posted as arXiv 2504.08165 in Apr 2025). Tracks: Strict (100M words), Strict-Small (10M words), Loose. — [arXiv 2504.08165](https://arxiv.org/html/2504.08165v1); [ACL Anthology CoNLL 2023 BabyLM volume](https://aclanthology.org/2023.conll-babylm.pdf)
- Winner: **ELC-BERT** (Charpentier & Samuel, LTG Oslo) won both Strict and Strict-Small. The runner-up, Boot-BERT, was also built on LTG-BERT. ELC-BERT lets each layer choose which earlier layers' outputs to use. Overall, LTG-BERT-based models did best. Loose-track models often *underperformed* despite using extra data types. The Loose winner was "Contextualizer," a data-augmentation scheme that mixes text chunks from different contexts. — [2023 findings](https://www.alphaxiv.org/abs/2504.08165); [ELC-BERT GitHub](https://github.com/ltgoslo/elc-bert)
- Score note: one snippet gives Strict aggregate ELC-BERT 0.85 vs Boot-BERT 0.86, which is inconsistent with ELC-BERT being the winner. Ranking may have used a different aggregate. **Treat the numbers as unreliable.** — [2023 findings](https://arxiv.org/html/2504.08165v1)
- In 2023 epochs were unlimited. Charpentier & Samuel trained for **>450 epochs** (Strict) and **>2,000 epochs** (Strict-Small). Data-efficient here does not mean compute-efficient. — reported in the [2025 findings](https://aclanthology.org/2025.babylm-main.28.pdf) and the [2023 findings](https://ar5iv.labs.arxiv.org/html/2504.08165)

**2024 (second round)**
- Findings: Hu et al., "Findings of the Second BabyLM Challenge" (arXiv 2412.05149, Dec 2024; CoNLL 2024 BabyLM proceedings, **[WS]**). **31 submissions.** Tracks: Strict (100M), Strict-Small (10M), new **Multimodal** track (aligned text–image data), and a Paper track. — [arXiv 2412.05149](https://arxiv.org/pdf/2412.05149); [ACL Anthology BabyLM 2024](https://aclanthology.org/events/babylm-2024/)
- A **hybrid causal–masked LM** architecture beat other approaches. This is **GPT-BERT** (Charpentier & Samuel, "GPT or BERT: why not both?", arXiv 2410.24159, Oct 2024), which won the Strict and Strict-Small tracks. — [2024 findings](https://arxiv.org/html/2412.05149v1); [GPT-BERT](https://arxiv.org/pdf/2410.24159)
- **Multimodal: no winner.** Only 3 teams (8 models) entered, and none beat the baselines. — [2024 findings](https://arxiv.org/html/2412.05149v1)
- **Curriculum learning:** 13 teams (41.9%) tried some form of it, but it "did not tend to produce high scores." One curriculum model did perform well. — [2024 findings](https://arxiv.org/pdf/2412.05149)

**2025 (third round, First BabyLM Workshop at EMNLP 2025)**
- Call for papers: "BabyLM Turns 3" (arXiv 2502.10645, Feb 2025). It added a new **INTERACTION** track that "encourages interactive behavior, learning from a teacher, and adapting the teaching material to the student." — [arXiv 2502.10645](https://arxiv.org/abs/2502.10645)
- Epoch cap: a 2026 paper says the standard tracks limit training to **10M or 100M words for at most 10 epochs, with no FLOP limit** (secondary source; I could not check it against the official rules). — ["Squeezing More from Limited Data with Recursive Transformers," arXiv 2608.26973 (Aug 2026), [PP]](https://arxiv.org/pdf/2608.26973)
- Findings: "Findings of the Third BabyLM Challenge: Accelerating Language Modeling Research with Cognitively Plausible Data" (**[WS]**, ACL Anthology 2025.babylm-main.28, Nov 2025). **32 models from 26 countries.** — [ACL Anthology](https://aclanthology.org/2025.babylm-main.28/); [PDF](https://aclanthology.org/2025.babylm-main.28.pdf); [ResearchGate](https://www.researchgate.net/publication/397422593_Findings_of_the_Third_BabyLM_Challenge_Accelerating_Language_Modeling_Research_with_Cognitively_Plausible_Data)
  - "Some models in the Strict and Interaction tracks are **comparable to a Llama 70B parameter model on the BLiMP task**." — [2025 findings](https://aclanthology.org/2025.babylm-main.28.pdf)
  - **Strict-Small winner:** "AMLM hard decay" (Edman & Fraser 2025), a masked LM that adapts its masking distribution during training. — [2025 findings](https://aclanthology.org/2025.babylm-main.28.pdf)
  - Strict track: a snippet points to a *diffusion-style masked model* among the winners, but the text was garbled. **Unverified.** — [2025 findings](https://aclanthology.org/2025.babylm-main.28.pdf)
  - **Human-likeness award (Strict):** CLASS-IT (Capone et al. 2025), which fine-tunes small LMs on a general instruction-following dataset. — [2025 findings](https://aclanthology.org/2025.babylm-main.28.pdf)
  - A snippet says an overall-winning team used **DeBERTa-v2 plus an auxiliary pinyin-prediction objective**. This is odd for an English task and **needs verification**. A Greek lab (Archimedes) announced a BabyLM award at EMNLP 2025. — [2025 findings](https://aclanthology.org/2025.babylm-main.28.pdf); [Archimedes news](https://archimedesai.gr/en/news/babylm-challenge-award-at-emnlp-2025-workshop)
  - **Interaction track:** single winner, reported as "BLM" (Martins et al. 2025). Interaction submissions beat their baselines, but "despite the greater flexibility and tailored supervision allowed in the interaction track, performance was generally higher in the strict track." — [2025 findings](https://aclanthology.org/2025.babylm-main.28.pdf)
  - **Multimodal: again no submission beat the baselines.** Earlier winners' methods (e.g., GPT-BERT, used as the 2025 Strict/Strict-Small baseline) "remained strong." — [2025 findings](https://aclanthology.org/2025.babylm-main.28.pdf)
- Interaction-track papers:
  - One BabyLM 2025 paper reports that teacher feedback is highly data-efficient: **"with just 1M words of input in interactive learning, storytelling skills can improve as much as with 410M words of next-word prediction."** GLUE dropped slightly (−0.7 pp) after interactive RL, and other BabyLM tasks were largely unaffected. Teachers were Llama-3.x Instruct models up to Llama 3.1 8B Instruct. (Snippets attribute this to ACL Anthology 2025.babylm-main.32; exact paper attribution not verified.) — [2025.babylm-main.32](https://aclanthology.org/2025.babylm-main.32v1.pdf); [BabyLM 2025 proceedings](https://aclanthology.org/volumes/2025.babylm-main/)
  - "Dialogue Is Not Enough to Make a Communicative BabyLM (But Neither Is Developmentally Inspired Reinforcement Learning)" (arXiv 2510.20358, Oct 2025, **[WS/PP]**). From the title, a negative result: training on dialogue data and developmentally inspired RL did not yield communicative competence. — [arXiv 2510.20358](https://arxiv.org/pdf/2510.20358)
- BabyLM-style work keeps going in 2026, e.g., recursive or weight-shared transformers to "squeeze more from limited data" (arXiv 2608.26973, **[PP]**). — [arXiv 2608.26973](https://arxiv.org/pdf/2608.26973)
- The methodology has spread to low-resource languages, e.g., "BabyLMs for isiXhosa" (arXiv 2501.03855, Jan 2025, **[PP]**). — [arXiv 2501.03855](https://arxiv.org/pdf/2501.03855)

### Inferences
- The consistent BabyLM lesson is that **the training objective and architecture matter more than clever data ordering at 10M–100M words.** Masked, hybrid and adaptive-masking objectives win. A companion trained from scratch is not the target, but the same toolkit (hybrid objectives, many epochs) applies to small personal-data fine-tuning.
- "Llama-70B-level BLiMP" measures **grammar only**. BabyLM models are not close on world knowledge, reasoning or instruction following. For a companion, linguistic competence is cheap, but knowledge and reasoning still need a large-data prior.
- The interaction-track result (1M interactive words ≈ 410M passive words for storytelling) is the most relevant signal for a companion that learns from feedback. It is a single workshop paper on one skill and needs replication.
- Multimodal BabyLM has failed to beat text-only baselines two years running. Grounding has not yet shown a data-efficiency win at these scales and with current recipes.

### Gaps
- Full 2025 winner list and per-track scores: not verified (could not open the PDF).
- I could not confirm whether the 2025 Interaction track had its own word budget (e.g., 100M words of student input plus unlimited teacher compute).
- **Distillation in BabyLM:** I recall "Baby Llama" (Timiryasov & Tastet, BabyLM 2023), where a student distilled from an ensemble of GPT-2 and LLaMA teachers on 10M words beat its teachers. **Not re-verified this session.**
- A search snippet said one curriculum paper beat random ordering by >10 pp when difficulty was defined relative to the model. I could not identify the source paper, so it is excluded from the findings.

---

## 3. Curriculum learning, developmentally plausible training, and grounded/multimodal learning (SAYCam, Vong et al. 2024)

### Takeaway
Grounded learning from a *single* child's egocentric video works for basic word–referent mapping: a generic contrastive model trained on 61 hours of one child's headcam learned many words. But it is far from full language. Curricula ordered "like child development" have mostly not helped in BabyLM, and age-ordered child-directed data is not special training data.

### Cited Findings
- Vong, Wang, Orhan & Lake, "Grounded language acquisition through the eyes and ears of a single child," *Science* 383:504–511, published **Feb 1, 2024** (**[PR]**), DOI 10.1126/science.adi1374. The data are longitudinal head-mounted camera recordings of **one child from 6 to 25 months** (the SAYCam dataset), recorded weekly. The model is trained on **61 hours** of paired visual–linguistic streams. — [Science](https://www.science.org/doi/10.1126/science.adi1374); [Semantic Scholar](https://www.semanticscholar.org/paper/Grounded-language-acquisition-through-the-eyes-and-Vong-Wang/2f90e89d7323006fd487715640b79d16e33a8122); [PDF mirror](https://gwern.net/doc/ai/nn/cnn/2024-vong.pdf)
- The model, **CVCL (Child's View for Contrastive Learning)**, embeds video frames and co-occurring utterances in a shared space, CLIP-style. It learned word–referent mappings, generalized beyond the specific objects in the child's environment, and aligned visual and linguistic representations. — [Science](https://www.science.org/doi/10.1126/science.adi1374); [ScienceDaily (NYU release), 2024-02-01](https://www.sciencedaily.com/releases/2024/02/240201212821.htm)
- Limitation stated by the authors: performance was "significantly above chance … but still struggled across many cases." — [OpenReview PDF (supplement)](https://openreview.net/pdf/27cf4cba6bd5d6594b40aa340092bbce7b86b3dc.pdf)
- Follow-up preprints on infant egocentric video:
  - "Do Blind Spots Matter for Word-Referent Mapping? A Computational Study with Infant Egocentric Video" (arXiv 2511.11725, Nov 2025, **[PP]**). — [arXiv](https://arxiv.org/pdf/2511.11725)
  - "Scaling cross-situational word learning on children's naturalistic experiences" (arXiv 2610.31842, Oct 2026, **[PP]**). — [arXiv](https://arxiv.org/pdf/2610.31842)
  - "Caregiver Talk Shapes Toddler Vision: A Computational Study of Dyadic Play" (arXiv 2312.04118, Dec 2023, **[PP]**). — [arXiv](https://arxiv.org/pdf/2312.04118)
  - The Lake lab's project page lists further work. — [Lake Lab projects](https://lake-lab.github.io/projects/)
- Curriculum learning in BabyLM 2024: 41.9% of teams tried it, and it mostly did not produce top scores. — [2024 findings](https://arxiv.org/pdf/2412.05149)
- Multimodal BabyLM tracks in 2024 and 2025: no submission beat the baselines. — [2024 findings](https://arxiv.org/html/2412.05149v1); [2025 findings](https://aclanthology.org/2025.babylm-main.28.pdf)
- Developmentally plausible *data*: natural child-directed speech is not uniquely good training data, and synthetic CDS-like dialogue beat it. — [Feng, Goodman & Frank, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.1231.pdf)
- 2025 BabyLM human-likeness award: instruction-tuned small LMs (CLASS-IT). — [2025 findings](https://aclanthology.org/2025.babylm-main.28.pdf)

### Inferences
- For a companion, grounding is useful mainly to *attach* the user's world (their objects, places, people) to an already-capable model through contrastive or vision-language adapters. It is not yet a route to cheaper base-model training.
- Fixed developmental curricula are a weak lever. Adaptive, model-centred difficulty (e.g., adaptive masking, the 2025 Strict-Small winner) and teacher feedback (the Interaction track) look more promising.

### Gaps
- Exact CVCL accuracy: I recall ~61.6% on a 22-concept 4-way evaluation vs ~66.7% for CLIP. **Not re-verified this session.**
- I did not find 2025–2026 results showing that multimodal grounding *reduces the text needed* for a given language skill. None was reported in BabyLM.

---

## 4. Synthetic data and distillation (TinyStories, Phi "textbooks are all you need")

### Takeaway
Curated or synthetic "textbook-quality" data lets very small models do well on narrow or STEM-heavy tasks. Under 10M parameters suffices for fluent children's stories (TinyStories). A 14B Phi-4 beats its GPT-4o teacher on GPQA and MATH. The limits: these models inherit the teacher's coverage, still trail on broad knowledge (Phi-4 is below GPT-4o on MMLU), and an active literature warns about diversity loss and model collapse.

### Cited Findings
- **TinyStories** (Eldan & Li, Microsoft Research, arXiv 2305.07759, May 2023, **[PP], older**). A GPT-3.5/GPT-4-generated dataset of short stories using only words a typical 3–4-year-old understands. Models **under 10M parameters** produce fluent, consistent multi-paragraph stories with basic reasoning. GPT-4 was used as the grader. — [arXiv 2305.07759](https://arxiv.org/pdf/2305.07759v1); [Semantic Scholar](https://www.semanticscholar.org/paper/TinyStories:-How-Small-Can-Language-Models-Be-and-Eldan-Li/28085f480ce456a376ebace9b899e3bc93dbc048)
- The authors' framing (secondary summary): small models struggle less from capacity limits than from being "overwhelmed" by complex, broad corpora. Narrowing the distribution makes small models work. — [Latent Space / Cognitive Revolution interview](https://www.latent.space/p/cogrev-tinystories); [alphaXiv overview](https://www.alphaxiv.org/overview/2305.07759v2)
- **Phi-4** (Microsoft, arXiv 2412.08905, Dec 2024, **[TR]**). 14B parameters, "centrally focused on data quality." **Synthetic data makes up the bulk of the training data**, generated by multi-agent prompting, self-revision workflows and instruction reversal. Earlier Phi models "largely distill the capabilities of a teacher (GPT-4)," while Phi-4 "substantially surpasses its teacher" on STEM QA. **GPQA: 56.1 vs GPT-4o 50.6**, it also beats GPT-4o on MATH, but **MMLU: 84.8 vs GPT-4o 88.1**. — [arXiv 2412.08905](https://arxiv.org/html/2412.08905v1); [Microsoft Research blog](https://www.microsoft.com/en-us/research/blog/phi-4-technical-report/)
- **phi-1, "Textbooks Are All You Need"** (Gunasekar et al., arXiv 2306.11644, Jun 2023, **[PP], older**): a 1.3B code model trained on ~7B tokens of filtered "textbook-quality" web data plus synthetic textbooks and exercises, reaching ~50% pass@1 on HumanEval. **[not re-verified this session]** — [arXiv 2306.11644](https://arxiv.org/abs/2306.11644)
- Limits literature (titles and abstracts only):
  - "Strong Model Collapse" (arXiv 2410.04840, Oct 2024) — [arXiv](https://arxiv.org/pdf/2410.04840)
  - "On the Diversity of Synthetic Data and its Impact on Training Large Language Models" (arXiv 2410.15226, Oct 2024) — [arXiv](https://arxiv.org/pdf/2410.15226)
- Synthetic data also helps at child scale: synthetic CDS-style dialogues beat natural child-directed speech as LM training data. — [Feng et al., EMNLP 2024](https://aclanthology.org/2024.emnlp-main.1231.pdf)

### Inferences
- For a personal companion, the practical recipe is **distill a narrow capability into a small on-device model** (TinyStories/Phi logic), then personalize. Synthetic data generated by a large "teacher" model about the user's domains is a plausible way to multiply scarce personal data. I found no direct evidence for this specific use.
- Phi-4's advantage over GPT-4o is concentrated in STEM, where synthetic data can be verified. It trails on MMLU, which tests broad knowledge. Expect synthetic-data small models to be **strong in verifiable reasoning and weaker in open-world knowledge**, which retrieval would need to supply.

### Gaps
- I did not retrieve Phi-4's own limitations section (e.g., factual hallucination, instruction-following weaknesses) or the 2025 Phi-4-mini / Phi-4-reasoning results.
- Quantitative model-collapse thresholds (how much synthetic data is safe) were not extracted.

---

## 5. Data-efficient adaptation to one person: ICL, LoRA/QLoRA on-device, retrieval memory, continual learning, privacy

### Takeaway
Shipping systems personalize with **frozen small base models plus adapters, retrieval and context**, and train population models with **federated learning plus differential privacy** (Gboard). 2025–2026 research is moving from per-user LoRA (~4M parameters per user) toward **compact user representations, soft-prompt memory and local parametric edits** that match LoRA with ~100× fewer parameters. **I found no source that states a minimum number of personal examples needed** for useful personalization. That remains an open empirical question.

### Cited Findings
**Industry on-device practice**
- **Google Gboard** (arXiv 2305.18465, May 2023, **[PP/industry], older**): "all the next word prediction neural network LMs in Gboard now have DP guarantees, and all future launches of Gboard neural network LMs will require DP guarantees." Training combines federated learning, differential privacy and secure aggregation. — [arXiv 2305.18465](https://arxiv.org/pdf/2305.18465)
- **Apple Intelligence Foundation Language Models Tech Report 2025** (arXiv 2507.13575, Jul 2025, revised Aug 2025, **[TR]**). A ~**3B-parameter on-device model** using **KV-cache sharing** and **2-bit quantization-aware training**. The server model uses a Parallel-Track Mixture-of-Experts. Supports 15 languages, with improved tool use and reasoning. A Swift **Foundation Models framework** exposes guided generation and **adapter fine-tuning** to developers. — [Apple ML Research](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025); [arXiv 2507.13575](https://arxiv.org/abs/2507.13575); [Apple 2025 updates post](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates)
- **Google research on on-device generative personalization:** "LoRA-generating hypernetworks for efficient on-device LLM generative personalization" (arXiv 2609.24979, Sep 2026, **[PP]**, Google-acknowledged). It *generates* a user's LoRA from user data instead of training it on the device. — [arXiv 2609.24979](https://arxiv.org/pdf/2609.24979)
- **Federated LoRA with DP:** DP-FedLoRA (arXiv 2509.09097, Sep 2025, **[PP]**) integrates LoRA with (ε, δ)-differential privacy for federated fine-tuning of on-device LLMs. — [arXiv 2509.09097](https://arxiv.org/pdf/2509.09097)
- Surveys: "A Survey on Foundation Models for Personalized Federated Intelligence" (arXiv 2505.06907, May 2025); "Synergizing Foundation Models and Federated Learning: A Survey" (arXiv 2406.12844, Jun 2024); and a curated list. — [arXiv 2505.06907](https://arxiv.org/pdf/2505.06907); [arXiv 2406.12844](https://arxiv.org/pdf/2406.12844); [awesome-fm-fl GitHub](https://github.com/lishenghui/awesome-fm-fl)
- **Personalization across base-model upgrades:** "Heirloom: Carrying a Phone's Personalization Across Base-Model Generations" (arXiv 2611.00650, Nov 2026, **[PP]**). It addresses the problem that per-user adapters break when the base model is replaced. — [arXiv 2611.00650](https://arxiv.org/pdf/2611.00650)

**Memory, retrieval and parametric personalization (2026 preprints)**
- Per-user LoRA (OPPU) needs a dedicated adapter of **~4.2M parameters per user**. "Beyond Retrieval: Learning Compact User Representations for Scalable LLM Personalization" (arXiv 2606.04547, Jun 2026, **[PP]**) claims to match or beat both retrieval-prompting and per-user adapters with a far smaller per-user state. A related result **matches LoRA accuracy on LoCoMo with ~120× fewer trainable parameters**. — [arXiv 2606.04547](https://arxiv.org/html/2606.04547v2)
- "Latent Personal Memory: Represent personal memory as dynamic soft prompts" (arXiv 2606.20911, Jun 2026, **[PP]**). — [arXiv](https://arxiv.org/html/2606.20911v1)
- "User as Engram: Internalizing Per-User Memory as Local Parametric Edits" (arXiv 2606.19172, Jun 2026, **[PP]**). — [arXiv](https://arxiv.org/pdf/2606.19172)
- "Evoking User Memory: Personalizing LLM via Recollection-Familiarity Adaptive Retrieval" (arXiv 2603.09250, Mar 2026, **[PP]**). Explicitly brain-inspired: dual-process recollection/familiarity retrieval. — [arXiv](https://arxiv.org/html/2603.09250v1)
- **PersonaMem** benchmark (2025, UPenn CogComp, **[status unclear]**): 180+ simulated user–LLM interaction histories with up to 60 multi-turn sessions. Retriever-based memory modules improved performance. Later work evaluates memory lengths of 32K, 128K and 1M tokens. — [PersonaMem PDF](https://cogcomp.seas.upenn.edu/papers/JHCLYCea25.pdf)
- Continual on-device learning: "Forget to Improve: On-Device LLM-Agent Continual Learning via Budget-Curated Memory" (arXiv 2606.25115, Jun 2026, **[PP]**). — [arXiv](https://arxiv.org/pdf/2606.25115)
- Risks:
  - "When Personalization Misleads: Understanding and Mitigating Hallucinations in Personalized LLMs" (arXiv 2601.11000, Jan 2026, **[PP]**). — [arXiv](https://arxiv.org/pdf/2601.11000)
  - "Know It, Act on It: Investigating Memory Utilization in LLM Personalization" (arXiv 2607.29433, Jul 2026, **[PP]**). Models often fail to *use* the memory they retrieve. — [arXiv](https://arxiv.org/pdf/2607.29433)
  - "Towards Root Memory: Benchmarking … Implicit Logical Memory Retrieval for Personalized LLMs" (arXiv 2606.23283, **[PP]**). — [arXiv](https://arxiv.org/pdf/2606.23283)
- Embedding adapters for personal-memory retrieval: "Axis-LoRA" (arXiv 2611.07820, Nov 2026, **[PP]**). — [arXiv](https://arxiv.org/html/2611.07820)

### Inferences
- A defensible 2026 companion architecture:
  - a frozen ~1–4B quantized base model (Apple-style 2-bit QAT shows ~3B fits on a phone);
  - an **episodic memory store with retrieval**, which needs zero training examples and works from the first conversation;
  - a **small parametric user state** (soft prompt, compact embedding or tiny LoRA) consolidated periodically from that memory, analogous to hippocampus-to-cortex consolidation;
  - optional federated, DP-protected updates to the shared base.
- Retrieval and in-context use need **zero training examples**. Parametric adaptation only becomes worthwhile once enough interaction history has accumulated. Where that crossover lies is not established in the sources I found.
- Version migration (Heirloom) is a real engineering issue. Keep the personal state in a **model-agnostic form** (text memories and preferences) so it can be re-distilled into new base models.

### Gaps
- **How much personal data is needed:** no source in my searches gave sample-efficiency curves (examples per user vs. quality) for per-user LoRA, retrieval or ICL. LaMP, OPPU and PER-PCS papers likely contain such curves but were not retrieved.
- I found nothing on Apple's *personal* on-device adaptation (as opposed to developer adapters) or Apple's Private Cloud Compute privacy design in this session.
- No head-to-head evidence on catastrophic forgetting rates for continual on-device fine-tuning over months of use.

---

## 6. Scaling-law evidence on data efficiency: data-constrained scaling, repeated epochs, small-model limits

### Takeaway
When unique data is scarce, **repeating it for up to ~4 epochs is almost as good as fresh data**, and gains fade toward zero with many more repeats (Muennighoff et al., NeurIPS 2023). The BabyLM 2023 winners nonetheless trained for hundreds to thousands of epochs on 10–100M words, and BabyLM 2025 capped training at 10 epochs. So "data-efficient" results often buy efficiency with compute.

### Cited Findings
- Muennighoff et al., "Scaling Data-Constrained Language Models," **NeurIPS 2023 (**[PR]**)**; extended version in **JMLR vol. 26 (2025) (**[PR]**)**; arXiv 2305.16264 (May 2023).
  - "With constrained data for a fixed compute budget, training with **up to 4 epochs of repeated data yields negligible changes to loss** compared to having unique data." One epoch is still best per unit of compute, but differences up to 4 epochs "do not lead to differences in downstream task performance." Beyond that, the value of repeated data "eventually diminish[es] to zero."
  - The study covered **400+ models from 10M to 9B parameters, trained for up to 1,500 epochs**. Runs and models are released.
  - It notes Galactica as an earlier exception to single-epoch training (4.25 epochs).
  - Sources: [arXiv 2305.16264](https://arxiv.org/abs/2305.16264); [JMLR v26 24-1000](https://jmlr.org/papers/volume26/24-1000/24-1000.pdf)
- BabyLM 2023 winners used >450 epochs (100M words) and >2,000 epochs (10M words). — [2025 findings](https://aclanthology.org/2025.babylm-main.28.pdf)
- BabyLM 2025 standard tracks: at most 10 epochs over 10M or 100M words, with no FLOP limit (secondary source). — [arXiv 2608.26973](https://arxiv.org/pdf/2608.26973)
- Recursive or parameter-shared transformers are being explored to extract more from limited data under this cap. — ["Squeezing More from Limited Data with Recursive Transformers," arXiv 2608.26973 (Aug 2026), [PP]](https://arxiv.org/pdf/2608.26973)
- "Classification-head fine-tuning of tiny LMs" (arXiv 2607.03801, Jul 2026, **[PP]**). Tiny models can "punch above their weight" on verifiable multiple-choice tasks when the output format is constrained. — [arXiv 2607.03801](https://arxiv.org/pdf/2607.03801)

### Inferences
- For a companion, personal data is extremely data-constrained: perhaps thousands to millions of tokens per year. Scaling-law evidence implies (a) **multi-epoch replay of personal data is fine up to a few passes**, (b) after that, returns come from **augmentation** (teacher-generated paraphrases or synthetic Q&A about the user's facts), not from more epochs, and (c) small models benefit from narrow output formats (Phi, TinyStories, classification-head results).
- Muennighoff's results are for *pretraining* loss. Their transfer to tiny personal fine-tuning sets, where overfitting and forgetting dominate, is plausible but **not directly established**.

### Gaps
- Small-model limits: I did not retrieve 2025–2026 scaling studies on "overtrained" small models, e.g., the tokens-per-parameter ratios of Llama 3 8B, Qwen or SmolLM, or the "densing law" of capability per parameter. These should be filled from the companion research topics.
- I did not retrieve explicit results on scaling laws for fine-tuning or personalization with very small datasets (thousands of examples).
