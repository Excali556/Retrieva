import os
import json
import random
import re

CORPUS_PATH = "data/bm25_corpus.json"
OUTPUT_TUNE = "mlops/eval/golden_tune.json"
OUTPUT_TEST = "mlops/eval/golden_test.json"
OUTPUT_ALL = "mlops/eval/golden_dataset.json"

def find_best_chunk(corpus, doc_keyword, search_phrase):
    """Finds the chunk from a document containing search_phrase."""
    candidates = []
    phrase_lower = search_phrase.lower()
    for chunk in corpus:
        fname = chunk["metadata"].get("filename", "").lower()
        if doc_keyword.lower() in fname:
            content_lower = chunk["page_content"].lower()
            if phrase_lower in content_lower:
                return chunk
    # Fallback to any chunk containing the phrase
    for chunk in corpus:
        if phrase_lower in chunk["page_content"].lower():
            return chunk
    return None

def build_dataset():
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(f"Missing {CORPUS_PATH}. Run ingestion first.")

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"Loaded {len(corpus)} chunks for alignment.")

    # 120 curated items:
    # 42 Exact Identifiers (~35%)
    # 42 Paraphrased Questions (~35%)
    # 36 Mixed / Multi-Hop (~30%)
    
    specs = [
        # --- EXACT IDENTIFIERS (42 questions) ---
        {
            "id": "exact_001_transformer_dmodel",
            "doc": "attention_is_all_you_need",
            "search": "dmodel = 512",
            "query": "In the Transformer architecture, what is the embedding and layer output dimension dmodel when using 8 heads?",
            "ground_truth_answer": "In the Transformer architecture, all sub-layers and embedding layers produce outputs of dimension dmodel = 512.",
            "expected_keywords": ["dmodel = 512", "512", "transformer", "dimension"],
            "expected_context_substr": "dmodel = 512",
            "category": "exact_identifier"
        },
        {
            "id": "exact_002_transformer_bleu",
            "doc": "attention_is_all_you_need",
            "search": "28.4",
            "query": "What BLEU score did the big Transformer model achieve on the WMT 2014 English-to-German translation task?",
            "ground_truth_answer": "The big Transformer model achieved a BLEU score of 28.4 on the WMT 2014 English-to-German translation task.",
            "expected_keywords": ["bleu", "28.4", "english-to-german", "wmt 2014"],
            "expected_context_substr": "28.4",
            "category": "exact_identifier"
        },
        {
            "id": "exact_003_bert_mask_rate",
            "doc": "bert_bidirectional",
            "search": "15%",
            "query": "What percentage of input tokens are uniformly selected for masking during the Masked Language Model MLM pre-training in BERT?",
            "ground_truth_answer": "BERT masks 15% of all input tokens at random during the Masked Language Model (MLM) pre-training step.",
            "expected_keywords": ["15%", "masked language model", "tokens"],
            "expected_context_substr": "15%",
            "category": "exact_identifier"
        },
        {
            "id": "exact_004_bert_large_params",
            "doc": "bert_bidirectional",
            "search": "340M",
            "query": "How many total parameters and layers does the BERT-Large model configuration contain?",
            "ground_truth_answer": "BERT-Large contains 340M total parameters, 24 layers (L=24), hidden size H=1024, and 16 attention heads (A=16).",
            "expected_keywords": ["340m", "24", "bert-large", "parameters"],
            "expected_context_substr": "340M",
            "category": "exact_identifier"
        },
        {
            "id": "exact_005_lora_decomposition",
            "doc": "lora_low_rank",
            "search": "BA",
            "query": "What low-rank matrix decomposition W0 + BA represents weight updates in the LoRA parameterization?",
            "ground_truth_answer": "LoRA decomposes the weight update delta W into BA, where B is d x r and A is r x k with rank r much smaller than d and k.",
            "expected_keywords": ["ba", "w0", "lora", "rank"],
            "expected_context_substr": "BA",
            "category": "exact_identifier"
        },
        {
            "id": "exact_006_lora_gpt3_rank",
            "doc": "lora_low_rank",
            "search": "intrinsic rank",
            "query": "What low intrinsic rank hypothesis explains why low-rank adaptation suffices even on large 175B models?",
            "ground_truth_answer": "The weight updates during task adaptation have a low intrinsic rank, allowing small rank r to capture the essential adaptation subspace.",
            "expected_keywords": ["intrinsic rank", "lora", "adaptation"],
            "expected_context_substr": "intrinsic rank",
            "category": "exact_identifier"
        },
        {
            "id": "exact_007_flash_sram",
            "doc": "flash_attention.pdf",
            "search": "SRAM",
            "query": "Which GPU on-chip fast memory does FlashAttention tile into to avoid materializing the full attention matrix in HBM?",
            "ground_truth_answer": "FlashAttention tiles inputs into fast on-chip SRAM (Static Random-Access Memory) to avoid reads/writes to high bandwidth memory (HBM).",
            "expected_keywords": ["sram", "hbm", "flashattention", "memory"],
            "expected_context_substr": "SRAM",
            "category": "exact_identifier"
        },
        {
            "id": "exact_008_flash2_speedup",
            "doc": "flash_attention_2",
            "search": "FlashAttention-2",
            "query": "How does FlashAttention-2 improve over FlashAttention-1 in terms of work partitioning across thread blocks and warps?",
            "ground_truth_answer": "FlashAttention-2 parallelizes over the sequence length dimension in addition to batch size and heads, reducing non-matmul FLOPs and increasing GPU occupancy.",
            "expected_keywords": ["flashattention-2", "sequence length", "warps", "thread blocks"],
            "expected_context_substr": "FlashAttention-2",
            "category": "exact_identifier"
        },
        {
            "id": "exact_009_react_thought_action",
            "doc": "react_reasoning",
            "search": "Thought",
            "query": "What are the two alternating textual tokens used in ReAct trajectories before receiving an Observation from the environment?",
            "ground_truth_answer": "ReAct alternates between 'Thought' (reasoning trace) and 'Action' (environment command) before receiving an 'Observation'.",
            "expected_keywords": ["thought", "action", "observation", "react"],
            "expected_context_substr": "Thought",
            "category": "exact_identifier"
        },
        {
            "id": "exact_010_toolformer_api_tags",
            "doc": "toolformer",
            "search": "[QA",
            "query": "What specific bracketed API call syntax does Toolformer insert to query external question-answering systems?",
            "ground_truth_answer": "Toolformer inserts calls formatted as [QA(query)] to invoke external QA tools.",
            "expected_keywords": ["[qa", "toolformer", "api"],
            "expected_context_substr": "[QA",
            "category": "exact_identifier"
        },
        {
            "id": "exact_011_dpo_beta",
            "doc": "direct_preference",
            "search": "beta",
            "query": "What hyperparameter beta governs the deviation penalty from the reference model pi_ref in the DPO objective?",
            "ground_truth_answer": "The temperature hyperparameter beta controls the strength of the KL divergence penalty from the reference policy pi_ref in Direct Preference Optimization (DPO).",
            "expected_keywords": ["beta", "pi_ref", "dpo", "kl divergence"],
            "expected_context_substr": "beta",
            "category": "exact_identifier"
        },
        {
            "id": "exact_012_qlora_nf4",
            "doc": "qlora",
            "search": "NF4",
            "query": "What novel 4-bit data type theoretically optimal for normally distributed weights is introduced in QLoRA?",
            "ground_truth_answer": "QLoRA introduces the 4-bit NormalFloat (NF4) data type, an information-theoretically optimal quantile quantization for zero-mean normal distributions.",
            "expected_keywords": ["nf4", "normalfloat", "qlora", "4-bit"],
            "expected_context_substr": "NF4",
            "category": "exact_identifier"
        },
        {
            "id": "exact_013_qlora_double_quant",
            "doc": "qlora",
            "search": "Double Quantization",
            "query": "How many bits per parameter on average does Double Quantization DQ save in QLoRA?",
            "ground_truth_answer": "Double Quantization (DQ) in QLoRA quantizes the quantization constants, saving approximately 0.37 bits per parameter.",
            "expected_keywords": ["double quantization", "0.37", "bits per parameter", "qlora"],
            "expected_context_substr": "Double Quantization",
            "category": "exact_identifier"
        },
        {
            "id": "exact_014_mistral_window_size",
            "doc": "mistral_7b",
            "search": "4096",
            "query": "What is the sliding window size W used by Mistral 7B in its Sliding Window Attention SWA layer?",
            "ground_truth_answer": "Mistral 7B uses a sliding window size W = 4096 in its Sliding Window Attention (SWA).",
            "expected_keywords": ["4096", "sliding window", "mistral 7b", "w"],
            "expected_context_substr": "4096",
            "category": "exact_identifier"
        },
        {
            "id": "exact_015_mistral_gqa_heads",
            "doc": "mistral_7b",
            "search": "Grouped-query attention",
            "query": "How many query heads and key/value heads are configured in Mistral 7B's Grouped Query Attention GQA?",
            "ground_truth_answer": "Mistral 7B uses 32 query attention heads and 8 key/value heads (4:1 ratio) with Grouped-Query Attention (GQA).",
            "expected_keywords": ["grouped-query attention", "gqa", "mistral 7b"],
            "expected_context_substr": "Grouped-query attention",
            "category": "exact_identifier"
        },
        {
            "id": "exact_016_colbert_maxsim",
            "doc": "colbert_late",
            "search": "MaxSim",
            "query": "What mathematical operator does ColBERT use to compute the sum of maximum cosine similarities between query tokens and document tokens?",
            "ground_truth_answer": "ColBERT utilizes the MaxSim operator, calculating the sum of maximum inner products between each query token vector and document token vectors.",
            "expected_keywords": ["maxsim", "colbert", "late interaction", "similarity"],
            "expected_context_substr": "MaxSim",
            "category": "exact_identifier"
        },
        {
            "id": "exact_017_bge_m3_context_len",
            "doc": "bge_m3",
            "search": "8192",
            "query": "What maximum input sequence length does BGE-M3 support for multi-granular text embeddings?",
            "ground_truth_answer": "BGE-M3 supports an extended context window length of up to 8192 tokens.",
            "expected_keywords": ["8192", "bge-m3", "tokens", "length"],
            "expected_context_substr": "8192",
            "category": "exact_identifier"
        },
        {
            "id": "exact_018_dpr_inbatch_negatives",
            "doc": "dpr_dense",
            "search": "in-batch",
            "query": "Which training technique does DPR use to efficiently treat passages paired with other queries in the same mini-batch as negative examples?",
            "ground_truth_answer": "DPR uses the 'in-batch negatives' technique to compute contrastive loss using other passages in the same batch as negative samples.",
            "expected_keywords": ["in-batch", "negatives", "dpr", "contrastive"],
            "expected_context_substr": "in-batch",
            "category": "exact_identifier"
        },
        {
            "id": "exact_019_raft_election_timeout",
            "doc": "raft_consensus",
            "search": "randomized election timeouts",
            "query": "What typical randomized range of election timeouts in milliseconds is chosen in the Raft consensus algorithm?",
            "ground_truth_answer": "Raft typically randomizes election timeouts between 150ms and 300ms to avoid split votes during leader election.",
            "expected_keywords": ["election timeouts", "raft", "randomized"],
            "expected_context_substr": "randomized election timeouts",
            "category": "exact_identifier"
        },
        {
            "id": "exact_020_raft_rpc_names",
            "doc": "raft_consensus",
            "search": "AppendEntries",
            "query": "What are the two basic Remote Procedure Calls RPCs used between servers in the Raft consensus algorithm?",
            "ground_truth_answer": "The two primary RPCs in Raft are RequestVote (initiated by candidates during elections) and AppendEntries (initiated by leaders for log replication and heartbeats).",
            "expected_keywords": ["appendentries", "requestvote", "raft", "rpc"],
            "expected_context_substr": "AppendEntries",
            "category": "exact_identifier"
        },
        {
            "id": "exact_021_adam_beta1_beta2",
            "doc": "adam_optimization",
            "search": "0.999",
            "query": "What are the default recommended values for hyper-parameters beta_1, beta_2, and epsilon in the Adam optimizer?",
            "ground_truth_answer": "In Adam, the default hyperparameter values are beta_1 = 0.9, beta_2 = 0.999, and epsilon = 10^-8.",
            "expected_keywords": ["0.9", "0.999", "adam", "beta_1", "beta_2"],
            "expected_context_substr": "0.999",
            "category": "exact_identifier"
        },
        {
            "id": "exact_022_resnet_layers",
            "doc": "deep_residual",
            "search": "152-layer",
            "query": "What is the deepest residual neural network architecture evaluated on the ImageNet dataset in the ResNet paper?",
            "ground_truth_answer": "ResNet evaluated a 152-layer deep residual network on the ImageNet dataset, winning the ILSVRC 2015 competition.",
            "expected_keywords": ["152-layer", "resnet", "imagenet", "residual"],
            "expected_context_substr": "152-layer",
            "category": "exact_identifier"
        },
        {
            "id": "exact_023_diffusion_t_steps",
            "doc": "diffusion_models",
            "search": "T = 1000",
            "query": "How many total diffusion timesteps T are used in DDPM for the forward noising process?",
            "ground_truth_answer": "DDPM uses T = 1000 total diffusion timesteps for the Gaussian noising schedule.",
            "expected_keywords": ["t = 1000", "timesteps", "ddpm", "diffusion"],
            "expected_context_substr": "T = 1000",
            "category": "exact_identifier"
        },
        {
            "id": "exact_024_roformer_theta",
            "doc": "roformer",
            "search": "10000",
            "query": "What base constant theta is commonly used in the rotary embedding frequency calculation in RoFormer?",
            "ground_truth_answer": "RoFormer uses a base frequency theta = 10000 to construct sinusoidal rotary position matrices.",
            "expected_keywords": ["10000", "roformer", "rotary", "rope"],
            "expected_context_substr": "10000",
            "category": "exact_identifier"
        },
        {
            "id": "exact_025_gpt3_params",
            "doc": "gpt3_few_shot",
            "search": "175 billion",
            "query": "How many total parameters does the largest GPT-3 autoregressive language model have?",
            "ground_truth_answer": "The largest GPT-3 model contains 175 billion parameters across 96 attention layers with 96 heads and d_model = 12288.",
            "expected_keywords": ["175 billion", "gpt-3", "parameters"],
            "expected_context_substr": "175 billion",
            "category": "exact_identifier"
        },
        {
            "id": "exact_026_llama2_token_count",
            "doc": "llama2",
            "search": "2 trillion",
            "query": "How many tokens of pre-training data were used to train Llama 2 models?",
            "ground_truth_answer": "Llama 2 was pre-trained on 2 trillion tokens of publicly available online data.",
            "expected_keywords": ["2 trillion", "tokens", "llama 2"],
            "expected_context_substr": "2 trillion",
            "category": "exact_identifier"
        },
        {
            "id": "exact_027_rag_seq_token",
            "doc": "rag_retrieval",
            "search": "RAG-Sequence",
            "query": "What are the two distinct model formulations proposed in RAG for conditioning on retrieved documents?",
            "ground_truth_answer": "The two formulations are RAG-Sequence (retrieves documents once per generated sequence) and RAG-Token (can retrieve different documents for each generated token).",
            "expected_keywords": ["rag-sequence", "rag-token", "documents", "generation"],
            "expected_context_substr": "RAG-Sequence",
            "category": "exact_identifier"
        },
        {
            "id": "exact_028_self_rag_tokens",
            "doc": "self_rag",
            "search": "Retrieve",
            "query": "What is the exact name of the special reflection token used in Self-RAG to determine whether retrieval should occur?",
            "ground_truth_answer": "Self-RAG uses the special reflection token Retrieve to dynamically decide whether retrieval is necessary.",
            "expected_keywords": ["retrieve", "self-rag", "reflection token"],
            "expected_context_substr": "Retrieve",
            "category": "exact_identifier"
        },
        {
            "id": "exact_029_graphrag_leiden",
            "doc": "graphrag",
            "search": "Leiden",
            "query": "Which community detection algorithm is used in GraphRAG to partition the knowledge graph into hierarchical communities?",
            "ground_truth_answer": "GraphRAG utilizes the Leiden community detection algorithm to partition entity-relationship graphs hierarchically.",
            "expected_keywords": ["leiden", "graphrag", "community detection", "graph"],
            "expected_context_substr": "Leiden",
            "category": "exact_identifier"
        },
        {
            "id": "exact_030_tree_of_thoughts_games",
            "doc": "tree_of_thoughts",
            "search": "Game of 24",
            "query": "Which mathematical puzzle game requiring reaching the number 24 from four numbers was used to evaluate Tree of Thoughts ToT?",
            "ground_truth_answer": "Tree of Thoughts was evaluated on Game of 24, where standard prompting achieved ~7% success while ToT achieved 74%.",
            "expected_keywords": ["game of 24", "tree of thoughts", "puzzle"],
            "expected_context_substr": "Game of 24",
            "category": "exact_identifier"
        },
        {
            "id": "exact_031_speculative_gamma",
            "doc": "speculative_decoding",
            "search": "gamma",
            "query": "In speculative decoding, what Greek letter variable gamma represents the number of lookahead candidate tokens generated by the draft model?",
            "ground_truth_answer": "The parameter gamma denotes the number of speculative tokens generated by the small draft model per iteration.",
            "expected_keywords": ["gamma", "speculative decoding", "draft model", "tokens"],
            "expected_context_substr": "gamma",
            "category": "exact_identifier"
        },
        {
            "id": "exact_032_gemma_geglu",
            "doc": "gemma_open",
            "search": "GeGLU",
            "query": "Which activation function replacing standard ReLU or GELU is used in the feed-forward layers of Gemma models?",
            "ground_truth_answer": "Gemma models use the GeGLU gated activation function in their feedforward blocks.",
            "expected_keywords": ["geglu", "activation", "gemma", "feed-forward"],
            "expected_context_substr": "GeGLU",
            "category": "exact_identifier"
        },
        {
            "id": "exact_033_scaling_law_power",
            "doc": "scaling_laws",
            "search": "power-law",
            "query": "According to Kaplan et al., how does cross-entropy test loss scale with model parameter count N and dataset size D?",
            "ground_truth_answer": "Cross-entropy loss scales as a power-law with model size N, dataset size D, and compute budget C.",
            "expected_keywords": ["power-law", "scaling laws", "cross-entropy", "parameters"],
            "expected_context_substr": "power-law",
            "category": "exact_identifier"
        },
        {
            "id": "exact_034_instruct_gpt_ppo",
            "doc": "instruct_gpt",
            "search": "PPO",
            "query": "Which reinforcement learning algorithm was used in InstructGPT to fine-tune the language model on human reward scores?",
            "ground_truth_answer": "InstructGPT uses Proximal Policy Optimization (PPO) to optimize policy outputs against the trained reward model.",
            "expected_keywords": ["ppo", "proximal policy optimization", "instructgpt", "reward model"],
            "expected_context_substr": "PPO",
            "category": "exact_identifier"
        },
        {
            "id": "exact_035_crag_evaluator_levels",
            "doc": "crag_corrective",
            "search": "Correct",
            "query": "What are the three confidence judgments output by the retrieval evaluator in Corrective RAG CRAG?",
            "ground_truth_answer": "The CRAG retrieval evaluator outputs one of three confidence levels: Correct, Incorrect, or Ambiguous.",
            "expected_keywords": ["correct", "incorrect", "ambiguous", "crag", "evaluator"],
            "expected_context_substr": "Correct",
            "category": "exact_identifier"
        },
        {
            "id": "exact_036_cve_2024_3094_id",
            "doc": "sample_retrieva",
            "search": "CVE-2024-3094",
            "query": "What exact identifier denotes the liblzma xz-utils upstream supply chain backdoor discovered in 2024?",
            "ground_truth_answer": "The backdoor vulnerability is designated as CVE-2024-3094 in upstream xz-utils tarballs.",
            "expected_keywords": ["cve-2024-3094", "xz-utils", "liblzma", "backdoor"],
            "expected_context_substr": "CVE-2024-3094",
            "category": "exact_identifier"
        },
        {
            "id": "exact_037_rfc_7636_pkce_id",
            "doc": "sample_retrieva",
            "search": "RFC 7636",
            "query": "Which IETF RFC specification defines Proof Key for Code Exchange PKCE for OAuth 2.0 public clients?",
            "ground_truth_answer": "OAuth 2.0 PKCE is standardized in IETF RFC 7636.",
            "expected_keywords": ["rfc 7636", "pkce", "oauth 2.0"],
            "expected_context_substr": "RFC 7636",
            "category": "exact_identifier"
        },
        {
            "id": "exact_038_http_504_gateway",
            "doc": "sample_retrieva",
            "search": "HTTP 504",
            "query": "Which standard HTTP status code indicates a Gateway Timeout from an intermediary proxy?",
            "ground_truth_answer": "HTTP 504 Gateway Timeout signifies that a proxy or gateway did not receive a timely response from upstream.",
            "expected_keywords": ["http 504", "gateway timeout", "proxy"],
            "expected_context_substr": "HTTP 504",
            "category": "exact_identifier"
        },
        {
            "id": "exact_039_rrf_k_val",
            "doc": "sample_retrieva",
            "search": "constant is 60",
            "query": "What default constant is assigned to rrf_k in the Reciprocal Rank Fusion calculation?",
            "ground_truth_answer": "In Reciprocal Rank Fusion, the smoothing parameter rrf_k default constant is 60.",
            "expected_keywords": ["rrf_k", "constant is 60", "reciprocal rank fusion"],
            "expected_context_substr": "constant is 60",
            "category": "exact_identifier"
        },
        {
            "id": "exact_040_cache_ttl_val",
            "doc": "sample_retrieva",
            "search": "3600 seconds",
            "query": "What value in seconds is assigned to CACHE_TTL_SECONDS in the caching configuration?",
            "ground_truth_answer": "CACHE_TTL_SECONDS is set to 3600 seconds (1 hour).",
            "expected_keywords": ["3600 seconds", "cache_ttl_seconds", "1 hour"],
            "expected_context_substr": "3600 seconds",
            "category": "exact_identifier"
        },
        {
            "id": "exact_041_llama_checkpoint_name",
            "doc": "sample_retrieva",
            "search": "llama-3.3-70b-versatile",
            "query": "What exact model identifier is configured as the default LLM_MODEL checkpoint?",
            "ground_truth_answer": "The default LLM model identifier is llama-3.3-70b-versatile served via Groq.",
            "expected_keywords": ["llama-3.3-70b-versatile", "checkpoint", "llm_model"],
            "expected_context_substr": "llama-3.3-70b-versatile",
            "category": "exact_identifier"
        },
        {
            "id": "exact_042_k8s_version_number",
            "doc": "sample_retrieva",
            "search": "Kubernetes v1.27",
            "query": "Which Kubernetes version promoted the CSI storage plugin migration to GA?",
            "ground_truth_answer": "Kubernetes v1.27 promoted in-tree storage plugin migration to CSI to GA.",
            "expected_keywords": ["kubernetes v1.27", "csi", "ga", "storage"],
            "expected_context_substr": "Kubernetes v1.27",
            "category": "exact_identifier"
        },

        # --- PARAPHRASED QUESTIONS (42 questions) ---
        {
            "id": "para_001_transformer_masking",
            "doc": "attention_is_all_you_need",
            "search": "subsequent positions",
            "query": "How is the auto-regressive decoder prevented from looking at future token positions during self-attention computation?",
            "ground_truth_answer": "Masked multi-head attention sets future attention weights to negative infinity before softmax, preventing positions from attending to subsequent tokens.",
            "expected_keywords": ["masking", "subsequent positions", "decoder", "attention"],
            "expected_context_substr": "subsequent positions",
            "category": "paraphrased"
        },
        {
            "id": "para_002_transformer_positional",
            "doc": "attention_is_all_you_need",
            "search": "sine and cosine",
            "query": "Since self-attention lacks recurrence and convolution, how does the Transformer inject word sequence order information?",
            "ground_truth_answer": "Positional encodings generated via sine and cosine functions of varying frequencies are added to input embeddings.",
            "expected_keywords": ["sine and cosine", "positional encodings", "order"],
            "expected_context_substr": "sine and cosine",
            "category": "paraphrased"
        },
        {
            "id": "para_003_bert_nsp_purpose",
            "doc": "bert_bidirectional",
            "search": "Next Sentence Prediction",
            "query": "What auxiliary binary classification objective helps language models understand relationships across consecutive sentence pairs?",
            "ground_truth_answer": "The Next Sentence Prediction (NSP) task trains the model to predict whether a second sentence is the actual continuation of the first.",
            "expected_keywords": ["next sentence prediction", "sentence pairs", "bert"],
            "expected_context_substr": "Next Sentence Prediction",
            "category": "paraphrased"
        },
        {
            "id": "para_004_lora_freeze_weights",
            "doc": "lora_low_rank",
            "search": "freezes the pre-",
            "query": "How does low-rank adaptation prevent catastrophic forgetting of knowledge when adapting huge foundational models?",
            "ground_truth_answer": "LoRA freezes the pre-trained weights entirely and only trains small auxiliary low-rank decomposition matrices.",
            "expected_keywords": ["freezes the pre-", "lora", "low-rank"],
            "expected_context_substr": "freezes the pre-",
            "category": "paraphrased"
        },
        {
            "id": "para_005_flash_io_awareness",
            "doc": "flash_attention.pdf",
            "search": "IO complexity",
            "query": "What principle explains why traditional transformer self-attention is bottlenecked by hardware memory bandwidth rather than compute FLOPs?",
            "ground_truth_answer": "Standard attention is memory-bound due to frequent reads and writes of the N x N attention matrix to GPU HBM rather than arithmetic compute throughput.",
            "expected_keywords": ["io complexity", "memory bandwidth", "hbm", "sram"],
            "expected_context_substr": "IO complexity",
            "category": "paraphrased"
        },
        {
            "id": "para_006_react_synergy",
            "doc": "react_reasoning",
            "search": "reasoning and acting",
            "query": "Why does combining internal model reasoning scratchpads with external tool interactions reduce factual hallucinations?",
            "ground_truth_answer": "Reasoning guides action planning and tracks progress, while external actions ground thoughts in real-time factual observations.",
            "expected_keywords": ["reasoning and acting", "hallucinations", "observations"],
            "expected_context_substr": "reasoning and acting",
            "category": "paraphrased"
        },
        {
            "id": "para_007_toolformer_filtering",
            "doc": "toolformer",
            "search": "loss",
            "query": "Under what condition does Toolformer keep an automatically generated tool call inside its fine-tuning dataset?",
            "ground_truth_answer": "Toolformer retains API calls if adding the call and its result reduces language modeling perplexity (loss) on future tokens.",
            "expected_keywords": ["loss", "toolformer", "api calls", "perplexity"],
            "expected_context_substr": "loss",
            "category": "paraphrased"
        },
        {
            "id": "para_008_cot_prompting_mechanism",
            "doc": "chain_of_thought",
            "search": "chain of thought",
            "query": "How can providing examples of intermediate derivation steps improve complex mathematical reasoning in neural models?",
            "ground_truth_answer": "Chain-of-thought demonstrations teach the model to decompose multi-step problems into sequential reasoning steps, allocating more compute tokens to the problem.",
            "expected_keywords": ["chain of thought", "intermediate", "reasoning", "multi-step"],
            "expected_context_substr": "chain of thought",
            "category": "paraphrased"
        },
        {
            "id": "para_009_dpo_no_rl",
            "doc": "direct_preference",
            "search": "without reinforcement learning",
            "query": "How does DPO eliminate the instability and complexity associated with classical actor-critic RL algorithms like PPO?",
            "ground_truth_answer": "DPO analytically solves the constrained reward maximization problem to express the loss directly as a simple binary cross-entropy on preference pairs without RL.",
            "expected_keywords": ["without reinforcement learning", "dpo", "preference"],
            "expected_context_substr": "without reinforcement learning",
            "category": "paraphrased"
        },
        {
            "id": "para_010_qlora_paged_optimizers",
            "doc": "qlora",
            "search": "Paged Optimizers",
            "query": "What technique prevents out-of-memory crashes during training by moving optimizer state pages to CPU RAM during memory spikes?",
            "ground_truth_answer": "Paged Optimizers utilize CUDA Unified Memory to automatically page optimizer states between GPU VRAM and CPU RAM during allocation spikes.",
            "expected_keywords": ["paged optimizers", "cuda unified memory", "qlora"],
            "expected_context_substr": "Paged Optimizers",
            "category": "paraphrased"
        },
        {
            "id": "para_011_colbert_retrieval_efficiency",
            "doc": "colbert_late",
            "search": "late interaction",
            "query": "How does late interaction achieve retrieval quality comparable to all-to-all cross-encoders while maintaining dense retrieval speed?",
            "ground_truth_answer": "It independently precomputes token representations for queries and documents, delaying interaction until a fast MaxSim operator is applied over vectors.",
            "expected_keywords": ["late interaction", "cross-encoders", "colbert"],
            "expected_context_substr": "late interaction",
            "category": "paraphrased"
        },
        {
            "id": "para_012_hyde_unsupervised_search",
            "doc": "hyde_zero_shot",
            "search": "hypothetical",
            "query": "How does generating a synthetic fictitious answer help dense retrieval when search queries lack document keyword overlap?",
            "ground_truth_answer": "HyDE generates a hypothetical document which captures the semantic neighborhood of the true answer, and embeds this passage to find real relevant documents.",
            "expected_keywords": ["hypothetical", "dense retrieval", "hyde"],
            "expected_context_substr": "hypothetical",
            "category": "paraphrased"
        },
        {
            "id": "para_013_self_rag_adaptive",
            "doc": "self_rag",
            "search": "adaptive retrieval",
            "query": "Why is always retrieving documents for every query suboptimal compared to selectively retrieving only when needed?",
            "ground_truth_answer": "Indiscriminate retrieval adds computational latency and risks introducing noisy, irrelevant contexts on simple queries that the model already knows.",
            "expected_keywords": ["adaptive retrieval", "self-rag", "retrieval"],
            "expected_context_substr": "adaptive retrieval",
            "category": "paraphrased"
        },
        {
            "id": "para_014_crag_web_fallback",
            "doc": "crag_corrective",
            "search": "web search",
            "query": "What recovery action is triggered in corrective retrieval when the internal corpus retrieval evaluator marks results as incorrect?",
            "ground_truth_answer": "CRAG falls back to an external web search to find accurate, up-to-date information when internal retrieved chunks are deemed incorrect.",
            "expected_keywords": ["web search", "crag", "retrieval evaluator"],
            "expected_context_substr": "web search",
            "category": "paraphrased"
        },
        {
            "id": "para_015_graphrag_global_sense",
            "doc": "graphrag",
            "search": "global",
            "query": "How does knowledge graph community summarization solve broad semantic questions that cannot be answered by isolated chunk retrieval?",
            "ground_truth_answer": "It builds summaries of entity clusters across the entire graph at multiple hierarchical levels, synthesizing answers for global sensemaking queries.",
            "expected_keywords": ["global", "graphrag", "summaries", "communities"],
            "expected_context_substr": "global",
            "category": "paraphrased"
        },
        {
            "id": "para_016_bge_m3_unification",
            "doc": "bge_m3",
            "search": "dense",
            "query": "What distinct retrieval paradigms are jointly unified into a single representation model in BGE-M3?",
            "ground_truth_answer": "BGE-M3 unifies dense vector retrieval, multi-vector ColBERT late interaction, and sparse lexical matching within a single embedding framework.",
            "expected_keywords": ["dense", "sparse", "multi-vector", "bge-m3"],
            "expected_context_substr": "dense",
            "category": "paraphrased"
        },
        {
            "id": "para_017_raft_leader_election",
            "doc": "raft_consensus",
            "search": "candidate",
            "query": "How does a follower node detect that the cluster leader has crashed and initiate a new election?",
            "ground_truth_answer": "If a follower's randomized election timer expires without receiving heartbeats from the leader, it transitions to candidate state and requests votes.",
            "expected_keywords": ["candidate", "election", "heartbeats", "raft"],
            "expected_context_substr": "candidate",
            "category": "paraphrased"
        },
        {
            "id": "para_018_raft_log_safety",
            "doc": "raft_consensus",
            "search": "Leader Append-Only",
            "query": "What invariant ensures that committed log entries in a distributed state machine are never overwritten or lost by future leaders?",
            "ground_truth_answer": "The Leader Append-Only property guarantees that leaders never overwrite or delete entries in their log, only appending new entries.",
            "expected_keywords": ["leader append-only", "committed", "log", "raft"],
            "expected_context_substr": "Leader Append-Only",
            "category": "paraphrased"
        },
        {
            "id": "para_019_adam_bias_correction",
            "doc": "adam_optimization",
            "search": "bias-corrected",
            "query": "Why are the moving averages of gradients initialized to zero scaled by 1 minus beta to the power t in early iterations?",
            "ground_truth_answer": "Bias correction counteracts the initialization bias toward zero during early training steps, producing unbiased first and second moment estimators.",
            "expected_keywords": ["bias-corrected", "adam", "moments", "zero"],
            "expected_context_substr": "bias-corrected",
            "category": "paraphrased"
        },
        {
            "id": "para_020_resnet_vanishing_gradients",
            "doc": "deep_residual",
            "search": "degradation",
            "query": "What unexpected degradation problem occurs when training standard deep feedforward networks with excessive depth?",
            "ground_truth_answer": "With increasing network depth, accuracy saturates and degrades rapidly, not from overfitting but due to vanishing/exploding gradients hindering optimization.",
            "expected_keywords": ["degradation", "depth", "resnet", "vanishing"],
            "expected_context_substr": "degradation",
            "category": "paraphrased"
        },
        {
            "id": "para_021_diffusion_reverse_process",
            "doc": "diffusion_models",
            "search": "reverse process",
            "query": "In generative diffusion models, how is clean data synthesized from pure random Gaussian noise?",
            "ground_truth_answer": "A neural network iteratively inverts the forward noise process step-by-step using learned conditional Gaussian transitions.",
            "expected_keywords": ["reverse process", "gaussian", "diffusion", "noise"],
            "expected_context_substr": "reverse process",
            "category": "paraphrased"
        },
        {
            "id": "para_022_roformer_relative_position",
            "doc": "roformer",
            "search": "relative position",
            "query": "What property of complex vector rotations allows RoPE to naturally encode token distance into inner product attention scores?",
            "ground_truth_answer": "Rotating query and key vectors before inner products ensures that their dot product depends strictly on relative position difference m - n.",
            "expected_keywords": ["relative position", "rotary", "inner product", "rope"],
            "expected_context_substr": "relative position",
            "category": "paraphrased"
        },
        {
            "id": "para_023_gpt3_few_shot_capabilities",
            "doc": "gpt3_few_shot",
            "search": "in-context learning",
            "query": "How can large language models solve novel tasks purely during inference without updating neural network weight parameters?",
            "ground_truth_answer": "Through in-context learning, models recognize patterns and instructions provided within the input context prompt at inference time.",
            "expected_keywords": ["in-context learning", "few-shot", "gpt-3", "inference"],
            "expected_context_substr": "in-context learning",
            "category": "paraphrased"
        },
        {
            "id": "para_024_llama2_safety_tuning",
            "doc": "llama2",
            "search": "safety",
            "query": "What methodologies are used to align generative responses to avoid producing harmful or dangerous user instructions?",
            "ground_truth_answer": "Safety fine-tuning uses dedicated adversarial safety datasets, safety-specific RLHF reward modeling, and red-teaming iterations.",
            "expected_keywords": ["safety", "llama 2", "alignment", "rlhf"],
            "expected_context_substr": "safety",
            "category": "paraphrased"
        },
        {
            "id": "para_025_constitutional_ai_principles",
            "doc": "constitutional_ai",
            "search": "constitution",
            "query": "How can an AI model be trained to critique and correct its own toxic responses without human crowd-worker annotations?",
            "ground_truth_answer": "The model uses a predefined set of written constitutional principles to critique, revise, and self-improve generated text.",
            "expected_keywords": ["constitution", "constitutional ai", "critique", "principles"],
            "expected_context_substr": "constitution",
            "category": "paraphrased"
        },
        {
            "id": "para_026_scaling_law_compute_balance",
            "doc": "scaling_laws",
            "search": "compute",
            "query": "When computational budget increases, should resources be allocated primarily to larger model parameters or longer dataset training?",
            "ground_truth_answer": "Optimal performance requires scaling both model parameters and dataset tokens concurrently, with larger models being more sample-efficient.",
            "expected_keywords": ["compute", "scaling laws", "parameters", "tokens"],
            "expected_context_substr": "compute",
            "category": "paraphrased"
        },
        {
            "id": "para_027_tot_exploration_strategy",
            "doc": "tree_of_thoughts",
            "search": "heuristic",
            "query": "How does exploring diverse alternative reasoning paths differ from traditional greedy linear autoregressive text generation?",
            "ground_truth_answer": "Tree of Thoughts uses search algorithms (BFS/DFS) with heuristic state evaluations, allowing lookahead and backtracking when paths prove unviable.",
            "expected_keywords": ["heuristic", "tree of thoughts", "backtracking", "search"],
            "expected_context_substr": "heuristic",
            "category": "paraphrased"
        },
        {
            "id": "para_028_speculative_lossless",
            "doc": "speculative_decoding",
            "search": "distribution",
            "query": "How does speculative decoding accelerate text generation speed without causing any degradation in the generated output text distribution?",
            "ground_truth_answer": "A modified rejection sampling scheme guarantees that accepted tokens mathematically follow the exact output distribution of the target model.",
            "expected_keywords": ["distribution", "speculative decoding", "rejection sampling"],
            "expected_context_substr": "distribution",
            "category": "paraphrased"
        },
        {
            "id": "para_029_rag_dpr_generator_link",
            "doc": "rag_retrieval",
            "search": "retriever",
            "query": "In retrieval-augmented generation systems, how does the parametric model interact with the non-parametric memory component?",
            "ground_truth_answer": "The dense passage retriever acts as non-parametric memory retrieving documents, which are prepended as context for the seq2seq generator.",
            "expected_keywords": ["retriever", "non-parametric", "generator", "rag"],
            "expected_context_substr": "retriever",
            "category": "paraphrased"
        },
        {
            "id": "para_030_dpr_vs_bm25_semantics",
            "doc": "dpr_dense",
            "search": "dense representation",
            "query": "Why do dual-encoder embeddings outperform lexical inverted indexes on paraphrased questions?",
            "ground_truth_answer": "Dense embeddings capture semantic concepts in continuous vector space, whereas lexical matching fails when synonyms do not share tokens.",
            "expected_keywords": ["dense representation", "dpr", "semantic", "bm25"],
            "expected_context_substr": "dense representation",
            "category": "paraphrased"
        },
        {
            "id": "para_031_mistral_byte_fallback",
            "doc": "mistral_7b",
            "search": "tokenizer",
            "query": "How does the tokenizer handle rare, unencountered Unicode characters without failing on out-of-vocabulary tokens?",
            "ground_truth_answer": "Mistral uses a byte-fallback BPE tokenizer that breaks down unseen characters into individual raw UTF-8 bytes.",
            "expected_keywords": ["tokenizer", "byte-fallback", "mistral 7b"],
            "expected_context_substr": "tokenizer",
            "category": "paraphrased"
        },
        {
            "id": "para_032_resnet_residual_identity",
            "doc": "deep_residual",
            "search": "identity",
            "query": "Why does adding shortcut skip connections make it easier for stacked layers to learn identity functions?",
            "ground_truth_answer": "If an identity mapping is optimal, it is easier for solvers to drive residual weights F(x) to zero than to fit an identity mapping from scratch.",
            "expected_keywords": ["identity", "shortcut", "resnet", "residual"],
            "expected_context_substr": "identity",
            "category": "paraphrased"
        },
        {
            "id": "para_033_instruct_gpt_human_alignment",
            "doc": "instruct_gpt",
            "search": "helpful",
            "query": "What discrepancy between language model pre-training goals and user expectations is addressed by RLHF?",
            "ground_truth_answer": "Next-token prediction on internet text does not align with following instructions safely and helpfully; RLHF aligns outputs with human intent.",
            "expected_keywords": ["helpful", "alignment", "instructgpt", "instructions"],
            "expected_context_substr": "helpful",
            "category": "paraphrased"
        },
        {
            "id": "para_034_cve_backdoor_mechanism",
            "doc": "sample_retrieva",
            "search": "sshd",
            "query": "In what manner did the compromised upstream compression utility tamper with secure remote shell authentication?",
            "ground_truth_answer": "The injected backdoor intercepted sshd authentication routines inside liblzma during OpenSSH connection handshakes.",
            "expected_keywords": ["sshd", "backdoor", "authentication", "liblzma"],
            "expected_context_substr": "sshd",
            "category": "paraphrased"
        },
        {
            "id": "para_035_oauth_pkce_defense",
            "doc": "sample_retrieva",
            "search": "code_verifier",
            "query": "How does Proof Key for Code Exchange shield mobile apps from authorization code interception attacks?",
            "ground_truth_answer": "By validating the cryptographically hashed code_challenge against the dynamically generated code_verifier on token exchange.",
            "expected_keywords": ["code_verifier", "pkce", "oauth", "authorization"],
            "expected_context_substr": "code_verifier",
            "category": "paraphrased"
        },
        {
            "id": "para_036_http_proxy_upstream_delay",
            "doc": "sample_retrieva",
            "search": "intermediary",
            "query": "What operational failure between frontend reverse proxies and backend worker services triggers a 504 error?",
            "ground_truth_answer": "An intermediary proxy server times out waiting for an HTTP response from an upstream backend server.",
            "expected_keywords": ["intermediary", "proxy", "upstream", "timeout"],
            "expected_context_substr": "intermediary",
            "category": "paraphrased"
        },
        {
            "id": "para_037_k8s_storage_csi_migration",
            "doc": "sample_retrieva",
            "search": "in-tree storage plugin",
            "query": "How did Kubernetes improve volume storage stability by replacing legacy in-tree drivers?",
            "ground_truth_answer": "By transitioning in-tree volume plugins to the standardized out-of-tree Container Storage Interface (CSI).",
            "expected_keywords": ["in-tree storage plugin", "csi", "kubernetes", "migration"],
            "expected_context_substr": "in-tree storage plugin",
            "category": "paraphrased"
        },
        {
            "id": "para_038_rrf_scoring_intuition",
            "doc": "sample_retrieva",
            "search": "Reciprocal Rank Fusion",
            "query": "How does rank-based fusion avoid calibration errors when combining disparate score distributions from keyword and vector models?",
            "ground_truth_answer": "RRF combines ordinal rankings rather than raw scores, summing 1/(k + rank) to prevent high-magnitude score imbalances.",
            "expected_keywords": ["reciprocal rank fusion", "rankings", "scores"],
            "expected_context_substr": "Reciprocal Rank Fusion",
            "category": "paraphrased"
        },
        {
            "id": "para_039_redis_ttl_rationale",
            "doc": "sample_retrieva",
            "search": "evict",
            "query": "What automated expiration mechanism prevents memory exhaustion in query response caches over time?",
            "ground_truth_answer": "Configuring time-to-live (TTL) expiration automatically evicts stale query keys and responses from memory.",
            "expected_keywords": ["evict", "ttl", "cache", "memory"],
            "expected_context_substr": "evict",
            "category": "paraphrased"
        },
        {
            "id": "para_040_observability_grafana_role",
            "doc": "sample_retrieva",
            "search": "Grafana",
            "query": "Which operational monitoring tool visualizes request latencies, QPS throughput, and cache metrics on live dashboards?",
            "ground_truth_answer": "Grafana provides real-time visualization dashboards connected to Prometheus metrics exporters.",
            "expected_keywords": ["grafana", "dashboard", "latency", "prometheus"],
            "expected_context_substr": "Grafana",
            "category": "paraphrased"
        },
        {
            "id": "para_041_data_integrity_validation",
            "doc": "sample_retrieva",
            "search": "Continuous evaluation",
            "query": "How do continuous evaluation pipelines safeguard against silent retrieval quality regressions during data updates?",
            "ground_truth_answer": "Continuous evaluation monitors RAG faithfulness, groundedness, and context relevance against a golden test set.",
            "expected_keywords": ["continuous evaluation", "faithfulness", "recall", "groundedness"],
            "expected_context_substr": "Continuous evaluation",
            "category": "paraphrased"
        },
        {
            "id": "para_042_sparse_lexical_advantage",
            "doc": "sample_retrieva",
            "search": "BM25",
            "query": "Why does classical term-frequency search remain essential alongside dense deep-learning vector models?",
            "ground_truth_answer": "BM25 excels at exact keyword, version number, acronym, and rare identifier lookups where dense vectors blur distinctions.",
            "expected_keywords": ["bm25", "lexical", "keyword", "exact"],
            "expected_context_substr": "BM25",
            "category": "paraphrased"
        },

        # --- MIXED & MULTI-HOP QUESTIONS (36 questions) ---
        {
            "id": "mixed_001_transformer_qlora",
            "doc": "attention_is_all_you_need",
            "search": "Multi-Head Attention",
            "query": "How is Multi-Head Attention parameterized in Transformers, and how can fine-tuning attention projection matrices be optimized via low-rank methods?",
            "ground_truth_answer": "Multi-Head Attention computes parallel attention projections using W_Q, W_K, W_V, and W_O matrices; low-rank methods (LoRA/QLoRA) freeze these matrices and adapt only low-rank decomposed adapters.",
            "expected_keywords": ["multi-head attention", "projections", "matrices", "attention"],
            "expected_context_substr": "Multi-Head Attention",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_002_rag_dpr_dual",
            "doc": "rag_retrieval",
            "search": "Dense Passage Retriever",
            "query": "What dual-encoder retriever component is used in RAG to index passages for the BART sequence-to-sequence generator?",
            "ground_truth_answer": "RAG uses the Dense Passage Retriever (DPR), which utilizes bi-encoder architectures to score question-passage similarity via dot product.",
            "expected_keywords": ["dense passage retriever", "dpr", "bart", "rag"],
            "expected_context_substr": "Dense Passage Retriever",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_003_dpr_vs_bm25_hybrid",
            "doc": "dpr_dense",
            "search": "BM25",
            "query": "How does Dense Passage Retrieval DPR compare against BM25 on top-20 passage retrieval accuracy for Natural Questions?",
            "ground_truth_answer": "DPR outperforms BM25 on Natural Questions top-20 retrieval accuracy by over 9-10% (achieving ~78-79% vs 59% for BM25).",
            "expected_keywords": ["bm25", "dpr", "natural questions", "retrieval"],
            "expected_context_substr": "BM25",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_004_flash_attention_exactness",
            "doc": "flash_attention.pdf",
            "search": "exact attention",
            "query": "How does FlashAttention maintain exact mathematical equivalence to standard self-attention while drastically reducing GPU memory consumption?",
            "ground_truth_answer": "It computes exact attention by recomputing intermediate softmax scaling statistics during backward passes rather than saving large intermediate attention matrices in HBM.",
            "expected_keywords": ["exact attention", "softmax", "flashattention", "hbm"],
            "expected_context_substr": "exact attention",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_005_lora_parameter_efficiency",
            "doc": "lora_low_rank",
            "search": "10,000 times",
            "query": "By what factor can LoRA reduce the number of trainable parameters compared to full fine-tuning on large 175B models?",
            "ground_truth_answer": "LoRA can reduce trainable parameters by up to 10,000 times and GPU memory requirements by 3 times compared to full fine-tuning.",
            "expected_keywords": ["10,000 times", "parameters", "lora", "fine-tuning"],
            "expected_context_substr": "10,000 times",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_006_dpo_vs_ppo_rewards",
            "doc": "direct_preference",
            "search": "PPO",
            "query": "How does the DPO loss formulation bypass the need for training a separate reward model compared to standard PPO RLHF?",
            "ground_truth_answer": "DPO reparameterizes the reward function using the policy itself, allowing direct optimization on human preference pairs without an explicit reward model or PPO training loop.",
            "expected_keywords": ["ppo", "reward model", "dpo", "preference"],
            "expected_context_substr": "PPO",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_007_qlora_memory_footprint",
            "doc": "qlora",
            "search": "single 48GB",
            "query": "What hardware GPU configuration is made capable of fine-tuning a 65B parameter model thanks to QLoRA?",
            "ground_truth_answer": "QLoRA enables fine-tuning a 65B parameter language model on a single 48GB GPU (like an NVIDIA A100) within 24 hours.",
            "expected_keywords": ["single 48gb", "65b", "qlora", "gpu"],
            "expected_context_substr": "single 48GB",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_008_mistral_gqa_throughput",
            "doc": "mistral_7b",
            "search": "inference speed",
            "query": "How does Grouped Query Attention GQA in Mistral 7B boost decoding throughput compared to full Multi-Head Attention MHA?",
            "ground_truth_answer": "By sharing key and value heads across multiple query heads, GQA substantially reduces KV cache memory footprint and bandwidth during autoregressive decoding.",
            "expected_keywords": ["inference speed", "gqa", "mistral 7b", "cache"],
            "expected_context_substr": "inference speed",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_009_colbert_indexing_compression",
            "doc": "colbert_late",
            "search": "vector",
            "query": "How does ColBERT index passages and manage memory storage requirements for multi-vector document representations?",
            "ground_truth_answer": "ColBERT offline encodes document tokens into vectors and uses vector quantization/pruning to store compact index representations for rapid MaxSim retrieval.",
            "expected_keywords": ["vector", "colbert", "index", "tokens"],
            "expected_context_substr": "vector",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_010_hyde_dense_encoder",
            "doc": "hyde_zero_shot",
            "search": "Contriever",
            "query": "Which unsupervised dense retriever model does HyDE pair with generative LLM prompts to perform zero-shot search?",
            "ground_truth_answer": "HyDE utilizes Contriever (an unsupervised contrastive dense encoder) to embed generated hypothetical documents without requiring relevance fine-tuning.",
            "expected_keywords": ["contriever", "hyde", "unsupervised", "encoder"],
            "expected_context_substr": "Contriever",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_011_self_rag_critique",
            "doc": "self_rag",
            "search": "reflection tokens",
            "query": "What specific reflection tokens does Self-RAG generate to evaluate whether retrieved passages are relevant and whether generated text is supported?",
            "ground_truth_answer": "Self-RAG generates [IsRel] to check document relevance, [IsSup] to check context groundedness, and [IsUse] to judge answer utility.",
            "expected_keywords": ["reflection tokens", "self-rag", "isrel", "issup"],
            "expected_context_substr": "reflection tokens",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_012_crag_decompose_refine",
            "doc": "crag_corrective",
            "search": "refine",
            "query": "How does Corrective RAG filter out irrelevant sentences from retrieved documents before passing them to the generator?",
            "ground_truth_answer": "CRAG uses a decompose-and-refine algorithm that breaks retrieved documents into smaller knowledge strips and filters out uninformative segments.",
            "expected_keywords": ["refine", "crag", "knowledge", "generator"],
            "expected_context_substr": "refine",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_013_graphrag_entities_relations",
            "doc": "graphrag",
            "search": "entity",
            "query": "What two core graph elements does GraphRAG extract from text chunks using LLM prompts before building community summaries?",
            "ground_truth_answer": "GraphRAG extracts entities (nodes) and relationships/claims (edges) along with descriptions from text chunks.",
            "expected_keywords": ["entity", "relationship", "graphrag", "graph"],
            "expected_context_substr": "entity",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_014_bge_m3_sparse_dense",
            "doc": "bge_m3",
            "search": "sparse",
            "query": "How does BGE-M3 compute lexical token importance weights alongside dense semantic embeddings in a single forward pass?",
            "ground_truth_answer": "BGE-M3 outputs a sparse weight vector over the vocabulary where each term weight reflects its lexical importance, complementing dense semantic representations.",
            "expected_keywords": ["sparse", "bge-m3", "dense", "embeddings"],
            "expected_context_substr": "sparse",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_015_raft_election_safety",
            "doc": "raft_consensus",
            "search": "Leader Election",
            "query": "How does Raft ensure Election Safety such that at most one leader can be elected in any given term?",
            "ground_truth_answer": "Each node votes for at most one candidate per term on a first-come-first-served basis, and a candidate requires a strict majority of cluster votes to win.",
            "expected_keywords": ["leader election", "term", "majority", "raft"],
            "expected_context_substr": "Leader Election",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_016_adam_momentum_scaling",
            "doc": "adam_optimization",
            "search": "second moment",
            "query": "How does Adam combine the concepts of gradient momentum with adaptive learning rate scaling across individual weights?",
            "ground_truth_answer": "Adam maintains an exponentially decaying average of past gradients (first moment/momentum) and past squared gradients (second moment/RMSProp).",
            "expected_keywords": ["second moment", "momentum", "adam", "gradients"],
            "expected_context_substr": "second moment",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_017_resnet_bottleneck",
            "doc": "deep_residual",
            "search": "bottleneck",
            "query": "What 3-layer bottleneck block design does ResNet introduce to reduce computational complexity in deeper 50/101/152 architectures?",
            "ground_truth_answer": "The bottleneck block consists of a 1x1 convolution for dimension reduction, a 3x3 convolution, and a 1x1 convolution for restoring dimensions.",
            "expected_keywords": ["bottleneck", "resnet", "convolution", "layers"],
            "expected_context_substr": "bottleneck",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_018_diffusion_forward_noise",
            "doc": "diffusion_models",
            "search": "forward process",
            "query": "How does the forward diffusion process progressively destroy data structure over T timesteps in DDPM?",
            "ground_truth_answer": "The forward process q adds small amounts of Gaussian noise at each step according to a variance schedule beta_t until the signal becomes pure isotropic Gaussian noise.",
            "expected_keywords": ["forward process", "gaussian", "beta", "diffusion"],
            "expected_context_substr": "forward process",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_019_roformer_inner_product",
            "doc": "roformer",
            "search": "RoPE",
            "query": "Why does RoPE rotary position embedding avoid adding positional vectors directly to token word embeddings?",
            "ground_truth_answer": "Direct addition can disturb semantic magnitude, whereas RoPE rotates token representations in 2D planes, preserving vector lengths while injecting position via rotational phase shifts.",
            "expected_keywords": ["rope", "rotary", "position", "embeddings"],
            "expected_context_substr": "RoPE",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_020_gpt3_few_shot_scaling",
            "doc": "gpt3_few_shot",
            "search": "few-shot",
            "query": "How does few-shot prompt performance scale as model capacity increases from 125 million to 175 billion parameters in GPT-3?",
            "ground_truth_answer": "Few-shot performance improves steeply with model parameter scale, with larger models demonstrating dramatic gains on reasoning, translation, and synthetic tasks.",
            "expected_keywords": ["few-shot", "gpt-3", "parameters", "scaling"],
            "expected_context_substr": "few-shot",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_021_llama2_gqa_benefit",
            "doc": "llama2",
            "search": "Grouped-Query Attention",
            "query": "Why did Llama 2 adopt Grouped-Query Attention GQA specifically for its largest 70B parameter variant?",
            "ground_truth_answer": "GQA drastically reduces the memory bandwidth requirement of the KV cache during multi-token generation, enabling longer context handling and larger batch sizes.",
            "expected_keywords": ["grouped-query attention", "llama 2", "70b", "kv cache"],
            "expected_context_substr": "Grouped-Query Attention",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_022_constitutional_ai_critique_loop",
            "doc": "constitutional_ai",
            "search": "critique and revision",
            "query": "What are the two phases of Constitutional AI (RLAIF) used to produce safe AI assistants?",
            "ground_truth_answer": "Phase 1 performs supervised learning using model critique and revision loops; Phase 2 trains a preference model from AI comparisons to perform reinforcement learning.",
            "expected_keywords": ["critique and revision", "constitutional ai", "reinforcement learning"],
            "expected_context_substr": "critique and revision",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_023_scaling_law_overfitting",
            "doc": "scaling_laws",
            "search": "overfitting",
            "query": "Under what condition does a large language model overfit according to Kaplan et al.'s scaling analysis?",
            "ground_truth_answer": "Overfitting occurs when model parameter capacity N is scaled without correspondingly scaling dataset tokens D according to the optimal power-law ratio.",
            "expected_keywords": ["overfitting", "scaling laws", "parameters", "dataset"],
            "expected_context_substr": "overfitting",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_024_tot_evaluation_states",
            "doc": "tree_of_thoughts",
            "search": "evaluation",
            "query": "How does Tree of Thoughts decide whether a given branch of reasoning should be pursued or pruned?",
            "ground_truth_answer": "An LLM acts as an evaluator to score or classify each intermediate thought state (e.g. sure, likely, impossible), pruning paths evaluated as dead ends.",
            "expected_keywords": ["evaluation", "tree of thoughts", "pruned", "states"],
            "expected_context_substr": "evaluation",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_025_speculative_draft_verification",
            "doc": "speculative_decoding",
            "search": "acceptance",
            "query": "In speculative decoding, what condition determines whether a draft token x is accepted by the target model?",
            "ground_truth_answer": "A token is accepted with probability min(1, p(x)/q(x)), where p is the target distribution and q is the draft distribution.",
            "expected_keywords": ["acceptance", "speculative decoding", "draft", "distribution"],
            "expected_context_substr": "acceptance",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_026_gemma_mqa_2b",
            "doc": "gemma_open",
            "search": "Multi-Query Attention",
            "query": "Why did the Gemma architecture use Multi-Query Attention MQA for the 2B model while using Multi-Head Attention for the 7B model?",
            "ground_truth_answer": "MQA uses a single key-value head, drastically reducing memory footprint for on-device and edge deployment targets where 2B models typically run.",
            "expected_keywords": ["multi-query attention", "gemma", "2b", "memory"],
            "expected_context_substr": "Multi-Query Attention",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_027_react_hotpotqa_multihop",
            "doc": "react_reasoning",
            "search": "HotpotQA",
            "query": "How does ReAct solve multi-hop question answering benchmarks like HotpotQA compared to standard chain-of-thought?",
            "ground_truth_answer": "ReAct uses search actions to retrieve facts across multiple Wikipedia articles, synthesizes the intermediate answers, and avoids hallucinating non-existent facts.",
            "expected_keywords": ["hotpotqa", "react", "multi-hop", "reasoning"],
            "expected_context_substr": "HotpotQA",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_028_toolformer_calculator",
            "doc": "toolformer",
            "search": "Calculator",
            "query": "How does Toolformer teach models to invoke an external Calculator tool when encountering arithmetic expressions?",
            "ground_truth_answer": "It inserts [Calculator(expr)] API annotations into math passages, evaluates expressions via Python, and keeps examples where calculator output lowers prediction loss.",
            "expected_keywords": ["calculator", "toolformer", "arithmetic", "api"],
            "expected_context_substr": "Calculator",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_029_instruct_gpt_rm_pairwise",
            "doc": "instruct_gpt",
            "search": "Reward Model",
            "query": "How was the reward model trained in InstructGPT using pairwise comparisons of model completions?",
            "ground_truth_answer": "Human labelers ranked K model completions from best to worst, and the reward model was trained using binary cross-entropy loss over all pairwise comparisons.",
            "expected_keywords": ["reward model", "pairwise", "instructgpt", "rankings"],
            "expected_context_substr": "Reward Model",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_030_cve_and_rrf_hybrid",
            "doc": "sample_retrieva",
            "search": "CVE-2024-3094",
            "query": "Why does searching for supply-chain vulnerability CVE-2024-3094 benefit from hybrid search with RRF over semantic-only search?",
            "ground_truth_answer": "Dense vector search often fails on rare alphanumeric strings like CVE-2024-3094, while BM25 anchors the exact identifier and RRF merges the ranking.",
            "expected_keywords": ["cve-2024-3094", "hybrid", "rrf", "bm25"],
            "expected_context_substr": "CVE-2024-3094",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_031_pkce_and_redis_cache",
            "doc": "sample_retrieva",
            "search": "PKCE",
            "query": "How does the Retrieva architecture handle repeated requests for technical protocols like OAuth 2.0 PKCE?",
            "ground_truth_answer": "Repeated queries are normalized and looked up in the Redis TTL response cache, bypassing the LLM and returning cached citations in under 3ms.",
            "expected_keywords": ["pkce", "redis", "cache", "latency"],
            "expected_context_substr": "PKCE",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_032_k8s_csi_and_prometheus",
            "doc": "sample_retrieva",
            "search": "rag_request_duration_seconds",
            "query": "What telemetry histogram does Prometheus scrape to record end-to-end request latency across duration buckets?",
            "ground_truth_answer": "Prometheus scrapes rag_request_duration_seconds across predefined latency buckets from 0.05s to 10.0s.",
            "expected_keywords": ["rag_request_duration_seconds", "prometheus", "latency"],
            "expected_context_substr": "rag_request_duration_seconds",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_033_bm25_dense_fusion_alpha",
            "doc": "sample_retrieva",
            "search": "Reciprocal Rank Fusion",
            "query": "How does varying the hybrid search parameter alpha between 0.0 and 1.0 balance lexical recall versus semantic generalization?",
            "ground_truth_answer": "Alpha=0.0 corresponds to pure BM25 lexical search, alpha=1.0 gives pure semantic dense search, and alpha=0.3 provides empirical optimal balance on technical workloads.",
            "expected_keywords": ["reciprocal rank fusion", "alpha", "bm25", "dense"],
            "expected_context_substr": "Reciprocal Rank Fusion",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_034_grafana_and_quality_gates",
            "doc": "sample_retrieva",
            "search": "data_validation.py",
            "query": "How are chunk integrity validations and boundary lengths verified before indexing in the Retrieva data pipeline?",
            "ground_truth_answer": "data_validation.py computes MD5 hashes of document chunks to detect duplicates and enforce minimum and maximum chunk length boundaries.",
            "expected_keywords": ["data_validation.py", "md5", "hashes", "duplicates"],
            "expected_context_substr": "data_validation.py",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_035_llama_temperature_and_fastapi",
            "doc": "sample_retrieva",
            "search": "Groq",
            "query": "Why is the Groq inference model temperature set to 0.0 when serving document question-answering over FastAPI?",
            "ground_truth_answer": "Temperature 0.0 enforces deterministic, greedy token generation to maximize factual groundedness and minimize hallucination.",
            "expected_keywords": ["groq", "temperature", "fastapi", "groundedness"],
            "expected_context_substr": "Groq",
            "category": "mixed_multihop"
        },
        {
            "id": "mixed_036_feedback_and_retrieval_tuning",
            "doc": "sample_retrieva",
            "search": "feedback",
            "query": "How does Retrieva capture end-user feedback to detect retrieval degradation in production?",
            "ground_truth_answer": "Users submit thumbs up/down ratings via /feedback endpoint, incrementing rag_feedback_total counters in Prometheus.",
            "expected_keywords": ["feedback", "rating", "prometheus", "counters"],
            "expected_context_substr": "feedback",
            "category": "mixed_multihop"
        }
    ]

    print(f"Total curated specifications: {len(specs)}")
    
    verified_items = []
    missing = []

    for item in specs:
        matched_chunk = find_best_chunk(corpus, item["doc"], item["search"])
        if matched_chunk:
            item_data = {
                "id": item["id"],
                "category": item["category"],
                "query": item["query"],
                "ground_truth_answer": item["ground_truth_answer"],
                "expected_keywords": item["expected_keywords"],
                "expected_context_substr": item["search"],
                "target_chunk_id": matched_chunk["metadata"].get("chunk_id", ""),
                "target_source": matched_chunk["metadata"].get("source", ""),
                "target_page": matched_chunk["metadata"].get("page", 1),
                "target_filename": matched_chunk["metadata"].get("filename", "")
            }
            verified_items.append(item_data)
        else:
            missing.append(f"{item['id']} (doc={item['doc']}, search='{item['search']}')")

    print(f"Verified against corpus: {len(verified_items)} / {len(specs)}")
    if missing:
        print(f"Warning: {len(missing)} items not found:")
        for m in missing[:10]:
            print(f"  - {m}")

    # Split into Tune (40%) and Test (60%) stratified by category
    tune_set = []
    test_set = []

    # Deterministic split using random seed
    random.seed(42)

    categories = set(item["category"] for item in verified_items)
    for cat in sorted(categories):
        cat_items = [it for it in verified_items if it["category"] == cat]
        random.shuffle(cat_items)
        n_tune = int(round(len(cat_items) * 0.40))
        tune_items = cat_items[:n_tune]
        test_items = cat_items[n_tune:]

        for it in tune_items:
            it["split"] = "tune"
            tune_set.append(it)
        for it in test_items:
            it["split"] = "test"
            test_set.append(it)

    print(f"\nFinal Split:")
    print(f"  Tune split: {len(tune_set)} items ({len(tune_set)/len(verified_items)*100:.1f}%)")
    print(f"  Test split: {len(test_set)} items ({len(test_set)/len(verified_items)*100:.1f}%)")
    print(f"  Total items: {len(verified_items)}")

    # Category breakdown
    print("\nCategory breakdown in splits:")
    for cat in sorted(categories):
        t_cnt = sum(1 for it in tune_set if it["category"] == cat)
        te_cnt = sum(1 for it in test_set if it["category"] == cat)
        print(f"  [{cat}]: tune={t_cnt}, test={te_cnt}, total={t_cnt+te_cnt}")

    # Save to disk
    os.makedirs(os.path.dirname(OUTPUT_TUNE), exist_ok=True)
    with open(OUTPUT_TUNE, "w", encoding="utf-8") as f:
        json.dump(tune_set, f, indent=2)
    print(f"Saved tune split to {OUTPUT_TUNE}")

    with open(OUTPUT_TEST, "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2)
    print(f"Saved test split to {OUTPUT_TEST}")

    with open(OUTPUT_ALL, "w", encoding="utf-8") as f:
        json.dump(verified_items, f, indent=2)
    print(f"Saved full golden dataset to {OUTPUT_ALL}")

if __name__ == "__main__":
    build_dataset()
