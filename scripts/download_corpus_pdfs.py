import os
import sys
import time
import urllib.request
import urllib.error

PDF_CATALOG = [
    ("1706.03762", "attention_is_all_you_need.pdf", "Attention Is All You Need (Transformer)"),
    ("2005.11401", "rag_retrieval_augmented_generation.pdf", "Retrieval-Augmented Generation for Knowledge-Intensive NLP"),
    ("1810.04805", "bert_bidirectional_transformers.pdf", "BERT: Pre-training of Deep Bidirectional Transformers"),
    ("2106.09685", "lora_low_rank_adaptation.pdf", "LoRA: Low-Rank Adaptation of Large Language Models"),
    ("2205.14135", "flash_attention.pdf", "FlashAttention: Fast and Memory-Efficient Exact Attention"),
    ("2307.08691", "flash_attention_2.pdf", "FlashAttention-2: Faster Attention with Work Partitioning"),
    ("2210.03629", "react_reasoning_and_acting.pdf", "ReAct: Synergizing Reasoning and Acting in Language Models"),
    ("2302.04761", "toolformer.pdf", "Toolformer: Language Models Can Teach Themselves to Use Tools"),
    ("2201.11903", "chain_of_thought.pdf", "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"),
    ("2305.18290", "direct_preference_optimization.pdf", "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"),
    ("2305.14314", "qlora_efficient_finetuning.pdf", "QLoRA: Efficient Finetuning of Quantized LLMs"),
    ("2310.06825", "mistral_7b.pdf", "Mistral 7B"),
    ("2004.12832", "colbert_late_interaction.pdf", "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction"),
    ("2212.10496", "hyde_zero_shot_dense_retrieval.pdf", "Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)"),
    ("2310.11511", "self_rag_learning_to_retrieve.pdf", "Self-RAG: Learning to Retrieve, Generate, and Critique"),
    ("2401.15884", "crag_corrective_rag.pdf", "Corrective Retrieval Augmented Generation (CRAG)"),
    ("2404.16130", "graphrag_local_to_global.pdf", "From Local to Global Grounded Text Generation (GraphRAG)"),
    ("2402.03216", "bge_m3_text_embeddings.pdf", "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity"),
    ("2004.04906", "dpr_dense_passage_retrieval.pdf", "Dense Passage Retrieval for Open-Domain Question Answering"),
    ("2305.10601", "tree_of_thoughts.pdf", "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"),
    ("2211.17192", "speculative_decoding.pdf", "Speculative Decoding: Fast Decoding by Fast Forwarding"),
    ("2307.09288", "llama2_open_foundation_models.pdf", "Llama 2: Open Foundation and Fine-Tuned Chat Models"),
    ("2212.08073", "constitutional_ai.pdf", "Constitutional AI: Harmlessness from AI Feedback"),
    ("2001.08361", "scaling_laws_neural_lms.pdf", "Scaling Laws for Neural Language Models"),
    ("2403.08295", "gemma_open_models.pdf", "Gemma: Open Models Based on Gemini Research and Technology"),
    ("1512.03385", "deep_residual_learning_resnet.pdf", "Deep Residual Learning for Image Recognition"),
    ("1412.6980", "adam_optimization.pdf", "Adam: A Method for Stochastic Optimization"),
    ("2006.11239", "diffusion_models_ddpm.pdf", "Denoising Diffusion Probabilistic Models"),
    ("2104.09864", "roformer_rotary_position_embedding.pdf", "RoFormer: Enhanced Transformer with Rotary Position Embedding"),
    ("2203.02155", "instruct_gpt.pdf", "Training Language Models to Follow Instructions with Human Feedback"),
    ("1405.0066", "raft_consensus_algorithm.pdf", "In Search of an Understandable Consensus Algorithm (Raft)"),
    ("2005.14165", "gpt3_few_shot_learners.pdf", "Language Models are Few-Shot Learners (GPT-3)")
]

def download_pdfs(output_dir: str = "data/pdfs"):
    os.makedirs(output_dir, exist_ok=True)
    # Remove test.pdf if it was in data/
    test_pdf = os.path.join("data", "test.pdf")
    if os.path.exists(test_pdf):
        try:
            os.remove(test_pdf)
        except OSError:
            pass

    print(f"Downloading {len(PDF_CATALOG)} real research PDFs to {output_dir}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    downloaded = 0
    skipped = 0
    failed = 0

    for idx, (arxiv_id, filename, title) in enumerate(PDF_CATALOG, start=1):
        target_path = os.path.join(output_dir, filename)
        if os.path.exists(target_path) and os.path.getsize(target_path) > 10000:
            print(f"[{idx}/{len(PDF_CATALOG)}] Already exists: {filename} ({os.path.getsize(target_path)} bytes)")
            skipped += 1
            continue

        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        print(f"[{idx}/{len(PDF_CATALOG)}] Downloading {filename} from {url} ({title})...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp, open(target_path, "wb") as out_file:
                out_file.write(resp.read())
            size = os.path.getsize(target_path)
            print(f" -> Saved {filename} ({size / 1024:.1f} KB)")
            downloaded += 1
            time.sleep(0.5) # Gentle rate limit
        except Exception as e:
            print(f" -> Error downloading {filename}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Download Summary: {downloaded} downloaded, {skipped} already present, {failed} failed.")
    total_files = len([f for f in os.listdir(output_dir) if f.endswith(".pdf")])
    print(f"Total PDFs in {output_dir}: {total_files}")
    print("=" * 60)

if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data/pdfs"
    download_pdfs(out_dir)
