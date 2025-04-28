import os
import textwrap
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer

def split_text(text, tokenizer_name="sshleifer/distilbart-cnn-12-6", max_tokens=1024):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs['input_ids'][0]

    safe_token_limit = max_tokens - 10
    chunks = []

    for i in range(0, len(input_ids), safe_token_limit):
        chunk_ids = input_ids[i:i + safe_token_limit]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text.strip())

    return chunks


def load_summarizer_model():
    model_name = "sshleifer/distilbart-cnn-12-6"
    try:
        print("[INFO] Trying to load model on CUDA...")
        summarizer = pipeline("summarization", model=model_name, device=0)
        print("[INFO] Model loaded on CUDA.")
    except Exception as e:
        print(f"[WARNING] CUDA failed: {e}")
        print("[INFO] Falling back to CPU...")
        summarizer = pipeline("summarization", model=model_name, device=-1)
    return summarizer


def summarize_text(text, cache_path="summary.txt"):
    if os.path.exists(cache_path):
        print(f"[INFO] Loading cached summary from {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as file:
            return file.read()

    summarizer = load_summarizer_model()

    print("[INFO] Chunking input text...")
    text_chunks = split_text(text, max_tokens=1024)

    print(f"[INFO] Summarizing {len(text_chunks)} chunks...")
    summaries = []
    for i, chunk in enumerate(tqdm(text_chunks, desc="Summarizing chunks")):
        try:
            summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            print(f"[ERROR] Skipping chunk {i+1}: {e}")

    final_summary = "\n\n".join(summaries)

    with open(cache_path, 'w', encoding='utf-8') as file:
        file.write(final_summary)

    print(f"[INFO] Summary saved to {cache_path}")
    return final_summary


def smart_clean_summary_chunked(input_file="summary.txt", output_file="cleaned.txt", chunk_size=700):
    if not os.path.exists(input_file):
        print(f"[ERROR] {input_file} does not exist.")
        return

    with open(input_file, "r", encoding="utf-8") as file:
        text = file.read()

    # Load summarization model
    try:
        print("[INFO] Loading summarization model on GPU...")
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0)
    except:
        print("[WARNING] GPU failed. Using CPU fallback...")
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

    instruction = ("Refine the following text by keeping only technical, academic, or educational content. "
                   "Remove introductions, greetings, conclusions, and any non-informative sentences.")

    # Split text into manageable chunks
    chunks = textwrap.wrap(text, width=chunk_size, break_long_words=False, replace_whitespace=False)
    print(f"[INFO] Split text into {len(chunks)} chunks.")

    cleaned_chunks = []

    for idx, chunk in enumerate(tqdm(chunks, desc="Cleaning chunks")):
        print(f"[INFO] Cleaning chunk {idx+1}/{len(chunks)}...")
        prompt = instruction + "\n\n" + chunk
        try:
            cleaned = summarizer(prompt, max_length=512, min_length=50, do_sample=False)[0]['summary_text']
            cleaned_chunks.append(cleaned.strip())
        except Exception as e:
            print(f"[ERROR] Cleaning chunk {idx+1} failed: {e}")
            continue

    final_cleaned_text = "\n\n".join(cleaned_chunks)

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(final_cleaned_text)

    print(f"[INFO] Final cleaned summary saved to {output_file}")