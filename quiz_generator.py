import os
import torch
import random
import re
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

def load_cleaned_text(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def smart_split(text, max_chunk_size=400):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) < max_chunk_size:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = sent
    if current:
        chunks.append(current.strip())
    return chunks


def generate_question(qg_model, qg_tokenizer, context):
    input_text = f"generate question: {context}"
    inputs = qg_tokenizer(input_text, return_tensors="pt").to(qg_model.device)
    outputs = qg_model.generate(**inputs, max_length=64)
    question_raw = qg_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Fix repeated questions:
    questions = question_raw.split("<sep>")
    questions = [q.strip() for q in questions if len(q.strip()) > 5]
    if questions:
        return questions[0]  # pick the first clean one
    else:
        return question_raw.strip()  # fallback


def generate_answer(qa_pipeline, context, question):
    result = qa_pipeline(question=question, context=context)
    return result['answer']


def generate_wrong_answers(context, correct_answer):
    # Extract nouns/keywords from context randomly
    words = list(set(re.findall(r'\b\w+\b', context)))
    words = [w for w in words if w.lower() not in correct_answer.lower() and len(w) > 3]
    random.shuffle(words)
    wrongs = words[:3]
    # if too few wrongs, pad manually
    while len(wrongs) < 3:
        wrongs.append("Unrelated concept")
    return wrongs


def make_real_mcq(question, correct_answer, wrong_answers):
    options = wrong_answers + [correct_answer]
    random.shuffle(options)
    correct_option = 'abcd'[options.index(correct_answer)]

    mcq = f"""Q: {question}
a) {options[0]}
b) {options[1]}
c) {options[2]}
d) {options[3]}
Answer: {correct_option}

"""
    return mcq


def smart_real_mcq_generator(input_file, output_file):
    # Load text
    text = load_cleaned_text(input_file)

    # Split into chunks
    chunks = smart_split(text)

    # Load models
    device = 0 if torch.cuda.is_available() else -1
    print(f"[INFO] Using device: {'GPU' if device==0 else 'CPU'}")

    print("[INFO] Loading models...")
    qg_tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-e2e-qg")
    qg_model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-e2e-qg").to(device)

    qa_pipeline = pipeline("question-answering", 
                          model="deepset/roberta-base-squad2", 
                          tokenizer="deepset/roberta-base-squad2", 
                          device=device)

    print("[INFO] Generating MCQs...")
    mcqs = []

    for chunk in tqdm(chunks, desc="Generating questions"):
        try:
            question = generate_question(qg_model, qg_tokenizer, chunk)
            answer = generate_answer(qa_pipeline, chunk, question)
            if answer.strip() == "" or len(answer.split()) > 25:
                continue
            wrongs = generate_wrong_answers(chunk, answer)
            mcq = make_real_mcq(question, answer, wrongs)
            mcqs.append(mcq)
        except Exception as e:
            print(f"[WARN] Skipping a chunk due to error: {e}")
            continue

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        for mcq in mcqs:
            f.write(mcq)

    print(f"[INFO] Saved {len(mcqs)} realistic MCQs to {output_file}")