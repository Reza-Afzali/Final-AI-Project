import os
import fitz  # PyMuPDF
import re
import json

DATASET_DIR = "dataset"
CHUNK_SIZE = 1000
OUTPUT_FILE = "processed_chunks.jsonl"

def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=CHUNK_SIZE):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        while end > start and text[end] != ' ':
            end -= 1
        chunks.append(text[start:end].strip())
        start = end
    return chunks

def process_pdf(file_path, company, year, filename):
    doc = fitz.open(file_path)
    all_chunks = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        raw_text = page.get_text()
        clean = clean_text(raw_text)
        chunks = chunk_text(clean)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "company": company,
                "year": year,
                "filename": filename,
                "page": page_num + 1,
                "chunk_id": i,
                "text": chunk
            })
    return all_chunks

def main():
    all_processed_chunks = []
    for company in os.listdir(DATASET_DIR):
        company_path = os.path.join(DATASET_DIR, company)
        if not os.path.isdir(company_path):
            continue
        for year in os.listdir(company_path):
            year_path = os.path.join(company_path, year)
            if not os.path.isdir(year_path):
                continue
            for file in os.listdir(year_path):
                if file.lower().endswith(".pdf"):
                    file_path = os.path.join(year_path, file)
                    print(f"Processing {file_path}...")
                    chunks = process_pdf(file_path, company, year, file)
                    all_processed_chunks.extend(chunks)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_processed_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Processed {len(all_processed_chunks)} chunks saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
