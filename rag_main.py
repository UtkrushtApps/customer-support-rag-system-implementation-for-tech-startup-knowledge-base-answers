# rag_main.py
import os
import json
from typing import List, Tuple, Dict

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from openai import OpenAI
from tqdm import tqdm
import numpy as np
from collections import namedtuple

# 1. Support documents: Load data (simulate as inline data for demo)
DOCUMENTS = [
    {
        'content': """Q: How do I reset my password on the SuperWidget platform?\nA: Go to the login screen, click 'Forgot Password', and follow the instructions sent to your email.""",
        'title': 'FAQ',
        'source': 'faq_v1.txt',
        'id': 'faq_001',
        'section': 'Authentication',
    },
    {
        'content': """Q: What should I do if the SuperWidget mobile app crashes?\nA: Force close the app and reopen it. If the issue persists, reinstall the app or contact support.""",
        'title': 'FAQ',
        'source': 'faq_v1.txt',
        'id': 'faq_002',
        'section': 'Troubleshooting',
    },
    {
        'content': """To set up your SuperWidget device:\n1. Plug in your device.\n2. Download the SuperWidget app.\n3. Follow the in-app pairing instructions.\n4. The indicator light turns green when setup is complete.""",
        'title': 'Product Manual',
        'source': 'manual_v1.txt',
        'id': 'manual_001',
        'section': 'Getting Started',
    },
    {
        'content': """Troubleshooting Wi-Fi Connection:\n- Ensure your router is online.\n- Place the device within 10m of the router.\n- Press the reset button for 5 seconds to restart the device and retry.""",
        'title': 'Troubleshooting Guide',
        'source': 'guide_wifi.txt',
        'id': 'guide_001',
        'section': 'Connectivity',
    },
    {
        'content': """SuperWidget only supports 2.4GHz Wi-Fi networks.\nDo not use 5GHz networks as setup will fail.""",
        'title': 'Product Manual',
        'source': 'manual_v1.txt',
        'id': 'manual_002',
        'section': 'Connectivity',
    },
    {
        'content': """Q: How can I check my subscription status?\nA: In the app, go to Settings > Subscription. Your current plan and renewal date are shown there.""",
        'title': 'FAQ',
        'source': 'faq_v1.txt',
        'id': 'faq_003',
        'section': 'Billing',
    },
]

# 2. Preprocessing and chunking with overlap
def preprocess_and_chunk(documents: List[Dict], chunk_size: int = 320, chunk_overlap: int = 64) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n", ". ", "\n\n", " "], length_function=len
    )
    all_chunks = []
    for doc in documents:
        cleaned_text = doc['content'].replace('\n', ' ').strip()
        splits = splitter.split_text(cleaned_text)
        for ix, chunk in enumerate(splits):
            metadata = {
                "source": doc['source'],
                "title": doc['title'],
                "id": doc['id'],
                "section": doc.get('section', ''),
                "chunk_id": f"{doc['id']}_chunk_{ix}",
            }
            all_chunks.append(Document(page_content=chunk, metadata=metadata))
    return all_chunks

# 3. Embeddings and Chroma vector DB
def get_vectorstore(persist_dir='./chroma_db'):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(
        collection_name="customer_support_docs",
        embedding_function=embeddings,
        persist_directory=persist_dir,
        client_settings={"host": "localhost", "port": 8000}
    )
    return db

# 4. Ingest and upsert to DB
def ingest_chunks_to_db(chunks: List[Document], chroma_db):
    print(f"Ingesting {len(chunks)} chunks...")
    chroma_db.add_documents(chunks)
    chroma_db.persist()
    print("Ingestion complete.")

# 5. Query: vector search, reranking, context assembly
def query_rag(
    question: str,
    chroma_db,
    top_k: int = 4,
    rerank: bool = True
):
    # Vector search (semantic)
    docs_and_scores = chroma_db.similarity_search_with_score(question, k=top_k*2)

    # Optional reranking (sort by score, not just embedding similarity)
    if rerank:
        # Simple rerank: sort by score ascending (smaller = more similar in Chroma)
        docs_and_scores.sort(key=lambda x: x[1])
    # Deduplicate chunks by id
    seen = set()
    unique_docs = []
    for doc, score in docs_and_scores:
        cid = doc.metadata["chunk_id"]
        if cid not in seen:
            unique_docs.append((doc, score))
            seen.add(cid)
        if len(unique_docs) >= top_k:
            break
    context = "\n".join([f"[{ix+1}] {d.page_content}" for ix, (d, s) in enumerate(unique_docs)])
    metadata_list = [d.metadata for d, s in unique_docs]
    return context, metadata_list, [d for d, s in unique_docs]

# 6. Generation (OpenAI, can be replaced with Llama2/Huggingface if needed)
def generate_answer(
    question: str,
    context: str,
    max_tokens: int = 256,
    api_key: str = None
):
    openai_api_key = api_key or os.getenv("OPENAI_API_KEY")
    assert openai_api_key, "OPENAI_API_KEY required."
    system_prompt = (
        "You are a helpful customer support agent for SuperWidget. "
        "Answer the question as accurately as possible using ONLY the information provided in the context. "
        "Always cite your answer with the relevant document number in square brackets (e.g., [1]). "
        "If you are unsure, politely say you don't know.\n\nContext:\n" + context
    )
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        max_tokens=max_tokens,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

# 7. Evaluation
EvaluationResult = namedtuple('EvaluationResult', ['query', 'ground_truth_ids', 'retrieved_ids', 'precision', 'recall'])
def evaluate_retrieval(
    queries_with_ground_truth: List[Tuple[str, List[str]]],
    chroma_db,
    top_k: int = 4
) -> List[EvaluationResult]:
    eval_results = []
    for query, gt_ids in queries_with_ground_truth:
        _, metas, _ = query_rag(query, chroma_db, top_k=top_k)
        retrieved_ids = [meta["id"] for meta in metas]
        precision = len(set(retrieved_ids) & set(gt_ids))/len(retrieved_ids) if retrieved_ids else 0
        recall = len(set(retrieved_ids) & set(gt_ids))/len(gt_ids) if gt_ids else 0
        eval_results.append(EvaluationResult(query, gt_ids, retrieved_ids, precision, recall))
    return eval_results

# 8. Main workflow: ingest, run, evaluate
def main():
    # Step 1: Preprocess and chunk
    print("Chunking and preprocessing...")
    chunks = preprocess_and_chunk(DOCUMENTS)
    print(f"Total Chunks: {len(chunks)}")

    # Step 2: Setup Chroma DB
    db = get_vectorstore()

    # Step 3: Ingest (check if already present; for demo, always ingest)
    ingest_chunks_to_db(chunks, db)

    # Step 4: Test Queries
    sample_queries = [
        ("How do I reset my SuperWidget password?", ["faq_001"]),
        ("What can I do if the app crashes on my phone?", ["faq_002"]),
        ("How do I set up my SuperWidget for the first time?", ["manual_001"]),
        ("Why does Wi-Fi setup fail for my SuperWidget?", ["manual_002", "guide_001"]),
        ("Where can I find my subscription details?", ["faq_003"]),
    ]

    print("\nSample Responses:")
    for query, gt_ids in sample_queries:
        context, metas, docs = query_rag(query, db, top_k=3)
        print(f"\n> Q: {query}")
        print(f"Context for Gen (truncated): {context[:150]}...")
        try:
            answer = generate_answer(query, context)
        except Exception as e:
            answer = f"[GENERATION ERROR: {e}]"
        print(f"A: {answer}")
        print(f"Ground Truth Doc IDs: {gt_ids} | Retrieved IDs: {[m['id'] for m in metas]}")

    # Step 5: Evaluation & Metrics
    eval_results = evaluate_retrieval(sample_queries, db)
    precisions = [res.precision for res in eval_results]
    recalls = [res.recall for res in eval_results]
    print("\nRetrieval Metrics:")
    print(f"Recall@3: {np.mean(recalls):.2f} | Precision@3: {np.mean(precisions):.2f}")
    for res in eval_results:
        print(f"Q: {res.query}\n  GT: {res.ground_truth_ids}\n  Retrieved: {res.retrieved_ids}\n  Precision: {res.precision:.2f} | Recall: {res.recall:.2f}")

if __name__ == "__main__":
    main()
