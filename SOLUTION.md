# Solution Steps

1. Create and collect representative customer support documents (FAQs, guides, manuals) as text, each with metadata fields (title, source, section).

2. Implement a preprocessing and chunking function that cleans text and splits long documents into overlapping, manageable chunks to maximize retrieval accuracy.

3. Select a suitable public text embedding model (such as sentence-transformers/all-MiniLM-L6-v2), and connect to the running Chroma vector database using the appropriate Python API/library.

4. For each chunk, generate its embedding, attach metadata, and batch insert all chunks with embeddings and metadata into Chroma while persisting the data.

5. For querying, build a working search function that computes the query embedding, performs top-k similarity search (optionally with reranking based on similarity score), and assembles a concise context from the most relevant retrieved chunks.

6. Integrate an LLM (e.g., OpenAI GPT-3.5, or a selected Huggingface/Llama2 model) as the generation module, and craft prompts that instruct the model to answer using only provided context (citing sources by reference numbers).

7. Implement retrieval evaluation by running sample queries, collecting retrieved document IDs, and computing precision@k and recall@k against ground truth for each test query.

8. Print and review the generated answers and evaluation metrics to ensure the RAG system produces accurate, well-cited, relevant responses for customer support use cases.

9. Document all functions with clear comments to ensure codebase is reproducible and easy for others to understand or extend.

