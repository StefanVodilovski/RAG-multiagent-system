import os
import torch
import numpy as np
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set up models
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')
rag_model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large')

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

# Load the collection
collection_name = "google_2022"
collection = Collection(name=collection_name)

def generate_embeddings(texts):
    embeddings = embedder.encode(texts, convert_to_tensor=True)
    return embeddings.numpy()

def retrieve_relevant_embeddings(query_text, top_k=5):
    query_embedding = generate_embeddings([query_text])[0]

    # Perform the search
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding.tolist()],
        anns_field="vector",
        param=search_params,
        limit=top_k
    )

    # Extract IDs from search results
    ids = [hit.id for hit in results[0]]
    print(f"Retrieved IDs: {ids}")

    # Fetch entities by IDs
    if ids:
        # Assuming 'text' is the field you want to retrieve
        # You might need to adjust this based on your collection schema
        df = collection.query(
            expr=f"id in {ids}",
            output_fields=["content"]
        )
        passages = [{'content': item['content']} for item in df]
    else:
        passages = []

    return passages

def generate_answer(question, passages):
    context = " ".join(passage['content'] for passage in passages)  # Adjust 'content' if necessary
    input_text = f"{question} {context}"

    # Encode the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = rag_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=5,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def answer_question(question):
    passages = retrieve_relevant_embeddings(question)
    answer = generate_answer(question, passages)
    return answer

# Example usage
if __name__ == "__main__":
    question = "What are the main environmental impacts discussed in the report?"
    answer = answer_question(question)
    print(answer)
