import torch
from transformers import RagTokenizer, RagTokenForGeneration
from pymilvus import connections, Collection


def connect_to_milvus():
    connections.connect(
        alias="default",
        uri="http://localhost:19530"
    )


def load_collection(collection_name):
    return Collection(collection_name)


def load_model_and_tokenizer():
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")
    return tokenizer, model


class RAGAgent:
    def __init__(self, collection, tokenizer, model, vector_dim):
        self.collection = collection
        self.tokenizer = tokenizer
        self.model = model
        self.vector_dim = vector_dim  # Store vector dimension

    def retrieve(self, query):
        # Perform search in Milvus
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        query_inputs = self.tokenizer(query, return_tensors='pt')
        vectors = query_inputs['input_ids']

        # Ensure vectors are of correct dimension
        if vectors.size(1) != self.vector_dim:
            raise ValueError(f"Query vector dimension mismatch: expected {self.vector_dim}, got {vectors.size(1)}")

        search_results = self.collection.search(
            vectors.tolist(),
            "vector",
            search_params,
            limit=5
        )
        docs = [hit.entity.get('content') for result in search_results for hit in result]
        return docs

    def generate(self, query, docs):
        # Generate a response using the retrieved documents
        inputs = self.tokenizer(query, return_tensors="pt")
        doc_inputs = self.tokenizer(docs, return_tensors="pt", padding=True, truncation=True)

        outputs = self.model.generate(
            input_ids=inputs['input_ids'],
            context_input_ids=doc_inputs['input_ids'],
            context_attention_mask=doc_inputs['attention_mask']
        )
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return response

    def ask(self, query):
        # Retrieve documents and generate an answer
        docs = self.retrieve(query)
        answer = self.generate(query, docs)
        return answer


if __name__ == "__main__":
    # Connect to Milvus
    connect_to_milvus()

    # Load the collection
    collection_name = "google_2022"
    collection = load_collection(collection_name)

    # Load the tokenizer and model
    tokenizer, model = load_model_and_tokenizer()

    # Get the vector dimension used in the collection schema
    vector_dim = 1536  # Ensure this matches the dimension of your vectors

    # Instantiate the RAG agent
    rag_agent = RAGAgent(collection, tokenizer, model, vector_dim)

    # Ask a question
    question = "What is the environmental impact of Google's data centers?"
    answer = rag_agent.ask(question)
    print(f"Question: {question}\nAnswer: {answer}")
