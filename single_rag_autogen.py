# Import necessary libraries
import os
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection
import autogen
from autogen.agentchat import GroupChat, GroupChatManager
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from transformers import AutoTokenizer, AutoModel
from PyPDF2 import PdfReader
import fitz  # Ensure you have PyMuPDF installed

# Set up the embedder (free model for embedding)
load_dotenv(Path(".env"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["AUTOGEN_USE_DOCKER"] = "False"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")
collection_name = "google_2022"
collection = Collection(name=collection_name)

# Configuration
config_list = [
    {"model": "gpt-3.5-turbo"},
]

llm_config = {
    "timeout": 60,
    "cache_seed": 42,
    "config_list":  config_list,
    "temperature": 0,
}


# Define termination message function
def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

# Function to process all files in a folder
def process_docs_folder(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Path to the folder containing your documents
docs_folder_path = "./google_data"
docs_files = process_docs_folder(docs_folder_path)

# Function to extract text from PDF using PyPDF2 and PyMuPDF as a fallback
def extract_text_from_pdf(file_path):
    try:
        # Try PyPDF2
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"PyPDF2 failed, trying PyMuPDF: {e}")
        try:
            # Try PyMuPDF if PyPDF2 fails
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            print(f"PDF extraction failed: {e}")
            return ""

# Define the agents
boss = RetrieveUserProxyAgent(
    name="Env_Report_Analyzer",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    default_auto_reply="Reply `TERMINATE` if the task is done.",
    description="Analyzes environmental reports and asks questions related to environmental impact.",
)

boss_aid = RetrieveUserProxyAgent(
    name="Env_Report_Assistant",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    default_auto_reply="Reply `TERMINATE` if the task is done.",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "qa",  # Ensure this task is implemented
        "docs_path": "./google_data",
        "chunk_token_size": 10000,  # Keep this high to handle large docs
        "model": config_list[0]["model"],  # Changed model
        "client": "milvus",
        "collection_name": "google_2022",
        "get_or_create": True,
        "max_tokens": 2048,  # Increase this value to handle large chunks
        "must_break_at_empty_line": False,  # Ensure this is False
    },
    code_execution_config=False,
    description="Assistant with extra content retrieval power for solving environmental report problems.",
)

# Define the group chat manager and agents
def _reset_agents():
    boss.reset()
    boss_aid.reset()

def rag_chat():
    _reset_agents()
    groupchat = GroupChat(
        agents=[boss, boss_aid], messages=[], max_round=12, speaker_selection_method="round_robin"
    )
    manager = GroupChatManager(groupchat=groupchat)

    # Start chatting with boss_aid as this is the user proxy agent.
    boss_aid.initiate_chat(
        manager,
        message=boss_aid.message_generator,
        problem="Analyze the Google Environmental Report 2022 and provide insights into its environmental impact. Extract key data and suggest improvements.",
        n_results=3,
    )

# Run the chat
rag_chat()
