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

from custom_model import CustomModelClient

# Set up the embedder (free model for embedding)

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")
collection_name = "google_2022"
collection = Collection(name=collection_name)

# Configuration
# config_list = [
#     {"model": "gpt-3.5-turbo"},
# ]
llm_config_local = {"config_list": [{
    "model": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
    "base_url": "http://172.23.192.1:1235/v1",
    "api_key": "lm-studio"
}]}

# Custom non gpt model FREE
config_list_custom = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={"model_client_cls": ["CustomModelClient"]},
)

llm_config = {
    "timeout": 600,
    "cache_seed": 42,
    "config_list":  llm_config_local["config_list"],
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


question_asker_env = autogen.AssistantAgent(
    name="Environmenalist",
    system_message=""""
      You are an expert environmentalist and care about the environemntal impacts of the big companies.  Your role is to analyze the Google Environmental Report 2022 and provide good environmental questions to the Google assistant.
    """,
    llm_config=llm_config,
    is_termination_msg=termination_msg,
    human_input_mode="NEVER"

)

google_env_expert = autogen.AssistantAgent(
    name="Google-expert",
    system_message="""You are an expert at Google who has deep knowledge of environmental reports. Your role is to analyze the Google Environmental Report 2022 and provide insights and recommendations based on the report's content. Reply `TERMINATE` in the end when everything is done.""",
    llm_config=llm_config,
    is_termination_msg=termination_msg,
    human_input_mode="NEVER"
)


google_agents_aid = RetrieveUserProxyAgent(
    name="Env_Report_Assistant",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    default_auto_reply="Reply `TERMINATE` if the task is done.",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "qa",  # Ensure this task is implemented
        "docs_path": "./google_data",
        "chunk_token_size": 10000,  # Keep this high to handle large docs
        "model": llm_config_local["config_list"][0]["model"],  # Changed model
        "client": "milvus",
        "collection_name": "google_2022",
        "get_or_create": True,
        "max_tokens": 2048,  # Increase this value to handle large chunks
        "must_break_at_empty_line": False,  # Ensure this is False
    },
    code_execution_config=False,
    description="Assistant with extra content retrieval power for solving environmental report problems.",
)


def termination_message(msg):
    return "TERMINATE" in str(msg.get("content", ""))


def _reset_agents():
    question_asker_env.reset()


def rag_chat():
    print("here")
    _reset_agents()
    groupchat = autogen.GroupChat(
        agents=[question_asker_env, google_env_expert],
        speaker_selection_method="round_robin",
        messages=[],
        max_round=12
    )
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        code_execution_config={"use_docker": False},
        is_termination_msg=termination_message
    )
    # Start chatting with boss_aid as this is the user proxy agent.
    manager.initiate_chat(
        question_asker_env,
        message="Environmenalist please start the chat by asking a single question",
        max_turns=12
    )


# Run the chat
rag_chat()
