from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from pydantic import BaseModel
from typing import List, Tuple, Union
from transformers import AutoTokenizer, AutoModel
import torch
import os
from dotenv import load_dotenv

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

# ----------------------------
# Hugging Face Embedding Wrapper (LangChain Compatible)
# ----------------------------
class HuggingFaceEmbeddingFunction(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

# ----------------------------
# Pydantic Response Schemas
# ----------------------------
class SQLResponse(BaseModel):
    sql_query: str
    explanation: str

class FollowUpResponse(BaseModel):
    explanation: str

class GeneralResponse(BaseModel):
    explanation: str

# ----------------------------
# Google Gemini LLM (Deterministic)
# ----------------------------
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro-latest",
    temperature=0.0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ----------------------------
# Output Parsers for Structured Responses
# ----------------------------
sql_parser = PydanticOutputParser(pydantic_object=SQLResponse)
followup_parser = PydanticOutputParser(pydantic_object=FollowUpResponse)
general_parser = PydanticOutputParser(pydantic_object=GeneralResponse)

# ----------------------------
# Chroma with Hugging Face Embeddings
# ----------------------------
embedding_function = HuggingFaceEmbeddingFunction()
chroma = Chroma(
    persist_directory="./chroma_northwind_full_hf/",
    embedding_function=embedding_function
)
retriever = chroma.as_retriever()

# ----------------------------
# Prompt Generators
# ----------------------------
def classify_question_prompt(question: str) -> str:
    return f"""Classify the user's intent as exactly one of the following categories:
- sql
- followup
- general

User input: "{question.strip()}"
Only respond with one word: sql, followup, or general."""

def generate_sql_prompt(question: str, tables: List[str]) -> str:
    tables_text = ", ".join(t.strip() for t in tables)
    format_instructions = sql_parser.get_format_instructions()
    return f"""You are an expert SQL assistant. The user asked a question about the following table(s): {tables_text}.
Return a valid SQL query and a brief explanation.

{format_instructions}

User question: {question.strip()}"""

def generate_followup_prompt(question: str, context: str, chat_history: str) -> str:
    format_instructions = followup_parser.get_format_instructions()
    return f"""You are a helpful assistant continuing a previous conversation.

Chat history:
{chat_history.strip()}

Relevant context:
{context.strip()}

User follow-up question: {question.strip()}
{format_instructions}"""

def generate_general_prompt(question: str) -> str:
    format_instructions = general_parser.get_format_instructions()
    return f"""You are a knowledgeable assistant. Provide a clear explanation.

User question: {question.strip()}
{format_instructions}"""

# ----------------------------
# Inference Logic
# ----------------------------
def generate_sql_or_followup(
    question: str,
    selected_tables: List[str],
    chat_history: List[str]
) -> Tuple[str, Union[SQLResponse, FollowUpResponse, GeneralResponse, str]]:

    try:
        classification = llm.invoke(classify_question_prompt(question)).content.strip().lower()

        if classification == "sql":
            sql_prompt = generate_sql_prompt(question, selected_tables)
            output = llm.invoke(sql_prompt)
            return "sql", sql_parser.parse(output.content.strip())

        elif classification == "followup":
            relevant_docs = retriever.invoke(question.strip())
            rag_context = "\n".join(doc.page_content for doc in relevant_docs)
            chat_hist = "\n".join([f"User: {msg.strip()}" for msg in chat_history])
            followup_prompt = generate_followup_prompt(question, rag_context, chat_hist)
            output = llm.invoke(followup_prompt)
            return "followup", followup_parser.parse(output.content.strip())

        elif classification == "general":
            general_prompt = generate_general_prompt(question)
            output = llm.invoke(general_prompt)
            return "general", general_parser.parse(output.content.strip())

        else:
            return "error", f"❌ Invalid classification: {classification}"

    except Exception as e:
        return "error", f"❌ Failed: {str(e)}"