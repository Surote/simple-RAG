# python3.13.3
# Author: Surote Wongpaiboon
# Testing
from flask import Flask, request, jsonify
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from bs4 import SoupStrainer
from datasets import load_dataset
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda


# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize the chat model
chat_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_retries=2,
)


os.environ["GUARDIAN_URL"] = "https://granite3-guardian-2b-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443"
GUARDIAN_URL = os.getenv('GUARDIAN_URL')
GUARDIAN_MODEL_NAME = "granite3-guardian-2b"
GUARDIAN_API_KEY = os.getenv('GUARDIAN_API_KEY')
# Initialize Guardian (Guardrails Model)
guardian = ChatOpenAI(
    openai_api_key=GUARDIAN_API_KEY,
    openai_api_base=f"{GUARDIAN_URL}/v1",
    model_name=GUARDIAN_MODEL_NAME,
    temperature=0.01,
    streaming=False,
)
# Define Tokens
SAFE_TOKEN = "No"
RISKY_TOKEN = "Yes"

def check_risk(user_query):
    """
    Step 1: Check risk using the Guardian model.
    Returns True (risky) or False (safe).
    """
    response = guardian.invoke(user_query)
    print(response)
    risk_label = response.content.strip().lower()

    return 'Cannot answer anything- guardrail' if risk_label == RISKY_TOKEN.lower() else user_query



# Set up the embeddings model using HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

# Load a dataset for additional use
dataset = load_dataset("flax-sentence-embeddings/stackexchange_math_jsonl", "title_answer")

# Create a loader to fetch content from a specified URL
web_loader = WebBaseLoader(
    web_paths=("https://www.couchbase.com/blog/what-is-vector-search/",),
    bs_kwargs=dict(parse_only=SoupStrainer())  # Parse specific parts of the HTML
)

# Load documents from the web
documents = web_loader.load()

# Initialize a text splitter to break documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

# Split documents into chunks
document_chunks = text_splitter.split_documents(documents)

# Create a vector store for document retrieval
vector_store = Chroma.from_documents(documents=document_chunks, embedding=embeddings, persist_directory="./chroma_langchain_db")

# Set up a retriever to get relevant document snippets
document_retriever = vector_store.as_retriever(k=3)

# Define a template for prompting the chat model
template = """Answer the question and explain the answer is coming from context or not and give me the answer you know:
{context}

Question: {messages}
"""

# Create a prompt using the template
chat_prompt = ChatPromptTemplate.from_template(template)

# Helper function to format document content
def format_documents(docs):
    return "\n\n".join(doc.page_content for doc in docs)

check_risk_runnable = RunnableLambda(lambda query: check_risk(query))
# Define a chain of processes for RAG (Retrieval-Augmented Generation)
rag_chain = (
    check_risk_runnable  # Step 1: Check risk
    | (  # Step 2: Conditional branching based on risk
        {
            "context": document_retriever | format_documents,
            "messages": RunnablePassthrough(),
        }
        | chat_prompt
        | chat_model
        | StrOutputParser()
    )
)

# Define the API endpoint
@app.route('/query', methods=['POST'])
def query():
    try:
        # Get the user input from the request
        user_input = request.json.get('question', '')
        if not user_input:
            return jsonify({"error": "Question is required"}), 400
        
        # Process the input through the RAG chain
        response = rag_chain.invoke(user_input)
        
        # Extract token usage (if supported by the model)
       # token_usage = chat_model.get_last_token_usage()  # Hypothetical method to get token usage
        
        # Return the response as JSON
        return jsonify({
            "response": response,
            #"tokens_used": token_usage
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)