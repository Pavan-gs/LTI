# =====================================================================
# HYBRID RAG (FAISS + BM25) + TOOL SUPPORT + MULTI-USER MEMORY (MISTRAL)
# =====================================================================
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import ConversationalRetrievalChain

from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import re
import json

# 1. LOAD CSV
loader = CSVLoader("incidents.csv")
docs = loader.load()

# 2. SPLIT DOCUMENTS
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
chunks = splitter.split_documents(docs)

# 3. EMBEDDINGS (MiniLM)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. VECTOR STORE (FAISS)
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("faiss_store")
faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# 5. BM25 RETRIEVER
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4

# 6. HYBRID RETRIEVER
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.4, 0.6]
)

# 7. LLM (MISTRAL)
llm = ChatOllama(model="mistral", temperature=0.7)

# TOOL
@tool
def kb_lookup(query: str) -> str:
    """Lookup internal knowledge base for fixes."""
    fixes = {
        "outlook crash": "Restart Outlook → Repair Office → Delete corrupt OST.",
        "printer issue": "Reinstall drivers and restart print spooler.",
    }
    for key in fixes:
        if key in query.lower():
            return fixes[key]
    return "No KB entry found."

llm_tools = llm.bind_tools([kb_lookup])

# ENTITY EXTRACTION
def extract_entities(text: str):
    entities = {}
    entities["ips"] = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text)
    entities["mitre"] = re.findall(r'T\d{4}', text)
    entities["os"] = re.findall(r'Windows|Linux|macOS', text)
    entities["severity"] = re.findall(r'Low|Medium|High|Critical', text)
    host_match = re.findall(r'Hostname[:\s]+(\S+)', text)
    if host_match:
        entities["hostname"] = host_match
    return entities

# THREAT SCORE
def compute_threat_score(entities):
    score = 0
    if entities.get("mitre"):
        score += 2 * len(entities["mitre"])
    if entities.get("severity"):
        sev_map = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
        score += sum(sev_map.get(sev, 0) for sev in entities["severity"])
    if entities.get("ips"):
        score += 2
    if "powershell" in str(entities).lower():
        score += 2
    if "brute-force" in str(entities).lower():
        score += 2
    return min(score, 10)

# 8. PROMPT
prompt_template = """
You are a “SOC Analyst Assistant – Incident RAG Investigation System”

Your response MUST ALWAYS follow this exact format:

From your past: <summarize relevant history or say 'No previous issues found'>.
Suggested: <the fix/solution in one short sentence>.
Entities: {entities}
Threat Score: {threat_score}

Use retrieved context + chat memory + tools when needed.
If unclear, infer from patterns.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}
"""

prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question", "entities", "threat_score"],
    template=prompt_template
)

# 9. BASE RAG CHAIN
base_rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm_tools,
    retriever=hybrid_retriever,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# 10. MEMORY
user_memory_store = {}
def get_user_memory(user_id: str):
    if user_id not in user_memory_store:
        user_memory_store[user_id] = InMemoryChatMessageHistory()
    return user_memory_store[user_id]

rag_chain = RunnableWithMessageHistory(
    base_rag_chain,
    get_user_memory,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# 11. CONSOLE LOOP
print("Multi-User RAG Chat — enter user_id & question\n")

while True:
    user_id = input("\nEnter user_id: ").strip()
    if user_id.lower() in ["exit", "quit", "stop"]:
        break

    query = input("You: ")

    # Retrieve context
    retrieved_docs = hybrid_retriever.invoke(query)
    context_text = "\n".join([doc.page_content for doc in retrieved_docs])

    # Extract entities and score
    entities = extract_entities(query + context_text)
    threat_score = compute_threat_score(entities)

    # Invoke chain
    response = rag_chain.invoke(
        {
            "question": query,
            "entities": json.dumps(entities),
            "threat_score": threat_score
        },
        config={"configurable": {"session_id": user_id}}
    )

    # Structured output
    structured_output = {
        "user": user_id,
        "query": query,
        "entities": entities,
        "threat_score": threat_score,
        "response": response["answer"]
    }

    print("\n--- Structured Response ---")
    print(json.dumps(structured_output, indent=2))