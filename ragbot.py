from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2", task="feature-extraction"
)

class IndexState(TypedDict):
    pdf_path: str
    output_dir: str


def create_vector(state: IndexState) -> FAISS:
    loader = PyPDFLoader(state["pdf_path"])
    pdf_file = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(pdf_file)

    db = FAISS.from_documents(documents= chunks, embedding=embeddings)
    db.save_local(state["output_dir"])
    return {}

def call_model(state: MessagesState):
    query = state["messages"][-1].content
    db = FAISS.load_local(r"./vectordb", embeddings=embeddings,
                           allow_dangerous_deserialization=True)
    docs = db.similarity_search(query=query, k=4)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(
        base_url="https://router.huggingface.co/v1",
        model="meta-llama/Llama-3.2-3B-Instruct:together",
        temperature=0.2, 
    )

    promt_template = ChatPromptTemplate.from_messages([
        ("system","You are an helpful assistant. Answer to the questions asked "
        "based on the given context. Question : {question}, context:{context}."
        "If the answer is not in the context just say I don't know!"),
        MessagesPlaceholder(variable_name="messages")
        ])
    
    prompt = promt_template.invoke({"question":query, "context": docs_page_content,
                                     "messages":state["messages"]})
    response = llm.invoke(prompt)
    return {"messages":[response]}

def index_pipeline():
    workflow1 = StateGraph(state_schema=IndexState)

    workflow1.add_node("vectorize", create_vector)
    workflow1.add_edge(START, "vectorize")
    workflow1.add_edge("vectorize", END)

    app1 = workflow1.compile()

    return app1


def chat_pipeline():
    workflow = StateGraph(state_schema=MessagesState)

    
    workflow.add_node("model", call_model)

    workflow.add_edge(START, "model")
    workflow.add_edge("model", END)

    app = workflow.compile(checkpointer=MemorySaver())
    return app



# indexpp = index_pipeline()
# vectordb = indexpp.invoke({"pdf_path":r".\Short Story.pdf", "output_dir":r".\vectordb"})

config = {"configurable": {"thread_id":"ragchat1"}}

query = "Do you remember my name and our past conversations?"
cp = chat_pipeline()
input_messages = [HumanMessage(query)]
output = cp.invoke({"messages":input_messages}, config)
output["messages"][-1].pretty_print()
