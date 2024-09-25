from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


#callback - llm의 event를 listen하는 class
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
        
    #새로운 tocken이 생성될 때마다 box에 message에 추가
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)
        

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True, # Set to True for real-time chat
    callbacks=[
        ChatCallbackHandler(),
    ],
)

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

# 메모리가 처음 설정될 때만 로드 - streamlit은 전체 refresh 되기에 ConversationSummaryBufferMemory 객체를 session_state에 넣어줌
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=100,
        return_messages=True,
    )

# session_state 에서 memory 가져오기
memory = st.session_state.memory

# 메모리에 대화 내용 저장 함수
def save_memory(human_message, ai_message):
    memory.save_context({"input": human_message}, {"output": ai_message})

def save_message(message, role):#
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

    
def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages(
    [
        (
             "system",
            """
            You are an AI assistant, and your task is to answer the user's question. You must only use the information provided in the "context" to answer the question. 
            Do not use any information that is not in the context. 
            If the answer is not available in the context, say "I don't know." 
            Do NOT make up any information.

            The following is the "context" you should use to answer the question:
            Context: {context}

            Below is the conversation history between you and the user, so you can understand the flow of the conversation:
            History: {chat_history}

            Now, based on this information, answer the user's question carefully.
            """,

        ),
        ("human", "{question}"),
    ]
)

st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)
# 사이드바 파일 업로드 설정
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

# 체인을 실행하고 메모리에 대화 저장
def chain_invoke(message):
    response = chain.invoke(message);
    save_memory(message, response.content) 
    
#session(memory)에서 이전 대화 내용을 가져오는 함수
def load_memory(_):
    return memory.load_memory_variables({})["history"]

# 파일이 업로드된 경우 실행
if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | RunnablePassthrough.assign(chat_history=load_memory) #prompt에 이전 대화 내용을 넣기 위한 설정
            | prompt
            | llm
        )
        #AI 봇 영역에서 token을 바로 바로 보여주기 위해 설정
        with st.chat_message("ai"):
             chain_invoke(message)

else:
    st.session_state["messages"] = []
    
