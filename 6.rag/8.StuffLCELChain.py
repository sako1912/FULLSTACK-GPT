from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

llm = ChatOpenAI(
    temperature=0.1,
)

cache_dir = LocalFileStore("./cache")

splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100
)

loader = UnstructuredFileLoader("./6.rag/Chapter 01.txt")

docs = loader.load_and_split(text_splitter=splitter)

embeddings = OpenAIEmbeddings()

#embedding을 재사용 하기 위해 cache에 저장 그렇지 않ㅇ면 매번 embedding을 다시 실행 (비용 발생)
cache_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embeddings,
    cache_dir
)

#embedding된 data(백터화)를 vectore store에 저장
vectorestore = FAISS.from_documents(docs, cache_embeddings)

retriver = vectorestore.as_retriever()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer questions using only the following context. If you don't know the answer just say you don't know, don't make it up:\n\n{context}"),
        ("human", "{question}")
    ]
)

#RunnablePassthrough 간단하게 설명하면 입력값을 말그대로 통과시켜주는 역할 ( "Describe Victory Mansions"을 넣어줌.)
chain = ({"context":retriver, "question": RunnablePassthrough()} | prompt | llm)

chain.invoke("Describe Victory Mansions")
