
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma #vectorstore로 사용
from langchain.storage import LocalFileStore
from langchain.chains import RetrievalQA

llm = ChatOpenAI()

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
vectorestore = Chroma.from_documents(docs, cache_embeddings)

chain = RetrievalQA.from_chain_type(llm,
                    chain_type="stuff", 
                    retriever=vectorestore.as_retriever() #vectoreStore 뿐만 아니라 DB나 Cloud에서도 조회가능
                    )

chain.run("where dose Winston live?")