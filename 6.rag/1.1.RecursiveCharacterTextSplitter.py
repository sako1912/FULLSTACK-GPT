from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#파일 분할 
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, #해당 글자 수 단위로 분할, 해당 옵션만 주게 되면 문장 중간이 잘리는 경우가 발생
    chunk_overlap=50 #해당 옵션을 주게 되면 문장 중간이 잘리는 경우를 방지할 수 있음 - 앞부분의 일부를 가져옴 (중복)
)

loader = UnstructuredFileLoader("./6.rag/Chapter 01.txt")

#방식 1
#docs = loader.load()
#splitter.split_documents(docs)

#방식 2
loader.load_and_split(text_splitter=splitter)

#spilt 길이 확인
#len(loader.load_and_split(text_splitter=splitter))
