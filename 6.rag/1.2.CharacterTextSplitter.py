from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter

#파일 분할 
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100
)

loader = UnstructuredFileLoader("./files/chapter_one.txt")

#방식 1
#docs = loader.load()
#splitter.split_documents(docs)

#방식 2
#loader.load_and_split(text_splitter=splitter)

#spilt 길이 확인
len(loader.load_and_split(text_splitter=splitter))
