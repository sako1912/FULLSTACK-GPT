from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

llm = ChatOpenAI(
    temperature=0.1,
)

cache_dir = LocalFileStore("./.cache/")

splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
)
loader = UnstructuredFileLoader("./files/chapter_one.txt")

docs = loader.load_and_split(text_splitter=splitter)

embeddings = OpenAIEmbeddings()

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

vectorstore = FAISS.from_documents(docs, cached_embeddings)

retriever = vectorstore.as_retriever()

#qustion에 관련 정보가 있으면 해당 부분을 반환 
map_doc_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use the following portion of a long document to see if any of the text is relevant to answer the question. Return any relevant text verbatim. If there is no relevant text, return : ''
            -------
            {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

#2. for doc in list of docs(이부분은 3번에 함수에서 같이 진행) | prompt | llm 
map_doc_chain = map_doc_prompt | llm

#3. for response in list of llm respone | put them all together
def map_docs(inputs):
    documents = inputs["documents"]
    question = inputs["question"]
    # inputs의 문서들과 질문을 담아 map_doc_chain을 호출하며 응답받은 content을 join으로 하나의 문서로 만듬
    # AI message이기에 content만 가져옴
    return "\n\n".join(
        map_doc_chain.invoke(
            {"context": doc.page_content, "question": question}
        ).content
        for doc in documents
    )


#  문서와 질문이 필요
# RunnableLambda : 내부 어디든 function 실행
map_chain = {
    "documents": retriever,
    "question": RunnablePassthrough(),
} | RunnableLambda(map_docs)

#주어진 document(final_doc) 발췌문들과 질문을 보고 최종 답변을 만들어라
# {context} = final_doc
final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Given the following extracted parts of a long document and a question, create a final answer. 
            If you don't know the answer, just say that you don't know. Don't try to make up an answer.
            ------
            {context} 
            """,
        ),
        ("human", "{question}"),
    ]
)
#4. final doc | promp | llm
chain = {"context": map_chain, "question": RunnablePassthrough()} | final_prompt | llm

chain.invoke("How many ministries are mentioned")