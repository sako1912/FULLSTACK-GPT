from langchain.embeddings import OpenAIEmbeddings

embedder = OpenAIEmbeddings()

#hi 표현하는 vector
#embedder.embed_query("hi!")

#여러 문장을 vector로 변환
vetcor = embedder.embed_documents([
    'hi!',
    'how',
    'are',
    'you',
])

#문장을 vector로 변환
#vetcor

#vector의 길이
#len(vetcor)

#vectore[0]가 가지고 있는 차원
print(len(vetcor[0]))
