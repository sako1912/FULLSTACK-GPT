from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("[INST]What is the meaning of {word}[/INST]")

llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model_kwargs={
        "max_new_tokens": 250,
    },
)
#Cannot override task for LLM models 에러 발생시 endpoint 변경
llm.client.api_url = 'https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3'

chain = prompt | llm

chain.invoke({"word": "potato"})