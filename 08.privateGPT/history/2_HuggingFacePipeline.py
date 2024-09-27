from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("A {word} is a")

llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation", #해당 모델이 어떤 일을 하길 원하는지 명시
    pipeline_kwargs={"max_new_tokens": 150}, #출력 문자열 길이 제한
    #device=0 #0 GPU 사용(mac pro사용 불가), -1 CPU 사용
)

chain = prompt | llm

chain.invoke({"word": "tomato"})