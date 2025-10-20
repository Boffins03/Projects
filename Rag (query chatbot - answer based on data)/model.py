from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# model_path = r"C:\Users\HP\.cache\huggingface\hub\models--google--gemma-2b-it\snapshots\96988410cbdaeb8d5093d1ebdc5a8fb563e02bad"
model_path = "google/gemma-2b-it"

def load_model(model_path=model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    device = 0 if torch.cuda.is_available() else -1
    
    llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    return HuggingFacePipeline(pipeline=llm_pipeline)



if __name__ == "__main__":
    llm = load_model()
    response = llm.invoke("Hello, how are you?")
    print(response)