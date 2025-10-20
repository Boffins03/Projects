from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

# Load Google Gemma (2B) - instruction tuned
model_name = "google/gemma-2b-it"

print("⏳ Loading model, this may take a few minutes...")

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Create a text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9
)

# Wrap with LangChain
local_llm = HuggingFacePipeline(pipeline=pipe)

# Test the model
response = local_llm("Explain what LangChain is in simple terms.")
print("\n🤖 Model response:\n", response)
