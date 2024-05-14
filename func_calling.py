from functions import load_model, create_chat_prompt
from langchain.chains import LLMChain

# Example usage
text = "Tell me about the Transformers architecture."  # This could be any other text you want to pass
chat_prompt = create_chat_prompt(text)
llm = load_model()

chain = LLMChain(llm=llm, prompt=chat_prompt, verbose=True)
response = chain.run(text=text)  # Passing the text parameter

print("LLM Generated Answer:", response)
