from langchain.chains import LLMChain
from langchain_together import Together
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Function to load the model
def load_model(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"):
    llm = Together(
        model=model_name,
        temperature=0.7,
        max_tokens=128,
        top_k=1,
        together_api_key=""  # use your together API key
    )
    return llm

# System prompt template
def system_prompt():
    system_template = "You are an AI assistant."
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    return system_message_prompt

# Human prompt template
def human_prompt(text):
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    return human_message_prompt.format(text=text)

# Chat prompt template
def create_chat_prompt(text):
    system_query = system_prompt()
    human_query = human_prompt(text)
    chat_prompt = ChatPromptTemplate.from_messages([system_query, human_query])
    return chat_prompt

