from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def run_local_gemma_with_langchain():
    try:
        # 1. Initialize the Ollama LLM
        # Point LangChain to your local Ollama instance and the model you pulled
        llm = OllamaLLM(model="gemma3:1b") # Use the model name as pulled from Ollama

        # 2. Create a Prompt Template
        # This helps structure your input to the model
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant that answers questions concisely."),
            ("user", "{question}")
        ])

        # 3. Create an Output Parser
        # This will simply extract the string content from the model's response
        output_parser = StrOutputParser()

        # 4. Create a LangChain Chain
        # This combines the prompt, LLM, and output parser into a single runnable sequence
        chain = prompt | llm | output_parser

        # 5. Invoke the chain with a question
        question1 = "why are you so difficult to trian on local and why you need so much GPU?"
        print(f"Question 1: {question1}")
        response1 = chain.invoke({"question": question1})
        print(f"AI's Response: {response1}")

        print("\n--- Second Interaction ---")
        question2 = "Tell me a fun fact about him."
        # For follow-up questions, in a real chat scenario, you'd want to pass previous messages
        # However, for simple single-turn interactions, you just invoke the chain again.
        # If you need memory/chat history, you'd use LangChain's conversational retrieval chain.
        response2 = chain.invoke({"question": question2})
        print(f"AI's Response: {response2}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure Ollama is running and you have pulled the 'gemma:2b' model.")

if __name__ == "__main__":
    run_local_gemma_with_langchain()