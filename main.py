import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # Load environment variables from .env file
api_key = os.getenv("OPENAI_API_KEY")  # Retrieve the API key from environment variables
client = OpenAI(api_key=api_key)  # Initialize OpenAI client with API key
system_prompt = "You are a helpful teacher assistant. " \
"your responsibilities Explain concepts clearly and provide helpful guidance." \
"Use examples and analogies to enhance understanding." \
"Use simple language when possible. " \
"if not sure say so instead of inventing information"  # Define the system prompt for the model
connversation_history = []  # Initialize an empty list to store conversation history
while True:  # Start an infinite loop to continuously prompt the user for questions
    question = input("you: ").strip()
    if question.lower() in {"quit", "exit"}:  # Check if the user wants to quit
        break  # Exit the loop
    if not question:  # Check if the question is empty
        print("Please enter a valid question.")  # Prompt the user to enter a valid question
        continue  # Continue to the next iteration of the loop
    connversation_history.append({"role": "user", "content": question})  # Add the user's question to the conversation history
    response = client.responses.create(
     model="gpt-5-mini",  # Specify the model to use0
     instructions=system_prompt,  # Provide instructions for the model
     input=connversation_history  # Provide the user's question as input to the model
 )
    answer = response.output_text  # Extract the model's response text
    print(f"assistant: {answer}")  # Print the model's response
    connversation_history.append({"role": "assistant", "content": answer})  # Add the model's response to the conversation history
print(response.output_text)  # Print the response from the model