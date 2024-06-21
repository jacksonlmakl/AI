from flask import Flask, request, jsonify
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import Runnable
import json
import os
import threading
import time
from dotenv import load_dotenv
import os 

app = Flask(__name__)
print("APP LOADED")

# Initialize the Hugging Face model
hf_token = os.getenv('HUGGINGFACE_API_KEY')

# model_name = "meta-llama/Meta-Llama-3-8B"  # You can replace this with any other model
model_name = "meta-llama/Meta-Llama-3-8B"  # You can replace this with any other model
model_pipeline = pipeline("text-generation", model=model_name, token=hf_token)

# Initialize LangChain components
memory = ChatMessageHistory()
print("MEMORY LOADED")
prompt_template = PromptTemplate(template="Answer the following question based on the context: {input}")

# Create a chain using Runnable
class CustomChain(Runnable):
    def __init__(self, model_pipeline, prompt_template, memory):
        self.model_pipeline = model_pipeline
        self.prompt_template = prompt_template
        self.memory = memory

    def invoke(self, input):
        # Use memory to load context
        context = [message.content for message in self.memory.messages]
        full_prompt = self.prompt_template.format(input=input, context=context)
        response = self.model_pipeline(full_prompt, max_length=50, num_return_sequences=1)
        self.memory.add_user_message(input)
        self.memory.add_ai_message(response[0]['generated_text'])
        return response[0]['generated_text']

chain = CustomChain(model_pipeline=model_pipeline, prompt_template=prompt_template, memory=memory)

# Load memory from cache if it exists
cache_file = "memory_cache.json"
if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
        cached_memory = json.load(f)
        memory.messages = [ChatMessageHistory.from_dict(msg) for msg in cached_memory]

# Function to save memory periodically
def save_memory_periodically():
    while True:
        with open(cache_file, "w") as f:
            json.dump([msg.to_dict() for msg in memory.messages], f)
        time.sleep(60)  # Save every 60 seconds

# Start the background thread to save memory periodically
thread = threading.Thread(target=save_memory_periodically)
thread.daemon = True  # Daemonize thread to ensure it exits when the main program does
thread.start()

@app.route('/api/prompt', methods=['POST'])
def handle_prompt():
    data = request.json
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    response = chain.invoke(prompt)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
