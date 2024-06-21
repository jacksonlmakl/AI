from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, LlamaForCausalLM
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import Runnable
import json
import os
import threading
import time
import faiss
import numpy as np
import sqlite3

app = Flask(__name__)
print("APP LOADED")

# Initialize the Hugging Face model
hf_token = os.getenv('HUGGINGFACE_API_KEY')
model_name = "meta-llama/Meta-Llama-3-8B"
# model_name = "meta-llama/Llama-2-7B-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token  # Add padding token
model = LlamaForCausalLM.from_pretrained(model_name, token=hf_token)
model_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Initialize LangChain components
memory = ChatMessageHistory()
print("MEMORY LOADED")
prompt_template = PromptTemplate(template="Answer the following prompt based on the context: {input}")

# Initialize FAISS index
d = model.config.hidden_size  # dimension of the model embeddings
index = faiss.IndexFlatL2(d)
print("INITIALIZED FAISS INDEX")

# Initialize SQLite database
def initialize_db(db_path="embeddings.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vector BLOB
        )
    """)
    conn.commit()
    return conn

def store_embeddings(conn, embeddings):
    cursor = conn.cursor()
    for embedding in embeddings:
        cursor.execute("INSERT INTO embeddings (vector) VALUES (?)", (embedding.tobytes(),))
    conn.commit()

# Load local information files and convert to embeddings
def load_information_files(directory="information"):
    info_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as file:
                info_data.append(file.read())
    return info_data

def load_embeddings(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT vector FROM embeddings")
    rows = cursor.fetchall()
    embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in rows]
    return np.vstack(embeddings) if embeddings else np.array([])

# Load or calculate embeddings
info_data = load_information_files()
conn = initialize_db()

embeddings = load_embeddings(conn)
if len(embeddings) == 0 and info_data:  # Calculate and save embeddings if not already saved
    embeddings = [model(**tokenizer(data, return_tensors="pt", padding=True)).last_hidden_state.mean(dim=1).detach().numpy() for data in info_data]
    embeddings = np.vstack(embeddings)
    store_embeddings(conn, embeddings)

if len(embeddings) > 0:  # Add embeddings to FAISS index if they exist
    index.add(embeddings)

# Function to retrieve relevant context using FAISS
def retrieve_context(query):
    query_embedding = model(**tokenizer(query, return_tensors="pt", padding=True)).last_hidden_state.mean(dim=1).detach().numpy()
    D, I = index.search(query_embedding, k=5)  # Retrieve top 5 relevant contexts
    return [info_data[i] for i in I[0]]

# Function to save memory cache to long-term storage
def save_memory_to_long_term(memory_cache, file_path):
    with open(file_path, 'w') as file:
        for msg in memory_cache:
            file.write(f"User: {msg['user']}\n")
            file.write(f"AI: {msg['ai']}\n\n")

# Function to check and save cache if it exceeds 100MB
def check_and_save_cache():
    cache_file = "memory_cache.json"
    max_size = 100 * 1024 * 1024  # 100MB

    if os.path.exists(cache_file) and os.path.getsize(cache_file) > max_size:
        with open(cache_file, 'r') as f:
            memory_cache = json.load(f)
        
        # Save to long-term memory
        long_term_file = f"information/memory_{int(time.time())}.txt"
        save_memory_to_long_term(memory_cache, long_term_file)
        
        # Clear memory cache
        with open(cache_file, 'w') as f:
            json.dump([], f)

# Create a chain using Runnable
class CustomChain(Runnable):
    def __init__(self, model_pipeline, prompt_template, memory, index):
        self.model_pipeline = model_pipeline
        self.prompt_template = prompt_template
        self.memory = memory
        self.index = index

    def invoke(self, input):
        # Use memory to load context
        context = [message.content for message in self.memory.messages]
        retrieved_contexts = retrieve_context(input)
        combined_context = ' '.join(context + retrieved_contexts)
        full_prompt = self.prompt_template.format(input=input, context=combined_context)
        response = self.model_pipeline(full_prompt, max_length=50, num_return_sequences=1)
        self.memory.add_user_message(input)
        self.memory.add_ai_message(response[0]['generated_text'])
        return response[0]['generated_text']

chain = CustomChain(model_pipeline=model_pipeline, prompt_template=prompt_template, memory=memory, index=index)

# Load memory from cache if it exists
cache_file = "memory_cache.json"
if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
        cached_memory = json.load(f)
        memory.messages = [ChatMessageHistory.from_dict(msg) for msg in cached_memory]

# Function to save memory periodically
def save_memory_periodically():
    while True:
        check_and_save_cache()
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
