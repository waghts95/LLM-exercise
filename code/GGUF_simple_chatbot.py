############################ GGUF Model ###############################
# GGUF is a compact format for storing AI models
# Think of GGUF as a ZIP file for AI models - it compresses huge models to run on your laptop
# This GGUF model : llama3.1:8b-instruct-q4_K_M
# This q4_K_M = 4-bit K-quant Medium (GGUF quantization type)
#####################################################################

## ollama is running through docker in background

import ollama

messages = []

print("Chatbot started. Type 'bye' to exit.\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ['bye', 'stop', 'exit']:
        print("Goodbye!")
        break
    
    messages.append({'role': 'user', 'content': user_input})
    
    response = ollama.chat(model='llama3.1:8b-instruct-q4_K_M', messages=messages)
    
    assistant_reply = response['message']['content']
    messages.append({'role': 'assistant', 'content': assistant_reply})
    
    print(f"Bot: {assistant_reply}\n")