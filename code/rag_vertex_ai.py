# ------------------------------
# Simple RAG example using GCP Vertex AI
# ------------------------------

from google import genai
from google.genai import types

def main():
    # 1️⃣ Create a client to talk to Google Vertex AI
    # No need for api_key if you’re using a service account or running in GCP
    client = genai.Client(vertexai=True)

    # 2️⃣ Ask the user for a question
    question = input("\nEnter your question: ")

    # 3️⃣ Choose the model to use (Gemini family)
    model_name = "gemini-2.5-flash-preview-09-2025"

    # “contents” is what we send to the model
    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=question)]
        )
    ]

    # 4️⃣ Connect your RAG (Retrieval-Augmented Generation) to vertex AI information retrieval module
    # We have developed information retrival module in Google cloud Vertex AI
     
    rag_tools = [
        types.Tool(
            retrieval=types.Retrieval(
                vertex_rag_store=types.VertexRagStore(
                    rag_resources=[
                        types.VertexRagStoreRagResource(
                            rag_corpus="projects/18551258701/locations/asia-south1/ragCorpora/6917529027641081856"
                        )
                    ]
                )
            )
        )
    ]

    # 5️⃣ Configure how the model should generate answers
    config = types.GenerateContentConfig(
        temperature=0.7,       # Lower = factual, Higher = creative
        max_output_tokens=1024, # Max length of the answer
        tools=rag_tools,        # Connects RAG store
    )

    print("\n🤖 Generating answer...\n")

    # 6️⃣ Send request and stream response as it comes
    for chunk in client.models.generate_content_stream(
        model=model_name,
        contents=contents,
        config=config,
    ):
        if chunk.candidates and chunk.candidates[0].content.parts:
            print(chunk.text, end="", flush=True)

    print("\n\n✅ Done!")

# Run the program
if __name__ == "__main__":
    main()
