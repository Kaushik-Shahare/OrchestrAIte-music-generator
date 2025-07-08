import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash"
    # google_api_key=os.environ["GOOGLE_API_KEY"]
)

def gemini_generate(prompt: str) -> str:
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response) 