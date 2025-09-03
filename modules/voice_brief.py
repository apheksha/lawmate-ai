from langchain import OpenAI
from modules.utils import speak_text

def generate_voice_brief(notes_text):
    prompt = f"Convert these notes into a structured legal brief:\n{notes_text}"
    llm = OpenAI(temperature=0)  # Requires free OPENAI_API_KEY
    brief = llm(prompt)
    speak_text(brief)
    return brief
