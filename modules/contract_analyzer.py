from transformers import pipeline
from modules.utils import generate_hash, cache, CACHE_FILE, pickle

generator = pipeline(
    "text2text-generation",
    model="mrm8488/T5-base-finetuned-cuad",
    device=-1
)

RISK_KEYWORDS = {
    "penalty": "High",
    "terminate": "High",
    "shall": "Medium",
    "may": "Low",
    "without consent": "High",
}

def analyze_contract(text, question):
    key = generate_hash(text, question)
    if key in cache:
        return cache[key]
    prompt = f"Text: {text}\nQuestion: {question}\nHighlight obligations, risks, penalties."
    result = generator(prompt, max_length=400)
    answer = result[0]['generated_text']
    cache[key] = answer
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    return answer

def risk_scoring(clause):
    for word, score in RISK_KEYWORDS.items():
        if word in clause.lower():
            return score
    return "Low"

def color_code_clause(clause, score):
    color = {"High": "red", "Medium": "orange", "Low": "green"}
    return f"<span style='color:{color[score]}'>{clause}</span>"
