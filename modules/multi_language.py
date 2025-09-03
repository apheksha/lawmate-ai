
from deep_translator import GoogleTranslator

# translation example
if language != "English":
    translated_answer = GoogleTranslator(source='auto', target=language.lower()).translate(raw_answer)
else:
    translated_answer = raw_answer
