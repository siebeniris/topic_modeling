import spacy 

nlp = spacy.load("en_core_web_sm")

text = "Translating the text from American English to British English"

doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)