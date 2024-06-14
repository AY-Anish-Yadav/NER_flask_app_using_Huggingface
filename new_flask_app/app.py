from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

app = Flask(__name__)

# Load the NER model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mdarhri00/named-entity-recognition")
model = AutoModelForTokenClassification.from_pretrained("mdarhri00/named-entity-recognition")
label_list = model.config.id2label

def get_entities(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    print(f"Tokenized inputs: {inputs}")

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs).logits
    print(f"Model outputs: {outputs}")

    # Get the predicted labels
    predictions = torch.argmax(outputs, dim=2)
    print(f"Predictions: {predictions}")

    entities = []
    for token, label_id in zip(inputs["input_ids"][0], predictions[0]):
        label = label_list[label_id.item()]
        if label != "O":  # O is for tokens with no entity
            entity = tokenizer.decode(token).strip()
            entities.append({"word": entity, "entity": label})
    return entities

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/extract_entities', methods=['POST'])
def extract_entities():
    text = request.form['text']
    print(f"Received text: {text}")  # Debugging statement
    entities = get_entities(text)
    print(f"Extracted entities: {entities}")  # Debugging statement
    print(entities)
    return render_template('index.html', entities=entities, text=text)

if __name__ == '__main__':
    app.run(debug=True)
