from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

tokenizer = AutoTokenizer.from_pretrained("Ateeqq/product-description-generator")
model = AutoModelForSeq2SeqLM.from_pretrained("Ateeqq/product-description-generator")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_description():
    data = request.json
    product_name = data.get('product_name', '')
    if not product_name:
        return jsonify({"error": "Please provide a product_name in the request"}), 400
    
    input_ids = tokenizer(f'description: {product_name}', return_tensors="pt", padding="longest", truncation=True, max_length=128)
    outputs = model.generate(
        input_ids['input_ids'],
        num_beams=5,
        num_beam_groups=5,
        num_return_sequences=5,
        repetition_penalty=10.0,
        diversity_penalty=3.0,
        no_repeat_ngram_size=2,
        temperature=0.7,
        max_length=128
    )
    
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"description": description})

if __name__ == '__main__':
    app.run(port=620, debug=True)
