from flask import Flask, render_template, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("startup-model")
tokenizer = T5Tokenizer.from_pretrained("startup-model")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate-report", methods=["POST"])
def generate_report():
    data = request.get_json()

    # Build input prompt
    prompt = f"""Generate a complete startup market report including:
- Market Analysis
- Competitor Analysis
- Market Gaps
- Validation Score

Startup:
Title: {data['title']}
Domain: {data['domain']}
Description: {data['description']}
Problem: {data['problem']}
Solution: {data['solution']}
Target Market: {data['target_market']}"""

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_new_tokens=300)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"report": decoded_output})

if __name__ == "__main__":
    app.run(debug=True)
