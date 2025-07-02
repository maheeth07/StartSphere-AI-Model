import json

# Load your dataset file (ensure it's named 'dataset.txt')
with open("dataset.txt", "r", encoding="utf-8") as f:
    startups = json.load(f)

# Prepare dataset.jsonl content
with open("dataset.jsonl", "w", encoding="utf-8") as f_out:
    for entry in startups:
        input_text = (
            f"Generate market report for a startup:\n"
            f"Title: {entry['title']}\n"
            f"Domain: {entry['domain']}\n"
            f"Description: {entry['description']}\n"
            f"Problem: {entry['problem']}\n"
            f"Solution: {entry['solution']}\n"
            f"Target Market: {entry['target_market']}"
        )
        output_text = (
            f"Market Analysis: {entry['market_analysis']}\n"
            f"Competitor Analysis: {entry['competitor_analysis']}\n"
            f"Market Gaps: {entry['market_gaps']}\n"
            f"Validation Score: {entry['validation_score']}"
        )
        json.dump({"input": input_text, "output": output_text}, f_out, ensure_ascii=False)
        f_out.write("\n")
