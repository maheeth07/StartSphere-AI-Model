from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer

dataset = load_dataset("json", data_files={"train": "dataset.jsonl"}, split="train")

model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def preprocess(example):
    return tokenizer(example["input"], text_target=example["output"],
                     truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(preprocess, batched=True)

training_args = TrainingArguments(
    output_dir="./startup-model",         
    overwrite_output_dir=True,            
    num_train_epochs=5,                  
    per_device_train_batch_size=2,        
    save_strategy="epoch",                
    logging_dir="./logs",                 
    save_total_limit=1,

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()

model.save_pretrained("startup-model")
tokenizer.save_pretrained("startup-model")
