
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load dataset
dataset = load_dataset("json", data_files={"train": "./augmented_final_empathy.jsonl"}, split="train")

# Preprocess dataset
def preprocess_function(examples):
    return tokenizer(examples["prompt"] + " " + examples["response"], truncation=True)

# Load tokenizer and model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is defined

model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./empathy-model-v1",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Train model
trainer.train()

# Save final model
trainer.save_model("./empathy-model-v1")
tokenizer.save_pretrained("./empathy-model-v1")
