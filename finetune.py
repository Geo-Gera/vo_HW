
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Загружаем модель и токенизатор
model_name = "google/bigbird-arithmetic"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Подготовка данных для дообучения
def preprocess_function(examples):
    inputs = examples["input_text"]
    targets = examples["output_text"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

    labels = tokenizer(targets, max_length=32, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Загрузка тренировочного датасета
train_data = load_dataset("json", data_files="training_dataset.json")["train"]
train_dataset = train_data.map(preprocess_function, batched=True)

# Настройки для дообучения
training_args = TrainingArguments(
    output_dir="./bigbird_trained",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Тренировка модели
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
model.save_pretrained("bigbird_trained_model")
tokenizer.save_pretrained("bigbird_trained_model")
print("Model fine-tuning complete and saved as 'bigbird_trained_model'.")
