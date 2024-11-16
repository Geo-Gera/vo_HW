
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Загрузка дообученной модели и токенизатора
model_path = "bigbird_trained_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Функция для выполнения сложения
def add_numbers(num1, num2):
    input_text = f"{num1} + {num2}"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.strip()

# Функция для валидации точности модели
def validate_model(test_data_path="test_dataset.json"):
    with open(test_data_path, "r") as f:
        test_data = json.load(f)
    
    correct = 0
    total = len(test_data)
    
    for example in test_data:
        num1, num2 = example["input"].split(" + ")
        expected_output = example["expected_output"]
        predicted_output = add_numbers(num1, num2)

        # Проверка корректности предсказания
        if predicted_output == expected_output:
            correct += 1
        else:
            print(f"Ошибка: {num1} + {num2} = {expected_output}, предсказано {predicted_output}")

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.2%}")

# Запуск валидации
if __name__ == "__main__":
    validate_model()
