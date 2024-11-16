
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Загрузка модели и токенизатора
model_path = "bigbird_trained_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Функция для сложения чисел
def add_numbers(num1, num2):
    input_text = f"{num1} + {num2}"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.strip()

# Интерфейс Gradio
def gradio_interface(num1, num2):
    try:
        # Проверка, что ввод состоит из чисел
        num1 = int(num1)
        num2 = int(num2)
        return add_numbers(num1, num2)
    except ValueError:
        return "Пожалуйста, введите два числа."

# Создаем Gradio интерфейс
iface = gr.Interface(
    fn=gradio_interface,
    inputs=["text", "text"],
    outputs="text",
    title="Сложение больших чисел",
    description="Введите два числа, и модель вернёт их сумму."
)

# Запуск интерфейса
if __name__ == "__main__":
    iface.launch()
