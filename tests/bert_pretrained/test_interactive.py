import torch
import os
import sys
import time
from transformers import AutoTokenizer, MobileBertForSequenceClassification

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

MODEL_STATE_DICT_PATH = "models/modelmobile_v1/sql-inject-detector_v1_epoch_5_sd.pt"
BASE_MODEL_NAME = 'google/mobilebert-uncased'
MAX_SEQ_LENGTH = 256

id2label = {0: "Нормальный запрос (Не инъекция)", 1: "SQL Инъекция"}
label2id = {"Нормальный запрос (Не инъекция)": 0, "SQL Инъекция": 1}

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Используется GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Используется CPU")
    return device

def load_trained_model(model_name, state_dict_path, num_labels, device):
    print(f"Загрузка базовой архитектуры: {model_name}")
    model = MobileBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    print(f"Загрузка обученных весов из: {state_dict_path}")
    try:
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
    except FileNotFoundError:
        print(f"ОШИБКА: Файл модели не найден по пути: {state_dict_path}")
        sys.exit(1)
    except Exception as e:
        print(f"ОШИБКА: Не удалось загрузить веса модели: {e}")
        sys.exit(1)
    model.to(device)
    model.eval()
    print("Модель MobileBERT успешно загружена и переведена в режим оценки.")
    return model

def predict_sql_injection(query: str, model, tokenizer, device, max_len: int):
    inputs = tokenizer(
        query,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).squeeze()
    predicted_class_id = torch.argmax(probabilities).item()
    predicted_label = id2label[predicted_class_id]
    probs_dict = {id2label[i]: prob.item() for i, prob in enumerate(probabilities)}
    return predicted_label, predicted_class_id, probs_dict

if __name__ == "__main__":
    device = setup_device()
    print(f"Загрузка токенизатора: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    num_labels = len(id2label)
    model = load_trained_model(BASE_MODEL_NAME, MODEL_STATE_DICT_PATH, num_labels, device)
    print("\n--- Интерактивное тестирование модели MobileBERT ---")
    print("Введите SQL-запрос для проверки или 'quit' для выхода.")
    while True:
        user_query = input("Запрос> ")
        if user_query.lower() == 'quit':
            break
        if not user_query.strip():
            continue
        start_pred_time = time.time()
        predicted_label, predicted_id, probabilities = predict_sql_injection(
            user_query, model, tokenizer, device, MAX_SEQ_LENGTH
        )
        end_pred_time = time.time()
        pred_duration = end_pred_time - start_pred_time
        print(f"  Предсказание: {predicted_label} (ID: {predicted_id})")
        print("  Вероятности:")
        for label, prob in probabilities.items():
            print(f"    - {label}: {prob:.4f}")
        print(f"  Время предсказания: {pred_duration:.4f} сек.")
        print("-" * 30)
    print("Тестирование завершено.")