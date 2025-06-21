import torch
import os
import sys
import time
import csv
from transformers import AutoTokenizer, MobileBertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

MODEL_STATE_DICT_PATH = "models/modelmobile_v1/sql-inject-detector_v1_epoch_3_sd.pt"
TEST_CSV_PATH = "datasets/test_data.csv"
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
    return predicted_class_id

if __name__ == "__main__":
    start_total_time = time.time()
    device = setup_device()
    print(f"Загрузка токенизатора: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    num_labels = len(id2label)
    model = load_trained_model(BASE_MODEL_NAME, MODEL_STATE_DICT_PATH, num_labels, device)
    print(f"\n--- Автоматическое тестирование модели MobileBERT ---")
    print(f"Тестовые данные: {TEST_CSV_PATH}")
    true_labels = []
    predicted_labels = []
    test_queries = []
    try:
        with open(TEST_CSV_PATH, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            print("Начало обработки тестовых запросов...")
            for i, row in enumerate(reader):
                try:
                    query = row['Query']
                    true_label = int(row['Label'])
                except KeyError:
                    print(f"ОШИБКА: В строке {i+1} CSV отсутствуют колонки 'Query' или 'Label'. Пропуск строки.")
                    continue
                except ValueError:
                     print(f"ОШИБКА: Не удалось преобразовать 'Label' в число в строке {i+1} CSV ('{row.get('Label', 'N/A')}'). Пропуск строки.")
                     continue
                predicted_id = predict_sql_injection(
                    query, model, tokenizer, device, MAX_SEQ_LENGTH
                )
                test_queries.append(query)
                true_labels.append(true_label)
                predicted_labels.append(predicted_id)
            print(f"Обработка {len(true_labels)} запросов завершена.")
    except FileNotFoundError:
        print(f"ОШИБКА: Тестовый CSV файл не найден по пути: {TEST_CSV_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"ОШИБКА: Произошла ошибка при чтении или обработке CSV: {e}")
        sys.exit(1)
    if not true_labels:
        print("Нет данных для расчета метрик.")
    else:
        accuracy = accuracy_score(true_labels, predicted_labels)
        report = classification_report(true_labels, predicted_labels, target_names=id2label.values(), digits=4, zero_division=0)
        print("\n--- Результаты Тестирования (MobileBERT) ---")
        print(f"Общая точность (Accuracy): {accuracy:.4f}")
        print("\nОтчет по классам:")
        print(report)
        print("\n--- Неверно классифицированные запросы (MobileBERT) ---")
        errors_found = False
        for i in range(len(test_queries)):
            if true_labels[i] != predicted_labels[i]:
                errors_found = True
                print(f"Запрос: {test_queries[i]}")
                print(f"  Ожидалось: {id2label[true_labels[i]]} ({true_labels[i]})")
                print(f"  Предсказано: {id2label[predicted_labels[i]]} ({predicted_labels[i]})")
                print("-" * 20)
        if not errors_found:
            print("Неверно классифицированных запросов не найдено.")
    end_total_time = time.time()
    total_duration = end_total_time - start_total_time
    print(f"\nОбщее время тестирования: {total_duration:.2f} сек.")