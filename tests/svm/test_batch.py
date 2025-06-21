import joblib
import os
import sys
import time
import csv
from sklearn.metrics import accuracy_score, classification_report

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

VECTORIZER_PATH = "models/svm/svm_vectorizer_best.joblib"
MODEL_PATH = "models/svm/svm_model_best.joblib"
TEST_CSV_PATH = "datasets/test_data.csv"

id2label = {0: "Нормальный запрос (Не инъекция)", 1: "SQL Инъекция"}

def load_vectorizer_and_model(vectorizer_path, model_path):
    print(f"Загрузка векторизатора из: {vectorizer_path}")
    try:
        vectorizer = joblib.load(vectorizer_path)
    except FileNotFoundError:
        print(f"ОШИБКА: Файл векторизатора не найден: {vectorizer_path}")
        sys.exit(1)
    except Exception as e:
        print(f"ОШИБКА: Не удалось загрузить векторизатор: {e}")
        sys.exit(1)

    print(f"Загрузка модели SVM из: {model_path}")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"ОШИБКА: Файл модели не найден: {model_path}")
        sys.exit(1)
    except Exception as e:
        print(f"ОШИБКА: Не удалось загрузить модель: {e}")
        sys.exit(1)

    print("Векторизатор и модель SVM успешно загружены.")
    return vectorizer, model

if __name__ == "__main__":
    start_total_time = time.time()
    vectorizer, model = load_vectorizer_and_model(VECTORIZER_PATH, MODEL_PATH)
    print(f"\n--- Автоматическое тестирование модели SVM ---")
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

                query_features = vectorizer.transform([query])
                prediction = model.predict(query_features)
                predicted_id = prediction[0]

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
        print("\n--- Результаты Тестирования (SVM) ---")
        print(f"Общая точность (Accuracy): {accuracy:.4f}")
        print("\nОтчет по классам:")
        print(report)
        print("\n--- Неверно классифицированные запросы (SVM) ---")
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