import joblib
import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

VECTORIZER_PATH = "models/svm/svm_vectorizer.joblib"
MODEL_PATH = "models/svm/svm_model.joblib"

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

def predict(query: str, vectorizer, model):
    query_features = vectorizer.transform([query])
    prediction = model.predict(query_features)
    predicted_id = prediction[0]
    predicted_label = id2label[predicted_id]
    return predicted_label, predicted_id

if __name__ == "__main__":
    vectorizer, model = load_vectorizer_and_model(VECTORIZER_PATH, MODEL_PATH)
    print("\n--- Интерактивное тестирование модели SVM ---")
    print("Введите SQL-запрос для проверки или 'quit' для выхода.")
    while True:
        user_query = input("Запрос> ")
        if user_query.lower() == 'quit':
            break
        if not user_query.strip():
            continue
        start_pred_time = time.time()
        predicted_label, predicted_id = predict(user_query, vectorizer, model)
        end_pred_time = time.time()
        pred_duration = end_pred_time - start_pred_time
        print(f"  Предсказание: {predicted_label} (ID: {predicted_id})")
        print(f"  Время предсказания: {pred_duration:.4f} сек.")
        print("-" * 30)
    print("Тестирование завершено.")