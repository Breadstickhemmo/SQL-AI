import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import sys
import time
import csv
import pickle
from sklearn.metrics import accuracy_score, classification_report

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

TOKENIZER_PATH = "models/lstm/lstm_tokenizer_best.pickle"
MODEL_PATH = "models/lstm/lstm_model_best.keras"
TEST_CSV_PATH = "datasets/test_data.csv"
MAX_LEN = 150

id2label = {0: "Нормальный запрос (Не инъекция)", 1: "SQL Инъекция"}

def load_tokenizer_and_model(tokenizer_path, model_path):
    print(f"Загрузка токенизатора из: {tokenizer_path}")
    try:
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
    except FileNotFoundError:
        print(f"ОШИБКА: Файл токенизатора не найден: {tokenizer_path}")
        sys.exit(1)
    except Exception as e:
        print(f"ОШИБКА: Не удалось загрузить токенизатор: {e}")
        sys.exit(1)

    print(f"Загрузка модели LSTM из: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
    except FileNotFoundError:
        if not os.path.exists(model_path) and os.path.isdir(model_path.replace('.keras','')):
             model_path = model_path.replace('.keras','')
             try:
                 model = tf.keras.models.load_model(model_path)
             except Exception as e_inner:
                 print(f"ОШИБКА: Не удалось загрузить модель из папки {model_path}: {e_inner}")
                 sys.exit(1)
        else:
            print(f"ОШИБКА: Файл или папка модели не найдены: {model_path}")
            sys.exit(1)
    except Exception as e:
        print(f"ОШИБКА: Не удалось загрузить модель LSTM: {e}")
        sys.exit(1)

    print("Токенизатор и модель LSTM успешно загружены.")
    return tokenizer, model

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')

    start_total_time = time.time()
    tokenizer, model = load_tokenizer_and_model(TOKENIZER_PATH, MODEL_PATH)
    print(f"\n--- Автоматическое тестирование модели LSTM ---")
    print(f"Тестовые данные: {TEST_CSV_PATH}")
    true_labels = []
    test_queries_text = []

    try:
        with open(TEST_CSV_PATH, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            print("Чтение тестовых запросов...")
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
                test_queries_text.append(query)
                true_labels.append(true_label)
            print(f"Прочитано {len(true_labels)} запросов.")

    except FileNotFoundError:
        print(f"ОШИБКА: Тестовый CSV файл не найден по пути: {TEST_CSV_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"ОШИБКА: Произошла ошибка при чтении CSV: {e}")
        sys.exit(1)

    if not true_labels:
        print("Нет данных для тестирования.")
    else:
        print("Преобразование запросов в последовательности и паддинг...")
        sequences = tokenizer.texts_to_sequences(test_queries_text)
        padded_sequences = tf.keras.utils.pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

        print("Предсказание на тестовом наборе...")
        y_pred_probs = model.predict(padded_sequences).flatten()
        y_pred_classes = (y_pred_probs > 0.5).astype(int)
        print("Предсказание завершено.")

        accuracy = accuracy_score(true_labels, y_pred_classes)
        report = classification_report(true_labels, y_pred_classes, target_names=id2label.values(), digits=4, zero_division=0)

        print("\n--- Результаты Тестирования (LSTM) ---")
        print(f"Общая точность (Accuracy): {accuracy:.4f}")
        print("\nОтчет по классам:")
        print(report)
        print("\n--- Неверно классифицированные запросы (LSTM) ---")
        errors_found = False
        for i in range(len(test_queries_text)):
            if true_labels[i] != y_pred_classes[i]:
                errors_found = True
                print(f"Запрос: {test_queries_text[i]}")
                print(f"  Ожидалось: {id2label[true_labels[i]]} ({true_labels[i]})")
                print(f"  Предсказано: {id2label[y_pred_classes[i]]} ({y_pred_classes[i]})")
                print("-" * 20)
        if not errors_found:
            print("Неверно классифицированных запросов не найдено.")

    end_total_time = time.time()
    total_duration = end_total_time - start_total_time
    print(f"\nОбщее время тестирования: {total_duration:.2f} сек.")