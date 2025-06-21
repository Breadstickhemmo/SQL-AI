import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import sys
import time
import pickle

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

TOKENIZER_PATH = "models/lstm/lstm_tokenizer.pickle"
MODEL_PATH = "models/lstm/lstm_model.keras"
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

def predict(query: str, tokenizer, model, max_len: int):
    sequence = tokenizer.texts_to_sequences([query])
    padded_sequence = tf.keras.utils.pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    prediction_prob = model.predict(padded_sequence, verbose=0)[0][0]
    predicted_id = int(prediction_prob > 0.5)
    predicted_label = id2label[predicted_id]
    probs_dict = {
        id2label[0]: 1.0 - prediction_prob,
        id2label[1]: prediction_prob
    }
    return predicted_label, predicted_id, probs_dict

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')

    tokenizer, model = load_tokenizer_and_model(TOKENIZER_PATH, MODEL_PATH)
    print("\n--- Интерактивное тестирование модели LSTM ---")
    print("Введите SQL-запрос для проверки или 'quit' для выхода.")
    while True:
        user_query = input("Запрос> ")
        if user_query.lower() == 'quit':
            break
        if not user_query.strip():
            continue
        start_pred_time = time.time()
        predicted_label, predicted_id, probabilities = predict(user_query, tokenizer, model, MAX_LEN)
        end_pred_time = time.time()
        pred_duration = end_pred_time - start_pred_time
        print(f"  Предсказание: {predicted_label} (ID: {predicted_id})")
        print("  Вероятности:")
        for label, prob in probabilities.items():
            print(f"    - {label}: {prob:.4f}")
        print(f"  Время предсказания: {pred_duration:.4f} сек.")
        print("-" * 30)
    print("Тестирование завершено.")