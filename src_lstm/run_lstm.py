import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from scikeras.wrappers import KerasClassifier
import time
import pickle

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU доступны: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("GPU не найдены, используется CPU.")


DATA_PATH = "datasets/SQL.csv"
RANDOM_STATE = 42
MODELS_DIR = "models/lstm"
TOKENIZER_NAME = "lstm_tokenizer_best.pickle"
MODEL_NAME = "lstm_model_best.keras"

N_SPLITS_CV = 3
PRIMARY_SCORING_METRIC = 'f1_weighted'

MAX_WORDS_VOCAB = 15000
MAX_SEQUENCE_LENGTH = 150

def create_lstm_model(embedding_dim=32, lstm_units=16, spatial_dropout_rate=0.3,
                      lstm_dropout_rate=0.3, bidirectional=False, optimizer='adam'):
    inputs = tf.keras.Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = tf.keras.layers.Embedding(
        input_dim=MAX_WORDS_VOCAB,
        output_dim=embedding_dim
    )(inputs)
    x = tf.keras.layers.SpatialDropout1D(spatial_dropout_rate)(x)
    
    lstm_layer_instance = tf.keras.layers.LSTM(
        lstm_units,
        dropout=lstm_dropout_rate,
        recurrent_dropout=0.2
    )
    
    if bidirectional:
        x = tf.keras.layers.Bidirectional(lstm_layer_instance)(x)
    else:
        x = lstm_layer_instance(x)
        
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    print("--- Запуск оптимизированного обучения LSTM ---")
    start_time = time.time()

    os.makedirs(MODELS_DIR, exist_ok=True)
    tokenizer_path = os.path.join(MODELS_DIR, TOKENIZER_NAME)
    model_path = os.path.join(MODELS_DIR, MODEL_NAME)

    param_grid_lstm = {
        'batch_size': [16, 32],
        'epochs': [5],
        'model__embedding_dim': [32, 64],
        'model__lstm_units': [16, 32],
        'model__spatial_dropout_rate': [0.2],
        'model__lstm_dropout_rate': [0.2],
        'model__bidirectional': [False],
        'optimizer': ['adam']
    }

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True,
        verbose=1
    )

    keras_model = KerasClassifier(
        model=create_lstm_model,
        verbose=0,
        callbacks=[early_stopping],
        validation_split=0.2
    )

    print("Загрузка и предобработка данных...")
    try:
        data = pd.read_csv(DATA_PATH)
        data['Query'] = data['Query'].fillna('').astype(str)
        data['Label'] = data['Label'].astype(int)
    except FileNotFoundError:
        print(f"ОШИБКА: Файл данных не найден: {DATA_PATH}")
        exit(1)
    except Exception as e:
        print(f"ОШИБКА при загрузке данных: {e}")
        exit(1)

    if len(data) == 0:
        print("ОШИБКА: Данные не загружены или файл пуст.")
        exit(1)
    
    if len(data['Label'].unique()) < 2:
        print(f"ОШИБКА: В данных присутствует только один класс метки. CV и обучение невозможны.")
        exit(1)
    
    if len(data) < N_SPLITS_CV :
         print(f"ОШИБКА: Недостаточно данных ({len(data)}) для {N_SPLITS_CV}-fold CV.")
         exit(1)

    print("Токенизация и паддинг...")
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_WORDS_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(data['Query'])
    X = tf.keras.preprocessing.sequence.pad_sequences(
        tokenizer.texts_to_sequences(data['Query']),
        maxlen=MAX_SEQUENCE_LENGTH,
        padding='post',
        truncating='post'
    )
    y = data['Label'].values

    print("Сохранение токенизатора...")
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)

    print("Начало RandomizedSearchCV...")
    cv_strategy = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    
    random_search = RandomizedSearchCV(
        estimator=keras_model,
        param_distributions=param_grid_lstm,
        n_iter=2,
        cv=cv_strategy,
        scoring=PRIMARY_SCORING_METRIC,
        verbose=2,
        n_jobs=1,
        random_state=RANDOM_STATE
    )
    try:
        random_search.fit(X, y)
    except Exception as e:
        print(f"ОШИБКА во время выполнения RandomizedSearchCV.fit: {e}")
        exit(1)


    print(f"Лучшие параметры: {random_search.best_params_}")
    print(f"Лучший скор ({PRIMARY_SCORING_METRIC}) на CV: {random_search.best_score_:.4f}")

    print("Сохранение лучшей модели...")
    best_tf_model = None
    if hasattr(random_search.best_estimator_, 'model_') and random_search.best_estimator_.model_ is not None:
        best_tf_model = random_search.best_estimator_.model_
    elif hasattr(random_search.best_estimator_, 'model') and random_search.best_estimator_.model is not None:
        best_tf_model = random_search.best_estimator_.model

    if best_tf_model:
        try:
            best_tf_model.save(model_path)
            print(f"Лучшая модель сохранена в: {model_path}")
        except Exception as e:
            print(f"Ошибка при сохранении лучшей модели Keras: {e}")
            print("Попытка создать и обучить модель с лучшими параметрами заново...")
            best_tf_model = None
    else:
        print("Не удалось получить обученную модель из best_estimator_. Попытка переобучить...")

    if not best_tf_model:
        try:
            print("Переобучение модели с лучшими найденными параметрами на полном наборе данных (с валидацией для early stopping)...")
            final_model_params = random_search.best_params_
            
            model_create_args = {k.replace('model__', ''): v for k, v in final_model_params.items() if k.startswith('model__')}
            optimizer_arg = final_model_params.get('optimizer', 'adam')
            epochs_arg = final_model_params.get('epochs', 10)
            batch_size_arg = final_model_params.get('batch_size', 128)

            final_model_instance = create_lstm_model(**model_create_args, optimizer=optimizer_arg)
            
            X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
                X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
            )

            print(f"Обучение финальной модели с параметрами: create_args={model_create_args}, optimizer={optimizer_arg}, epochs={epochs_arg}, batch_size={batch_size_arg}")
            final_model_instance.fit(
                X_train_final, y_train_final,
                epochs=epochs_arg,
                batch_size=batch_size_arg,
                callbacks=[early_stopping],
                validation_data=(X_val_final, y_val_final),
                verbose=1
            )
            final_model_instance.save(model_path)
            best_tf_model = final_model_instance
            print(f"Финальная модель (переобученная) сохранена в: {model_path}")
        except Exception as e:
            print(f"Ошибка при переобучении и сохранении финальной модели: {e}")


    if best_tf_model:
        print("\nОценка лучшей модели на всем датасете:")
        y_pred_probs = best_tf_model.predict(X)
        if y_pred_probs.ndim > 1 and y_pred_probs.shape[1] == 1:
            y_pred_probs = y_pred_probs.flatten()
        y_pred_classes = (y_pred_probs > 0.5).astype(int)
        print(classification_report(y, y_pred_classes, target_names=['Normal', 'Injection'], digits=4, zero_division=0))
    else:
        print("Не удалось получить или обучить финальную модель для оценки.")

    end_time = time.time()
    print(f"\nОбщее время выполнения RandomizedSearchCV и сохранения: {end_time - start_time:.2f} сек")