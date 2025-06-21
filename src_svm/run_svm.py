import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
import os
import joblib
import time

DATA_PATH = "datasets/SQL.csv"
RANDOM_STATE = 42
MODELS_DIR = "models/svm"
VECTORIZER_NAME = "svm_vectorizer_best.joblib"
MODEL_NAME = "svm_model_best.joblib"
N_SPLITS_CV = 5
PRIMARY_SCORING_METRIC = 'f1_weighted'

if __name__ == "__main__":
    print("--- Запуск обучения с CV и HPT: Метод Опорных Векторов (SVM) ---")
    start_time = time.time()

    os.makedirs(MODELS_DIR, exist_ok=True)
    vectorizer_path = os.path.join(MODELS_DIR, VECTORIZER_NAME)
    model_path = os.path.join(MODELS_DIR, MODEL_NAME)

    print(f"Загрузка данных из: {DATA_PATH}")
    try:
        data = pd.read_csv(DATA_PATH)
        if 'Query' not in data.columns or 'Label' not in data.columns:
            raise ValueError("CSV файл должен содержать колонки 'Query' и 'Label'")
        data.dropna(subset=['Query'], inplace=True)
        data['Query'] = data['Query'].astype(str)
        data['Label'] = data['Label'].astype(int)
    except FileNotFoundError:
        print(f"ОШИБКА: Файл данных не найден: {DATA_PATH}")
        exit(1)
    except Exception as e:
        print(f"ОШИБКА при загрузке данных: {e}")
        exit(1)

    print(f"Количество записей: {len(data)}")
    if len(data) < N_SPLITS_CV:
        print(f"ОШИБКА: Недостаточно данных ({len(data)}) для {N_SPLITS_CV}-fold CV.")
        exit(1)
    if len(data['Label'].unique()) < 2:
        print(f"ОШИБКА: В данных присутствует только один класс метки. CV и обучение невозможны.")
        exit(1)

    print("Извлечение признаков (CountVectorizer)...")
    vectorizer = CountVectorizer(token_pattern=r'\b\w+\b', max_features=10000)
    try:
        X = vectorizer.fit_transform(data['Query'])
    except Exception as e:
        print(f"ОШИБКА при векторизации данных: {e}")
        exit(1)

    y = data['Label']
    print(f"Размерность матрицы признаков: {X.shape}")

    print(f"Сохранение векторизатора в: {vectorizer_path}")
    joblib.dump(vectorizer, vectorizer_path)

    print("Определение сетки гиперпараметров для SVC...")
    param_grid_svm = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    print(f"Запуск GridSearchCV с {N_SPLITS_CV}-fold CV (scoring: {PRIMARY_SCORING_METRIC})...")
    cv_strategy = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    
    grid_search_svm = GridSearchCV(
        SVC(random_state=RANDOM_STATE, probability=True),
        param_grid_svm,
        cv=cv_strategy,
        scoring=PRIMARY_SCORING_METRIC,
        verbose=1,
        n_jobs=-1
    )

    try:
        grid_search_svm.fit(X, y)
    except Exception as e:
        print(f"ОШИБКА во время GridSearchCV.fit: {e}")
        exit(1)

    print("GridSearchCV завершен.")
    print(f"Лучшие параметры: {grid_search_svm.best_params_}")
    print(f"Лучшая оценка ({PRIMARY_SCORING_METRIC}) на CV: {grid_search_svm.best_score_:.4f}")

    best_svm_model = grid_search_svm.best_estimator_

    print(f"Сохранение лучшей модели в: {model_path}")
    joblib.dump(best_svm_model, model_path)

    print("\n--- Результаты оценки лучшей модели на данных, использованных для GridSearchCV ---")
    y_pred_on_full_data = best_svm_model.predict(X)
    
    accuracy = accuracy_score(y, y_pred_on_full_data)
    recall = recall_score(y, y_pred_on_full_data, average='weighted', zero_division=0)
    precision = precision_score(y, y_pred_on_full_data, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred_on_full_data, average='weighted', zero_division=0)

    print(f"Точность (Accuracy): {accuracy:.4f}")
    print(f"Полнота (Recall weighted):    {recall:.4f}")
    print(f"Точность (Precision weighted): {precision:.4f}")
    print(f"F1-мера (weighted):             {f1:.4f}")
    print("\nОтчет по классам (на данных GridSearchCV):")
    print(classification_report(y, y_pred_on_full_data, target_names=['Normal (0)', 'Injection (1)'], digits=4, zero_division=0))

    print("\nРезультаты кросс-валидации для всех комбинаций:")
    cv_results_df = pd.DataFrame(grid_search_svm.cv_results_)
    top_n = 5
    relevant_cols = ['params'] + [col for col in cv_results_df.columns if 'split' in col and f'test_{PRIMARY_SCORING_METRIC}' in col] + [f'mean_test_{PRIMARY_SCORING_METRIC}', f'std_test_{PRIMARY_SCORING_METRIC}']
    rank_col_name = f'rank_test_{PRIMARY_SCORING_METRIC}' if f'rank_test_{PRIMARY_SCORING_METRIC}' in cv_results_df.columns else 'rank_test_score'
    if rank_col_name not in relevant_cols:
        relevant_cols.append(rank_col_name)

    print(cv_results_df[relevant_cols].sort_values(by=rank_col_name).head(top_n))

    end_time = time.time()
    print(f"\nОбщее время выполнения: {end_time - start_time:.2f} сек.")
    print(f"--- Завершено: Метод Опорных Векторов (SVM) с CV и HPT ({PRIMARY_SCORING_METRIC}) ---")
    print(f"Лучшая модель сохранена в {model_path}, векторизатор в {vectorizer_path}")