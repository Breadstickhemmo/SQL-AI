import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
import os
import joblib
import time

DATA_PATH = "datasets/SQL.csv"
RANDOM_STATE = 42
MODELS_DIR = "models/decision_tree"
VECTORIZER_NAME = "dt_vectorizer_best.joblib"
MODEL_NAME = "decision_tree_model_best.joblib"
N_SPLITS_CV = 5
PRIMARY_SCORING_METRIC = 'f1_weighted'

if __name__ == "__main__":
    print("--- Запуск обучения с CV и HPT: Дерево Решений (Decision Tree) ---")
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

    print("Определение сетки гиперпараметров для DecisionTreeClassifier...")
    param_grid_dt = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 8]
    }

    print(f"Запуск GridSearchCV с {N_SPLITS_CV}-fold CV (scoring: {PRIMARY_SCORING_METRIC})...")
    cv_strategy = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    
    grid_search_dt = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid_dt,
        cv=cv_strategy,
        scoring=PRIMARY_SCORING_METRIC,
        verbose=1,
        n_jobs=-1 
    )

    try:
        grid_search_dt.fit(X, y)
    except Exception as e:
        print(f"ОШИБКА во время GridSearchCV.fit: {e}")
        exit(1)

    print("GridSearchCV завершен.")
    print(f"Лучшие параметры: {grid_search_dt.best_params_}")
    print(f"Лучшая оценка ({PRIMARY_SCORING_METRIC}) на CV: {grid_search_dt.best_score_:.4f}")

    best_dt_model = grid_search_dt.best_estimator_

    print(f"Сохранение лучшей модели в: {model_path}")
    joblib.dump(best_dt_model, model_path)

    print("\n--- Результаты оценки лучшей модели на данных, использованных для GridSearchCV ---")
    y_pred_on_full_data = best_dt_model.predict(X)
    
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
    cv_results_df = pd.DataFrame(grid_search_dt.cv_results_)
    top_n = 5
    
    base_relevant_cols = ['params']
    split_cols = [col for col in cv_results_df.columns if 'split' in col and f'test_{PRIMARY_SCORING_METRIC}' in col]
    mean_std_rank_cols = []
    mean_col = f'mean_test_{PRIMARY_SCORING_METRIC}'
    std_col = f'std_test_{PRIMARY_SCORING_METRIC}'
    rank_col_name = f'rank_test_{PRIMARY_SCORING_METRIC}'
    if rank_col_name not in cv_results_df.columns:
        if 'rank_test_score' in cv_results_df.columns:
            rank_col_name = 'rank_test_score'
        else:
            rank_col_name = None 
            print(f"Предупреждение: столбец ранга для {PRIMARY_SCORING_METRIC} или 'rank_test_score' не найден в cv_results_df.")


    if mean_col in cv_results_df.columns:
        mean_std_rank_cols.append(mean_col)
    if std_col in cv_results_df.columns:
        mean_std_rank_cols.append(std_col)
    if rank_col_name and rank_col_name in cv_results_df.columns:
         mean_std_rank_cols.append(rank_col_name)
    
    relevant_cols = base_relevant_cols + split_cols + mean_std_rank_cols
    relevant_cols = sorted(list(set(relevant_cols)), key=relevant_cols.index)

    existing_relevant_cols = [col for col in relevant_cols if col in cv_results_df.columns]

    if rank_col_name and rank_col_name in existing_relevant_cols:
        print(cv_results_df[existing_relevant_cols].sort_values(by=rank_col_name).head(top_n))
    elif mean_col in existing_relevant_cols:
        print(f"Сортировка по {mean_col}, так как столбец ранга '{rank_col_name}' не найден.")
        print(cv_results_df[existing_relevant_cols].sort_values(by=mean_col, ascending=False).head(top_n))
    else:
        print("Не удалось определить столбец для сортировки результатов CV. Вывод первых N строк:")
        print(cv_results_df[existing_relevant_cols].head(top_n))


    end_time = time.time()
    print(f"\nОбщее время выполнения: {end_time - start_time:.2f} сек.")
    print(f"--- Завершено: Дерево Решений с CV и HPT ({PRIMARY_SCORING_METRIC}) ---")
    print(f"Лучшая модель сохранена в {model_path}, векторизатор в {vectorizer_path}")