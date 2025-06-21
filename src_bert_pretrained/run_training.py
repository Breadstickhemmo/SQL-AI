import logging
import sys
import os
import time
import itertools
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Subset

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    from src_bert_pretrained.config import config as base_config
    from src_bert_pretrained.pipeline import SQLInjectionPipeline, collate_fn
    from src_bert_pretrained.utils import set_seed, ensure_dir
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что вы запускаете скрипт из папки 'code' или что папка 'src' доступна в PYTHONPATH.")
    sys.exit(1)

log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
log_level = logging.INFO
log_dir = "logs"
ensure_dir(log_dir)
log_file_base_name_template = "hpt_cv_training_bert" 

logger = logging.getLogger("run_training_cv_hpt_bert")

N_SPLITS_CV = 3 
PRIMARY_METRIC = 'f1' 

param_grid_bert = {
    'lr': [1e-5, 3e-5],
    'batch_size': [8, 16],
    'EPOCHS': [3, 4], 
    'warmup_steps': [0, 50],
    'weight_decay': [0.0, 0.01]
}

def setup_file_logger(log_file_path):
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            root_logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)
    return file_handler

def print_results_table(results, title_prefix=""):
    if not results: 
        logger.warning(f"Нет результатов для таблицы '{title_prefix}'")
        return
        
    table_str = f"\n" + "="*30 + f" {title_prefix} РЕЗУЛЬТАТЫ " + "="*30 + "\n"
    header = f"{'Эпоха':<7} | {'Train Loss':<12} | {'Eval Loss':<11} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'ROC AUC':<10} | {'Время (сек)':<12}"
    table_str += header + "\n"
    table_str += "-" * len(header) + "\n"
    for epoch_result in results:
        log_msg = (
            f"{epoch_result.get('epoch', '?'):<7} | "
            f"{epoch_result.get('avg_train_loss', float('nan')):.4f}".ljust(12) + " | "
            f"{epoch_result.get('avg_eval_loss', float('nan')):.4f}".ljust(11) + " | "
            f"{epoch_result.get('precision', float('nan')):.4f}".ljust(10) + " | "
            f"{epoch_result.get('recall', float('nan')):.4f}".ljust(10) + " | "
            f"{epoch_result.get('f1', float('nan')):.4f}".ljust(10) + " | "
            f"{epoch_result.get('roc_auc', float('nan')):.4f}".ljust(10) + " | "
            f"{epoch_result.get('epoch_duration_sec', float('nan')):.2f}".ljust(12)
        )
        table_str += log_msg + "\n"
    table_str += "=" * (len(header)) + "\n"
    logger.info(table_str)
    print(table_str)

def main_hpt_cv():
    overall_log_file_path = os.path.join(log_dir, f"{log_file_base_name_template}_{time.strftime('%Y%m%d_%H%M%S')}_MAIN.log")
    main_file_handler = setup_file_logger(overall_log_file_path)
    logger.info(f"Главный лог HPT процесса: {overall_log_file_path}")

    overall_start_time = time.time()
    set_seed(base_config['seed'])

    logger.info("Загрузка полного датасета для CV...")
    temp_pipeline_for_data_load = SQLInjectionPipeline(base_config)
    temp_pipeline_for_data_load.load_and_prepare_data()
    full_dataset_objects = temp_pipeline_for_data_load.full_dataset_for_cv
    
    if not full_dataset_objects:
        logger.error("Не удалось загрузить полный датасет. Прерывание HPT.")
        if main_file_handler: main_file_handler.close()
        return

    try:
        all_labels = [sample[2] for sample in full_dataset_objects]
    except (IndexError, TypeError) as e:
        logger.error(f"Ошибка извлечения меток из full_dataset_objects: {e}. Убедитесь, что данные корректны.")
        if main_file_handler: main_file_handler.close()
        return
    
    if len(np.unique(all_labels)) < 2:
        logger.error("В датасете должен быть как минимум два класса для StratifiedKFold и бинарной классификации.")
        if main_file_handler: main_file_handler.close()
        return

    skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=base_config['seed'])

    best_avg_metric_val = -1.0
    best_hyperparams = None
    all_hpt_runs_summary = []

    keys, values = zip(*param_grid_bert.items())
    hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    logger.info(f"Начало подбора гиперпараметров и {N_SPLITS_CV}-fold кросс-валидации для BERT...")
    logger.info(f"Всего комбинаций гиперпараметров: {len(hyperparam_combinations)}")

    for i, params_combo in enumerate(hyperparam_combinations):
        hpt_run_start_time = time.time()
        current_config = base_config.copy()
        current_config.update(params_combo)
        current_config['num_labels'] = 2

        logger.info(f"--- HPT Прогон {i+1}/{len(hyperparam_combinations)} ---")
        logger.info(f"Текущие гиперпараметры: {params_combo}")

        fold_metric_values = []

        for fold_idx, (train_indices, val_indices) in enumerate(skf.split(np.zeros(len(full_dataset_objects)), all_labels)):
            fold_start_time = time.time()
            logger.info(f"--- Фолд {fold_idx + 1}/{N_SPLITS_CV} (для HPT {i+1}) ---")
            
            fold_train_dataset = Subset(full_dataset_objects, train_indices)
            fold_val_dataset = Subset(full_dataset_objects, val_indices)

            pipeline_fold = SQLInjectionPipeline(current_config)
            fold_epoch_results = pipeline_fold.run(
                train_dataset_fold=fold_train_dataset,
                val_dataset_fold=fold_val_dataset
            )

            if fold_epoch_results:
                best_epoch_metric_for_fold = max(res.get(PRIMARY_METRIC, 0.0) for res in fold_epoch_results)
                fold_metric_values.append(best_epoch_metric_for_fold)
                logger.info(f"Фолд {fold_idx + 1}: Лучшая метрика ({PRIMARY_METRIC}) = {best_epoch_metric_for_fold:.4f}")
            else:
                logger.warning(f"Фолд {fold_idx + 1} (для HPT {i+1}) не вернул результатов.")
                fold_metric_values.append(0.0)
            
            fold_duration = time.time() - fold_start_time
            logger.info(f"Фолд {fold_idx + 1} завершен за {fold_duration:.2f} сек.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_metric_for_current_params = np.mean(fold_metric_values) if fold_metric_values else 0.0
        logger.info(f"Средняя метрика ({PRIMARY_METRIC}) по фолдам для {params_combo}: {avg_metric_for_current_params:.4f}")
        
        current_hpt_run_summary = {
            'params': params_combo,
            f'avg_cv_{PRIMARY_METRIC}': avg_metric_for_current_params,
            'fold_metrics': fold_metric_values,
            'duration_sec': time.time() - hpt_run_start_time
        }
        all_hpt_runs_summary.append(current_hpt_run_summary)

        if avg_metric_for_current_params > best_avg_metric_val:
            best_avg_metric_val = avg_metric_for_current_params
            best_hyperparams = params_combo
            logger.info(f"!!! Новые лучшие гиперпараметры найдены: {best_hyperparams} с {PRIMARY_METRIC}={best_avg_metric_val:.4f} !!!")
        
        hpt_run_duration = time.time() - hpt_run_start_time
        logger.info(f"HPT Прогон {i+1} завершен за {hpt_run_duration:.2f} сек.")


    logger.info("--- Подбор гиперпараметров и CV для BERT завершены ---")
    logger.info("Сводка по всем HPT прогонам:")
    for run_summary in sorted(all_hpt_runs_summary, key=lambda x: x[f'avg_cv_{PRIMARY_METRIC}'], reverse=True):
        logger.info(f"  Параметры: {run_summary['params']}, Средний {PRIMARY_METRIC}: {run_summary[f'avg_cv_{PRIMARY_METRIC}']:.4f}, Длительность: {run_summary['duration_sec']:.2f} сек")

    logger.info(f"Лучшие гиперпараметры: {best_hyperparams}")
    logger.info(f"Лучшая средняя CV метрика ({PRIMARY_METRIC}): {best_avg_metric_val:.4f}")
    
    print("\n" + "="*30 + " ИТОГИ ПОДБОРА ГИПЕРПАРАМЕТРОВ (BERT) " + "="*30)
    print(f"Лучшие гиперпараметры: {best_hyperparams}")
    print(f"Лучшая средняя CV метрика ({PRIMARY_METRIC}): {best_avg_metric_val:.4f}")

    if best_hyperparams:
        logger.info("\nОбучение финальной модели BERT с лучшими гиперпараметрами...")
        final_config = base_config.copy()
        final_config.update(best_hyperparams)
        final_config['num_labels'] = 2 
        
        final_epochs = best_hyperparams.get('EPOCHS', base_config['EPOCHS'])
        final_config['EPOCHS'] = final_epochs

        final_model_save_dir = os.path.join(base_config['save_loc'], "best_hpt_bert_model")
        final_config['save_loc'] = final_model_save_dir
        ensure_dir(final_model_save_dir)

        logger.info(f"Обучение финальной модели с параметрами: {final_config}")
        
        final_pipeline = SQLInjectionPipeline(final_config)
        logger.info("Используется стандартное train/val разделение для обучения финальной модели.")
        final_results = final_pipeline.run()
        
        if final_results:
            logger.info("Финальная модель BERT обучена.")
            print_results_table(final_results, "ФИНАЛЬНАЯ МОДЕЛЬ BERT")
            logger.info(f"Финальная модель (state_dict последней эпохи) должна быть сохранена в: {final_model_save_dir}")
        else:
            logger.error("Обучение финальной модели BERT не дало результатов.")
    else:
        logger.warning("Лучшие гиперпараметры не найдены, обучение финальной модели BERT пропущено.")

    overall_duration = time.time() - overall_start_time
    logger.info(f"Выполнение всего скрипта HPT и CV для BERT завершено за {overall_duration:.2f} секунд.")
    print(f"\nВыполнение всего скрипта HPT и CV для BERT завершено за {overall_duration:.2f} секунд.")
    print(f"Подробные логи сохранены в: {overall_log_file_path}")
    if best_hyperparams and final_results:
         print(f"Финальная модель BERT сохранена в директории: {final_model_save_dir}")
    
    if main_file_handler: main_file_handler.close()


if __name__ == "__main__":
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(stdout_handler)
    logging.getLogger().setLevel(log_level)

    main_hpt_cv()