import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split, Dataset, Subset
from transformers import AutoTokenizer, MobileBertForSequenceClassification
from torch.optim import AdamW
from transformers import Adafactor, get_linear_schedule_with_warmup
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.amp import GradScaler, autocast
import numpy as np
import time
import datetime
import csv
import logging
import os

from .utils import ensure_dir

def collate_fn(batch):
    input_ids, attention_masks, labels = zip(*batch)
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return input_ids_padded, attention_masks_padded, labels_tensor

def create_dataloaders(train_dataset, val_dataset, batch_size, collate_fn_func):
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size,
        collate_fn=collate_fn_func
    )
    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size,
        collate_fn=collate_fn_func
    )
    return train_dataloader, val_dataloader

class SQLInjectionPipeline:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.device = self._device_setup()
        self.training_results = []
        ensure_dir(self.config['save_loc'])
        self.full_dataset_for_cv = None

    def _device_setup(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logging.warning("GPU (CUDA) не обнаружен. Используется CPU. Обучение может быть медленным.")
        print(f"Используемое устройство: {device}")
        return device

    def load_and_prepare_data(self, train_dataset_fold=None, val_dataset_fold=None):
        if train_dataset_fold and val_dataset_fold:
            logging.info("Использование предварительно разделенных датасетов для обучения и валидации (режим CV).")
            batch_size = self.config['batch_size']
            self.train_dataloader, self.val_dataloader = create_dataloaders(
                train_dataset_fold, val_dataset_fold, batch_size, collate_fn
            )
            logging.info(f"Размер обучающего DataLoader (CV fold): {len(self.train_dataloader.dataset)}")
            logging.info(f"Размер валидационного DataLoader (CV fold): {len(self.val_dataloader.dataset)}")
            return

        start_time = time.time()
        
        if self.full_dataset_for_cv is None:
            input_ids_list = []
            attention_masks_list = []
            labels_list = []
            data_file = self.config['data_file']
            data_format = self.config['data_format']
            query_key, label_key = self.config['data_keys']
            max_len = self.config['max_seq_length']

            if data_format == 'csv':
                try:
                    with open(data_file, 'r', encoding='utf-8') as file:
                        reader = csv.DictReader(file)
                        rows_processed = 0
                        rows_skipped = 0
                        for i, row in enumerate(reader):
                            rows_processed += 1
                            try:
                                query = row[query_key]
                                label = int(row[label_key])
                                inputs = self.tokenizer(
                                    query,
                                    add_special_tokens=True,
                                    max_length=max_len,
                                    padding=False, 
                                    truncation=True,
                                    return_tensors=None 
                                )
                                input_ids_list.append(torch.tensor(inputs['input_ids']))
                                attention_masks_list.append(torch.tensor(inputs['attention_mask']))
                                labels_list.append(label)
                            except KeyError as e:
                                logging.warning(f"Пропущена строка {i+1}: отсутствует ключ {e} в файле {data_file}.")
                                rows_skipped += 1
                                continue
                            except ValueError as e:
                                logging.warning(f"Пропущена строка {i+1}: ошибка преобразования метки '{row[label_key]}' в число. Ошибка: {e}.")
                                rows_skipped += 1
                                continue
                            except Exception as e:
                                logging.warning(f"Пропущена строка {i+1}: непредвиденная ошибка обработки: {e}.")
                                rows_skipped += 1
                                continue
                except FileNotFoundError:
                    logging.error(f"Файл данных не найден: {data_file}")
                    raise
                except Exception as e:
                    logging.error(f"Ошибка чтения CSV файла {data_file}: {e}")
                    raise
            else:
                logging.error(f"Неподдерживаемый формат данных: {data_format}")
                raise ValueError(f"Неподдерживаемый формат данных: {data_format}")
            
            if not input_ids_list:
                logging.error("Данные не загружены. Проверьте путь к файлу, формат и содержимое.")
                raise ValueError("Данные не загружены.")
            
            self.full_dataset_for_cv = list(zip(input_ids_list, attention_masks_list, labels_list))
            logging.info(f"Полный датасет загружен. Количество записей: {len(self.full_dataset_for_cv)}")
        
        if not train_dataset_fold and not val_dataset_fold:
            total_size = len(self.full_dataset_for_cv)
            train_size = int(self.config['train_split_ratio'] * total_size)
            val_size = total_size - train_size
            if train_size == 0 or val_size == 0:
                 logging.error(f"Размер датасета слишком мал для разделения. Обуч.: {train_size}, Валид.: {val_size}")
                 raise ValueError("Датасет слишком мал для разделения на обучающую/валидационную выборки.")
            
            generator = torch.Generator().manual_seed(self.config['seed'])
            train_dataset_standard, val_dataset_standard = random_split(
                self.full_dataset_for_cv, [train_size, val_size], generator=generator
            )
            batch_size = self.config['batch_size']
            self.train_dataloader, self.val_dataloader = create_dataloaders(
                train_dataset_standard, val_dataset_standard, batch_size, collate_fn
            )
            logging.info(f"Стандартное разделение: Обуч. DL: {len(self.train_dataloader.dataset)}, Валид. DL: {len(self.val_dataloader.dataset)}")

        elapsed = time.time() - start_time
        logging.info(f"Подготовка данных завершена за {elapsed:.2f} сек.")


    def initialize_model(self):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        model_source = ""
        if self.config['ft_model']:
            try:
                self.model = torch.load(self.config['ft_model'], map_location=self.device)
                model_source = f"полная модель из файла: {self.config['ft_model']}"
            except Exception as e:
                logging.error(f"Не удалось загрузить полную модель из {self.config['ft_model']}: {e}. Загружается базовая модель {self.config['model_name']}.")
                self.model = MobileBertForSequenceClassification.from_pretrained(self.config['model_name'], num_labels=self.config.get('num_labels', 2))
                model_source = f"базовая модель {self.config['model_name']} (ошибка загрузки ft_model)"
        elif self.config['ft_model_sd']:
            self.model = MobileBertForSequenceClassification.from_pretrained(self.config['model_name'], num_labels=self.config.get('num_labels', 2))
            try:
                self.model.load_state_dict(torch.load(self.config['ft_model_sd'], map_location=self.device))
                model_source = f"state_dict из файла: {self.config['ft_model_sd']} (на базе {self.config['model_name']})"
            except Exception as e:
                logging.error(f"Не удалось загрузить state dict из {self.config['ft_model_sd']}: {e}. Используются веса базовой модели.")
                model_source = f"базовая модель {self.config['model_name']} (ошибка загрузки ft_model_sd)"
        else:
            self.model = MobileBertForSequenceClassification.from_pretrained(self.config['model_name'], num_labels=self.config.get('num_labels', 2))
            model_source = f"предобученная модель: {self.config['model_name']}"
        
        self.model.to(self.device)
        
        opt_name = self.config['optimizer']
        if opt_name == 'Adafactor':
            self.optimizer = Adafactor(
                self.model.parameters(),
                lr=self.config.get('lr'),
                weight_decay=self.config.get('weight_decay', 0.0),
                relative_step=self.config.get('adafactor_relative_step', True),
                warmup_init=self.config.get('adafactor_warmup_init', False)
            )
        elif opt_name == 'AdamW':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config['lr'],
                weight_decay=self.config['weight_decay']
            )
        else:
            logging.error(f"Неподдерживаемый оптимизатор: {opt_name}")
            raise ValueError(f"Неподдерживаемый оптимизатор: {opt_name}")
        
        if opt_name == 'AdamW' and self.config['scheduler'] == 'linear':
            if self.train_dataloader is None or len(self.train_dataloader) == 0:
                 logging.error("Невозможно настроить планировщик: DataLoader для обучения не инициализирован или пуст.")
                 raise ValueError("DataLoader для обучения должен быть инициализирован и не пуст перед настройкой планировщика.")
            total_steps = len(self.train_dataloader) * self.config['EPOCHS']
            if total_steps == 0 :
                logging.warning("Total_steps для планировщика равен 0. Планировщик не будет создан.")
                self.scheduler = None
            else:
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=self.config['warmup_steps'],
                    num_training_steps=total_steps
                )
        else:
            self.scheduler = None
        logging.info(f"Модель инициализирована с источником: {model_source}")


    def evaluate(self):
        start_time = time.time()
        self.model.eval()
        all_pred_ids = []
        all_label_ids = []
        all_probs = []
        total_eval_loss = 0

        if not self.val_dataloader or len(self.val_dataloader) == 0:
            logging.warning("Валидационный DataLoader недоступен или пуст. Оценка пропускается.")
            return 0.0, 0.0, 0.0, 0.0, 0.0
            
        for batch_idx, batch in enumerate(self.val_dataloader):
            b_input_ids = batch[0].to(self.device)
            b_attention_masks = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=b_input_ids,
                    attention_mask=b_attention_masks,
                    labels=b_labels
                )
            loss = outputs.loss
            logits = outputs.logits
            total_eval_loss += loss.item()
            
            pred_ids = torch.argmax(logits, dim=1)
            all_pred_ids.extend(pred_ids.cpu().numpy())
            all_label_ids.extend(b_labels.cpu().numpy())
            
            if logits.shape[1] == 2:
                 probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                 all_probs.extend(probs)
            elif logits.shape[1] == 1:
                 probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                 all_probs.extend(probs)


        avg_eval_loss = total_eval_loss / len(self.val_dataloader) if len(self.val_dataloader) > 0 else 0
        
        precision = precision_score(all_label_ids, all_pred_ids, average='weighted', zero_division=0)
        recall = recall_score(all_label_ids, all_pred_ids, average='weighted', zero_division=0)
        f1 = f1_score(all_label_ids, all_pred_ids, average='weighted', zero_division=0)
        
        roc_auc = 0.0
        unique_labels = np.unique(all_label_ids)
        if len(all_probs) == len(all_label_ids) and len(unique_labels) > 1:
             try:
                 roc_auc = roc_auc_score(all_label_ids, all_probs)
             except ValueError as e:
                 logging.warning(f"Не удалось вычислить ROC AUC: {e}. Установлено значение 0.")
                 roc_auc = 0.0
        elif len(unique_labels) <= 1:
             logging.warning(f"В валидационных метках присутствует только один класс ({unique_labels}) или нет вероятностей. ROC AUC не определен, установлено значение 0.")
             roc_auc = 0.0
        
        elapsed = time.time() - start_time
        logging.info(f"  Оценка: Loss={avg_eval_loss:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f}, Время={elapsed:.2f} сек")
        return avg_eval_loss, precision, recall, f1, roc_auc

    def train(self):
        if not self.train_dataloader or not self.val_dataloader or len(self.train_dataloader) == 0:
            logging.error("DataLoader'ы не инициализированы или пусты. Обучение невозможно.")
            return []
        if not self.model or not self.optimizer:
             logging.error("Модель или оптимизатор не инициализированы. Обучение невозможно.")
             return []

        epochs = self.config['EPOCHS']
        use_amp = self.config['use_amp'] and self.device.type == 'cuda'
        log_batches = self.config['log_batches']
        model_save_name_base = f"{self.config.get('model_display_name', 'model')}_{self.config.get('version', 'vx')}"
        
        scaler = GradScaler(enabled=use_amp)
        
        start_train_time = datetime.datetime.now()
        logging.info(f"Начало обучения. Количество эпох: {epochs}. Время: {start_train_time}")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            logging.info(f"--- НАЧАЛО ЭПОХИ {epoch + 1}/{epochs} ---")
            self.model.train()
            total_train_loss = 0
            batch_step = 0
            epoch_batch_start_time = time.time()
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                b_input_ids = batch[0].to(self.device)
                b_attention_masks = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)
                
                with autocast(device_type=self.device.type, enabled=use_amp):
                    outputs = self.model(
                        input_ids=b_input_ids,
                        attention_mask=b_attention_masks,
                        labels=b_labels
                    )
                    loss = outputs.loss
                
                scaler.scale(loss).backward()
                if use_amp:
                    scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
                
                if self.scheduler:
                    self.scheduler.step()
                
                current_loss = loss.item()
                total_train_loss += current_loss
                batch_step += 1
                
                if batch_step % log_batches == 0 and batch_step > 0:
                    elapsed_batch_time = time.time() - epoch_batch_start_time
                    avg_batch_time = elapsed_batch_time / log_batches
                    try:
                       current_lr = self.optimizer.param_groups[0]['lr']
                    except:
                       current_lr = float('nan') 
                    logging.info(f"  Эпоха {epoch+1}/{epochs}, Шаг {batch_step}/{len(self.train_dataloader)}, Потеря (Loss): {current_loss:.4f}, Среднее время/батч: {avg_batch_time:.3f} сек, LR: {current_lr:.2e}")
                    epoch_batch_start_time = time.time()

            avg_train_loss = total_train_loss / len(self.train_dataloader) if len(self.train_dataloader) > 0 else 0
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            
            logging.info(f"--- ЗАВЕРШЕНИЕ ЭПОХИ {epoch + 1}/{epochs} ---")
            logging.info(f"  Средняя потеря (Loss) на обучении: {avg_train_loss:.4f}")
            logging.info(f"  Длительность эпохи: {epoch_duration:.2f} сек.")
            
            avg_eval_loss, precision, recall, f1, roc_auc = self.evaluate()
            
            epoch_results = {
                'epoch': epoch + 1,
                'avg_train_loss': avg_train_loss,
                'avg_eval_loss': avg_eval_loss,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'epoch_duration_sec': epoch_duration
            }
            self.training_results.append(epoch_results)
            
            if self.config.get('save_each_epoch', True):
                ep_model_sd_loc = os.path.join(self.config['save_loc'], f"{model_save_name_base}_epoch_{epoch+1}_sd.pt")
                try:
                    torch.save(self.model.state_dict(), ep_model_sd_loc)
                    logging.info(f"  Модель сохранена: {ep_model_sd_loc}")
                except Exception as e:
                    logging.error(f"Ошибка сохранения state_dict модели для эпохи {epoch+1}: {e}")
        
        end_train_time = datetime.datetime.now()
        total_training_duration = end_train_time - start_train_time
        logging.info(f"--- ОБУЧЕНИЕ ЗАВЕРШЕНО ---")
        logging.info(f"Время окончания: {end_train_time}")
        logging.info(f"Общая длительность обучения: {total_training_duration}")
        
        return self.training_results

    def run(self, train_dataset_fold=None, val_dataset_fold=None):
        try:
            self.training_results = [] 
            self.load_and_prepare_data(train_dataset_fold, val_dataset_fold)
            self.initialize_model()
            results = self.train()
            return results
        except Exception as e:
            logging.exception("Произошла критическая ошибка во время выполнения пайплайна.")
            logging.error(f"Конфигурация во время ошибки: {self.config}")
            return []