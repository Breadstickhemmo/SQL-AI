config = {
    'model_display_name': 'sql-inject-detector',
    'model_name': 'google/mobilebert-uncased',
    'ft_model': '',
    'ft_model_sd': '',

    'optimizer': 'AdamW',
    'lr': 3e-5,
    'weight_decay': 0.01,
    'scheduler': 'linear',
    'warmup_steps': 94,

    'EPOCHS': 5,
    'batch_size': 16,
    'max_seq_length': 256,
    'use_amp': False,

    'data_file': 'datasets/SQL.csv',
    'data_format': 'csv',
    'data_keys': ['Query', 'Label'],
    'train_split_ratio': 0.8,

    'save_loc': "models/modelmobile_v1",
    'log_batches': 50,
    'version': 'v1',
    'seed': 42,
}