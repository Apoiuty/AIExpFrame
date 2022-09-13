import os

import wandb

from utils.func import train_model

os.environ['WANDB_API_KEY'] = '929a39a3325d5090f997653bf3dacad61eb58687'

model_config = {
    'hidden_channels': 16,
    'num_layers': 2,
}
run = wandb.init(project='test', tags=['GNN'], group='GCN', name='GIN', id='GIN',)
print(run.id)
train_model(model_config, 10, 'MUTAG', 42, 'Logs/GIN', wandb_run=run)
