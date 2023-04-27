import gin
import torch
import wandb
import lightning as lit

import numpy as np

from fire import Fire
from loguru import logger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from ModelManager import get_model, LighntingE2EModelUnfolding
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger

from data import batch_preparation_ctc, load_dataset

import os

torch.set_float32_matmul_precision('high')

@gin.configurable
def main(data_path=None, corpus_name=None, model_name=None):
    outpath = f"out/{corpus_name}/{model_name}"
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(f"{outpath}/hyp", exist_ok=True)
    os.makedirs(f"{outpath}/gt", exist_ok=True)

    train_dataset, val_dataset, test_dataset = load_dataset(data_path=data_path, corpus_name=corpus_name)
    
    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=20, collate_fn=batch_preparation_ctc)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=20, collate_fn=batch_preparation_ctc)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=20, collate_fn=batch_preparation_ctc)
    
    m_width, m_height = train_dataset.get_max_hw()

    wandb_logger = WandbLogger(project=f'AMNLT_ICDAR2023', name=model_name, group=corpus_name)
#
    model = get_model(maxwidth=m_width, maxheight=m_height, 
                      in_channels=1, out_size=train_dataset.vocab_size()+1, 
                      blank_idx=train_dataset.vocab_size(), i2w=train_dataset.get_i2w(), 
                      model_name=model_name, output_path=outpath)
    
    early_stopping = EarlyStopping(monitor='val_KER', min_delta=0.01, patience=5, mode="min", verbose=True)
    checkpointer = ModelCheckpoint(dirpath=f"weights/{corpus_name}", filename=f"{model_name}", 
                                   monitor="val_KER", mode='min',
                                   save_top_k=1, verbose=True)
    
    trainer = lit.Trainer(max_epochs=1000, callbacks=[early_stopping, checkpointer], logger=wandb_logger)
#
    trainer.fit(model, train_dataloader, val_dataloader)
    model = model.load_from_checkpoint(checkpointer.best_model_path)
    trainer.test(model, test_dataloader)
    wandb.finish()


def launch(config):
    gin.parse_config_file(config)
    main()

if __name__ == "__main__":
    Fire(launch)