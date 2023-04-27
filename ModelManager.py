import gin
import torch
import wandb
import lightning as L
import random
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from itertools import groupby
from evaluation_metrics import compute_metrics

from models.E2EScoreUnfolding import get_FCN_model, get_CRNN_model

CONST_MODEL_IMPLEMENTATIONS = {
    "FCN": get_FCN_model,
    "CRNN": get_CRNN_model
}

class LighntingE2EModelUnfolding(L.LightningModule):
    def __init__(self, model, blank_idx, i2w, output_path) -> None:
        super(LighntingE2EModelUnfolding, self).__init__()
        self.model = model
        self.loss = nn.CTCLoss(blank=blank_idx)
        self.blank_idx = blank_idx
        self.i2w = i2w
        self.accum_ed = 0
        self.accum_len = 0
        
        self.dec_val_ex = []
        self.gt_val_ex = []
        self.img_val_ex = []
        self.ind_val_ker = []

        self.out_path = output_path

        self.save_hyperparameters(ignore=['model'])

    def forward(self, input):
        return self.model(input)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, train_batch, batch_idx):
         X_tr, Y_tr, L_tr, T_tr = train_batch
         predictions = self.forward(X_tr)
         loss = self.loss(predictions, Y_tr, L_tr, T_tr)
         self.log('loss', loss, on_epoch=True, batch_size=1, prog_bar=True)
         return loss

    def compute_prediction(self, batch):
        X, Y, _, _ = batch
        pred = self.forward(X)
        pred = pred.permute(1,0,2).contiguous()
        pred = pred[0]
        out_best = torch.argmax(pred,dim=1)
        out_best = [k for k, g in groupby(list(out_best))]
        decoded = []
        for c in out_best:
            if c.item() != self.blank_idx:
                decoded.append(c.item())
        
        decoded = [self.i2w[tok] for tok in decoded]
        gt = [self.i2w[int(tok.item())] for tok in Y[0]]

        return decoded, gt

    def validation_step(self, val_batch, batch_idx):
        dec, gt = self.compute_prediction(val_batch)
        
        dec = "".join(dec)
        dec = dec.replace("<t>", "\t")
        dec = dec.replace("<b>", "\n")

        gt = "".join(gt)
        gt = gt.replace("<t>", "\t")
        gt = gt.replace("<b>", "\n")

        self.dec_val_ex.append(dec)
        self.gt_val_ex.append(gt)

    def on_validation_epoch_end(self):        
        
        mer, wer, ler, ker = compute_metrics(self.dec_val_ex, self.gt_val_ex)

        self.log('val_MER', mer)
        self.log('val_WER', wer)
        self.log('val_LER', ler)
        self.log('val_KER', ker)

        return ker

    def test_step(self, test_batch, batch_idx):
        dec, gt = self.compute_prediction(test_batch)
        
        dec = "".join(dec)
        dec = dec.replace("<t>", "\t")
        dec = dec.replace("<b>", "\n")

        gt = "".join(gt)
        gt = gt.replace("<t>", "\t")
        gt = gt.replace("<b>", "\n")

        with open(f"{self.out_path}/hyp/{batch_idx}.krn", "w+") as krnfile:
            krnfile.write(dec)
        
        with open(f"{self.out_path}/gt/{batch_idx}.krn", "w+") as krnfile:
            krnfile.write(gt)

        self.dec_val_ex.append(dec)
        self.gt_val_ex.append(gt)
        self.img_val_ex.append((255.*test_batch[0].squeeze(0)))
    
    def on_test_epoch_end(self) -> None:
        mer, wer, ler, ker = compute_metrics(self.dec_val_ex, self.gt_val_ex)

        self.log('val_MER', mer)
        self.log('val_WER', wer)
        self.log('val_LER', ler)
        self.log('val_KER', ker)
        
        columns = ['Image', 'PRED', 'GT']
        data = []

        nsamples = len(self.dec_val_ex) if len(self.dec_val_ex) < 5 else 5
        random_indices = random.sample(range(len(self.dec_val_ex)), nsamples)

        for index in random_indices:
            data.append([wandb.Image(self.img_val_ex[index]), "".join(self.dec_val_ex[index]), "".join(self.gt_val_ex[index])])
        
        table = wandb.Table(columns= columns, data=data)
        
        self.logger.experiment.log(
            {'Validation samples': table}
        )

        self.gt_val_ex = []
        self.dec_val_ex = []
        self.ind_val_ker = []

        return ker

def get_model(maxwidth, maxheight, in_channels, out_size, blank_idx, i2w, model_name, output_path, maxlen=None):
    model = CONST_MODEL_IMPLEMENTATIONS[model_name](in_channels=in_channels, out_size=out_size, maxlen=maxlen)
    lighningModel = LighntingE2EModelUnfolding(model=model, blank_idx=blank_idx, i2w=i2w, output_path=output_path)
    summary(lighningModel, input_size=([1, in_channels, maxheight, maxwidth]))
    return lighningModel