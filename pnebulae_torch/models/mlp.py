import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
import pytorch_lightning as L
from torch.optim import Adam
import segmentation_models_pytorch as smp
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
from skimage import morphology
from pnebulae_torch.utils import plot_all


class MLPClassifier(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        self.layer1_linear = nn.Linear(input_dim, hidden_dims[0])
        
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims)-1)])
        
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x):
        x = F.relu(self.layer1_linear(x))
        
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            
        return self.output_layer(x)

metrics_fncs = {
    "accuracy": tm.Accuracy, 
    "precision": tm.Precision, 
    "recall": tm.Recall, 
    "f1": tm.F1Score
    }

class MLPModel(L.LightningModule):
    def __init__(self, model, learning_rate = 1e-4, loss_fn = nn.BCEWithLogitsLoss, optimizer = Adam, scheduler = None, threshold = 0.5, current_fold = 0):
        super().__init__()
        self.model = model
        
        self.learning_rate = learning_rate
        
        if smp.losses.__name__ in loss_fn.__module__:
            self.loss_fn = loss_fn(smp.losses.BINARY_MODE, from_logits=True)
        else:
            self.loss_fn = loss_fn()
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.threshold = threshold
        
        self.current_fold = current_fold
        
        self.metrics_dict = nn.ModuleDict({
            "train_metrics": nn.ModuleDict({f"train_{metric_name}": metric_fnc(task="binary", threshold=self.threshold) for metric_name, metric_fnc in metrics_fncs.items()}),
            "val_metrics": nn.ModuleDict({f"val_{metric_name}": metric_fnc(task="binary", threshold=self.threshold) for metric_name, metric_fnc in metrics_fncs.items()}),
            "test_metrics": nn.ModuleDict({f"test_{metric_name}": metric_fnc(task="binary", threshold=self.threshold) for metric_name, metric_fnc in metrics_fncs.items()})
        })
        
        self.stage_step_outputs = {'train': [], 'val': [], 'test': []}
        self.preds = []
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.model(x)
    
    def shared_step(self, batch, stage):
        x, y = batch
        y_logits = self(x)
        
        loss = self.loss_fn(y_logits, y)
        # self.log(f"{stage}_loss", loss)
        
        y_hat = torch.sigmoid(y_logits)
        
        for metric_name, metric_fnc in self.metrics_dict[stage+"_metrics"].items():
            metric_fnc(y_hat, y)
            self.log(f"{metric_name}", metric_fnc, sync_dist=True, on_step=False, on_epoch=True)
        
        self.stage_step_outputs[stage].append({"loss": loss})
        
        return {"loss": loss}
        
    def personal_test_step(self, batch, stage):
        x, y = batch
        y_logits = self(x)
        
        loss = self.loss_fn(y_logits, y)
        # self.log(f"{stage}_loss", loss)
        
        y_hat = torch.sigmoid(y_logits)
        
        self.preds.append(y_hat)
        
        for metric_name, metric_fnc in self.metrics_dict[stage+"_metrics"].items():
            metric_fnc(y_hat, y)
            self.log(f"{metric_name}", metric_fnc, sync_dist=True, on_step=False, on_epoch=True)
        
        self.stage_step_outputs[stage].append({"loss": loss})
        
        return {"loss": loss}
    
    def shared_epoch_end(self, outputs, stage):
        total_loss = 0
        iter_count = len(outputs)
        
        for idx in range(iter_count):
            total_loss += outputs[idx]['loss'].item()
        
        metrics = {
            f"{stage}_fold": self.current_fold,
            f"{stage}_loss": total_loss/iter_count,
        }
        
        self.log_dict(metrics, sync_dist = True)
        
        self.stage_step_outputs[stage].clear()
        
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def on_train_epoch_end(self):
        outputs = self.stage_step_outputs['train']
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def on_validation_epoch_end(self):
        outputs = self.stage_step_outputs['val']
        return self.shared_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self.personal_test_step(batch, "test")  

    def on_test_epoch_end(self):
        outputs = self.stage_step_outputs['test']
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer, **self.kwargs)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler,'monitor': 'val_loss'}
        else:
            return optimizer
        
def classBalancing(subset, size = None):
    x_trainBal = torch.empty((0, subset[0][0].shape[0]))
    y_trainBal = torch.empty((0, 1))
    
    for x, y in subset:
        xFlat = x.permute(1,2,0).view(-1, x.shape[0])
        yFlat = y.permute(1,2,0).view(-1, 1)
        
        idxObject = np.where(yFlat == 1)[0]
        idxBackground = np.where(yFlat == 0)[0]
        
        np.random.shuffle(idxBackground)
        if size == None or size > len(idxObject) or size > len(idxBackground):
            if len(idxObject) <= len(idxBackground):
                idxs = np.concatenate((idxObject, idxBackground[:len(idxObject)]))
            else:
                idxs = np.concatenate((idxObject[:len(idxBackground)], idxBackground))
        else:
            np.random.shuffle(idxObject)
            idxs = np.concatenate((idxObject[:size], idxBackground[:size]))

        np.random.shuffle(idxs)
        
        x_trainBal = torch.concatenate((x_trainBal, xFlat[idxs,:]), dim=0)
        y_trainBal = torch.concatenate((y_trainBal, yFlat[idxs,:]), dim=0)
        
    return x_trainBal, y_trainBal

def testFlatten(subset):
    x_testFlat = torch.empty((0, subset[0][0].shape[0]))
    y_testFlat = torch.empty((0, 1))
    for x, y in subset:
        xFlat = x.permute(1,2,0).view(-1, x.shape[0])
        yFlat = y.permute(1,2,0).view(-1, 1)
        
        x_testFlat = torch.concatenate((x_testFlat, xFlat), dim = 0)
        y_testFlat = torch.concatenate((y_testFlat, yFlat), dim = 0)
        
    return x_testFlat, y_testFlat

class FlattenSubset(Dataset):
    """
    Dataset wrapper personalizado que aplica la transformación (si se le introduce) a un Subset dado.

    Args:
        subset (torch.utils.data.Dataset): Subset del dataset original.
        transform (callable, optional): Función de transformación para aplicar a los datos introducidos. Default is None.
    """
    def __init__(self, subset, train = False, size = None):
        self.subset = subset
        
        if train:
            self.x_flatten, self.y_flatten = classBalancing(subset, size)
        else:
            self.x_flatten, self.y_flatten = testFlatten(subset)
            
    def __getitem__(self, index):
        x = self.x_flatten[index]
        y = self.y_flatten[index]
        return x, y

    def __len__(self):
        return len(self.x_flatten)
    
def evaluate_rebuild_images(dataset, model, preprocessing = False, plot = False):
    results = {"iou": [], "f1": [], "precision": [], "accuracy": [], "recall": []}

    for i in range(len(dataset)):
        image, mask = dataset[i]
        
        x_flatten, y_flatten = testFlatten(torch.utils.data.Subset(dataset, [i]))

        y_logits = model(x_flatten)
        y_hat = torch.sigmoid(y_logits)
        y_pred = y_hat > 0.5
        output = y_pred.view((1, image.shape[1], image.shape[2]))
        
        if preprocessing:
            output = morphology.remove_small_objects(output.permute(1,2,0).numpy()[:,:,0], min_size=image.shape[1]*image.shape[2]*0.01)
            output = torch.tensor(output).unsqueeze(0)
        
        tp, fp, fn, tn = smp.metrics.get_stats(output, mask, mode='binary')
        
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")        # Índice de Jaccard
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")          # F1-Score
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")          # Accuracy
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")    # Sensibilidad
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")        # Precisión

        results["iou"].append(iou_score)
        results["f1"].append(f1_score)
        results["precision"].append(precision)
        results["accuracy"].append(accuracy)
        results["recall"].append(recall)
        
        if plot:
            plot_all(image, mask, cmap = "gray")
            plt.figure()
            plt.imshow(output.permute(1,2,0)[:,:,0], cmap = "gray")
            plt.title(f"Segmentation")
            plt.show()

    df_results = pd.DataFrame(results)
    print(df_results.astype(float).describe().loc[['mean', 'std']].transpose().to_markdown())