import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.optim as optim
import inspect

def get_segmentation_masks(outputs, threshold=0.5):
    probs = torch.sigmoid(outputs)
    masks = (probs > threshold)*1.0
    return masks

class YOLOAdapter(L.LightningModule):
    
    def __init__(self, model, learning_rate = 0.0001, loss_fn = smp.losses.DiceLoss, optimizer = optim.Adam, scheduler = None, threshold = 0.5, current_fold = 0, **kwargs):
        super().__init__()
        self.model = model
        
        self.learning_rate = learning_rate
        
        if "mode" not in inspect.signature(loss_fn).parameters:
            self.loss_fn = loss_fn()
        else:
            self.loss_fn = loss_fn(smp.losses.BINARY_MODE, from_logits=True)
            # self.loss_fn = loss_fn(smp.losses.BINARY_MODE, from_logits=True, ignore_index=-1)
            
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.kwargs = kwargs
        
        self.threshold = threshold
    
        self.current_fold = current_fold
        
        self.stage_step_outputs = {'train': [], 'val': [], 'test': []}
        
        self.save_hyperparameters(ignore=['model'])
        
    def forward(self, x):
        return self.model(x)
    
    def shared_step(self, batch, stage):
        x, y = batch
        y_logits = self(x)
        
        loss = self.loss_fn(y_logits, y)
        
        # Descomentar si se utiliza el parametro ignore_index=-1
        # y[y == -1] = 0
        
        y_hat = get_segmentation_masks(y_logits, self.threshold)
        
        tp, fp, fn, tn = smp.metrics.get_stats(y_hat.long(), y.long(), mode="binary")
        
        self.stage_step_outputs[stage].append({"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
        
        return {"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        
    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([m["tp"] for m in outputs])
        fp = torch.cat([m["fp"] for m in outputs])
        fn = torch.cat([m["fn"] for m in outputs])
        tn = torch.cat([m["tn"] for m in outputs])
        
        total_loss = 0
        iter_count = len(outputs)
        
        for idx in range(iter_count):
            total_loss += outputs[idx]['loss'].item()
        
        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")

        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="macro")
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="macro")
        
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        
        metrics = {
            f"{stage}_fold": self.current_fold,
            f"{stage}_loss": total_loss/iter_count,
            f"{stage}_precision": precision,
            f"{stage}_recall": recall,
            f"{stage}_accuracy": accuracy,
            f"{stage}_f1_score": f1_score,
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
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
        return self.shared_step(batch, "test")  

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