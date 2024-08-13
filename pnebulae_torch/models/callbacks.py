import wandb
from pnebulae_torch.models.smp import get_segmentation_masks
import lightning as L
import skimage.morphology as morphology
import torch

class PrintCallback(L.pytorch.Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")
        
    def on_train_end(self, trainer, pl_module):
        print("Training is done.")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when the validation batch ends."""
 
        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case
        try:
            # Let's log 20 sample image predictions from first batch
            if (trainer.current_epoch % 250 == 0 and trainer.current_epoch != 0) or trainer.current_epoch == trainer.max_epochs-1:
                if batch_idx == 0:
                    n = 4
                    x, y = batch
                    
                    outputs = get_segmentation_masks(pl_module(x))
                    
                    if pl_module.postprocess:
                        y_hat_post = outputs.clone().cpu().permute(0,2,3,1).squeeze(3).numpy().astype(int)
                        for i in range(outputs.shape[0]):
                            y_hat_post[i] = morphology.binary_opening(y_hat_post[i], footprint=morphology.disk(5))
                            y_hat_post[i] = morphology.remove_small_objects(y_hat_post[i], min_size=y_hat_post[i].shape[0]*y_hat_post[i].shape[1]*0.01)
                        outputs = torch.tensor(y_hat_post).unsqueeze_(-1).permute(0, 3, 1, 2).cuda()
                    
                    if x.shape[1] > 1:
                        columns = ["image", 'equalize_image', "ground truth", "prediction"]
                        data = [
                            [wandb.Image(x_i[0].float()), wandb.Image(x_i[1].float()), wandb.Image(y_i.float()), wandb.Image(y_pred.float())] for x_i,
                            y_i,
                            y_pred in list(zip(x, y, outputs))
                        ]
                    else:
                        columns = ["image", "ground truth", "prediction"]
                        data = [
                            [wandb.Image(x_i.float()), wandb.Image(y_i.float()), wandb.Image(y_pred.float())] for x_i,
                            y_i,
                            y_pred in list(zip(x, y, outputs))
                        ]
                    
                    trainer.loggers[-1].log_table(key=f"table_epoch_{trainer.current_epoch}_fold_{pl_module.current_fold}", columns=columns, data=data)
        except:
            pass
        
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when the validation batch ends."""
 
        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case
        try:
            x, y = batch
            outputs = get_segmentation_masks(pl_module(x))
            
            if pl_module.postprocess:
                y_hat_post = outputs.clone().cpu().permute(0,2,3,1).squeeze(3).numpy().astype(int)
                for i in range(outputs.shape[0]):
                    y_hat_post[i] = morphology.binary_opening(y_hat_post[i], footprint=morphology.disk(5))
                    y_hat_post[i] = morphology.remove_small_objects(y_hat_post[i], min_size=y_hat_post[i].shape[0]*y_hat_post[i].shape[1]*0.01)
                outputs = torch.tensor(y_hat_post).unsqueeze_(-1).permute(0, 3, 1, 2).cuda()
            
            if x.shape[1] > 1:
                columns = ["image", 'equalize_image', "ground truth", "prediction"]
                data = [
                    [wandb.Image(x_i[0].float()), wandb.Image(x_i[1].float()), wandb.Image(y_i.float()), wandb.Image(y_pred.float())] for x_i,
                    y_i,
                    y_pred in list(zip(x, y, outputs))
                ]
            else:
                columns = ["image", "ground truth", "prediction"]
                data = [
                    [wandb.Image(x_i[0].float()), wandb.Image(y_i.float()), wandb.Image(y_pred.float())] for x_i,
                    y_i,
                    y_pred in list(zip(x, y, outputs))
                ]
            
            trainer.loggers[-1].log_table(key=f"table_test_fold_{pl_module.current_fold}_batch_{batch_idx}", columns=columns, data=data)
        except:
            print("Error")
            pass