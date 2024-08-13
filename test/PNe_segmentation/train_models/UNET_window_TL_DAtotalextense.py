from api_keys import set_wandb_api_key
from pnebulae_torch.dataset import NebulaeDataset
from pnebulae_torch.preprocess import ApplyMorphology, ApplyIntensityTransformation, ApplyFilter, CustomPad, CutValues
from pnebulae_torch.normalize import TypicalImageNorm, MinMaxImageNorm
from pnebulae_torch.models.callbacks import PrintCallback
from pnebulae_torch.models import basicUNet, smpAdapter, ConvNet
from pnebulae_torch.utils import DivideWindowsSubset
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from sklearn.model_selection import KFold
from torchvision import transforms
from skimage import morphology, exposure
from scipy import ndimage
from lightning.pytorch import seed_everything
from segmentation_models_pytorch.losses import DiceLoss
from lightning.pytorch.loggers import WandbLogger
import torch
import os
import pandas as pd
import lightning as L
import wandb
import inspect
import time
import gc
import numpy as np
import segmentation_models_pytorch as smp

if __name__ == "__main__":
    ########## CONFIGURACIÓN SCRIPT ##########
    # Establecemos la clave de la API de W&B
    set_wandb_api_key()
    ruta_logs_wandb = os.environ["STORE"] + "/TFG/logs_wandb/"
    
    working_directory = "../"
    
    os.chdir(working_directory)
    
    working_directory = os.getcwd()
    
    masks_directory = working_directory+"/masks"
    data_directory = working_directory+"/data"
    
    torch.set_float32_matmul_precision('high')
    
    ####### CONFIGURACIÓN ENTRENAMIENTO #######
    model_name = "UNet_mobilenet_v2_cut2_hist_TL_DAtotalextense"
    
    BATCH_SIZE = 52
    num_epochs = 500
    lr = 1e-4
    weights = 'imagenet'
    window_shape = 512
    
    k = 5
    
    loss_fn = DiceLoss
    activation_layer=torch.nn.ReLU
    
    if "mode" in inspect.signature(loss_fn).parameters:
        type_fnc = torch.Tensor.int
    else:
        type_fnc = torch.Tensor.float
        
    ############# CARGA DATASET #############
    transform_x = transforms.Compose([
                        # MinMaxNorm,
                        CutValues(factor = 2),
                        TypicalImageNorm(factor = 1, substract=0),
                        # MinMaxImageNorm(min = -88.9933, max=125873.7500),
                        # ApplyMorphology(operation = morphology.binary_opening, concat = True, footprint = morphology.disk(2)),
                        # ApplyMorphology(operation = morphology.area_opening, concat = True, area_threshold = 200, connectivity = 1),
                        # ApplyIntensityTransformation(transformation = exposure.equalize_hist, concat = True, nbins = 4096),
                        # ApplyIntensityTransformation(transformation = exposure.equalize_adapthist, concat = True, nbins = 640, kernel_size = 5),
                        # ApplyMorphology(operation = morphology.area_opening, concat = True, area_threshold = 200, connectivity = 1),
                        # ApplyFilter(filter = ndimage.gaussian_filter, concat = True, sigma = 5),
                        # transforms.ToTensor(),
                        # CustomPad(target_size = (1984, 1984), fill_min=True, tensor_type=torch.Tensor.float)
                        ApplyIntensityTransformation(transformation = exposure.equalize_hist, concat = True, nbins = 256),
                        transforms.ToTensor(),
                        ])

    transform_y = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: type_fnc(x.round())),
                        # CustomPad(target_size = (1984, 1984), fill = 0, tensor_type=torch.Tensor.int)
                        ])

    transform_x_aug = transforms.Compose([
                        # MinMaxNorm,
                        CutValues(factor = 2),
                        TypicalImageNorm(factor = 1, substract=0),
                        
                        transforms.ToPILImage(),
                        
                        transforms.RandomHorizontalFlip(),  # Voltear la imagen horizontalmente con una probabilidad del 50%
                        transforms.RandomRotation(180),  # Rotar la imagen aleatoriamente en un rango de -20 a 20 grados
                        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.75, 1.5)),  # Traslación aleatoria del 10%
                        # transforms.ColorJitter(brightness = (0.9, 1.25), contrast = (0.9, 1.25)),  # Cambios aleatorios en la saturación, brillo y contraste
                        transforms.Lambda(lambda x: np.array(x)),
                        
                        ApplyIntensityTransformation(transformation = exposure.equalize_hist, concat = True, nbins = 256),
                        transforms.ToTensor(),
                        ])

    transform_y_aug = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.RandomHorizontalFlip(),  # Voltear la imagen horizontalmente con una probabilidad del 50%
                        transforms.RandomRotation(180),  # Rotar la imagen aleatoriamente en un rango de -20 a 20 grados
                        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.75, 1.5)),  # Traslación aleatoria del 10%
                        
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: type_fnc(x.round())),
                        # CustomPad(target_size = (1984, 1984), fill = 0, tensor_type=torch.Tensor.int)
                        ])
    
    df_train = pd.read_csv("data_files_1c_train.csv")
    dataset_train = NebulaeDataset(data_directory, masks_directory, df_train, transform = (transform_x, transform_y))
    
    dataset_train_aug = NebulaeDataset(data_directory, masks_directory, df_train, transform = (transform_x_aug, transform_y_aug))
    
    df_test = pd.read_csv("data_files_1c_test.csv")
    dataset_test = NebulaeDataset(data_directory, masks_directory, df_test, transform = (transform_x, transform_y))

    df_train_ext = pd.read_csv("data_files_1c_train_da.csv")
    
    seed_everything(42, workers = True)
    
    ########## ENTRENAMIENTO MODELO ##########
    # Definimos el K-fold Cross Validator
    kfold = KFold(n_splits=k, shuffle=True, random_state = 42)
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset_train)):
        if fold==0:
            continue
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=os.environ["STORE"] + f"/TFG/model_checkpoints/{model_name}",
            filename='best_model-{epoch:02d}-'+str(fold),
            save_top_k=1,
            mode='min',
        )
        
        checkpoint_callback_last = ModelCheckpoint(
            monitor=None,
            dirpath=os.environ["STORE"] + f"/TFG/model_checkpoints/{model_name}",
            filename='last_model_fold'+str(fold),
        )
        
        callbacks = [PrintCallback(), LearningRateMonitor(logging_interval='epoch'), checkpoint_callback, checkpoint_callback_last]
        
        model = smp.Unet(
                        encoder_name = "mobilenet_v2",
                        encoder_weights = weights,
                        in_channels = dataset_train[0][0].shape[0],
                        classes = 1
                        )
        
        # Definimos el modelo con los pesos inicializados aleatoriamente (sin preentrenar)
        model = smpAdapter(model = model, learning_rate=lr, threshold=0.5, current_fold=fold, loss_fn=loss_fn, scheduler=None)
        # model = smpAdapter(model = model, learning_rate=lr, threshold=0.5, current_fold=fold, loss_fn=loss_fn, scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau, mode='min', factor=0.1, patience=500, cooldown=150, verbose=False)
        # model = smpAdapter(model = model, learning_rate=lr, threshold=0.5, current_fold=fold, loss_fn=loss_fn, scheduler=torch.optim.lr_scheduler.StepLR, step_size = 2000, gamma = 0.1, verbose=False)
        # model = smpAdapter(model = model, learning_rate=lr, threshold=0.5, current_fold=fold, loss_fn=loss_fn, scheduler=torch.optim.lr_scheduler.MultiStepLR, milestones = [1000, 4000], gamma = 0.1, verbose=False)
        
        ruta_logs_wandb = os.environ["STORE"] + "/TFG/logs_wandb/"
        logger_wandb = WandbLogger(project="segmentation_TFG", log_model = False, name=model_name, save_dir=ruta_logs_wandb)
        logger_wandb.experiment.config.update({"model_name": model_name})

        # log gradients, parameter histogram and model topology
        logger_wandb.watch(model, log="all")

        trainer = L.Trainer(strategy='auto', max_epochs=num_epochs, accelerator='cuda', log_every_n_steps=2, logger= logger_wandb, callbacks=callbacks)

        # Imprimimos el fold del que van a mostrarse los resultados
        print('--------------------------------')
        print(f"Model info:\n\t- Batch Size: {BATCH_SIZE}\n\t- GPUs on use: {torch.cuda.device_count()}")

        # Creamos nuestros propios Subsets de PyTorch aplicando a cada conjunto la transformacion deseada
        train_subset = torch.utils.data.Subset(dataset_train, train_ids)
        
        indices_ext = list(df_train[df_train['name'].isin(df_train_ext['name'])].dropna().index)
        for i in val_ids:
            indices_ext.remove(i) if i in indices_ext else None
            
        train_subset_ext = torch.utils.data.Subset(dataset_train_aug, indices_ext)
        train_subset_ext2 = torch.utils.data.Subset(dataset_train_aug, indices_ext)
        
        # train_subset = torch.utils.data.ConcatDataset([train_subset, train_subset_ext, train_subset_ext2])
        
        train_subset_aug = torch.utils.data.Subset(dataset_train_aug, train_ids)
        train_subset = torch.utils.data.ConcatDataset([train_subset, train_subset_aug, train_subset_ext, train_subset_ext2])
        
        val_subset = torch.utils.data.Subset(dataset_train, val_ids)
        
        if window_shape is not None:
            train_subset = DivideWindowsSubset(train_subset, window_shape = window_shape, fill_min = True)
            val_subset = DivideWindowsSubset(val_subset, window_shape = window_shape, fill_min = True)
        
        # Definimos un data loader por cada conjunto de datos que vamos a utilizar.
        trainloader = torch.utils.data.DataLoader(
                                train_subset,
                                batch_size=BATCH_SIZE, num_workers=6, shuffle=True, persistent_workers=False)

        valloader = torch.utils.data.DataLoader(
                                val_subset,
                                batch_size=BATCH_SIZE, num_workers=6, shuffle=False, persistent_workers=False)
        
        # Entrenamos el modelo, extrayendo los resultados y guardandolos en la variable result, y evaluamos en el conjunto de test.
        trainer.fit(model, trainloader, valloader) 

        logger_wandb.experiment.unwatch(model)

        # testloader = torch.utils.data.DataLoader(
        #                         dataset_test,
        #                         batch_size=BATCH_SIZE, num_workers=8, shuffle=False, persistent_workers=True)
        
        # Creamos un nuevo entrenador con una sola GPU para la fase de prueba
        # trainer_test = L.Trainer(devices = 1, strategy='auto', max_epochs=num_epochs, accelerator='cuda', log_every_n_steps=1, logger=logger_wandb, callbacks=callbacks)
        # trainer_test.test(model, valloader)

        logger_wandb.finalize("success")
        wandb.finish()
        
        del model
        del trainer
        
        torch.cuda.empty_cache()
        time.sleep(30)