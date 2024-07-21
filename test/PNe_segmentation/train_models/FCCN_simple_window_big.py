from api_keys import set_wandb_api_key
from pnebulae_torch.dataset import NebulaeDataset
from pnebulae_torch.preprocess import ApplyMorphology, ApplyIntensityTransformation, ApplyFilter, CustomPad
from pnebulae_torch.normalize import TypicalImageNorm
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
    
    torch.set_float32_matmul_precision('medium')
    
    ####### CONFIGURACIÓN ENTRENAMIENTO #######
    model_name = "FCCN_simple_window_dice_relu_512_ks9"
    
    BATCH_SIZE = 256
    num_epochs = 2000
    lr = 1e-4
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
                        TypicalImageNorm(factor = 1, substract=0),
                        # ApplyMorphology(operation = morphology.binary_opening, concat = True, footprint = morphology.disk(2)),
                        # ApplyMorphology(operation = morphology.area_opening, concat = True, area_threshold = 200, connectivity = 1),
                        # ApplyIntensityTransformation(transformation = exposure.equalize_hist, concat = True, nbins = 4096),
                        # ApplyIntensityTransformation(transformation = exposure.equalize_adapthist, concat = True, nbins = 640, kernel_size = 5),
                        # ApplyMorphology(operation = morphology.area_opening, concat = True, area_threshold = 200, connectivity = 1),
                        # ApplyFilter(filter = ndimage.gaussian_filter, concat = True, sigma = 5),
                        transforms.ToTensor(),
                        # CustomPad(target_size = (1984, 1984), fill_min=True, tensor_type=torch.Tensor.float)
                        ])

    transform_y = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: type_fnc(x)),
                        # CustomPad(target_size = (1984, 1984), fill = 0, tensor_type=torch.Tensor.int)
                        ])

    df_train = pd.read_csv("data_files_1c_train.csv")
    dataset_train = NebulaeDataset(data_directory, masks_directory, df_train, transform = (transform_x, transform_y))
    
    df_test = pd.read_csv("data_files_1c_test.csv")
    dataset_test = NebulaeDataset(data_directory, masks_directory, df_test, transform = (transform_x, transform_y))

    seed_everything(42, workers = True)
    
    ########## ENTRENAMIENTO MODELO ##########
    # Definimos el K-fold Cross Validator
    kfold = KFold(n_splits=5, shuffle=True, random_state = 42)
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset_train)):
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=os.environ["STORE"] + f"/TFG/model_checkpoints/{model_name}",
            filename='best_model-{epoch:02d}-'+str(fold),
            save_top_k=1,
            mode='min',
        )
        
        callbacks = [PrintCallback(), LearningRateMonitor(logging_interval='epoch'), checkpoint_callback]
        
        model = ConvNet(input_dim = 1, hidden_dims = [8, 8, 8, 8, 8], output_dim = 1, transposeConv=False, separable_conv=False, activation_layer=activation_layer, kernel_size = 9, padding = 'same')
        
        # Definimos el modelo con los pesos inicializados aleatoriamente (sin preentrenar)
        model = smpAdapter(model = model, learning_rate=lr, threshold=0.5, current_fold=fold, loss_fn=loss_fn, scheduler=None)
        # model = smpAdapter(model = model, learning_rate=lr, threshold=0.5, current_fold=fold, loss_fn=loss_fn, scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau, mode='min', factor=0.1, patience=20, cooldown=5, verbose=False)
        # model = UNETModel(model = model, learning_rate=5e-6, current_fold=fold, loss_fn=loss_fn, scheduler=optim.lr_scheduler.StepLR, step_size = 15, gamma = 0.1, verbose=False)
        # model = UNETModel(model = model, learning_rate=1e-6, current_fold=fold, loss_fn=loss_fn, scheduler=optim.lr_scheduler.MultiStepLR, milestones = [91], gamma = 0.1, verbose=False)
        
        ruta_logs_wandb = os.environ["STORE"] + "/TFG/logs_wandb/"
        logger_wandb = WandbLogger(project="segmentation_TFG", log_model = False, name=model_name, save_dir=ruta_logs_wandb)
        logger_wandb.experiment.config.update({"model_name": model_name})

        # log gradients, parameter histogram and model topology
        logger_wandb.watch(model, log="all")

        trainer = L.Trainer(strategy='auto', max_epochs=num_epochs, accelerator='cuda', log_every_n_steps=1, logger= logger_wandb, callbacks=callbacks)

        # Imprimimos el fold del que van a mostrarse los resultados
        print('--------------------------------')
        print(f"Model info:\n\t- Batch Size: {BATCH_SIZE}\n\t- GPUs on use: {torch.cuda.device_count()}")

        # Creamos nuestros propios Subsets de PyTorch aplicando a cada conjunto la transformacion deseada
        train_subset = torch.utils.data.Subset(dataset_train, train_ids)
        val_subset = torch.utils.data.Subset(dataset_train, val_ids)
        
        if window_shape is not None:
            train_subset = DivideWindowsSubset(train_subset, window_shape = window_shape, fill_min = True)
            val_subset = DivideWindowsSubset(val_subset, window_shape = window_shape, fill_min = True)
        
        # Definimos un data loader por cada conjunto de datos que vamos a utilizar.
        trainloader = torch.utils.data.DataLoader(
                                train_subset,
                                batch_size=BATCH_SIZE, num_workers=8, shuffle=True, persistent_workers=True)

        valloader = torch.utils.data.DataLoader(
                                val_subset,
                                batch_size=BATCH_SIZE, num_workers=8, shuffle=False, persistent_workers=True)
        
        # Entrenamos el modelo, extrayendo los resultados y guardandolos en la variable result, y evaluamos en el conjunto de test.
        trainer.fit(model, trainloader, valloader) 

        logger_wandb.experiment.unwatch(model)

        testloader = torch.utils.data.DataLoader(
                                dataset_test,
                                batch_size=BATCH_SIZE, num_workers=8, shuffle=False, persistent_workers=True)
        
        # Creamos un nuevo entrenador con una sola GPU para la fase de prueba
        # trainer_test = L.Trainer(devices = 1, strategy='auto', max_epochs=num_epochs, accelerator='cuda', log_every_n_steps=1, logger=logger_wandb, callbacks=callbacks)
        # trainer_test.test(model, testloader)

        logger_wandb.finalize("success")
        wandb.finish() 
        break