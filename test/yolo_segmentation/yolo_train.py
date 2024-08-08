# from ultralytics import YOLO, checks, hub
# checks()

# hub.login('71454859f9a2ce11ea390ba04b99fac877ea55f8f7')

# model = YOLO('https://hub.ultralytics.com/models/TTIEfE32Y0EgTAgR3kec')
# results = model.train()

import os
from pnebulae_torch.dataset import NebulaeDataset
from pnebulae_torch.preprocess import  CutValues
from pnebulae_torch.normalize import TypicalImageNorm
from pnebulae_torch.utils import DivideWindowsSubset
from sklearn.model_selection import KFold
from torchvision import transforms
from lightning.pytorch import seed_everything
import torch
import os
import pandas as pd
import time
import shutil
from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback

if __name__ == "__main__":
    os.chdir(os.environ["HOME"])
    print("Vamos a cambiar el directorio de trabajo")

    # Indicamos la ruta del directorio de trabajo
    route = os.getcwd()+"/TFG/test/PNe_segmentation"
    os.chdir(route)

    current_directory = os.getcwd()
    print(" El directorio actual es:", current_directory)

    # Listamos el contenido del directorio
    files = os.listdir(current_directory)
    print(" Contenido del directorio actual:")
    for file in files:
        print("\t",file)
        
    # Listamos el contenido del directorio de las máscaras
    # masks_directory = route+"TFG\\test\\PNe_segmentation\\masks"
    # data_directory = route+"TFG\\test\\PNe_segmentation\\data"
    ## Ejecución en el CESGA Finisterrae III
    masks_directory = current_directory+"/masks"
    data_directory = current_directory+"/data"

    os.chdir(route+"/../yolo_segmentation")
    save_directory = os.getcwd()

    # Nuevo código para mover archivos de vuelta a las carpetas de entrenamiento
    val_folders = ['images', 'labels', 'masks']
    for folder in val_folders:
        val_path = os.path.join(save_directory, "segment_kfold", folder, "val")
        train_path = os.path.join(save_directory, "segment_kfold", folder, "train")
        
        for file_name in os.listdir(val_path):
            shutil.move(os.path.join(val_path, file_name), os.path.join(train_path, file_name))

    os.environ["WANDB_API_KEY"] = "21924e6e134841c5c16842c4ac42fcbe5a66feb2"
    ruta_logs_wandb = os.environ["STORE"] + "/TFG/logs_wandb/"

    torch.set_float32_matmul_precision('high')

    ####### CONFIGURACIÓN ENTRENAMIENTO #######
    BATCH_SIZE = -1
    num_epochs = 1000
    lr = 1e-4
    window_shape = 512

    k = 5
        
    ############# CARGA DATASET #############
    transform_x = transforms.Compose([
                        CutValues(factor = 2),
                        TypicalImageNorm(factor = 1, substract=0),
                        transforms.ToTensor(),
                        ])

    transform_y = transforms.Compose([
                        transforms.ToTensor()
                        ])

    df_train = pd.read_csv(route+"/data_files_1c_train.csv")
    dataset_train = NebulaeDataset(data_directory, masks_directory, df_train, transform = (transform_x, transform_y))

    seed_everything(42, workers = True)

    ########## ENTRENAMIENTO MODELO ##########
    # Definimos el K-fold Cross Validator
    kfold = KFold(n_splits=k, shuffle=True, random_state = 42)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset_train)):
        
        for val_id in val_ids:
            val_subset = torch.utils.data.Subset(dataset_train, [val_id])
            val_subset = DivideWindowsSubset(val_subset, window_shape = window_shape, fill_min = True)
            
            for i in range(len(val_subset)):
                _ = shutil.move(save_directory + "/segment_kfold/images/train/" + str(val_id+1+i).zfill(3) + ".png", save_directory + "/segment_kfold/images/val/" + str(val_id+1+i).zfill(3) + ".png")
                _ = shutil.move(save_directory + "/segment_kfold/labels/train/" + str(val_id+1+i).zfill(3) + ".txt", save_directory + "/segment_kfold/labels/val/" + str(val_id+1+i).zfill(3) + ".txt")
                _ = shutil.move(save_directory + "/segment_kfold/masks/train/" + str(val_id+1+i).zfill(3) + ".png", save_directory + "/segment_kfold/masks/val/" + str(val_id+1+i).zfill(3) + ".png")
        
        # Acceder a la capa de convolución inicial
        model = YOLO("yolov8n-seg.pt", task='segment')
        
        add_wandb_callback(model, enable_model_checkpointing=False)
        
        # Imprimimos el fold del que van a mostrarse los resultados
        print('--------------------------------')
        print(f"Model info:\n\t- Batch Size: {BATCH_SIZE}\n\t- GPUs on use: {torch.cuda.device_count()}")

        # Entrenamos el modelo, extrayendo los resultados y guardandolos en la variable result, y evaluamos en el conjunto de test.
        results = model.train(data = save_directory + "/segment_kfold/segment.yaml", 
                            pretrained = False, lr0 = 0.001, lrf = 0.01, 
                            epochs = num_epochs, batch = BATCH_SIZE, imgsz = window_shape, 
                            seed = 42, 
                            single_cls = True, 
                            workers = 8,  
                            mask_ratio = 1, close_mosaic = num_epochs//10, 
                            verbose = False, plots = True, 
                            project = "YOLOv8_PNeSegm", name = 'box3_cls05_dfl1_rect_noTL', 
                            patience = 0, optimizer = "AdamW",
                            box = 3, cls = 0.5, dfl = 1,
                            rect = True,
                            save_dir = os.environ["STORE"]+ "/TFG/YOLOv8")

        # testloader = torch.utils.data.DataLoader(
        #                         dataset_test,
        #                         batch_size=BATCH_SIZE, num_workers=8, shuffle=False, persistent_workers=True)
        
        # Creamos un nuevo entrenador con una sola GPU para la fase de prueba
        # trainer_test = L.Trainer(devices = 1, strategy='auto', max_epochs=num_epochs, accelerator='cuda', log_every_n_steps=1, logger=logger_wandb, callbacks=callbacks)
        # trainer_test.test(model, testloader)

        wandb.finish()
        
        del model

        # Nuevo código para mover archivos de vuelta a las carpetas de entrenamiento
        val_folders = ['images', 'labels', 'masks']
        for folder in val_folders:
            val_path = os.path.join(save_directory, "segment_kfold", folder, "val")
            train_path = os.path.join(save_directory, "segment_kfold", folder, "train")
            
            for file_name in os.listdir(val_path):
                shutil.move(os.path.join(val_path, file_name), os.path.join(train_path, file_name))
            
        torch.cuda.empty_cache()
        time.sleep(30)

    