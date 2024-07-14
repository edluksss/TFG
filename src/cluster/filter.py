import numpy as np
import torch
from skimage import measure
from skimage.morphology import binary_closing, binary_erosion, binary_dilation, disk

def find_closest_points(point, contour):
    """
    Encuentra el punto más cercano en un contorno dado a un punto dado.

    Args:
        point (np.array): El punto de referencia, debe ser un array de numpy de forma (n,), donde n es la dimensión del espacio.
        contour (np.array): El contorno de puntos, debe ser un array de numpy de forma (m, n), donde m es el número de puntos en el contorno y n es la dimensión del espacio.

    Returns:
        np.array: El punto en el contorno que está más cerca del punto de referencia. Tiene la misma dimensión que el punto de referencia.
    """
    distances = np.linalg.norm(contour - point, axis=1)
    return contour[np.argmin(distances)]

def filter_cluster(image, threshold=0.90, morphology_percentage_alpha = 0.025, mask_probs = None, mode = "star_background", channel_index = 0, metric_fnc = lambda x_mean, x_std, y_mean, y_std: (x_mean / x_std) - (y_mean / y_std)) :
    """
    Filtra los clusters de una imagen binarizada para obtener el fondo.
    
    Parámetros:
    image (torch.Tensor): Imagen binarizada.
    min_background_percentage (float, opcional): Porcentaje mínimo de píxeles de fondo. Por defecto es 0.90.
    
    Retorna:
    torch.Tensor: Imagen binarizada con el fondo.
    
    """
    image_knn = image[-1]
    # Sort unique cluster values in ascending order
    unique_values = image_knn.unique(sorted=True)

    background = torch.where(image_knn == unique_values[0], torch.tensor(0), torch.tensor(1))

    old_metric = -np.inf
    
    if mode == "star_background":
        # Mientras que el porcentaje de píxeles de fondo sea menor que el porcentaje mínimo, seguimos añadiendo clusters al fondo
        for cluster_value in unique_values[1:]:
            
            add_background = torch.where(image_knn == cluster_value, torch.tensor(0), torch.tensor(1))
            new_background = background * add_background
            
            if (1 - new_background.sum() / new_background.numel()) > threshold:
                break
            
            background = new_background
        
        if mask_probs is not None:
            background = background * mask_probs
    
    elif mode == "contrast_difference":
        final_background = background.clone()
        for cluster_value in unique_values[1:]:
            bg_wo_holes = binary_closing(background, footprint=disk(image.shape[1]*morphology_percentage_alpha))
            bg_erosion = binary_erosion(bg_wo_holes, footprint=disk(image.shape[1]*morphology_percentage_alpha))

            bg_dilation = binary_dilation(background, disk(image.shape[1]*morphology_percentage_alpha*2))

            if sum(bg_erosion.flatten()) >= len(bg_erosion.flatten()) * 0.99 or sum(bg_dilation.flatten()) >= len(bg_dilation.flatten())*0.99:
                add_background = torch.where(image_knn == cluster_value, torch.tensor(0), torch.tensor(1))
                background *= add_background
                continue
            
            elif sum(bg_erosion.flatten()) == 0 or sum(bg_dilation.flatten()) == 0:
                break
            
            contour_bg_erosion = measure.find_contours(bg_erosion)
            contour_bg_dilation = measure.find_contours(bg_dilation)

            # Repetir el proceso para la máscara erosionada
            blank_image_erosion = np.zeros_like(bg_erosion, dtype=np.uint8)

            contour_max_length = max(contour_bg_erosion, key=len)
            for point in contour_max_length:
                blank_image_erosion[int(point[0]), int(point[1])] = 1

            # Repetir el proceso para la máscara dilatada
            blank_image_dilation = np.zeros_like(bg_dilation, dtype=np.uint8)

            contour_max_length = max(contour_bg_dilation, key=len)
            for point in contour_max_length:
                blank_image_dilation[int(point[0]), int(point[1])] = 1
                    
            # Calcular la diferencia entre los contornos erosionados y dilatados
            contour_values_dilation = image.permute(1,2,0).numpy()[:,:,channel_index] * blank_image_dilation
            contour_values_erosion = image.permute(1,2,0).numpy()[:,:,channel_index] * blank_image_erosion

            bp_erosion = contour_values_erosion.flatten()[contour_values_erosion.flatten()!=0]
            bp_dilation = contour_values_dilation.flatten()[contour_values_dilation.flatten()!=0]

            # Calcular la media y la desviación típica de los valores de contraste
            mean_erosion = np.mean(bp_erosion)
            std_erosion = np.std(bp_erosion)
            mean_dilation = np.mean(bp_dilation)
            std_dilation = np.std(bp_dilation)

            metric = metric_fnc(mean_erosion, std_erosion, mean_dilation, std_dilation)
            add_background = torch.where(image_knn == cluster_value, torch.tensor(0), torch.tensor(1))
            new_background = background * add_background
            
            if  metric >= threshold:
                final_background = background.clone()
                break
            
            elif metric > old_metric:
                old_metric = metric
                final_background = background.clone()
            
            background = new_background
        background = final_background
    elif mode == "mixed":
        background_percentage = threshold[0]
        threshold = threshold[1]
        
        cnt = 0
        
        # Mientras que el porcentaje de píxeles de fondo sea menor que el porcentaje mínimo, seguimos añadiendo clusters al fondo
        for cluster_value in unique_values[1:]:
            
            add_background = torch.where(image_knn == cluster_value, torch.tensor(0), torch.tensor(1))
            new_background = background * add_background
            
            cnt += 1
            if (1 - new_background.sum() / new_background.numel()) > background_percentage:
                break
            
            background = new_background
            
        if mask_probs is not None:
            background = background * mask_probs
        
        final_background = background.clone()
        
        for cluster_value in unique_values[cnt:]:
            bg_wo_holes = binary_closing(background, footprint=disk(image.shape[1]*morphology_percentage_alpha))
            bg_erosion = binary_erosion(bg_wo_holes, footprint=disk(image.shape[1]*morphology_percentage_alpha))

            bg_dilation = binary_dilation(background, disk(image.shape[1]*morphology_percentage_alpha*2))
            
            if sum(bg_erosion.flatten()) == 0 or sum(bg_dilation.flatten()) == 0:
                break
            
            contour_bg_erosion = measure.find_contours(bg_erosion)
            contour_bg_dilation = measure.find_contours(bg_dilation)

            # Repetir el proceso para la máscara erosionada
            blank_image_erosion = np.zeros_like(bg_erosion, dtype=np.uint8)

            contour_max_length = max(contour_bg_erosion, key=len)
            for point in contour_max_length:
                blank_image_erosion[int(point[0]), int(point[1])] = 1

            # Repetir el proceso para la máscara dilatada
            blank_image_dilation = np.zeros_like(bg_dilation, dtype=np.uint8)

            contour_max_length = max(contour_bg_dilation, key=len)
            for point in contour_max_length:
                blank_image_dilation[int(point[0]), int(point[1])] = 1
                    
            # Calcular la diferencia entre los contornos erosionados y dilatados
            contour_values_dilation = image.permute(1,2,0).numpy()[:,:,channel_index] * blank_image_dilation
            contour_values_erosion = image.permute(1,2,0).numpy()[:,:,channel_index] * blank_image_erosion

            bp_erosion = contour_values_erosion.flatten()[contour_values_erosion.flatten()!=0]
            bp_dilation = contour_values_dilation.flatten()[contour_values_dilation.flatten()!=0]

            # Calcular la media y la desviación típica de los valores de contraste
            mean_erosion = np.mean(bp_erosion)
            std_erosion = np.std(bp_erosion)
            mean_dilation = np.mean(bp_dilation)
            std_dilation = np.std(bp_dilation)

            metric = metric_fnc(mean_erosion, std_erosion, mean_dilation, std_dilation)
            
            add_background = torch.where(image_knn == cluster_value, torch.tensor(0), torch.tensor(1))
            new_background = background * add_background
            
            if  metric >= threshold:
                final_background = background.clone()
                break
            
            elif metric > old_metric:
                old_metric = metric
                final_background = background.clone()
                
            background = new_background
        background = final_background
    elif mode == "contrast_difference_np":
        final_background = background.clone()
        for cluster_value in unique_values[1:]:
            bg_wo_holes = binary_closing(background, footprint=disk(image.shape[1]*morphology_percentage_alpha))
            bg_erosion = binary_erosion(bg_wo_holes, footprint=disk(image.shape[1]*morphology_percentage_alpha))

            bg_dilation = binary_dilation(background, disk(image.shape[1]*morphology_percentage_alpha*2))

            if sum(bg_erosion.flatten()) >= len(bg_erosion.flatten()) * 0.99 or sum(bg_dilation.flatten()) >= len(bg_dilation.flatten())*0.99:
                add_background = torch.where(image_knn == cluster_value, torch.tensor(0), torch.tensor(1))
                background *= add_background
                continue
            
            elif sum(bg_erosion.flatten()) == 0 or sum(bg_dilation.flatten()) == 0:
                break
            
            contour_bg = measure.find_contours(background.numpy())
            contour_bg = max(contour_bg, key=len)
            
            contour_bg_erosion = measure.find_contours(bg_erosion)
            contour_bg_erosion = max(contour_bg_erosion, key=len)
            
            contour_bg_dilation = measure.find_contours(bg_dilation)
            contour_bg_dilation = max(contour_bg_dilation, key=len)

            closest_points_erosion = map(lambda x: find_closest_points(x, contour_bg_erosion), contour_bg)
            closest_points_dilation = map(lambda x: find_closest_points(x, contour_bg_dilation), contour_bg)
        
            differences_points = []
        
            for point_erosion, point_dilation in zip(closest_points_erosion, closest_points_dilation):
                value_erosion = image.permute(1,2,0).numpy()[int(point_erosion[0]), int(point_erosion[1]), channel_index]
                value_dilation = image.permute(1,2,0).numpy()[int(point_dilation[0]), int(point_dilation[1]), channel_index]
                differences_points.append(value_erosion - value_dilation)

            # Calcular la media y la desviación típica de los valores de contraste
            mean_difference = np.mean(differences_points)
            std_difference = np.std(differences_points)

            metric = metric_fnc(mean_difference, std_difference)
            
            add_background = torch.where(image_knn == cluster_value, torch.tensor(0), torch.tensor(1))
            new_background = background * add_background
            
            if  metric >= threshold:
                final_background = background.clone()
                break
            
            elif metric > old_metric:
                old_metric = metric
                final_background = background.clone()
            
            background = new_background
        background = final_background
    
    elif mode == "mixed_np":
        background_percentage = threshold[0]
        threshold = threshold[1]
        
        cnt = 0
        
        # Mientras que el porcentaje de píxeles de fondo sea menor que el porcentaje mínimo, seguimos añadiendo clusters al fondo
        for cluster_value in unique_values[1:]:
            
            add_background = torch.where(image_knn == cluster_value, torch.tensor(0), torch.tensor(1))
            new_background = background * add_background
            
            cnt += 1
            if (1 - new_background.sum() / new_background.numel()) > background_percentage:
                break
            
            background = new_background
            
        if mask_probs is not None:
            background = background * mask_probs
        
        final_background = background.clone()
        
        for cluster_value in unique_values[cnt:]:
                bg_wo_holes = binary_closing(background, footprint=disk(image.shape[1]*morphology_percentage_alpha))
                bg_erosion = binary_erosion(bg_wo_holes, footprint=disk(image.shape[1]*morphology_percentage_alpha))

                bg_dilation = binary_dilation(background, disk(image.shape[1]*morphology_percentage_alpha*2))
                
                if sum(bg_erosion.flatten()) == 0 or sum(bg_dilation.flatten()) == 0:
                    break
                
                try:
                    contour_bg = measure.find_contours(background.numpy())
                    contour_bg = max(contour_bg, key=len)
                    
                    contour_bg_erosion = measure.find_contours(bg_erosion)
                    contour_bg_erosion = max(contour_bg_erosion, key=len)
                    
                    contour_bg_dilation = measure.find_contours(bg_dilation)
                    contour_bg_dilation = max(contour_bg_dilation, key=len)
                except ValueError:
                    add_background = torch.where(image_knn == cluster_value, torch.tensor(0), torch.tensor(1))
                    background *= add_background
                    continue
                    
                closest_points_erosion = map(lambda x: find_closest_points(x, contour_bg_erosion), contour_bg)
                closest_points_dilation = map(lambda x: find_closest_points(x, contour_bg_dilation), contour_bg)

                differences_points = []
            
                for point_erosion, point_dilation in zip(closest_points_erosion, closest_points_dilation):
                    value_erosion = image.permute(1,2,0).numpy()[int(point_erosion[0]), int(point_erosion[1]), channel_index]
                    value_dilation = image.permute(1,2,0).numpy()[int(point_dilation[0]), int(point_dilation[1]), channel_index]
                    differences_points.append(value_erosion - value_dilation)

                # Calcular la media y la desviación típica de los valores de contraste
                mean_difference = np.mean(differences_points)
                std_difference = np.std(differences_points)

                metric = metric_fnc(mean_difference, std_difference)
                
                add_background = torch.where(image_knn == cluster_value, torch.tensor(0), torch.tensor(1))
                new_background = background * add_background
                
                if  metric >= threshold:
                    final_background = background.clone()
                    break
                
                elif metric > old_metric:
                    old_metric = metric
                    final_background = background.clone()
                    
                background = new_background
        background = final_background
        
    else:
        raise ValueError("mode must be 'star_background', 'contrast_difference', 'contrast_difference_np, 'mixed' or 'mixed_np'")
    
    return background