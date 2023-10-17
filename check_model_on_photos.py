import os
import random
import shutil
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO

import json
import time

PATH_TO_CONFIG = r'configuration\configuration.yaml'


def separate_image_paths(image_source):
    processed_images = []
    non_processed_images = []
    for path in image_source:
        if path.endswith("_PREPROCESSED.jpg"):
            processed_images.append(path)
        else:
            non_processed_images.append(path)
    return processed_images, non_processed_images


def load_vars_from_yaml(yaml_path: str) -> dict[str, Any]:
    """
    Load variables from yaml file as global variables.
    """
    # Load the YAML file
    with open(yaml_path, 'r') as yaml_file:
        data = yaml.load(yaml_file, Loader=yaml.FullLoader)
    # Create a dictionary to store the variables
    variables = {}
    for key, value in data.items():
        variables[key] = value
    return variables


def get_image_paths(folder_path: str) -> list[str]:
    """
    Returns all images path names at folder_path as a list.
    """
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


def create_results_dict(cls, conf, xywh):
    results = {}
    for i in range(len(cls)):
        key = f"Detect{i+1}"
        value = (int(cls[i]), conf[i], xywh[i])
        results[key] = value
    return results


def get_central_point_and_dim_normalized(start_point, end_point):
    x_box_center = round(
        (start_point[0] + 0.5*(end_point[0] - start_point[0])), 6)
    y_box_center = round(
        (start_point[1] + 0.5*(end_point[1] - start_point[1])), 6)
    w_box = round((end_point[0] - start_point[0]), 6)
    h_box = round((end_point[1] - start_point[1]), 6)
    return x_box_center, y_box_center, w_box, h_box


def write_labels_line(path, filename, label):
    if not os.path.exists(path):
        os.makedirs(path)
    # Open the file for appending (or create a new one if it doesn't exist)
    with open(f"{path}\\{filename}.txt", "a+") as f:
        # Write the label to a new line in the file
        f.write(" ".join(map(str, label)) + "\n")


def extract_parent_folder_from_path(imagepath):
    return os.path.basename(os.path.dirname(imagepath))


def extract_start_end_points(results_dict):
    start_point = ()
    end_point = ()
    for key, value in results_dict.items():
        start_point = (value[2][0], value[2][1])
        end_point = (value[2][2], value[2][3])
    return start_point, end_point


def extract_file_name(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    return file_name


def create_yolo_training_folder(dataset_path: str,
                                validation_percentage: float,
                                seed: int = 42):
    
    # create directories if they don't exist
    for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
        if os.path.exists(os.path.join(dataset_path, folder)):
            shutil.rmtree(os.path.join(dataset_path, folder))
        os.makedirs(os.path.join(dataset_path, folder), exist_ok=True)
    # get list of image and label files
    image_files = [f for f in os.listdir(dataset_path) if f.endswith(".jpg")]
    label_files = [f for f in os.listdir(dataset_path) if f.endswith(".txt")]
    # sort lists to ensure matching pairs
    image_files.sort()
    label_files.sort()
    # shuffle lists using given seed
    random.seed(seed)
    random.shuffle(image_files)
    random.seed(seed)
    random.shuffle(label_files)
    # calculate number of validation files
    num_val_files = int(len(image_files) * validation_percentage)
    # copy validation files to corresponding directories
    for i in range(num_val_files):
        shutil.copy(os.path.join(dataset_path, image_files[i]), os.path.join(dataset_path, "images/val", image_files[i]))
        shutil.copy(os.path.join(dataset_path, label_files[i]), os.path.join(dataset_path, "labels/val", label_files[i]))
    # copy training files to corresponding directories
    for i in range(num_val_files, len(image_files)):
        shutil.copy(os.path.join(dataset_path, image_files[i]), os.path.join(dataset_path, "images/train", image_files[i]))
        shutil.copy(os.path.join(dataset_path, label_files[i]), os.path.join(dataset_path, "labels/train", label_files[i]))


def write_train_yaml_file(yaml_path: str,
                          yaml_name: str,
                          yaml_root: str,
                          train_path_images: str,
                          val_path_images: str,
                          dictionary: dict[str, int]):
    if not os.path.exists(yaml_path):
        os.makedirs(yaml_path)
    # Create dictionary for the YAML file
    yaml_dict = {
        'path': yaml_root,
        'train': train_path_images,
        'val': val_path_images,
        'names': {}
    }
    # Add elements and their values to the dictionary
    for key, value in dictionary.items():
        yaml_dict['names'][value] = key
    # Create the full path for the YAML file
    full_path = os.path.join(yaml_path, yaml_name)
    # Write the YAML file
    with open(full_path, 'w') as file:
        yaml.dump(yaml_dict, file)


def copy_training_images(classes: dict[str, int],
                         src_train_images_folder: str,
                         dest_train_images_folder: str):
    """
    Copy training images from either shared or local folder to destination folder.

    Parameters:
    - classes: a dictionary mapping class names to integers IDs
    - src_train_images_folder: a string representing the source folder paths
    - dest_train_images: a string representing the destination folder path
    """
    if src_train_images_folder is not None and os.path.exists(src_train_images_folder):
        # Check that all required class folders exist and are non-empty
        for class_name, class_id in classes.items():
            class_folder_path = os.path.join(src_train_images_folder, class_name)
            if not os.path.exists(class_folder_path) or not os.path.isdir(class_folder_path) or len(os.listdir(class_folder_path)) == 0:
                break
            else:
                # All class folders exist and are non-empty, copy them to destination folder
                for class_name, class_id in classes.items():
                    class_folder_path = os.path.join(src_train_images_folder, class_name)
                    dest_class_folder_path = os.path.join(dest_train_images_folder, str(class_name))
                    shutil.copytree(class_folder_path, dest_class_folder_path, dirs_exist_ok=True)
    else:
        raise Exception("Cannot copy training photos!")


def preprocess_training_images(training_images_path: str,
                               classes: dict[str, int]
                               ):
    for class_name, class_id in classes.items():
        preprocessed_model_folder=os.path.join(training_images_path,class_name)
        for image_name in os.listdir(preprocessed_model_folder):
            if image_name.endswith('.jpg'):
                image_path = os.path.join(preprocessed_model_folder, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                clahe_image = clahe.apply(image)
                preprocessed_image_name = image_name[:-4] + 'HE_PREPROCESSED.jpg'
                preprocessed_image_path = os.path.join(
                    preprocessed_model_folder, preprocessed_image_name)
                cv2.imwrite(preprocessed_image_path, clahe_image)
                gamma_1 = np.power(image / 255.0, config_vars['TRAINING_YOLO']['GAMMA_1'])
                gamma_1_image = (gamma_1 * 255.0).astype(np.uint8)
                preprocessed_image_name = image_name[:-4] + 'GAMMA_1_PREPROCESSED.jpg'
                preprocessed_image_path = os.path.join(
                    preprocessed_model_folder, preprocessed_image_name)
                cv2.imwrite(preprocessed_image_path, gamma_1_image)
                gamma_2 = np.power(image / 255.0, config_vars['TRAINING_YOLO']['GAMMA_2'])
                gamma_2_image = (gamma_2 * 255.0).astype(np.uint8)
                preprocessed_image_name = image_name[:-4] + 'GAMMA_2_PREPROCESSED.jpg'
                preprocessed_image_path = os.path.join(
                    preprocessed_model_folder, preprocessed_image_name)
                cv2.imwrite(preprocessed_image_path, gamma_2_image)


def create_training_photos_folder(source_path_training_folder: str,
                                  dest_path_training_folder: str,
                                  classes: dict[str, int]):
    
    # Create folder containing all training photos
    if os.path.exists(dest_path_training_folder):
        shutil.rmtree(dest_path_training_folder)
    os.makedirs(dest_path_training_folder)

    # Loop over the folders in 'path_training_folder'
    for folder_name in os.listdir(source_path_training_folder):
        folder_path = os.path.join(source_path_training_folder, folder_name)
        # Check if this folder is in classes_dict
        if folder_name in classes.keys():
            # Loop over the files in the folder
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                # Check if the file is a .jpg or .txt file
                if file_name.endswith('.jpg') or file_name.endswith('.txt'):
                    new_file_path = os.path.join(
                        dest_path_training_folder, file_name)
                    shutil.copyfile(file_path, new_file_path)


def auto_create_labels_for_yolo(training_images_path: str,
                                undetected_labels_images_folder:str ,
                                classes: dict[str, int],
                                model_for_labels: YOLO,
                                confidence_threshold_boxes: int,
                                device_boxes: Any):
    
    model_labels = YOLO(model=model_for_labels)

    if os.path.exists(undetected_labels_images_folder):
        shutil.rmtree(undetected_labels_images_folder)
    os.makedirs(undetected_labels_images_folder)
    
    for class_name , class_id in classes.items():
        class_path = os.path.join(training_images_path, class_name)
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                if file_name.endswith(".jpg"):
                    
                    image_path=os.path.join(class_path,
                                            file_name)
                    
                    results = model_labels(image_path,
                                           conf=confidence_threshold_boxes/100,
                                           verbose=False,
                                           device=device_boxes)
                    
                    results_dict_norm=create_results_dict(results[0].boxes.cls.tolist(),
                                                          results[0].boxes.conf.tolist(),
                                                          results[0].boxes.xyxyn.tolist())
                    
                    if len(results_dict_norm) == 1:
                        
                        start_point, end_point = extract_start_end_points(results_dict_norm)

                        box_centre_and_size=get_central_point_and_dim_normalized(
                            start_point,
                            end_point)
                        
                        class_name_for_detect=extract_parent_folder_from_path(image_path)
                        class_number=classes[class_name_for_detect]
                        
                        label = (class_number,
                                box_centre_and_size[0],
                                box_centre_and_size[1],
                                box_centre_and_size[2],
                                box_centre_and_size[3])
                        
                        write_labels_line(
                            path=class_path,
                            filename=extract_file_name(image_path),
                            label=label)
                        
                        results_frame = results[0].plot()
                        
                        # Show images
                        cv2.imshow(f'Boxes Results', results_frame)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("Closing python!")
                            exit()
                    else:
                        shutil.move(image_path,
                                    os.path.join(undetected_labels_images_folder,
                                                 file_name))
    cv2.destroyAllWindows()


def generate_yolo_weigths(path_images,
                          yaml_path,
                          yaml_name,
                          yaml_root,
                          validation_images_percentage,
                          seed,
                          dictionary_objects_boxes,
                          starting_model_for_labels,
                          training_epochs,
                          training_patience,
                          training_batch,
                          training_imgsz,
                          training_save_period,
                          training_device,
                          training_workers,
                          training_project_name,
                          training_folder_name,
                          training_optimizer,
                          training_dropout):
    
    write_train_yaml_file(yaml_path=yaml_path,
                          yaml_name=yaml_name,
                          yaml_root=yaml_root,
                          train_path_images=r'images\train',
                          val_path_images=r'images\val',
                          dictionary=dictionary_objects_boxes,
                          )
    
    create_yolo_training_folder(path_images,
                                validation_percentage=validation_images_percentage,
                                seed=seed)
    
    model = YOLO(starting_model_for_labels)
    
    model.train(
        data=os.path.join(yaml_path,yaml_name),
        epochs=training_epochs,
        patience=training_patience,
        batch=training_batch,
        imgsz=training_imgsz,
        save_period=training_save_period,
        device=training_device,
        workers=training_workers,
        project=training_project_name,
        name=training_folder_name,
        exist_ok=True,
        pretrained=False,
        optimizer=training_optimizer,
        rect= False,
        dropout=training_dropout,
        )

    if os.path.exists('yolov8n.pt'):
        print("Deleting downloaded yolo weights..")
        os.remove('yolov8n.pt')


def get_image_pixel_map(image_input):
    if isinstance(image_input, str):
        # If input is a file path, load the image using OpenCV
        image = cv2.imread(image_input)
    elif isinstance(image_input, np.ndarray):
        # If input is already an OpenCV image, use it directly
        image = image_input
    else:
        raise ValueError("Input must be a file path (str) or an OpenCV image (numpy array)")
    if image is None:
        raise ValueError("Image not found or could not be loaded")
    # Convert the image to RGB format if it's in BGR
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert the RGB image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Convert the image to a NumPy array
    image_array = np.array(image)
    return image_array


def plot_image_map(pixel_map):
    # Check if the input is a valid NumPy array
    if not isinstance(pixel_map, np.ndarray):
        raise ValueError("Input must be a NumPy array")
    # Check if the input has 2 or 3 dimensions (grayscale or RGB)
    if pixel_map.ndim not in [2, 3]:
        raise ValueError("Input must have 2 or 3 dimensions")
    # Determine the number of color channels (1 for grayscale, 3 for RGB)
    num_channels = 1 if pixel_map.ndim == 2 else pixel_map.shape[2]
    # Create a figure and axis based on the number of color channels
    if num_channels == 1:
        plt.imshow(pixel_map, cmap='gray')
    else:
        plt.imshow(pixel_map)
    # Display the image
    plt.axis('off')
    plt.show()


if __name__ == '__main__':

    config_vars = load_vars_from_yaml(PATH_TO_CONFIG)

    if config_vars['NETWORK_USED'] == 'yolov8':

        if config_vars['TESTING_YOLO']['WEIGHTS'] == 'old':
            weights_path = rf"YOLO\Camera_{config_vars['STATION']}_weights_webb\best.pt"
        elif config_vars['TESTING_YOLO']['WEIGHTS'] == 'new':
            weights_path = rf"YOLO\Camera_{config_vars['STATION']}_weights_new\best.pt"

        if config_vars['USAGE'] == 'test':
            
            model = YOLO(weights_path)

            if config_vars['STATION'] == 1:
                paths = config_vars['TESTING_YOLO']['PATHS_CAM_1']

            elif config_vars['STATION'] == 2:
                paths = config_vars['TESTING_YOLO']['PATHS_CAM_2']

            if config_vars['TESTING_YOLO']['ENABLE_P_R_COMPUTATION']:
                prec_rec_dict = {} # to compute precision and recall
                prec_dict = {}
                # pred != gt # si riferisce alle altre classi trovate
                FP_dict = {class_model: 0\
                            for class_model in list(config_vars['TRAINING_YOLO']['MODELS_TRAIN_DICT'].keys())}
                # pred = gt
                TP_dict = {class_model: 0\
                            for class_model in list(config_vars['TRAINING_YOLO']['MODELS_TRAIN_DICT'].keys())}

            for photos_path in paths:
                if config_vars['TESTING_YOLO']['ENABLE_P_R_COMPUTATION']:
                    recall_dict = {}
                
                #Get initial image paths
                image_paths = get_image_paths(photos_path)
 
                for image_path in image_paths:
                    #Clear all _PREPROCESSED images
                    if image_path.endswith('_PREPROCESSED.jpg'):
                        os.remove(image_path)

                if config_vars['TESTING_YOLO']['TEST_ON_PREPROCESSED']:
                    
                    #Get new image paths
                    image_paths = get_image_paths(photos_path)

                    #Create a preprocessed copy of each
                    for image_path in image_paths:
                        shutil.copy(image_path, image_path.split('.')[0]+'_PREPROCESSED.jpg')
                    
                    #Get all paths (original + _PREPROCESSED)
                    image_paths = get_image_paths(photos_path)
                    
                    if config_vars["TESTING_YOLO"]["ENABLE_TIME_COMPUTATION"]:
                        start_time = time.time()

                    #Iterate on original +_PREPROCESSED
                    for image_path in image_paths:
                        
                        if image_path.endswith('_PREPROCESSED.jpg'):
                            
                            image_source_cv2=cv2.imread(image_path)

                            if config_vars['TESTING_YOLO']['ENABLE_RESIZING']:
                                image_source_cv2 = cv2.resize(image_source_cv2, 
                                                              (config_vars['TESTING_YOLO']['RESIZED_WIDTH'], 
                                                               config_vars['TESTING_YOLO']['RESIZED_HEIGHT']))
                            
                            if config_vars['TESTING_YOLO']['HISTOGRAM_EQUALIZATION']:
                                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                                image_source_cv2=cv2.cvtColor(image_source_cv2,cv2.COLOR_RGB2GRAY)
                                image_source_cv2 = clahe.apply(image_source_cv2)
                                image_source_cv2=cv2.cvtColor(image_source_cv2,cv2.COLOR_GRAY2RGB)
                            
                            if config_vars['TESTING_YOLO']['SET_GAIN']:
                                image_source_cv2=np.clip(image_source_cv2 * config_vars['TESTING_YOLO']['GAIN'], 0, 255).astype(np.uint8)
                            
                            if config_vars['TESTING_YOLO']['SET_BRIGHTNESS']:
                                # Apply brightness adjustment
                                image_source_cv2 = np.clip(image_source_cv2 + config_vars['TESTING_YOLO']['BRIGHTNESS'], 0, 255).astype(np.uint8)
                            
                            if config_vars['TESTING_YOLO']['SET_SHARPNESS']:
                                # Create a parametrized sharpening kernel
                                kernel = np.array([[-1, -1, -1],
                                                [-1, 1 + config_vars['TESTING_YOLO']['SHARPNESS'], -1],
                                                [-1, -1, -1]])
                                # Apply the sharpening filter using convolution
                                image_source_cv2 = cv2.filter2D(image_source_cv2, -1, kernel)
                            
                            if config_vars['TESTING_YOLO']['SET_SATURATION']:
                                # Convert the image from BGR to HSV color space
                                image_source_cv2 = cv2.cvtColor(image_source_cv2, cv2.COLOR_BGR2HSV)
                                # Modify the saturation channel
                                image_source_cv2[:, :, 1] = np.clip(image_source_cv2[:, :, 1] * config_vars['TESTING_YOLO']['SATURATION'], 0, 255).astype(np.uint8)
                                # Convert the image back to BGR color space
                                result_image = cv2.cvtColor(image_source_cv2, cv2.COLOR_HSV2BGR)
                            
                            if config_vars['TESTING_YOLO']['SET_CONTRAST']:
                                # Apply contrast adjustment
                                image_source_cv2 = np.clip(image_source_cv2 * config_vars['TESTING_YOLO']['CONTRAST'], 0, 255).astype(np.uint8)
                            
                            if config_vars['TESTING_YOLO']['SET_GAMMA']:
                                # Apply gamma correction
                                image_source_cv2 = np.power(image_source_cv2 / 255.0, config_vars['TESTING_YOLO']['GAMMA'])
                                image_source_cv2 = (image_source_cv2 * 255.0).astype(np.uint8)
                            
                            cv2.imwrite(image_path, image_source_cv2)
                
                else:
                    if config_vars["TESTING_YOLO"]["ENABLE_TIME_COMPUTATION"]:
                        start_time = time.time()

                image_source=get_image_paths(photos_path)
                if config_vars['TESTING_YOLO']['TEST_ON_PREPROCESSED']:
                    image_source=separate_image_paths(image_source)[0]
                else:
                    image_source=separate_image_paths(image_source)[1]

                results = model.predict(source=image_source[0 : len(image_source) // 2],
                                        conf=config_vars['TESTING_YOLO']['CONF_LIMIT'],
                                        iou=config_vars['TESTING_YOLO']['IOU_LIMIT'],
                                        device=config_vars['TESTING_YOLO']['DEVICE'],
                                        save=True,
                                        save_conf=True,
                                        save_txt=False)
                
                results += model.predict(source=image_source[len(image_source) // 2 : ],
                                        conf=config_vars['TESTING_YOLO']['CONF_LIMIT'],
                                        iou=config_vars['TESTING_YOLO']['IOU_LIMIT'],
                                        device=config_vars['TESTING_YOLO']['DEVICE'],
                                        save=True,
                                        save_conf=True,
                                        save_txt=False)
                
                if config_vars["TESTING_YOLO"]["ENABLE_TIME_COMPUTATION"]:
                    end_time = time.time()
                    diff_time = end_time - start_time

                if config_vars['TESTING_YOLO']['ENABLE_P_R_COMPUTATION']:
                    gt_class = photos_path.split("\\")[-1]
                    n_images = len(results) # num images
                    count_FN = 0 # pred = 'undetected'  
                    
                    pred_confidence_list = []

                    for result in results:

                        if len(result.boxes.cls) == 1:
                            pred_class = str(result.names.get(result.boxes.cls.item()))
                            pred_confidence_list.append(round(float(result.boxes.conf.item()), 3))

                            if pred_class == gt_class: # TP
                                if pred_class in TP_dict:
                                    TP_dict[pred_class] += 1
                                else:
                                    TP_dict[pred_class] = 1
                            else: # FP
                                if pred_class in FP_dict:
                                    FP_dict[pred_class] += 1
                                else:
                                    FP_dict[pred_class] = 1
                            
                        else:
                            pred_class = 'Undetected'
                            pred_confidence_list.append(None)
                            count_FN += 1
                    
                    # Calcola la media delle confidence
                    pred_confidence_list = [x for x in pred_confidence_list if x is not None]
                    try:
                        avg_confidence = round(sum(pred_confidence_list) / len(pred_confidence_list), 3)
                    except ZeroDivisionError:
                        avg_confidence = 0.0         

                    recall_dict[gt_class] = round(float(TP_dict[gt_class] / (TP_dict[gt_class] + count_FN)), 3)

                    if config_vars["TESTING_YOLO"]["ENABLE_TIME_COMPUTATION"]:
                        prec_rec_dict[gt_class] = {'N_images': n_images,\
                                                'Avg_conf': avg_confidence,\
                                                'P': 0.0,\
                                                'R': recall_dict[gt_class],\
                                                'time': round(diff_time, 3)}
                    else:
                        prec_rec_dict[gt_class] = {'N_images': n_images,\
                                                'Avg_conf': avg_confidence,\
                                                'P': 0.0,\
                                                'R': recall_dict[gt_class]}
                    
                    filename = rf'runs\Camera{config_vars["STATION"]}_{config_vars["TESTING_YOLO"]["WEIGHTS"]}'

                    if config_vars["TESTING_YOLO"]["TEST_ON_PREPROCESSED"]:
                        if config_vars["TESTING_YOLO"]["HISTOGRAM_EQUALIZATION"]:
                            filename += rf'_he'
                        if config_vars["TESTING_YOLO"]["SET_GAMMA"]:
                            filename += rf'_gamma{config_vars["TESTING_YOLO"]["GAMMA"]}'
                        if config_vars["TESTING_YOLO"]["ENABLE_RESIZING"]:
                            filename += rf'_{config_vars["TESTING_YOLO"]["RESIZED_WIDTH"]},{config_vars["TESTING_YOLO"]["RESIZED_HEIGHT"]}'

                    filename += '.json'

            if config_vars['TESTING_YOLO']['ENABLE_P_R_COMPUTATION']:
                class_models_list = [p.split("\\")[-1] for p in paths]
                for cm in class_models_list:
                    try:
                        prec_dict[cm] = {'P': round(float(TP_dict.get(cm) / (FP_dict.get(cm) + TP_dict.get(cm))), 3)}
                    except ZeroDivisionError:
                        prec_dict[cm] = {'P': 0.0}

                # inserisco le precision in prec_rec_dict prima di salvarlo in .json
                for key, sub_dict2 in prec_dict.items():
                    if key in prec_rec_dict:
                        sub_dict1 = prec_rec_dict[key]
                        for sub_key, value2 in sub_dict2.items():
                            if sub_key in sub_dict1:
                                sub_dict1[sub_key] = value2
                
                with open(filename, 'w') as file:
                    json.dump(prec_rec_dict, file, indent=4)
              



        elif config_vars['USAGE'] == 'training':
            
            classes_dictionary = config_vars['TRAINING_YOLO']['MODELS_TRAIN_DICT']
            
            if not config_vars['TRAINING_YOLO']['ONLY_TRAINING']:
                # Create folder containing all training photos
                if os.path.exists(r'YOLO\training_images'):
                    shutil.rmtree(r'YOLO\training_images')
                os.makedirs(r'YOLO\training_images')

                copy_training_images(src_train_images_folder=rf'training_images\Camera_{config_vars["STATION"]}_training_photos',
                                    dest_train_images_folder=r'YOLO\training_images',
                                    classes=classes_dictionary)

                preprocess_training_images(training_images_path=r'YOLO\training_images',
                                        classes=classes_dictionary)

                auto_create_labels_for_yolo(training_images_path=r'YOLO\training_images',
                                            undetected_labels_images_folder=rf'YOLO\training_images\undetected_boxes_by_yolo',
                                            classes=classes_dictionary,
                                            model_for_labels=r'YOLO\weights_boxes_exp\weights_boxes\weights\best.pt',
                                            confidence_threshold_boxes=config_vars['TRAINING_YOLO']['CONFIDENCE_AUTOLABELING'],
                                            device_boxes=config_vars['TRAINING_YOLO']['DEVICE'])
                
                create_training_photos_folder(source_path_training_folder=r'YOLO\training_images',
                                            dest_path_training_folder=r'YOLO\training_images\all_models_folder',
                                            classes=classes_dictionary)
            
            generate_yolo_weigths(
                                  path_images=r'YOLO\training_images\all_models_folder',
                                  yaml_path=r'YOLO\training_files',
                                  yaml_name=f"dataset_generic_CAM{config_vars['STATION']}.yaml",
                                  yaml_root=r'C:\Users\GEFIT\Desktop\Workspace\02-PYVISIND\pyvis-ind\pyvis-ind\test_on_images\YOLO\training_images\all_models_folder',
                                  validation_images_percentage=config_vars['TRAINING_YOLO']['VAL_PERCENTAGE'],
                                  dictionary_objects_boxes=config_vars['TRAINING_YOLO']['MODELS_TRAIN_DICT'],
                                  seed=config_vars['TRAINING_YOLO']['SEED'],
                                  starting_model_for_labels=rf"YOLO\original_weights\yolov8{config_vars['TRAINING_YOLO']['MODEL_SIZE']}.pt",
                                  training_epochs=config_vars['TRAINING_YOLO']['EPOCHS'],
                                  training_patience=config_vars['TRAINING_YOLO']['PATIENCE'],
                                  training_batch=config_vars['TRAINING_YOLO']['BATCH_SIZE'],
                                  training_imgsz=config_vars['TRAINING_YOLO']['IMG_SIZE'],
                                  training_save_period=config_vars['TRAINING_YOLO']['SAVE_PERIOD'],
                                  training_device=config_vars['TRAINING_YOLO']['DEVICE'],
                                  training_workers=config_vars['TRAINING_YOLO']['WORKERS'],
                                  training_project_name='last_training_results',
                                  training_folder_name='training_results',
                                  training_optimizer=config_vars['TRAINING_YOLO']['OPTIMIZER'],
                                  training_dropout=config_vars['TRAINING_YOLO']['DROPOUT']
                                  )
            
            trained_model_path=r'last_training_results\training_results\weights\best.pt'
            destination_model_path=rf'YOLO\Camera_{config_vars["STATION"]}_weights\best.pt'
            shutil.copy2(trained_model_path, destination_model_path)
         
    
    elif config_vars['NETWORK_USED'] == 'mediapipe':
        #TODO manage test and train for mediapipe
        raise Exception("Mediapipe still not supported")
