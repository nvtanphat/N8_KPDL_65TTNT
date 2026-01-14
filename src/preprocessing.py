import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_df(data_dir):
    """
    Tạo DataFrame từ cấu trúc thư mục.
    """
    filepaths = []
    labels = []
    if os.path.exists(data_dir):
        for category in os.listdir(data_dir):
            category_path = os.path.join(data_dir, category)
            if os.path.isdir(category_path):
                for img_name in os.listdir(category_path):
                    filepaths.append(os.path.join(category_path, img_name))
                    labels.append(category)
    return pd.DataFrame({'path_full': filepaths, 'category_str': labels})

def get_data_generators(train_df, val_df, config):
    """
    Tạo các generators cho training và validation với augmentation bảo thủ.
    """
    # Conservative Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0,
        zoom_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.9, 1.1],
        fill_mode='reflect'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='path_full',
        y_col='category_str',
        target_size=(config['IMG_SIZE'], config['IMG_SIZE']),
        batch_size=config['BATCH_SIZE'],
        class_mode='categorical',
        shuffle=True,
        seed=config['SEED']
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='path_full',
        y_col='category_str',
        target_size=(config['IMG_SIZE'], config['IMG_SIZE']),
        batch_size=config['BATCH_SIZE'],
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator
