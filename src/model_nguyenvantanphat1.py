import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV3Large

def build_model(config, freeze_base=True):
    """
    Xây dựng mô hình MobileNetV3Large.
    """
    base_model = MobileNetV3Large(
        include_top=False,
        weights='imagenet',
        input_shape=(config['IMG_SIZE'], config['IMG_SIZE'], 3)
    )
    base_model.trainable = not freeze_base
    
    inputs = layers.Input(shape=(config['IMG_SIZE'], config['IMG_SIZE'], 3))
    x = layers.Rescaling(255.0)(inputs) # MobileNetV3 expects [0, 255] then internal preprocessing if built-in
    # Note: ImageDataGenerator rescales to [0,1]. MobileNetV3 usually expects inputs to be roughly in that range or [0,255] 
    # depending on implementation. If using 'imagenet' weights, standard keras app expects specific prep.
    # However, in your notebook you used Rescaling(255.0) which implies inputs were [0,1].
    # Let's keep consistency with your notebook.
    
    x = base_model(x, training=not freeze_base)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(config['NUM_CLASSES'], activation='softmax')(x)
    model = models.Model(inputs, outputs, name='BeanLeaf_MobileNetV3Large')
    
    optimizer = tf.keras.optimizers.AdamW(learning_rate=config['LEARNING_RATE'])
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model
