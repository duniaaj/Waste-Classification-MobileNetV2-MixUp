import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import numpy as np
import os
from sklearn.utils.class_weight import compute_class_weight

# Custom MixUp Generator
def mixup_generator(generator, alpha=0.2):
    while True:
        x_batch, y_batch = next(generator)
        batch_size = len(x_batch)
        lam = np.random.beta(alpha, alpha)
        index_array = np.random.permutation(batch_size)

        x1 = x_batch
        x2 = x_batch[index_array]
        y1 = y_batch
        y2 = y_batch[index_array]

        x = lam * x1 + (1 - lam) * x2
        y = lam * y1 + (1 - lam) * y2

        yield x, y


# Paths to training and validation directories
train_dir = 'waste_split/train'
val_dir = 'waste_split/val'

# ============================
# Data Augmentation
# ============================
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# ============================
# Class Weights
# ============================
labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights_dict = dict(enumerate(class_weights))
print("Computed class weights:", class_weights_dict)

# ============================
# Base Model (MobileNetV2)
# ============================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base initially

# ============================
# Top Layers
# ============================
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# ============================
# Set Up TensorBoard
# ============================
tensorboard_callback = TensorBoard(log_dir='logs')

# ============================
# Train the Model with MixUp
# ============================
model.fit(
    mixup_generator(train_generator),
    epochs=10,
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    class_weight=class_weights_dict,
    callbacks=[tensorboard_callback]
)

# ============================
# Fine-Tune the Base Model
# ============================
base_model.trainable = True

# Optionally freeze first N layers (e.g., 100)
for layer in base_model.layers[:100]:
    layer.trainable = False

# Compile with lower learning rate for fine-tuning
model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Fine-tuning training
model.fit(
    mixup_generator(train_generator),
    validation_data=validation_generator,
    epochs=20,  # You can go higher if GPU available
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[tensorboard_callback, early_stop],
    class_weight=class_weights_dict
)

# Save the final model
model.save('waste_classification_model_finetuned_with_mixup.keras')
