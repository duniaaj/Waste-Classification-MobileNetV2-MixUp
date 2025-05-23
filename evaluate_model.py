import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = tf.keras.models.load_model("waste_classification_model_finetuned_with_mixup.keras")

# Class names (make sure these are in the same order as during training)
class_names = ['metal', 'paper', 'plastic']
num_classes = len(class_names)

# Load test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "waste_split/test",
    image_size=(224, 224),
    batch_size=32,
    label_mode='int',  # still load as int so we can one-hot encode
    shuffle=False
)

# Normalize the images and convert labels to one-hot
test_ds = test_ds.map(lambda x, y: (x / 255.0, tf.one_hot(y, depth=num_classes)))

# Evaluate the model
loss, accuracy = model.evaluate(test_ds)
print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")

# Extract true labels and predictions for confusion matrix
y_true = []
y_pred = []

for batch_x, batch_y in test_ds:
    preds = model.predict(batch_x)
    y_true.extend(tf.argmax(batch_y, axis=1).numpy())  # convert one-hot back to int
    y_pred.extend(tf.argmax(preds, axis=1).numpy())

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
