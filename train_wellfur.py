import tensorflow as tf
import numpy as np
import os
import json
import pathlib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# Setup Logging
def setup_logging(model_save_path):
    os.makedirs(model_save_path, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(model_save_path, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Logging setup complete.")

# Custom dataset splitting to handle "Healthy" class imbalance
# splits the data into training and testing/ validation
# 
def custom_split_dataset(dataset_path, healthy_ratio_train=0.2):
    data_dir = pathlib.Path(dataset_path)
    train_files, train_labels, val_files, val_labels = [], [], [], []
    
    # Get class names and index mapping
    class_names = sorted([item.name for item in data_dir.glob('*') if item.is_dir()])
    if not class_names:
        raise ValueError(f"No valid directories found in {dataset_path}")
    
    class_to_index = {name: i for i, name in enumerate(class_names)}
    logging.info(f"Found classes: {class_names}")
    
    for class_name in class_names:
        class_dir = data_dir / class_name
        # Get only image files
        files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + \
               list(class_dir.glob('*.png')) + list(class_dir.glob('*.bmp'))
        
        if not files:
            raise ValueError(f"No valid image files found in class: {class_name}")
        
        files = [str(f) for f in files]
        np.random.shuffle(files)
        
        # Custom split for Healthy class
        split_idx = int(len(files) * (healthy_ratio_train if class_name == "Healthy" else 0.8))
        split_idx = max(1, split_idx)  # Ensure at least one sample
        
        train_files.extend(files[:split_idx])
        train_labels.extend([class_to_index[class_name]] * split_idx)
        val_files.extend(files[split_idx:])
        val_labels.extend([class_to_index[class_name]] * (len(files) - split_idx))
    
    if not train_files or not val_files:
        raise ValueError("No files found for training or validation")
    
    logging.info(f"Training samples: {len(train_files)}, Validation samples: {len(val_files)}")
    return (train_files, train_labels), (val_files, val_labels), class_names


# Dataset Pipeline
# converts the file paths and labels into a TensorFlow dataset
# basically image -> 1 or 0 so its easier for the computer to understand
# 
def create_dataset(file_paths, labels, image_size, num_classes, batch_size, shuffle=True):
    if not file_paths:
        raise ValueError("Empty file paths provided")
    
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    if shuffle:
        # Use a reasonable buffer size
        buffer_size = min(len(file_paths), 1000)
        dataset = dataset.shuffle(buffer_size=buffer_size)

    def load_and_preprocess(path, label):
        try:
            image = preprocess_image(path, image_size)
            label = tf.one_hot(label, num_classes)
            return image, label
        except tf.errors.InvalidArgumentError:
            logging.error(f"Error loading image: {path}")
            return None
    
    dataset = dataset.map(load_and_preprocess, 
                         num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(lambda x: x is not None)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Image Preprocessing
# loads and preprocesses the images for the model
# 
def preprocess_image(image_path, image_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Change to decode_png if needed
    image = tf.image.resize(image, image_size)
    image = tf.keras.applications.mobilenet_v3.preprocess_input(image)
    return image

# Dataset Pipeline
def create_dataset(file_paths, labels, image_size, num_classes, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(file_paths))

    def load_and_preprocess(path, label):
        image = preprocess_image(path, image_size)
        label = tf.one_hot(label, num_classes)
        return image, label

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Build MobileNetV3 Model pretrained model
# loading the pretrained model and processing it 
# like evaluting the weights and notusing the initial layers
def build_finetune_model(image_size, num_classes, layers_to_freeze=30):
    base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(*image_size, 3), minimalistic=True)
    
    # Freeze all layers except last `layers_to_freeze`
    base_model.trainable = True
    for layer in base_model.layers[:-layers_to_freeze]:
        layer.trainable = False
        
    inputs = Input(shape=(*image_size, 3))
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(1024, kernel_regularizer=l2(0.01))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(x)
    
    model = tf.keras.Model(inputs, outputs)
    optimizer = Adam(learning_rate=1e-4, clipnorm=1.0, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
    logging.info("Model built and compiled.")
    return model

# Evaluate Model
# visualization only to see metrics no evident effect in the model
def evaluate_model(model, dataset, class_names, model_save_path):
    all_y_true, all_y_pred = [], []
    for images, labels in dataset:
        preds = model.predict(images)
        all_y_true.extend(tf.argmax(labels, axis=1).numpy())
        all_y_pred.extend(tf.argmax(preds, axis=1).numpy())
        
    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, 'confusion_matrix.png'))
    plt.close()
    
    # Metrics Calculation
    accuracy = np.mean(np.array(all_y_true) == np.array(all_y_pred))
    f1_macro = f1_score(all_y_true, all_y_pred, average='macro')
    precision_macro = precision_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    
    metrics_dict = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro
    }
    
    with open(os.path.join(model_save_path, 'model_metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    
    logging.info("Evaluation metrics computed and saved.")
    return metrics_dict

# Train the Model
# training the overall model with the dataset
# with the freeze layers and the pretrained model, weights, and layers and ewan ko na din
def train_finetune_model(dataset_path, model_save_path, image_size=(224,224), batch_size=16, epochs=10):
    setup_logging(model_save_path)
    
    # Dataset Splitting
    (train_files, train_labels), (val_files, val_labels), class_names = custom_split_dataset(dataset_path, healthy_ratio_train=0.2)
    num_classes = len(class_names)
    
    train_ds = create_dataset(train_files, train_labels, image_size, num_classes, batch_size, shuffle=True)
    val_ds = create_dataset(val_files, val_labels, image_size, num_classes, batch_size, shuffle=False)
    
    model = build_finetune_model(image_size, num_classes, layers_to_freeze=30)
    
    # Train the Model
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
    # Evaluate Model
    metrics_dict = evaluate_model(model, val_ds, class_names, model_save_path)
    
    # Save Model & Labels
    model.save(os.path.join(model_save_path, 'train_wellfur.keras'))
    with open(os.path.join(model_save_path, 'labels.json'), 'w') as f:
        json.dump(class_names, f, indent=4)

# Run Training
if __name__ == "__main__":
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create paths relative to the script location
    dataset_path = os.path.join(current_dir, "data")
    model_save_path = os.path.join(current_dir, "a_model")
    
    try:
        train_finetune_model(
            dataset_path=dataset_path,
            model_save_path=model_save_path,
            image_size=(224, 224),
            batch_size=16,
            epochs=10
        )
    except Exception as e:
        print(f"Training failed: {str(e)}")
