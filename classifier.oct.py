import os
import sys
import cv2
import json
import numpy as np
import random
import logging
from pathlib import Path

# Suppress CUDA-related warnings (if no GPU available)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Suppress CUDA warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO and WARNING logs
os.environ['TF_ABSL_LOG_LEVEL'] = '3'    # Suppress absl-related logs
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'    # Disable CUDA debugging

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import (
   ModelCheckpoint,
   ReduceLROnPlateau,
   LearningRateScheduler,
)
from tensorflow.keras.utils import Sequence
from tensorflow.keras import models, layers
import tensorflow as tf
from PIL import Image

tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# Locate the root directory and dataset
project_dir = (Path(__file__).parent).resolve()
dataset_dir = (project_dir / 'OCT2017').resolve()
train_dir = os.path.abspath((dataset_dir / 'train').resolve())
test_dir = os.path.abspath((dataset_dir / 'test').resolve())
val_dir = os.path.abspath((dataset_dir / 'val').resolve())

predict_dir = os.path.abspath((project_dir / 'predict').resolve())
prediction_file = os.path.abspath(
   (project_dir / 'prediction.json').resolve()
)

dataset_file = os.path.abspath(
   (project_dir / 'dataset.image.json').resolve()
)

# Paths for saving/loading models (relative to project_dir)
model_loss_path = os.path.abspath(
   (project_dir / 'model.loss.h5.keras').resolve()
)
model_accuracy_path = os.path.abspath(
   (project_dir / 'model.accuracy.h5.keras').resolve()
)
model_accuracy_categorical_path = os.path.abspath(
   (project_dir / 'model.accuracy.categorical.h5.keras').resolve()
)
model_mae_path = os.path.abspath(
   (project_dir / 'model.mae.h5.keras').resolve()
)
model_mse_path = os.path.abspath(
   (project_dir / 'model.mse.h5.keras').resolve()
)
model_final_path = os.path.abspath(
   (project_dir / 'model.result.final.h5.keras').resolve()
)
model_iteration_path = os.path.abspath(
   (project_dir / 'model.iteration.h5.keras').resolve()
)

# globals
class_mapping = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
dataset_info = None
dataset = None
Z = 0
K = 0

# Pre-processors
def preprocess_1(image):
   """ Placeholder for future pre-processor, returns image unchanged """
   return image

def preprocess_2(image, augment_params):
   """
   Augment the image with specified transformations: rotation, translation, etc.
   `augment_params` can include rotation angle, translation amount, etc.
   """
   if augment_params['rotate']:
      image = np.array(
         tf.image.rot90(
            image,
            k=(augment_params['rotate'] // 90),
         ),
      )  # Rotate by k * 90 degrees
   
   if augment_params['translate']:
      translation = augment_params['translate']
      matrix = np.float32(
         [
            [1, 0, translation[0]],
            [0, 1, translation[1]],
         ]
      )  # Translation matrix
      image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
   
   return image

def preprocess_3(image, Z):
   """
   Padding the image to global Z x Z dimensions.
   """
   # print(f"Original image shape: {image.shape}")
   if len(image.shape) == 2:
      image = np.expand_dims(image, axis=-1)
   # print(f"Image after adding channel: {image.shape}")
   h, w = image.shape[:2]
   pad_h = (Z - h) // 2
   pad_w = (Z - w) // 2
   # Ensure padding is applied symmetrically
   if (Z - h) % 2 != 0:
      pad_h1, pad_h2 = pad_h, pad_h + 1
   else:
      pad_h1, pad_h2 = pad_h, pad_h
   
   if (Z - w) % 2 != 0:
      pad_w1, pad_w2 = pad_w, pad_w + 1
   else:
      pad_w1, pad_w2 = pad_w, pad_w
   
   # Apply padding
   padded_image = np.pad(
      image,
      ((pad_h1, pad_h2), (pad_w1, pad_w2), (0, 0)),
      mode='constant',
      constant_values=0,
   )
   # print(f"Padded image shape: {padded_image.shape}")
   return padded_image

def preprocess_4(image, K):
   """
   Resize the image to global K x K dimensions.
   """
   if image.shape[0] == K:
      return image
   return cv2.resize(image, (K, K))  # Resize to K x K

# Lazy Image Loading Dataset Class
class LazyImageDataset(Sequence):
   def __init__(
      self,
      dataset,
      # batch_size=32,
      batch_size=4,
      dataset_type='train',
   ):
      global class_mapping
      self.dataset = dataset[dataset_type].copy()
      random.shuffle(self.dataset)
      self.batch_size = batch_size
      self.dataset_type = dataset_type
      # self.classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
      self.classes = class_mapping.copy()
      self.indexes = np.arange(len(self.dataset))
      # self.indexes = np.arange((len(self.dataset) // self.batch_size))
   
   def __len__(self):
      # Return the number of batches per epoch
      return int(np.floor(len(self.dataset) / self.batch_size))
   
   def __getitem__(self, idx):
      """
      Load a single batch lazily. Each item contains a single image and its label.
      """
      # Start of batch
      batch_paths = self.dataset[
         idx * self.batch_size: (idx + 1) * self.batch_size
      ]
      batch_labels = []
      batch_images = []
      
      for img_path, label, augment_type, augment_params in batch_paths:
         # Apply the appropriate augmentation (real or augmented)
         batch_labels.append(self.classes.index(label))
         
         # Lazy loading of single image
         img = self.load_image(img_path, augment_type, augment_params)
         batch_images.append(img)
      
      batch_labels = tf.keras.utils.to_categorical(
         batch_labels, num_classes=4,
      )
      
      return np.array(batch_images), np.array(batch_labels)

   def on_epoch_end(self):
      # Shuffle indexes at the end of each epoch
      np.random.shuffle(self.indexes)

   def load_image(self, img_path, augment_type, augment_params):
      global Z, K
      
      """Helper function to load and process images lazily based on metadata"""
      # Load the original image
      img = load_img(
         img_path,
         # target_size=(Z, Z),
         color_mode='grayscale',
      )  # Convert to grayscale
      img = img_to_array(img)
      
      # Apply preprocessors
      img = preprocess_1(img)  # Placeholder for future use
      
      # Apply augmentation before padding and resizing
      if augment_type == 'augmented':
         img = preprocess_2(img, augment_params)
      
      # Apply padding and resizing (common for both real and augmented images)
      img = preprocess_3(img, Z)
      img = preprocess_4(img, K)
      
      return img

# Compute Z and K when generating dataset
def generate_dataset_info(dataset_dirs):
   dataset = {
      'train' : [],
      'test'  : [],
      'val'   : [],
   }
   max_width, max_height = 0, 0
   
   augment_params_list = [
      {'rotate': 90, 'translate': (5, 5)},   # Rotate by 90 degrees and translate by 5px
      {'rotate': 180, 'translate': (0, 0)},  # Rotate by 180 degrees, no translation
      {'rotate': 0, 'translate': (10, 10)},  # No rotation, translate by 10px
      {'rotate': 270, 'translate': (-5, -5)},# Rotate by 270 degrees, translate by -5px
   ]
   
   # Loop through train, test, and val directories
   for dataset_type, dataset_path in dataset_dirs.items():
      # Traverse each class folder (CNV, DME, DRUSEN, NORMAL)
      for class_dir in os.listdir(dataset_path):
         class_path = os.path.join(dataset_path, class_dir)
         if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
               img_path = os.path.join(class_path, image_name)
               if img_path.lower().endswith(('.jpeg', '.jpg', '.png')):
                  with Image.open(img_path) as img:
                     # Get image size (width, height)
                     width, height = img.size
                     max_width = max(max_width, width)
                     max_height = max(max_height, height)
                  
                  # Add the real image entry
                  dataset[dataset_type].append(
                     (img_path, class_dir, 'real', None)
                  )  # Real image
                  
                  # Add augmented images (create several augmented versions per real image)
                  for augment_params in augment_params_list:
                     dataset[dataset_type].append(
                        (img_path, class_dir, 'augmented', augment_params)
                     )  # Augmented image
   
   # Calculate global Z (max dimension) and global K (for resizing)
   Z = max(max_width, max_height)
   # K = Z // 2 if Z // 2 > 32 else 32  # For memory optimization
   K = 120
   
   dataset_info = {'Z': Z, 'K': K, 'dataset': dataset}
   return dataset_info

def dataset_init ():
   global dataset_file, dataset_info, dataset, Z, K
   # Check if dataset preprocessed file exists
   if os.path.exists(dataset_file):
      print('[ dataset ]: found, loading existing')
      with open(dataset_file, 'r') as f:
         dataset_info = json.load(f)
      Z = dataset_info['Z']
      K = dataset_info['K']
   else:
      print('[ dataset ]: not found, generating')
      dataset_info = None
   
   # Generate or load dataset info
   if dataset_info is None:
      dataset_dirs = {
         'train': train_dir,
         'test': test_dir,
         'val': val_dir,
      }
      dataset_info = generate_dataset_info(dataset_dirs)
      with open(dataset_file, 'w') as f:
         json.dump(dataset_info, f)

   Z = dataset_info['Z']
   K = dataset_info['K']
   dataset = dataset_info['dataset']

# Define the CNN model
def create_model(input_shape):
   model = models.Sequential()
   model.add(
      # layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)
      layers.Conv2D(
         32,
         (3, 3),
         activation='relu',
         input_shape=input_shape,
      )
   )
   model.add(layers.MaxPooling2D((2, 2)))
   model.add(layers.Conv2D(64, (3, 3), activation='relu'))
   model.add(layers.MaxPooling2D((2, 2)))
   model.add(layers.Conv2D(128, (3, 3), activation='relu'))
   model.add(layers.MaxPooling2D((2, 2)))
   model.add(layers.Flatten())
   model.add(layers.Dense(128, activation='relu'))
   model.add(layers.Dense(4, activation='softmax'))  # 4 classes
   model.compile(
      optimizer='adam',
      loss='categorical_crossentropy',
      metrics=[
         'accuracy',
         'categorical_accuracy',
         'mae',
         'mse',
      ],
   )
   return model

def model_train ():
   global model_loss_path, model_accuracy_path, model_final_path
   global model_accuracy_categorical_path, model_mae_path, model_mse_path
   global model_iteration_path, dataset, Z, K
   
   # Load model if it exists
   if os.path.exists(model_accuracy_categorical_path):
      print('[ model ]: found, loading existing')
      model = tf.keras.models.load_model(model_loss_path)
   else:
      print('[ model ]: not found, generating')
      model = create_model((K, K, 1))  # Input shape for grayscale images
   
   # Callbacks for saving best models
   checkpoint_loss = ModelCheckpoint(
      model_loss_path,
      monitor='loss',
      save_best_only=True,
      save_weights_only=False,
      mode='min',
      verbose=1,
   )
   checkpoint_accuracy = ModelCheckpoint(
      model_accuracy_path,
      monitor='accuracy',
      save_best_only=True,
      save_weights_only=False,
      mode='max',
      verbose=1,
   )
   checkpoint_accuracy_categorical = ModelCheckpoint(
      model_accuracy_categorical_path,
      monitor='categorical_accuracy',
      save_best_only=True,
      save_weights_only=False,
      mode='max',
      verbose=1,
   )
   checkpoint_mae = ModelCheckpoint(
      model_mae_path,
      monitor='mae',
      save_best_only=True,
      save_weights_only=False,
      mode='min',
      verbose=1,
   )
   checkpoint_mse = ModelCheckpoint(
      model_mse_path,
      monitor='mse',
      save_best_only=True,
      save_weights_only=False,
      mode='min',
      verbose=1,
   )
   checkpoint_final = ModelCheckpoint(
      model_final_path,
      save_best_only=False,
      save_weights_only=False,
      mode='auto',
      verbose=1,
   )
   checkpoint_iteration= ModelCheckpoint(
      model_iteration_path,
      save_freq='epoch',
      # verbose=1,
   )
   
   # Train the model
   # train_batch_size = 32
   train_batch_size = 500
   train_steps = len(dataset['train']) // train_batch_size
   train_gen = LazyImageDataset(
      dataset,
      batch_size=train_batch_size,
      dataset_type='train',
   )
   
   val_batch_size = 4
   val_steps = len(dataset['val']) // val_batch_size
   val_gen = LazyImageDataset(
      dataset,
      batch_size=val_batch_size,
      dataset_type='val',
   )
   
   model.fit(
      train_gen,
      # steps_per_epoch=train_steps,
      # epochs=50,
      epochs=20,
      validation_data=val_gen,
      # validation_steps=val_steps,
      callbacks=[
         checkpoint_loss,
         checkpoint_accuracy,
         checkpoint_accuracy_categorical,
         checkpoint_mae,
         checkpoint_mse,
         checkpoint_final,
         checkpoint_iteration,
         ReduceLROnPlateau(                                
            monitor='val_accuracy',  # or 'val_accuracy'
            factor=0.8, # 0.5,       # Reduce the learning rate by half
            patience=0,          # Number of epochs with no improvement after which learning rate will be reduced
            min_lr=1e-9,         # Lower bound on learning rate
            verbose=1            # Print a message when the learning rate is reduced
         ),
         LearningRateScheduler((
            lambda epoch, lr: (lr * 0.94)
         )),
      ]
   )
   
   # Save the final model
   model.save(model_final_path)
   
   # Evaluate the model on the test set
   test_batch_size = 120
   test_steps = len(dataset['test']) // test_batch_size
   test_gen = LazyImageDataset(
      dataset,
      batch_size=test_batch_size,
      dataset_type='test',
   )
   
   (
      test_loss,
      test_acc,
      test_cat_acc,
      test_mae,
      test_mse,
   ) = model.evaluate(test_gen, steps=test_steps)
   
   print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")
   print(f"Test categorical accuracy: {test_cat_acc}")
   print(f"Test MAE: {test_mae}, Test MSE: {test_mse}")

def train ():
   dataset_init()
   model_train()

def predict ():
   global predict_dir, prediction_file, class_mapping, Z, K
   
   if (
         (Z == 0)
      or (K == 0)
   ):
      dataset_init()
   
   dataset_prediction = list()
   models = dict()
   model_paths = {
      'loss'                  : model_loss_path,
      'accuracy'              : model_accuracy_path,
      'accuracy.categorical'  : model_accuracy_categorical_path,
      'mae'                   : model_mae_path,
      'mse'                   : model_mse_path,
      'final'                 : model_final_path,
      'iteration'             : model_iteration_path,
   }
   
   for model_name, model_path in model_paths.items():
      model = None
      try:
         model = tf.keras.models.load_model(
            model_path,
         )
      except:
         model = None
      
      if (model is not None):
         models[model_name] = model
   
   for image_name in os.listdir(predict_dir):
      img_path = os.path.join(predict_dir, image_name)
      if img_path.lower().endswith(('.jpeg', '.jpg', '.png')):
         img = load_img(
            img_path,
            # target_size=(Z, Z),
            color_mode='grayscale',
         )  # Convert to grayscale
         img = img_to_array(img)
         
         # Apply preprocessors
         img = preprocess_1(img)  # Placeholder for future use
         
         # Apply padding and resizing (common for both real and augmented images)
         img = preprocess_3(img, Z)
         img = preprocess_4(img, K)
         # Reshape image to be compatible with model input (batch_size, height, width, channels)
         # img = np.expand_dims(img, axis=0)  # Add batch dimension (1, K, K, 1)
         img = np.reshape(img, (1, *img.shape))
         
         prediction = {
            'image' : image_name,
         }
         
         for model_key, model in models.items():
            prediction[model_key] = class_mapping[
               np.argmax(
                  model.predict(img),
                  axis=-1,
               )[0]
            ]
         
         dataset_prediction.append(prediction)
   
   # save prediction
   with open(prediction_file, 'w') as f:
      json.dump(dataset_prediction, f)

if __name__ == '__main__':
   print('[ status ] : starting')
   
   choice_model_train = False
   choice_model_predict = False
   
   if (len(sys.argv) > 1):
      for argv in sys.argv[1:]:
         if (str(argv).lower()[:1]) in (
            't', '0', 'm'
         ):
            choice_model_train = True
         
         if (str(argv).lower()[:1]) in (
            'p', '1', 'u'
         ):
            choice_model_predict = True
   else:
      choice_model_train = True
      choice_model_predict = True
   
   if (choice_model_train):
      print('[ state ] : training')
      train()
      print('[ state ] : trained')
   
   if (choice_model_predict):
      print('[ state ] : predicting')
      predict()
      print('[ state ] : predicted')
   
   print('[ status ] : done')
