######
README
######

project.classifier.oct
**********************
OCT image classifier to classifiy OCT images into 4 labels - CNV, DME, DRUSEN
and NORMAL.

Requirements
============
Install requirements as::
   
   # requirements:
   pip install opencv-python
   pip install pillow
   pip install numpy
   pip install keras
   pip install tensorflow

   # or:
   pip install -r project.classifier.oct/requirements.txt

Dataset
=======
Kaggle: `link <https://www.kaggle.com/datasets/paultimothymooney/kermany2018>`_.
Hyperparameter tuning
=====================
Tune hyperparameters at following locations in file::
   
   # potential locations in file (project.classifier.oct/classifier.oct.py)
   # which you need to change
   line 134: batch_size
   line 248: K
   line 286: layers ... # model layers
   line 289: model.add ... # model layers
   line 290: model.add ... # model layers
   line 291: model.add ... # model layers
   line 292: model.add ... # model layers
   line 347: train_batch_size
   line 366: epochs

Cleanup
=======
Files to remove to clean up::
   
   # Files to remove to freshly build / train.
   rm project.classifier.oct/prediction.json
   rm project.classifier.oct/dataset.image.json
   rm project.classifier.oct/model.*.h5

File system (dataset)
=====================
File system required for the program to run::
   
   # Expected file structure
   
   # training dataset (downloaded from kaggle)
   project.classifier.oct/OCT2017/
      train/
         CNV/
            *.jpeg # or .png or .jpg
         DME/
            *.jpeg # or .png or .jpg
         DRUSEN/
            *.jpeg # or .png or .jpg
         NORMAL/
            *.jpeg # or .png or .jpg
      test/
         CNV/
            *.jpeg # or .png or .jpg
         DME/
            *.jpeg # or .png or .jpg
         DRUSEN/
            *.jpeg # or .png or .jpg
         NORMAL/
            *.jpeg # or .png or .jpg
      val/
         CNV/
            *.jpeg # or .png or .jpg
         DME/
            *.jpeg # or .png or .jpg
         DRUSEN/
            *.jpeg # or .png or .jpg
         NORMAL/
            *.jpeg # or .png or .jpg
   
   
   # prediction dataset (create one yourself
   #     by copy-pasting some image from training dataset)
   project.classifier.oct/predict/
      *.jpeg # or .png or .jpg

Running
=======
Running the code::
   
   # How to run:
   # train and predict
   python project.classifier.oct/classifier.oct.py
   
   # train only
   python project.classifier.oct/classifier.oct.py t
   
   # predict only
   python project.classifier.oct/classifier.oct.py p

Predictions
===========
Prediction storage format::
   
   # project.classifier.oct/prediction.json
   [ # list
      { # row
         "image"     : "<file_name>",  # image file name (used)
         "loss"      : "<PREDICTION>", # class predicted by best 'loss' model
         "accuracy"  : "<PREDICTION>", # class predicted by best 'accuracy' model
         "final"     : "<PREDICTION>", # class predicted by best 'final' model
         "iteration" : "<PREDICTION>", # class predicted by best 'iteration' model
      },
   ]
