import os
from pathlib import Path
# import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(2024)

path_plot_dir = (Path(__file__).parent / '..').resolve()

def plot_curve (plot_title, plot_type, Y_label, X, Y, Y_legends, colors):
   # plot config
   fig, axs = plt.subplots()
   
   # plot - data points
   # axs.scatter(X, Y)
   
   for y, color in zip(Y, colors):
      # plot - values
      axs.plot(
         X,
         y,
         color=color,
      )
   
   plt.xticks(X)
   # setting labels
   axs.set_xlabel(
      'Epoch(s)'
   )
   axs.set_ylabel(
      Y_label
   )
   axs.legend(
      Y_legends,
   )
   axs.set_title(
      plot_title,
   )
   
   global path_plot_dir
   plt.savefig(os.path.abspath(
      (
            path_plot_dir
         /  'plot.curve.training.dataset.{0}.png'.format(plot_type)
      ).resolve()
   ))
   plt.close(fig)
   # plt.show()

def plot ():
   data_plot = [
      {
         'plot_title': 'Model Training: Epochs Vs Loss Curve',
         'plot_type' : 'loss',
         'Y_label'   : 'Loss',
         'X'         : [
            i
            for i in range(1, 21)
         ],
         'Y'         : [
            [  # train augmented
               0.5250, 0.4546, 0.4062, 0.3736, 0.3358,
               0.3054, 0.2857, 0.2638, 0.2397, 0.2209,
               0.2044, 0.1920, 0.1808, 0.1705, 0.1620,
               0.1556, 0.1522, 0.1498, 0.1466, 0.1440,
            ],
            [  # val augmented
               0.9766, 0.9268, 0.7379, 0.7582, 0.6680,
               0.6224, 0.7197, 0.6903, 0.5847, 0.6441,
               0.6647, 0.6697, 0.6531, 0.6900, 0.6670,
               0.6865, 0.7070, 0.7071, 0.7014, 0.6729,
            ],
            [  # train original
               3.7559, 0.6106, 0.5328, 0.4762, 0.4449,
               0.4072, 0.3902, 0.3724, 0.3516, 0.3457,
               0.3321, 0.3328, 0.3164, 0.3068, 0.3023,
               0.2934, 0.2874, 0.2888, 0.2826, 0.2853,
            ],
            [  # val original
               1.1818, 0.8215, 0.8026, 0.5917, 0.6457,
               0.6049, 0.6223, 0.6209, 0.6902, 0.6424,
               0.6393, 0.6125, 0.6160, 0.6634, 0.6338,
               0.5979, 0.5839, 0.6071, 0.5605, 0.5975,
            ],
         ],
         'Y_legends' : [
            'loss.train.augmented',
            'loss.val.augmented',
            'loss.train.original',
            'loss.val.original',
         ],
         'colors'    : [
            'blue',
            'purple',
            'orange',
            'red',
         ],
      },
      {
         'plot_title': 'Model Training: Epochs Vs Accuracy Curve',
         'plot_type' : 'accuracy',
         'Y_label'   : 'Accuracy',
         'X'         : [
            i
            for i in range(1, 21)
         ],
         'Y'         : [
            [  # train augmented
               0.7939, 0.8219, 0.8435, 0.8560, 0.8712,
               0.8842, 0.8921, 0.9006, 0.9105, 0.9183,
               0.9254, 0.9297, 0.9346, 0.9387, 0.9421,
               0.9451, 0.9465, 0.9479, 0.9492, 0.9500,
            ],
            [  # val augmented
               0.6687, 0.7312, 0.8000, 0.7625, 0.8000,
               0.8687, 0.8188, 0.8500, 0.8687, 0.8687,
               0.8750, 0.8750, 0.8625, 0.8750, 0.8750,
               0.8750, 0.8750, 0.8750, 0.8687, 0.8813,
            ],
            [  # train original
               0.4754, 0.7592, 0.7883, 0.8123, 0.8273,
               0.8406, 0.8491, 0.8571, 0.8658, 0.8667,
               0.8750, 0.8732, 0.8817, 0.8848, 0.8860,
               0.8925, 0.8928, 0.8929, 0.8941, 0.8944,
            ],
            [  # val original
               0.5312, 0.6250, 0.7188, 0.8125, 0.8125,
               0.8125, 0.8125, 0.8125, 0.8125, 0.8438,
               0.8750, 0.8438, 0.9062, 0.8750, 0.8438,
               0.8750, 0.8750, 0.8750, 0.8750, 0.8750,
            ],
         ],
         'Y_legends' : [
            'accuracy.train.augmented',
            'accuracy.val.augmented',
            'accuracy.train.original',
            'accuracy.val.original',
         ],
         'colors'    : [
            'blue',
            'purple',
            'orange',
            'red',
         ],
      },
      {
         'plot_title': 'Model Training: Epochs Vs Error(s) Curve',
         'plot_type' : 'error',
         'Y_label'   : 'Error',
         'X'         : [
            i
            for i in range(1, 21)
         ],
         'Y'         : [
            [  # mae train augmented
               0.1452, 0.1267, 0.1135, 0.1045, 0.0941,
               0.0861, 0.0807, 0.0753, 0.0690, 0.0642,
               0.0600, 0.0568, 0.0538, 0.0510, 0.0489,
               0.0471, 0.0461, 0.0454, 0.0445, 0.0438,
            ],
            [  # mae val augmented
               0.1940, 0.1739, 0.1429, 0.1434, 0.1273,
               0.1038, 0.1148, 0.1027, 0.0908, 0.0889,
               0.0903, 0.0862, 0.0827, 0.0823, 0.0789,
               0.0784, 0.0788, 0.0781, 0.0765, 0.0747,
            ],
            [  # mae train original
               0.3103, 0.1699, 0.1492, 0.1337, 0.1251,
               0.1157, 0.1116, 0.1071, 0.1019, 0.1005,
               0.0962, 0.0963, 0.0924, 0.0900, 0.0888,
               0.0865, 0.0854, 0.0853, 0.0838, 0.0844,
            ],
            [  # mae val original
               0.2307, 0.1931, 0.1767, 0.1364, 0.1389,
               0.1309, 0.1180, 0.1237, 0.1277, 0.1122,
               0.1098, 0.1053, 0.1036, 0.1057, 0.1055,
               0.1036, 0.0970, 0.0979, 0.0967, 0.0977,
            ],
            
            [  # mse train augmented
               0.0721, 0.0631, 0.0563, 0.0518, 0.0466,
               0.0422, 0.0395, 0.0366, 0.0332, 0.0305,
               0.0281, 0.0264, 0.0247, 0.0233, 0.0221,
               0.0211, 0.0206, 0.0202, 0.0197, 0.0194,
            ],
            [  # mse val augmented
               0.1229, 0.1044, 0.0833, 0.0884, 0.0751,
               0.0604, 0.0707, 0.0629, 0.0555, 0.0542,
               0.0563, 0.0547, 0.0535, 0.0536, 0.0505,
               0.0514, 0.0520, 0.0515, 0.0511, 0.0491,
            ],
            [  # mse train original
               0.1772, 0.0837, 0.0736, 0.0660, 0.0616,
               0.0565, 0.0541, 0.0514, 0.0486, 0.0480,
               0.0457, 0.0460, 0.0436, 0.0423, 0.0417,
               0.0402, 0.0395, 0.0396, 0.0388, 0.0392,
            ],
            [  # mse val original
               0.1443, 0.1112, 0.1047, 0.0733, 0.0814,
               0.0775, 0.0677, 0.0735, 0.0765, 0.0670,
               0.0672, 0.0636, 0.0618, 0.0641, 0.0652,
               0.0635, 0.0589, 0.0600, 0.0597, 0.0600,
            ],
         ],
         'Y_legends' : [
            'error.mae.train.augmented',
            'error.mae.val.augmented',
            'error.mae.train.original',
            'error.mae.val.original',
            
            'error.mse.train.augmented',
            'error.mse.val.augmented',
            'error.mse.train.original',
            'error.mse.val.original',
         ],
         'colors'    : [
            'blue',
            'purple',
            'orange',
            'red',
            
            'yellow',
            'green',
            'black',
            'magenta',
         ],
      },
   ]
   
   for plot_data in data_plot:
      plot_curve(
         **plot_data,
      )
   
   return None

if __name__ == '__main__':
   print('Plotting ...')
   plot()
   print('Done !')
