## FeatureCloud CNN Watermarking App

This app embeds a watermark into a given CNN model. 

### Client's data 

The app is designed for single-client usage. 
A client's data folder should contain two files:
- config.yml 
- model.pth (the name could be different but should coincide with the 'model_name' given in config.yml)

### Config file 
In the config file the following information should be provided:

#### Model Parameters
  - **model_name** (default *'model.pth'*)  
  The name of the file in the client's data folder that contains a model.

- **architecture** (default *'resnet34'*)  
  The architecture of the model. Models from *timm* package are supported. The given name should be the same as the one used in *timm* for that architecture type.

#### Watermarking Parameters
- **wm_type** (default *'ood_abstract'*)  
  A watermark embedding method. Currently, three methods are supported:
   - *'ood_abstract'*  
   Implements an approach described in the paper [Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring](https://www.usenix.org/system/files/conference/usenixsecurity18/sec18-adi.pdf). The dataset used for watermarking is given within the app. The number of trigger images (see below) should not exceed 100.
   - *'ood_torchvision'*  
  Implements an approach described in the paper [Protecting Intellectual Property of Deep Neural Networks with Watermarking](https://www.doi.org/10.1145/3196494.3196550). In the paper, only MNIST and CIFAR10 were considered as trigger datasets. This app supports the majority of *torchvision* image datasets:
      -     'caltech101': torchvision.datasets.Caltech101,
      -     'caltech256': torchvision.datasets.Caltech256,
      -     'celeba': torchvision.datasets.CelebA,
      -     'cifar10': torchvision.datasets.CIFAR10,
      -     'cifar100': torchvision.datasets.CIFAR100,
      -     'cityscapes': torchvision.datasets.Cityscapes,
      -     'coco': torchvision.datasets.CocoDetection,
      -     'emnist': torchvision.datasets.EMNIST,
      -     'fakedata': torchvision.datasets.FakeData,
      -     'fmnist': torchvision.datasets.FashionMNIST,
      -     'flickr': torchvision.datasets.Flickr8k,
      -     'inaturalist': torchvision.datasets.INaturalist,
      -     'kitti': torchvision.datasets.Kitti,
      -     'kmnist': torchvision.datasets.KMNIST,
      -     'lfw': torchvision.datasets.LFWPeople,
      -     'lsun': torchvision.datasets.LSUN,
      -     'mnist': torchvision.datasets.MNIST,
      -     'omniglot': torchvision.datasets.Omniglot,
      -     'phototour': torchvision.datasets.PhotoTour,
      -     'place365': torchvision.datasets.Places365,
      -     'qmnist': torchvision.datasets.QMNIST,
      -     'sbd': torchvision.datasets.SBDataset,
      -     'sbu': torchvision.datasets.SBU,
      -     'semeion': torchvision.datasets.SEMEION,
      -     'stl10': torchvision.datasets.STL10,
      -     'svhn': torchvision.datasets.SVHN,
      -     'usps': torchvision.datasets.USPS,
      -     'voc': torchvision.datasets.VOCDetection,
      -     'widerface': torchvision.datasets.WIDERFace.
   - *'custom'*  
  Train a model on a custom trigger set. For this method, additional hyperparameters should be provided (see below).
- **wm_classes** (default *[0]*)  
  Labels from the original dataset used to label the trigger set. The amount of samples is equally distributed among classes. 
- **wm_th** (default *80*)  
  A threshold for watermark verification. A watermark is considered successfully embedded if the accuracy of the model on the trigger set is greater or equal to *wm_th*. As soon as the threshold is reached, the training is stopped.
- **batch_size** (default *32*)  
A batch size used for training a model on a trigger set.
- **optimizer** (default *'sgd'*)   
An optimizer used for training a model on a trigger set. Supported values: 
  - 'adam'  
  - 'sgd'  
- **lr** (default *0.001*)  
A learning rate used for training a model on a trigger set.
- **momentum** (default *0.9*)  
If the optimizer is *'sgd'* and *momentum* is non-zero, SGD with momentum is used.
- **max_epochs** (default *100*)  
The maximum number of epochs for training on the trigger set. The training is stopped even if the model did not reach the desired performance on the trigger set.
  
#### Trigges Set Parameters
- **dataset_name** (default *null*)  
A *torchvision* dataset used as a trigger set. This parameters is required if *wm_type* is *'ood_torchvision'*.
- **dataset_folder** (default *null*)  
The name of a folder that contains the trigger set. This parameters is required if *wm_type* is *'custom'*.
- **data_extensions** (default *null*)  
A list of possible extensions of images in the custom trigger set.
- **trigger_set_size** (default *100*)  
The size of the trigger set.
- **image_size** 
  - **height** (default *224*)  
  - **width** (default *224*)  
The size of input images (should correspond to the model architecture).
- **num_channels** (default *3*)  
The number of input channels (should correspond to the model architecture).
- **num_classes** (default *10*)  
The number of classes (should correspond to the model architecture).
- **mean** (default *null*)  
The mean value for data normalization.
- **std** (default *null*)  
The std value for data normalization.

### Output data

The app produces four items:
- *'wm_model.pth'* - watermarked model.
- *'images'* - a folder with trigger set saved as images.
- *'tensors'* - a folder with trigger set saved as tensors.
- *'labels.csv'* - labels for the trigger set.
