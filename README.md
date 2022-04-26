# pytorch_lite

- flutter package to help run pytorch lite models classification and yolov5
- ios support (can be added following this https://github.com/pytorch/ios-demo-app) PR will be appreciated  


# example for Classification
![image](https://user-images.githubusercontent.com/25157308/165343107-85bc8d7f-3db2-425e-bcbc-6a4c18c77947.png)

# example for Object detection
![image](https://user-images.githubusercontent.com/25157308/165341783-3296579c-bbb5-47ff-9588-d34fb143e6c9.png)



## Usage

### Installation

To use this plugin, add `pytorch_mobile` as a [dependency in your pubspec.yaml file](https://flutter.dev/docs/development/packages-and-plugins/using-packages).

Create a `assets` folder with your pytorch model and labels if needed. Modify `pubspec.yaml` accoringly.

```yaml
assets:
 - assets/models/model_classification.pt
 - assets/labels_classification.txt
 - assets/models/model_objectDetection.torchscript
 - assets/labels_objectDetection.txt
```

Run `flutter pub get`

### Import the library

```dart
import 'package:pytorch_lite/pytorch_lite.dart';
```

### Load model

Either classification model:
```dart
ClassificationModel classificationModel= await PytorchLite.loadClassificationModel(
          "assets/models/model_classification.pt", 224, 224,
          labelPath: "assets/labels/label_classification_imageNet.txt");
```
Or objectDetection model:
```dart
ModelObjectDetection objectModel = await PytorchLite.loadObjectDetectionModel(
          "assets/models/yolov5s.torchscript", 80, 640, 640,
          labelPath: "assets/labels/labels_objectDetection_Coco.txt");
```

### Get classification prediction as label

```dart
String imagePrediction = await classificationModel.getImagePrediction(File(image.path));
```
### Get classification prediction as raw output layer

```dart
List<double?>? predictionList = await _imageModel!.getImagePredictionList(
      File(image.path),
    );
```

### Get object detection prediction for an image
```dart
 List<ResultObjectDetection?> objDetect = await _objectModel.getImagePrediction(File(image!.path),
        minimumScore: 0.1, IOUThershold: 0.3);
```

### Image prediction for an image with custom mean and std
```dart
final mean = [0.5, 0.5, 0.5];
final std = [0.5, 0.5, 0.5];
String prediction = await classificationModel
        .getImagePrediction(image, mean: mean, std: std);
```



#References 
- code used the samme strucute as the package https://pub.dev/packages/pytorch_mobile
- while using the updated code from https://github.com/pytorch/android-demo-app
