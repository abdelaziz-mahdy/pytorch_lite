# pytorch_lite

- flutter package to help run pytorch lite models classification and YoloV5 and YoloV8.

# example for Classification

![image](https://user-images.githubusercontent.com/25157308/165343107-85bc8d7f-3db2-425e-bcbc-6a4c18c77947.png)

# example for Object detection

![image](https://user-images.githubusercontent.com/25157308/165341783-3296579c-bbb5-47ff-9588-d34fb143e6c9.png)

## Usage

## preparing the model

- classification

```python
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile


model = torch.load('model_scripted.pt',map_location="cpu")
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
optimized_traced_model = optimize_for_mobile(traced_script_module)
optimized_traced_model._save_for_lite_interpreter("model.pt")
```

- object detection (yolov5)

```python
!python export.py --weights "the weights of your model" --include torchscript --img 640 --optimize
```

example

```python
!python export.py --weights yolov5s.pt --include torchscript --img 640 --optimize
```

- object detection (yolov8)

```python
!yolo mode=export model="your model" format=torchscript optimize
```

example

```python
!yolo mode=export model=yolov8s.pt format=torchscript optimize
```

### Installation

To use this plugin, add `pytorch_lite` as a [dependency in your pubspec.yaml file](https://flutter.dev/docs/development/packages-and-plugins/using-packages).

Create a `assets` folder with your pytorch model and labels if needed. Modify `pubspec.yaml` accordingly.

```yaml
assets:
  - assets/models/model_classification.pt
  - assets/labels_classification.txt
  - assets/models/model_objectDetection.torchscript
  - assets/labels_objectDetection.txt
```

Run `flutter pub get`

#### For release

- Go to android/app/build.gradle
- Add those next lines in the release config

```
shrinkResources false
minifyEnabled false
```

example

```
    buildTypes {
        release {
            shrinkResources false
            minifyEnabled false
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig signingConfigs.debug
        }
    }
```

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
          labelPath: "assets/labels/labels_objectDetection_Coco.txt",
          objectDetectionModelType: ObjectDetectionModelType.yolov5);
```

### Get classification prediction as label

```dart
String imagePrediction = await classificationModel.getImagePrediction(await File(image.path).readAsBytes());
```

### Get classification prediction as label from camera image

```dart
String imagePrediction = await _objectModel.getCameraImagePrediction(
        cameraImage,
        rotation, // check example for rotation values
        );
```

### Get classification prediction as raw output layer

```dart
List<double>? predictionList = await _imageModel!.getImagePredictionList(
      await File(image.path).readAsBytes(),
    );
```

### Get classification prediction as raw output layer from camera image
```dart
List<double>? predictionList = await _imageModel!.getCameraImagePredictionList(
        cameraImage,
        rotation, // check example for rotation values
    );
```

### Get classification prediction as Probabilities (incase model is not using softmax)
```dart
List<double>? predictionListProbabilities = await _imageModel!.getImagePredictionListProbabilities(
      await File(image.path).readAsBytes(),
    );
```
### Get classification prediction as Probabilities (incase model is not using softmax)
```dart
List<double>? predictionListProbabilities = await _imageModel!.getCameraPredictionListProbabilities(
        cameraImage,
        rotation, // check example for rotation values
    );
```
### Get object detection prediction for an image
```dart
 List<ResultObjectDetection> objDetect = await _objectModel.getImagePrediction(await File(image.path).readAsBytes(),
        minimumScore: 0.1, IOUThershold: 0.3);
```

### Get object detection prediction from camera image

```dart
 List<ResultObjectDetection> objDetect = await _objectModel.getCameraImagePrediction(
        cameraImage,
        rotation, // check example for rotation values
        minimumScore: 0.1, IOUThershold: 0.3);
```

### Get render boxes with image

```dart
objectModel.renderBoxesOnImage(_image!, objDetect)
```

### Image prediction for an image with custom mean and std

```dart
final mean = [0.5, 0.5, 0.5];
final std = [0.5, 0.5, 0.5];
String prediction = await classificationModel
        .getImagePrediction(image, mean: mean, std: std);
```