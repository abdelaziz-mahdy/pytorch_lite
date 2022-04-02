import 'package:pigeon/pigeon.dart';

/*
flutter pub run pigeon --input pigeons/messages.dart --dart_out lib/pigeon.dart --objc_header_out ios/Runner/pigeon.h --objc_source_out ios/Runner/pigeon.m --java_out android/src/main/java/com/zezo789/pytorch_lite/Pigeon.java --java_package "com.zezo789.pytorch_lite"

flutter pub run pigeon --input pigeons/messages.dart --dart_out lib/pigeon.dart  --java_out android/src/main/java/com/zezo789/pytorch_lite/Pigeon.java --java_package "com.zezo789.pytorch_lite"

*/

enum DType {
  float32,
  float64,
  int32,
  int64,
  int8,
  uint8,
}

@HostApi()
abstract class ModelApi {
  int loadModel(
    String modelPath,
    String labelsPath,
  );

  ///predicts abstract number input
  @async
  List? getPredictionCustom(
      int index, List<double> input, List<int> shape, String dtype);

  ///predicts image and returns the supposed label belonging to it
  @async
  String getImagePrediction(int index, String imagePath, int width, int height,
      List<double> mean, List<double> std);

  ///predicts image but returns the raw net output
  @async
  List? getImagePredictionList(int index, String imagePath, int width,
      int height, List<double> mean, List<double> std);

  ///predicts image and returns the path of the image with detection on it
  @async
  String getImagePredictionObjectDetection(int index, String imagePath,
      int width, int height, List<double> mean, List<double> std);

  ///predicts image but returns the output detections
  @async
  List? getImagePredictionListObjectDetection(int index, String imagePath,
      int width, int height, List<double> mean, List<double> std);
}
