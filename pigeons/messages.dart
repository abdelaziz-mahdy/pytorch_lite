import 'package:pigeon/pigeon.dart';
// Build android only
// flutter pub run pigeon --input pigeons/messages.dart --dart_out lib/pigeon.dart  --java_out android/src/main/java/com/zezo357/pytorch_lite/Pigeon.java --java_package "com.zezo357.pytorch_lite"

// Build android and ios (it fails)

//flutter pub run pigeon --input pigeons/messages.dart --dart_out lib/pigeon.dart --objc_header_out ios/Classes/pigeon.h --objc_source_out ios/Classes/pigeon.m --java_out android/src/main/java/com/zezo357/pytorch_lite/Pigeon.java --java_package "com.zezo357.pytorch_lite"

class Rect {
  double left;
  double top;
  double right;
  double bottom;
  double width;
  double height;
  Rect(this.left, this.top, this.width, this.height, this.right, this.bottom);
}

class ResultObjectDetection {
  int classIndex;
  String? className;
  double score;
  Rect rect;

  ResultObjectDetection(this.classIndex, this.score, this.rect);
}

@HostApi()
abstract class ModelApi {
  @TaskQueue(type: TaskQueueType.serialBackgroundThread)
  int loadModel(String modelPath, int? numberOfClasses, int? imageWidth,
      int? imageHeight);

  ///predicts abstract number input
  @TaskQueue(type: TaskQueueType.serialBackgroundThread)
  @async
  List? getPredictionCustom(
      int index, List<double> input, List<int> shape, String dtype);

  ///predicts image but returns the raw net output
  @TaskQueue(type: TaskQueueType.serialBackgroundThread)
  @async
  List<double>? getImagePredictionList(
      int index, Uint8List imageData, List<double> mean, List<double> std);

  ///predicts image but returns the output detections
  @TaskQueue(type: TaskQueueType.serialBackgroundThread)
  @async
  List<ResultObjectDetection> getImagePredictionListObjectDetection(
      int index,
      Uint8List imageData,
      double minimumScore,
      double IOUThreshold,
      int boxesLimit);
}
