import 'package:pigeon/pigeon.dart';

/*
flutter pub run pigeon --input pigeons/messages.dart --dart_out lib/pigeon.dart --objc_header_out ios/Runner/pigeon.h --objc_source_out ios/Runner/pigeon.m --java_out android/src/main/java/com/zezo789/pytorch_lite/Pigeon.java --java_package "com.zezo789.pytorch_lite"

flutter pub run pigeon --input pigeons/messages.dart --dart_out lib/pigeon.dart  --java_out android/src/main/java/com/zezo789/pytorch_lite/Pigeon.java --java_package "com.zezo789.pytorch_lite"

*/
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
  int loadModel(
      String modelPath, int? numberOfClasses, int imageWidth, int imageHeight);

  ///predicts abstract number input
  @async
  List? getPredictionCustom(
      int index, List<double> input, List<int> shape, String dtype);

  ///predicts image but returns the raw net output
  @async
  List<double>? getImagePredictionList(int index, Uint8List imageData,
      int width, int height, List<double> mean, List<double> std);

  ///predicts image but returns the output detections
  @async
  List<ResultObjectDetection> getImagePredictionListObjectDetection(
      int index,
      Uint8List imageData,
      double minimumScore,
      double IOUThreshold,
      int boxesLimit);
}
