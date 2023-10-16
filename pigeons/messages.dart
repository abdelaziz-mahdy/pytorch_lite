// ignore_for_file: public_member_api_docs, sort_constructors_first

import 'package:pigeon/pigeon.dart';

//Build android only
// flutter pub run pigeon --input pigeons/messages.dart --dart_out lib/pigeon.dart  --java_out android/src/main/java/com/zezo357/pytorch_lite/Pigeon.java --java_package "com.zezo357.pytorch_lite"

// Build android and ios

// flutter pub run pigeon --input pigeons/messages.dart --dart_out lib/pigeon.dart --objc_header_out ios/Classes/pigeon.h --objc_source_out ios/Classes/pigeon.mm --java_out android/src/main/java/com/zezo357/pytorch_lite/Pigeon.java --java_package "com.zezo357.pytorch_lite"

class PyTorchRect {
  double left;
  double top;
  double right;
  double bottom;
  double width;
  double height;
  PyTorchRect(
      this.left, this.top, this.width, this.height, this.right, this.bottom);
}

class ResultObjectDetection {
  int classIndex;
  String? className;
  double score;
  PyTorchRect rect;

  ResultObjectDetection(this.classIndex, this.score, this.rect);
}

// enum ObjectDetectionModelType { yolov5, yolov8 }

@HostApi()
abstract class ModelApi {
  @TaskQueue(type: TaskQueueType.serialBackgroundThread)
  @async
  int loadModel(String modelPath, int? numberOfClasses, int? imageWidth,
      int? imageHeight, int? objectDetectionModelType);

  ///predicts abstract number input
  @TaskQueue(type: TaskQueueType.serialBackgroundThread)
  @async
  List? getPredictionCustom(
      int index, List<double> input, List<int> shape, String dtype);

  ///predicts raw image but returns the raw net output
  @TaskQueue(type: TaskQueueType.serialBackgroundThread)
  @async
  List<double> getRawImagePredictionList(
    int index,
    Float64List imageData,
    bool isTupleOutput,
    int tupleIndex,
  );

  ///predicts raw image but returns the raw net output
  @TaskQueue(type: TaskQueueType.serialBackgroundThread)
  @async
  List<ResultObjectDetection> getRawImagePredictionListObjectDetection(
    int index,
    Uint8List imageData,
    double minimumScore,
    double IOUThreshold,
    int boxesLimit,
    bool isTupleOutput,
    int tupleIndex,
  );

  ///predicts image but returns the raw net output
  @TaskQueue(type: TaskQueueType.serialBackgroundThread)
  @async
  List<double> getImagePredictionList(
    int index,
    Uint8List? imageData,
    List<Uint8List>? imageBytesList,
    int? imageWidthForBytesList,
    int? imageHeightForBytesList,
    List<double> mean,
    List<double> std,
    bool isTupleOutput,
    int tupleIndex,
  );

  ///predicts image but returns the output detections
  @TaskQueue(type: TaskQueueType.serialBackgroundThread)
  @async
  List<ResultObjectDetection> getImagePredictionListObjectDetection(
    int index,
    Uint8List? imageData,
    List<Uint8List>? imageBytesList,
    int? imageWidthForBytesList,
    int? imageHeightForBytesList,
    double minimumScore,
    double IOUThreshold,
    int boxesLimit,
    bool isTupleOutput,
    int tupleIndex,
  );
}
