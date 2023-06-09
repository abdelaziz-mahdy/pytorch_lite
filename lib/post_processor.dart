import 'dart:math';

import 'package:pytorch_lite/classes/rect.dart';
import 'package:pytorch_lite/classes/result_object_detection.dart';
import 'package:pytorch_lite/enums/model_type.dart';

class PostProcessorObjectDetection {
  late int numberOfClasses;
  late int outputRow;
  late int outputColumn;
  late int modelOutputLength;
  double scoreThreshold = 0.30;
  double IOUThreshold = 0.30;
  int imageWidth;
  int imageHeight;
  int nmsLimit = 15;
  ObjectDetectionModelType objectDetectionModelType;

  PostProcessorObjectDetection(
    this.numberOfClasses,
    this.imageWidth,
    this.imageHeight,
    this.objectDetectionModelType,
  ) {
    if (objectDetectionModelType == ObjectDetectionModelType.yolov5) {
      outputRow = 25200;
      outputColumn = (numberOfClasses + 5);
      modelOutputLength = outputRow * outputColumn;
    } else {
      outputRow = 8400;
      outputColumn = (numberOfClasses + 4);
            modelOutputLength = outputRow * outputColumn;

    }
  }

  List<ResultObjectDetection> nonMaxSuppression(
      List<ResultObjectDetection> boxes) {
    // Sort the boxes based on the score in descending order
    boxes.sort((boxA, boxB) => boxB.score.compareTo(boxA.score));

    List<ResultObjectDetection> selected = [];
    List<bool> active = List<bool>.filled(boxes.length, true);
    int numActive = active.length;

    bool done = false;
    for (int i = 0; i < boxes.length && !done; i++) {
      if (active[i]) {
        ResultObjectDetection boxA = boxes[i];
        selected.add(boxA);
        if (selected.length >= nmsLimit) break;

        for (int j = i + 1; j < boxes.length; j++) {
          if (active[j]) {
            ResultObjectDetection boxB = boxes[j];
            if (iou(boxA.rect, boxB.rect) > IOUThreshold) {
              active[j] = false;
              numActive -= 1;
              if (numActive <= 0) {
                done = true;
                break;
              }
            }
          }
        }
      }
    }
    print("Result length after processing ${selected.length}");

    return selected;
  }

  double iou(PyTorchRect a, PyTorchRect b) {
    double areaA = (a.right - a.left) * (a.bottom - a.top);
    if (areaA <= 0.0) {
      return 0.0;
    }

    double areaB = (b.right - b.left) * (b.bottom - b.top);
    if (areaB <= 0.0) {
      return 0.0;
    }

    double intersectionMinX = max(a.left, b.left);
    double intersectionMinY = max(a.top, b.top);
    double intersectionMaxX = min(a.right, b.right);
    double intersectionMaxY = min(a.bottom, b.bottom);
    double intersectionArea = max(intersectionMaxY - intersectionMinY, 0.0) *
        max(intersectionMaxX - intersectionMinX, 0.0);
    return intersectionArea / (areaA + areaB - intersectionArea);
  }

  List<ResultObjectDetection> outputsToNMSPredictionsYoloV8(
      List<double> outputs) {
    List<ResultObjectDetection> results = [];
    for (int i = 0; i < outputRow; i++) {
      double x = outputs[i];
      double y = outputs[outputRow + i];
      double w = outputs[2 * outputRow + i];
      double h = outputs[3 * outputRow + i];

      double left = (x - w / 2);
      double top = (y - h / 2);
      double right = (x + w / 2);
      double bottom = (y + h / 2);

      double max = outputs[4 * outputRow + i];
      int cls = 0;
      for (int j = 4; j < outputColumn; j++) {
        if (outputs[j * outputRow + i] > max) {
          max = outputs[j * outputRow + i];
          cls = j - 4;
        }
      }

      if (max > scoreThreshold) {
        PyTorchRect rect = PyTorchRect(
            left: left / imageWidth,
            top: top / imageHeight,
            right: right / imageWidth,
            bottom: bottom / imageHeight,
            width: w,
            height: h);
        ResultObjectDetection result =
            ResultObjectDetection(classIndex: cls, score: max, rect: rect);

        results.add(result);
      }
    }

    print("Result length before processing ${results.length}");
    return nonMaxSuppression(results); // Please implement this method
  }

  List<ResultObjectDetection> outputsToNMSPredictionsYolov5(
      List<double> outputs) {
    List<ResultObjectDetection> results = [];
    for (int i = 0; i < outputRow; i++) {
      if (outputs[i * outputColumn + 4] > scoreThreshold) {
        double x = outputs[i * outputColumn];
        double y = outputs[i * outputColumn + 1];
        double w = outputs[i * outputColumn + 2];
        double h = outputs[i * outputColumn + 3];

        double left = (x - w / 2);
        double top = (y - h / 2);
        double right = (x + w / 2);
        double bottom = (y + h / 2);

        double max = outputs[i * outputColumn + 5];
        int cls = 0;
        for (int j = 0; j < outputColumn - 5; j++) {
          if (outputs[i * outputColumn + 5 + j] > max) {
            max = outputs[i * outputColumn + 5 + j];
            cls = j;
          }
        }

        PyTorchRect rect = PyTorchRect(
          left: left / imageWidth,
          top: top / imageHeight,
          width: w / imageWidth,
          height: h / imageHeight,
          bottom: bottom / imageHeight,
          right: right / imageWidth,
        );
        ResultObjectDetection result = ResultObjectDetection(
          classIndex: cls,
          score: outputs[i * outputColumn + 4],
          rect: rect,
        );

        results.add(result);
      }
    }

    print("Result length before processing ${results.length}");
    return nonMaxSuppression(results);
  }

  List<ResultObjectDetection> outputsToNMSPredictions(List<double> outputs) {
    DateTime startTime = DateTime.now(); // Record the start time

    List<ResultObjectDetection> predictions;

    if (objectDetectionModelType == ObjectDetectionModelType.yolov5) {
      predictions = outputsToNMSPredictionsYolov5(outputs);
    } else {
      predictions = outputsToNMSPredictionsYoloV8(outputs);
    }

    DateTime endTime = DateTime.now(); // Record the end time
    int executionTime = endTime
        .difference(startTime)
        .inMilliseconds; // Calculate the execution time in milliseconds

    print(
        " outputsToNMSPredictions: Execution time: $executionTime milliseconds");

    return predictions;
  }
}
