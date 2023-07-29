// This is a basic Flutter integration test.
//
// Since integration tests run in a full Flutter application, they can interact
// with the host side of a plugin implementation, unlike Dart unit tests.
//
// For more information about Flutter integration tests, please see
// https://docs.flutter.dev/cookbook/testing/integration/introduction

import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:pytorch_lite/image_utils_isolate.dart';
import 'package:pytorch_lite/native_wrapper.dart';

import 'package:pytorch_lite/pytorch_lite.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

// Defining global paths
const String pathClassificationModel = "assets/models/model_classification.pt";
const String pathObjectDetectionModel = "assets/models/yolov5s.torchscript";
const String pathObjectDetectionModelYolov8 = "assets/models/yolov8s.torchscript";
const String labelPathClassification = "assets/labels/label_classification_imageNet.txt";
const String labelPathObjectDetection = "assets/labels/labels_objectDetection_Coco.txt";
const String pathToSmallTestImage = "assets/kitten.jpeg";
const String pathToLargeTestImage = "assets/8k-test.jpg";

// Global model variables
late ClassificationModel imageModel;
late ModelObjectDetection objectModel;
late ModelObjectDetection objectModelYolov8;

Future<Uint8List> _getImageBytes(String path) async {
  final byteData = await rootBundle.load(path);
  return byteData.buffer.asUint8List();
}
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  PytorchFfi.init();
  ImageUtilsIsolate.init();

  group('Loading Models', () {
    testWidgets('Load Classification Model', (WidgetTester tester) async {
      int startTime = DateTime.now().millisecondsSinceEpoch;
      imageModel = await PytorchLite.loadClassificationModel(
        pathClassificationModel, 224, 224, 1000,
        labelPath: labelPathClassification);
      int endTime = DateTime.now().millisecondsSinceEpoch;
      print("Time taken for 'Load Classification Model': ${endTime - startTime} ms");
      expect(imageModel, isNotNull);
    });

    testWidgets('Load Object Detection Model', (WidgetTester tester) async {
      int startTime = DateTime.now().millisecondsSinceEpoch;
      objectModel = await PytorchLite.loadObjectDetectionModel(
        pathObjectDetectionModel, 80, 640, 640,
        labelPath: labelPathObjectDetection);
      int endTime = DateTime.now().millisecondsSinceEpoch;
      print("Time taken for 'Load Object Detection Model': ${endTime - startTime} ms");
      expect(objectModel, isNotNull);
    });

    testWidgets('Load Object Detection Model YOLOv8', (WidgetTester tester) async {
      int startTime = DateTime.now().millisecondsSinceEpoch;
      objectModelYolov8 = await PytorchLite.loadObjectDetectionModel(
        pathObjectDetectionModelYolov8, 80, 640, 640,
        labelPath: labelPathObjectDetection,
        objectDetectionModelType: ObjectDetectionModelType.yolov8);
      int endTime = DateTime.now().millisecondsSinceEpoch;
      print("Time taken for 'Load Object Detection Model YOLOv8': ${endTime - startTime} ms");
      expect(objectModelYolov8, isNotNull);
    });
  });

  group('Testing Model Runs', () {
    testWidgets('Run Classification Model on Small Image', (WidgetTester tester) async {
      int startTime = DateTime.now().millisecondsSinceEpoch;
      Uint8List smallImageData = await _getImageBytes(pathToSmallTestImage);
      var imagePrediction = await imageModel.getImagePrediction(smallImageData);
      int endTime = DateTime.now().millisecondsSinceEpoch;
      print("Time taken for 'Run Classification Model on Small Image': ${endTime - startTime} ms");
      expect(imagePrediction, isNotNull);
    });

    testWidgets('Run Classification Model on Large Image', (WidgetTester tester) async {
      int startTime = DateTime.now().millisecondsSinceEpoch;
      Uint8List largeImageData = await _getImageBytes(pathToLargeTestImage);
      var imagePrediction = await imageModel.getImagePrediction(largeImageData);
      int endTime = DateTime.now().millisecondsSinceEpoch;
      print("Time taken for 'Run Classification Model on Large Image': ${endTime - startTime} ms");
      expect(imagePrediction, isNotNull);
    });

    testWidgets('Run Object Detection Model on Small Image', (WidgetTester tester) async {
      int startTime = DateTime.now().millisecondsSinceEpoch;
      Uint8List smallImageData = await _getImageBytes(pathToSmallTestImage);
      var objDetect = await objectModel.getImagePrediction(smallImageData, minimumScore: 0.1, iOUThreshold: 0.3);
      int endTime = DateTime.now().millisecondsSinceEpoch;
      print("Time taken for 'Run Object Detection Model on Small Image': ${endTime - startTime} ms");
      expect(objDetect, isNotNull);
    });

    testWidgets('Run Object Detection Model on Large Image', (WidgetTester tester) async {
      int startTime = DateTime.now().millisecondsSinceEpoch;
      Uint8List largeImageData = await _getImageBytes(pathToLargeTestImage);
      var objDetect = await objectModel.getImagePrediction(largeImageData, minimumScore: 0.1, iOUThreshold: 0.3);
      int endTime = DateTime.now().millisecondsSinceEpoch;
      print("Time taken for 'Run Object Detection Model on Large Image': ${endTime - startTime} ms");
      expect(objDetect, isNotNull);
    });

    testWidgets('Run Object Detection Model YOLOv8 on Small Image', (WidgetTester tester) async {
      int startTime = DateTime.now().millisecondsSinceEpoch;
      Uint8List smallImageData = await _getImageBytes(pathToSmallTestImage);
      var objDetect = await objectModelYolov8.getImagePrediction(smallImageData, minimumScore: 0.1, iOUThreshold: 0.3);
      int endTime = DateTime.now().millisecondsSinceEpoch;
      print("Time taken for 'Run Object Detection Model YOLOv8 on Small Image': ${endTime - startTime} ms");
      expect(objDetect, isNotNull);
    });

    testWidgets('Run Object Detection Model YOLOv8 on Large Image', (WidgetTester tester) async {
      int startTime = DateTime.now().millisecondsSinceEpoch;
      Uint8List largeImageData = await _getImageBytes(pathToLargeTestImage);
      var objDetect = await objectModelYolov8.getImagePrediction(largeImageData, minimumScore: 0.1, iOUThreshold: 0.3);
      int endTime = DateTime.now().millisecondsSinceEpoch;
      print("Time taken for 'Run Object Detection Model YOLOv8 on Large Image': ${endTime - startTime} ms");
      expect(objDetect, isNotNull);
    });
  });
}
