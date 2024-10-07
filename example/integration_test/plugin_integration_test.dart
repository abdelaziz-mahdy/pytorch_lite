// This is a basic Flutter integration test.
//
// Since integration tests run in a full Flutter application, they can interact
// with the host side of a plugin implementation, unlike Dart unit tests.
//
// For more information about Flutter integration tests, please see
// https://docs.flutter.dev/cookbook/testing/integration/introduction

import 'dart:convert';
import 'dart:developer';
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'package:pytorch_lite/pytorch_lite.dart';

import 'data.dart';

import 'utils.dart';

// Defining global paths
const String pathClassificationModel = "assets/models/model_classification.pt";
const String pathObjectDetectionModel = "assets/models/yolov5s.torchscript";
const String pathObjectDetectionModelYolov8 =
    "assets/models/yolov8s.torchscript";
const String labelPathClassification =
    "assets/labels/label_classification_imageNet.txt";
const String labelPathObjectDetection =
    "assets/labels/labels_objectDetection_Coco.txt";
const String pathToSmallTestImage = "assets/kitten.jpeg";
const String pathToLargeTestImage = "assets/8k-test.jpg";

const String resultsFilePath =
    "path_to_your_results_file.json"; // Define the path to your results JSON file

Map<String, dynamic> testResults =
    {}; // This map will store all your test results

// Global model variables
late ClassificationModel imageModel;
late ModelObjectDetection objectModel;
late ModelObjectDetection objectModelYolov8;

Future<Uint8List> _getImageBytes(String path) async {
  final byteData = await rootBundle.load(path);
  return byteData.buffer.asUint8List();
}

void loadResults() {
  testResults = data;
}

void printWrapped(String text) =>
    RegExp('.{1,800}').allMatches(text).map((m) => m.group(0)).forEach(print);

void saveResults() {
  final File file = File(resultsFilePath);
  String data = json.encode(
    testResults,
    toEncodable: (object) {
      print(object.runtimeType);
      if (object is ResultObjectDetection) {
        print("HERE I AM ");
        print("map ${object.toMap()}");
        print("json ${object.toJson()}");
        return object.toMap();
      } else {
        print("HERE I AM 2");
      }

      return object.toMap();
    },
  );
  printWrapped("file data is $data");
  log("file data is $data");

  file.writeAsStringSync(data);
}

Future<void> runTestWithWrapper({
  required WidgetTester tester,
  required String testName,
  required Future<dynamic> Function() testBody,
}) async {
  final startTime = DateTime.now().millisecondsSinceEpoch;

  // Execute the test body and get the results
  var result = await testBody();

  final endTime = DateTime.now().millisecondsSinceEpoch;
  final elapsedTime = endTime - startTime;

  print("Time taken for '$testName': $elapsedTime ms");
  if (testResults.containsKey(testName)) {
    final previousResult = testResults[testName];
    if (result is List<ResultObjectDetection>) {
      result = result.map((e) => e.toMap()).toList();
      expect(listEquals(result, previousResult), true,
          reason: listDifferences(result, previousResult).join("\n"));
    } else {
      if (result is List) {
        expect(listEquals(result, previousResult), true,
            reason: listDifferences(result, previousResult).join("\n"));
      } else {
        expect(result, previousResult);
      }
    }
  } else {
    print("Warning: Results for '$testName' not found.");
    printWrapped("please add to data: ${json.encode({testName: result})}");
    // This test was not previously run, so store its result
    testResults[testName] = result;
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  setUpAll(() {
    loadResults();
  });
  // tearDownAll(() {
  //   saveResults();
  // });
  group('Loading Models', () {
    testWidgets('Load Classification Model', (WidgetTester tester) async {
      int startTime = DateTime.now().millisecondsSinceEpoch;
      imageModel = await PytorchLite.loadClassificationModel(
          pathClassificationModel, 224, 224, 1000,
          labelPath: labelPathClassification);
      int endTime = DateTime.now().millisecondsSinceEpoch;
      print(
          "Time taken for 'Load Classification Model': ${endTime - startTime} ms");
      expect(imageModel, isNotNull);
    });

    testWidgets('Load Object Detection Model', (WidgetTester tester) async {
      int startTime = DateTime.now().millisecondsSinceEpoch;
      objectModel = await PytorchLite.loadObjectDetectionModel(
          pathObjectDetectionModel, 80, 640, 640,
          labelPath: labelPathObjectDetection);
      int endTime = DateTime.now().millisecondsSinceEpoch;
      print(
          "Time taken for 'Load Object Detection Model': ${endTime - startTime} ms");
      expect(objectModel, isNotNull);
    });

    testWidgets('Load Object Detection Model YOLOv8',
        (WidgetTester tester) async {
      int startTime = DateTime.now().millisecondsSinceEpoch;
      objectModelYolov8 = await PytorchLite.loadObjectDetectionModel(
          pathObjectDetectionModelYolov8, 80, 640, 640,
          labelPath: labelPathObjectDetection,
          objectDetectionModelType: ObjectDetectionModelType.yolov8);
      int endTime = DateTime.now().millisecondsSinceEpoch;
      print(
          "Time taken for 'Load Object Detection Model YOLOv8': ${endTime - startTime} ms");
      expect(objectModelYolov8, isNotNull);
    });
  });

  group('Testing Model Runs', () {
    testWidgets('Run Classification Model on Small Image (raw list)',
        (WidgetTester tester) async {
      await runTestWithWrapper(
          tester: tester,
          testName: 'Run Classification Model on Small Image (raw list)',
          testBody: () async {
            Uint8List smallImageData =
                await _getImageBytes(pathToSmallTestImage);
            return await imageModel.getImagePredictionList(smallImageData);
          });
    });
    testWidgets('Run Classification Model on Small Image',
        (WidgetTester tester) async {
      await runTestWithWrapper(
          tester: tester,
          testName: 'Run Classification Model on Small Image',
          testBody: () async {
            Uint8List smallImageData =
                await _getImageBytes(pathToSmallTestImage);
            return await imageModel.getImagePrediction(smallImageData);
          });
    });

    testWidgets('Run Classification Model on Large Image',
        (WidgetTester tester) async {
      await runTestWithWrapper(
          tester: tester,
          testName: 'Run Classification Model on Large Image',
          testBody: () async {
            Uint8List largeImageData =
                await _getImageBytes(pathToLargeTestImage);
            return await imageModel.getImagePrediction(largeImageData);
          });
    });

    testWidgets('Run Object Detection Model on Small Image',
        (WidgetTester tester) async {
      await runTestWithWrapper(
          tester: tester,
          testName: 'Run Object Detection Model on Small Image',
          testBody: () async {
            Uint8List smallImageData =
                await _getImageBytes(pathToSmallTestImage);
            return await objectModel.getImagePrediction(smallImageData,
                minimumScore: 0.1, iOUThreshold: 0.3);
          });
    });

    testWidgets('Run Object Detection Model on Large Image',
        (WidgetTester tester) async {
      await runTestWithWrapper(
          tester: tester,
          testName: 'Run Object Detection Model on Large Image',
          testBody: () async {
            Uint8List largeImageData =
                await _getImageBytes(pathToLargeTestImage);
            return await objectModel.getImagePrediction(largeImageData,
                minimumScore: 0.1, iOUThreshold: 0.3);
          });
    });

    testWidgets('Run Object Detection Model YOLOv8 on Small Image',
        (WidgetTester tester) async {
      await runTestWithWrapper(
          tester: tester,
          testName: 'Run Object Detection Model YOLOv8 on Small Image',
          testBody: () async {
            Uint8List smallImageData =
                await _getImageBytes(pathToSmallTestImage);
            return await objectModelYolov8.getImagePrediction(smallImageData,
                minimumScore: 0.1, iOUThreshold: 0.3);
          });
    });

    testWidgets('Run Object Detection Model YOLOv8 on Large Image',
        (WidgetTester tester) async {
      await runTestWithWrapper(
          tester: tester,
          testName: 'Run Object Detection Model YOLOv8 on Large Image',
          testBody: () async {
            Uint8List largeImageData =
                await _getImageBytes(pathToLargeTestImage);
            return await objectModelYolov8.getImagePrediction(largeImageData,
                minimumScore: 0.1, iOUThreshold: 0.3);
          });
    });
  });
}

