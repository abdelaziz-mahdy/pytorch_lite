import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:pytorch_lite/enums/model_type.dart';
import 'package:pytorch_lite/image_utils_isolate.dart';
import 'package:pytorch_lite/pigeon.dart';
import 'package:collection/collection.dart';

export 'enums/dtype.dart';
export 'package:pytorch_lite/pigeon.dart';
export 'extensions/to_map_json.dart';

const List<double> torchVisionNormMeanRGB = [0.485, 0.456, 0.406];
const List<double> torchVisionNormSTDRGB = [0.229, 0.224, 0.225];
const List<double> noMeanRGB = [0, 0, 0];
const List<double> noSTDRGB = [1, 1, 1];

enum ObjectDetectionModelType { yolov5, yolov8 }

class PytorchLite {
  /*
  ///Sets pytorch model path and returns Model
  static Future<CustomModel> loadCustomModel(String path) async {
    String absPathModelPath = await _getAbsolutePath(path);
    int index = await ModelApi().loadModel(absPathModelPath, null, 0, 0);
    return CustomModel(index);
  }
   */

  ///Sets pytorch model path and returns Model
  static Future<ClassificationModel> loadClassificationModel(
      String path, int imageWidth, int imageHeight, int numberOfClasses,
      {String? labelPath, bool ensureMatchingNumberOfClasses = true}) async {
    String absPathModelPath = await _getAbsolutePath(path);
    int index = await ModelApi()
        .loadModel(absPathModelPath, null, imageWidth, imageHeight, null);
    List<String> labels = [];
    if (labelPath != null) {
      if (labelPath.endsWith(".txt")) {
        labels = await _getLabelsTxt(labelPath);
      } else {
        labels = await _getLabelsCsv(labelPath);
      }
      if (ensureMatchingNumberOfClasses) {
        if (labels.length != numberOfClasses) {
          throw Exception(
              "Number of labels does not match number of classes ,labels ${labels.length} classes $numberOfClasses");
        }
      }
    }

    return ClassificationModel(index, labels, imageWidth, imageHeight);
  }

  ///Sets pytorch object detection model (path and lables) and returns Model
  static Future<ModelObjectDetection> loadObjectDetectionModel(
      String path, int numberOfClasses, int imageWidth, int imageHeight,
      {String? labelPath,
      ObjectDetectionModelType objectDetectionModelType =
          ObjectDetectionModelType.yolov5}) async {
    String absPathModelPath = await _getAbsolutePath(path);

    int index = await ModelApi().loadModel(absPathModelPath, numberOfClasses,
        imageWidth, imageHeight, objectDetectionModelType.index);
    List<String> labels = [];
    if (labelPath != null) {
      if (labelPath.endsWith(".txt")) {
        labels = await _getLabelsTxt(labelPath);
      } else {
        labels = await _getLabelsCsv(labelPath);
      }
    }
    return ModelObjectDetection(index, imageWidth, imageHeight, labels,
        modelType: objectDetectionModelType);
  }

  static Future<String> _getAbsolutePath(String path) async {
    Directory dir = await getApplicationDocumentsDirectory();
    String dirPath = join(dir.path, path);
    ByteData data = await rootBundle.load(path);
    //copy asset to documents directory
    List<int> bytes =
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);

    //create non existant directories
    List split = path.split("/");
    String nextDir = "";
    for (int i = 0; i < split.length; i++) {
      if (i != split.length - 1) {
        nextDir += split[i];
        await Directory(join(dir.path, nextDir)).create();
        nextDir += "/";
      }
    }
    await File(dirPath).writeAsBytes(bytes);

    return dirPath;
  }
}

///get labels in csv format
///labels are separated by commas
Future<List<String>> _getLabelsCsv(String labelPath) async {
  String labelsData = await rootBundle.loadString(labelPath);
  return labelsData.split(",");
}

///get labels in txt format
///each line is a label
Future<List<String>> _getLabelsTxt(String labelPath) async {
  String labelsData = await rootBundle.loadString(labelPath);
  return labelsData.split("\n");
}

/*
class CustomModel {
  final int _index;

  CustomModel(this._index);

  ///predicts abstract number input
  Future<List?> getPrediction(
      List<double> input, List<int> shape, DType dtype) async {
    final List? prediction = await ModelApi().getPredictionCustom(
        _index, input, shape, dtype.toString().split(".").last);
    return prediction;
  }
}
*/
class ClassificationModel {
  final int _index;
  final List<String> labels;
  final int imageWidth;
  final int imageHeight;

  ClassificationModel(
      this._index, this.labels, this.imageWidth, this.imageHeight);

  /// Returns the index of the maximum value in the prediction list using the softmax function.
  ///
  /// The softmax function takes a list of double values and returns a probability distribution
  /// over the elements by exponentiating each value and normalizing it by the sum of all
  /// exponentiated values.
  ///
  /// The `prediction` parameter is a list of double values representing the predicted scores
  /// for each class.
  ///
  /// Returns the index of the maximum value in the prediction list.
  int softMax(List<double?> prediction) {
    double maxScore = double.negativeInfinity;
    int maxScoreIndex = -1;

    for (int i = 0; i < prediction.length; i++) {
      if (prediction[i]! > maxScore) {
        maxScore = prediction[i]!;
        maxScoreIndex = i;
      }
    }

    return maxScoreIndex;
  }

  /// Returns the probabilities of each element in the prediction list using the softmax function.
  ///
  /// The softmax function takes a list of double values and returns a probability distribution
  /// over the elements by exponentiating each value and normalizing it by the sum of all
  /// exponentiated values.
  ///
  /// The `prediction` parameter is a list of double values representing the predicted scores
  /// for each class.
  ///
  /// Returns a list of double values representing the probabilities of each element in the
  /// prediction list.
  List<double> getProbabilities(List<double> prediction) {
    List<double> predictionProbabilities = [];
    double? sumExp;

    // Getting sum of exponentiated values
    for (var element in prediction) {
      if (sumExp == null) {
        sumExp = exp(element);
      } else {
        sumExp = sumExp + exp(element);
      }
    }

    // Calculating probabilities
    for (var element in prediction) {
      predictionProbabilities.add(exp(element) / sumExp!);
    }

    return predictionProbabilities;
  }

  /// Returns the predicted image label using the given [imageAsBytes].
  ///
  /// The [mean] and [std] parameters are optional and default to the values of [torchVisionNormMeanRGB] and [torchVisionNormSTDRGB].
  /// The [preProcessingMethod] parameter is optional and defaults to [PreProcessingMethod.imageLib].
  /// Returns a [Future] that completes with a [String] representing the predicted image label.
  Future<String> getImagePrediction(
    Uint8List imageAsBytes, {
    List<double> mean = torchVisionNormMeanRGB,
    List<double> std = torchVisionNormSTDRGB,
    PreProcessingMethod preProcessingMethod = PreProcessingMethod.imageLib,
    bool isTupleOutput = false,
    int tupleIndex = 0,
  }) async {
    // Assert mean std
    assert(mean.length == 3, "mean should have size of 3");
    assert(std.length == 3, "std should have size of 3");

    final List<double> prediction = await getImagePredictionList(
      imageAsBytes,
      mean: mean,
      std: std,
      preProcessingMethod: preProcessingMethod,
      isTupleOutput: isTupleOutput,
      tupleIndex: tupleIndex,
    );

    int maxScoreIndex = softMax(prediction);
    return labels[maxScoreIndex];
  }

  /// Returns the predicted image as a list of scores using the given [imageAsBytes].
  ///
  /// The [mean] and [std] parameters are optional and default to the values of [torchVisionNormMeanRGB] and [torchVisionNormSTDRGB].
  /// The [preProcessingMethod] parameter is optional and defaults to [PreProcessingMethod.imageLib].
  /// Returns a [Future] that completes with a [List<double>] representing the predicted scores.
  Future<List<double>> getImagePredictionList(
    Uint8List imageAsBytes, {
    List<double> mean = torchVisionNormMeanRGB,
    List<double> std = torchVisionNormSTDRGB,
    PreProcessingMethod preProcessingMethod = PreProcessingMethod.imageLib,
    bool isTupleOutput = false,
    int tupleIndex = 0,
  }) async {
    // Assert mean std
    assert(mean.length == 3, "Mean should have size of 3");
    assert(std.length == 3, "STD should have size of 3");

    if (preProcessingMethod == PreProcessingMethod.imageLib) {
      Float64List data = await ImageUtilsIsolate.convertImageBytesToFloatBuffer(
          imageAsBytes, imageWidth, imageHeight, mean, std);
      return (await ModelApi().getRawImagePredictionList(
        _index,
        data,
        isTupleOutput,
        tupleIndex,
      ))
          .whereNotNull()
          .toList();
    }
    return (await ModelApi().getImagePredictionList(
      _index,
      imageAsBytes,
      null,
      null,
      null,
      mean,
      std,
      isTupleOutput,
      tupleIndex,
    ))
        .whereNotNull()
        .toList();
  }

  /// Returns the predicted image probabilities using the given [imageAsBytes].
  ///
  /// The [mean] and [std] parameters are optional and default to the values of [torchVisionNormMeanRGB] and [torchVisionNormSTDRGB].
  /// The [preProcessingMethod] parameter is optional and defaults to [PreProcessingMethod.imageLib].
  /// Returns a [Future] that completes with a [List<double>] representing the predicted probabilities.
  Future<List<double>> getImagePredictionListProbabilities(
    Uint8List imageAsBytes, {
    List<double> mean = torchVisionNormMeanRGB,
    List<double> std = torchVisionNormSTDRGB,
    PreProcessingMethod preProcessingMethod = PreProcessingMethod.imageLib,
    bool isTupleOutput = false,
    int tupleIndex = 0,
  }) async {
    List<double> prediction = await getImagePredictionList(
      imageAsBytes,
      mean: mean,
      std: std,
      preProcessingMethod: preProcessingMethod,
      isTupleOutput: isTupleOutput,
      tupleIndex: tupleIndex,
    );

    return getProbabilities(prediction);
  }

  /// Returns a list of predictions for an image as a bytes list.
  /// The image are passed as a list of [Uint8List] objects.
  /// The [imageWidth] and [imageHeight] parameters specify the dimensions of the image.
  /// The optional [mean] and [std] parameters can be used to normalize the image.
  /// Returns a [Future] that resolves to a list of [double] values representing the predictions.
  Future<List<double>> getImagePredictionListFromBytesList(
    List<Uint8List> imageAsBytesList,
    int imageWidth,
    int imageHeight, {
    List<double> mean = torchVisionNormMeanRGB,
    List<double> std = torchVisionNormSTDRGB,
    bool isTupleOutput = false,
    int tupleIndex = 0,
  }) async {
    // Assert mean std
    assert(mean.length == 3, "Mean should have size of 3");
    assert(std.length == 3, "STD should have size of 3");

    // Call the getImagePredictionList method of the ModelApi class to get the predictions
    final List<double> prediction = (await ModelApi().getImagePredictionList(
      _index,
      null,
      imageAsBytesList,
      imageWidth,
      imageHeight,
      mean,
      std,
      isTupleOutput,
      tupleIndex,
    ))
        .whereNotNull()
        .toList();

    return prediction;
  }

  /// Returns the predicted label for an image as a bytes list..
  /// The image are passed as a list of [Uint8List] objects.
  /// The [imageWidth] and [imageHeight] parameters specify the dimensions of the image.
  /// The optional [mean] and [std] parameters can be used to normalize the image.
  /// Returns a [Future] that resolves to a [String] representing the predicted label.
  Future<String> getImagePredictionFromBytesList(
    List<Uint8List> imageAsBytesList,
    int imageWidth,
    int imageHeight, {
    List<double> mean = torchVisionNormMeanRGB,
    List<double> std = torchVisionNormSTDRGB,
    bool isTupleOutput = false,
    int tupleIndex = 0,
  }) async {
    // Get the predictions using the getImagePredictionListFromBytesList method
    final List<double> prediction = await getImagePredictionListFromBytesList(
      imageAsBytesList,
      imageWidth,
      imageHeight,
      mean: mean,
      std: std,
      isTupleOutput: isTupleOutput,
      tupleIndex: tupleIndex,
    );

    // Find the index of the prediction with the maximum score
    int maxScoreIndex = softMax(prediction);

    // Return the label corresponding to the maximum score
    return labels[maxScoreIndex];
  }

  /// Returns a list of probabilities for an image as a bytes list.
  /// The image are passed as a list of [Uint8List] objects.
  /// The [imageWidth] and [imageHeight] parameters specify the dimensions of the image.
  /// The optional [mean] and [std] parameters can be used to normalize the image.
  /// Returns a [Future] that resolves to a list of [double] values representing the probabilities.
  Future<List<double>> getImagePredictionListProbabilitiesFromBytesList(
    List<Uint8List> imageAsBytesList,
    int imageWidth,
    int imageHeight, {
    List<double> mean = torchVisionNormMeanRGB,
    List<double> std = torchVisionNormSTDRGB,
    bool isTupleOutput = false,
    int tupleIndex = 0,
  }) async {
    // Get the predictions using the getImagePredictionListFromBytesList method
    final List<double> prediction = await getImagePredictionListFromBytesList(
      imageAsBytesList,
      imageWidth,
      imageHeight,
      mean: mean,
      std: std,
      isTupleOutput: isTupleOutput,
      tupleIndex: tupleIndex,
    );

    // Return the probabilities derived from the predictions
    return getProbabilities(prediction);
  }

  /// Retrieves a list of predictions for a camera image.
  ///
  /// Takes a [cameraImage] and [rotation] as input. Optional parameters include [mean], [std],
  /// [cameraPreProcessingMethod], and [preProcessingMethod].
  /// Returns a [Future] that resolves to a [List] of [double] values representing the predictions.
  /// Throws an [Exception] if unable to process the image bytes.
  Future<List<double>> getCameraImagePredictionList(
    CameraImage cameraImage,
    int rotation, {
    List<double> mean = torchVisionNormMeanRGB,
    List<double> std = torchVisionNormSTDRGB,
    CameraPreProcessingMethod cameraPreProcessingMethod =
        CameraPreProcessingMethod.imageLib,
    PreProcessingMethod preProcessingMethod = PreProcessingMethod.imageLib,
    bool isTupleOutput = false,
    int tupleIndex = 0,
  }) async {
    // Perform preprocessing based on the chosen camera pre-processing method
    if (cameraPreProcessingMethod == CameraPreProcessingMethod.imageLib) {
      Uint8List? bytes =
          await ImageUtilsIsolate.convertCameraImageToBytes(cameraImage);
      if (bytes == null) {
        throw Exception("Unable to process image bytes");
      }
      // Retrieve the image predictions for the preprocessed image bytes
      return await getImagePredictionList(
        bytes,
        mean: mean,
        std: std,
        preProcessingMethod: preProcessingMethod,
        isTupleOutput: isTupleOutput,
        tupleIndex: tupleIndex,
      );
    }
    // Retrieve the image predictions for the camera image planes
    return await getImagePredictionListFromBytesList(
      cameraImage.planes.map((e) => e.bytes).toList(),
      cameraImage.width,
      cameraImage.height,
      mean: mean,
      std: std,
      isTupleOutput: isTupleOutput,
      tupleIndex: tupleIndex,
    );
  }

  /// Retrieves the top prediction label for a camera image.
  ///
  /// Takes a [cameraImage] and [rotation] as input. Optional parameters include [mean], [std],
  /// [cameraPreProcessingMethod], and [preProcessingMethod].
  /// Returns a [Future] that resolves to a [String] representing the top prediction label.
  Future<String> getCameraImagePrediction(
    CameraImage cameraImage,
    int rotation, {
    List<double> mean = torchVisionNormMeanRGB,
    List<double> std = torchVisionNormSTDRGB,
    CameraPreProcessingMethod cameraPreProcessingMethod =
        CameraPreProcessingMethod.imageLib,
    PreProcessingMethod preProcessingMethod = PreProcessingMethod.imageLib,
    bool isTupleOutput = false,
    int tupleIndex = 0,
  }) async {
    // Retrieve the prediction list for the camera image
    final List<double> prediction = await getCameraImagePredictionList(
      cameraImage,
      rotation,
      mean: mean,
      std: std,
      cameraPreProcessingMethod: cameraPreProcessingMethod,
      preProcessingMethod: preProcessingMethod,
      isTupleOutput: isTupleOutput,
      tupleIndex: tupleIndex,
    );

    // Get the index of the maximum score from the prediction list
    int maxScoreIndex = softMax(prediction);
    // Return the label corresponding to the maximum score index
    return labels[maxScoreIndex];
  }

  /// Retrieves the probabilities of predictions for a camera image.
  ///
  /// Takes a [cameraImage] and [rotation] as input. Optional parameters include [mean], [std],
  /// [cameraPreProcessingMethod], and [preProcessingMethod].
  /// Returns a [Future] that resolves to a [List] of [double] values representing the prediction probabilities.
  Future<List<double>> getCameraImagePredictionProbabilities(
    CameraImage cameraImage,
    int rotation, {
    List<double> mean = torchVisionNormMeanRGB,
    List<double> std = torchVisionNormSTDRGB,
    CameraPreProcessingMethod cameraPreProcessingMethod =
        CameraPreProcessingMethod.imageLib,
    PreProcessingMethod preProcessingMethod = PreProcessingMethod.imageLib,
    bool isTupleOutput = false,
    int tupleIndex = 0,
  }) async {
    // Retrieve the prediction list for the camera image
    final List<double> prediction = await getCameraImagePredictionList(
      cameraImage,
      rotation,
      mean: mean,
      std: std,
      cameraPreProcessingMethod: cameraPreProcessingMethod,
      preProcessingMethod: preProcessingMethod,
      isTupleOutput: isTupleOutput,
      tupleIndex: tupleIndex,
    );

    return getProbabilities(prediction);
  }
}

class ModelObjectDetection {
  final int _index;
  final int imageWidth;
  final int imageHeight;
  final List<String> labels;
  final ObjectDetectionModelType modelType;
  ModelObjectDetection(
      this._index, this.imageWidth, this.imageHeight, this.labels,
      {this.modelType = ObjectDetectionModelType.yolov5});

  /// Adds labels to the given list of [prediction] objects.
  ///
  /// The labels are added based on the class index of each prediction object.
  /// The class name is retrieved from the [labels] list using the class index,
  /// and assigned to the [className] property of each prediction object.
  ///
  /// Parameters:
  /// - [prediction]: The list of prediction objects to add labels to.
  void addLabels(List<ResultObjectDetection> prediction) {
    for (var element in prediction) {
      // Retrieve the class name from the labels list using the class index
      element.className = labels[element.classIndex];
    }
  }

  /// Performs object detection on an image and returns a list of [ResultObjectDetection] with its assigned labels.
  ///
  /// Parameters:
  /// - [imageAsBytes]: The image as bytes in Uint8List format.
  /// - [minimumScore]: The minimum confidence score for a detected object to be included in the results. Default is 0.5.
  /// - [iOUThreshold]: The threshold for intersection over union (IOU) to filter out redundant bounding boxes. Default is 0.5.
  /// - [boxesLimit]: The maximum number of bounding boxes to return. Default is 10.
  /// - [preProcessingMethod]: The preprocessing method to apply to the image before object detection. Default is [PreProcessingMethod.imageLib].
  ///
  /// Returns:
  /// A list of [ResultObjectDetection] containing the detected objects and their bounding boxes.
  Future<List<ResultObjectDetection>> getImagePrediction(
    Uint8List imageAsBytes, {
    double minimumScore = 0.5,
    double iOUThreshold = 0.5,
    int boxesLimit = 10,
    PreProcessingMethod preProcessingMethod = PreProcessingMethod.imageLib,
    bool isTupleOutput = false,
    int tupleIndex = 0,
  }) async {
    // Perform object detection on the image
    List<ResultObjectDetection> prediction = await getImagePredictionList(
        imageAsBytes,
        minimumScore: minimumScore,
        iOUThreshold: iOUThreshold,
        boxesLimit: boxesLimit,
        preProcessingMethod: preProcessingMethod);

    // Add labels to the detected objects
    addLabels(prediction);

    return prediction;
  }

  /// Performs object detection on an image as bytesList and returns a list of [ResultObjectDetection] with its assigned labels.
  ///
  /// Parameters:
  /// - [imageAsBytes]: The image as bytes in Uint8List format.
  /// - [imageWidth]: The width of the image.
  /// - [imageHeight]: The height of the image.
  /// - [minimumScore]: The minimum confidence score for a detected object to be included in the results. Default is 0.5.
  /// - [iOUThreshold]: The threshold for intersection over union (IOU) to filter out redundant bounding boxes. Default is 0.5.
  /// - [boxesLimit]: The maximum number of bounding boxes to return. Default is 10.
  /// - [preProcessingMethod]: The preprocessing method to apply to the image before object detection. Default is [PreProcessingMethod.imageLib].
  ///
  /// Returns:
  /// A list of [ResultObjectDetection] containing the detected objects and their bounding boxes.
  Future<List<ResultObjectDetection>> getImagePredictionFromBytesList(
    List<Uint8List> imageAsBytesList,
    int imageWidth,
    int imageHeight, {
    double minimumScore = 0.5,
    double iOUThreshold = 0.5,
    int boxesLimit = 10,
    bool isTupleOutput = false,
    int tupleIndex = 0,
  }) async {
    List<ResultObjectDetection> prediction =
        await getImagePredictionListFromBytesList(
      imageAsBytesList,
      imageWidth,
      imageHeight,
      minimumScore: minimumScore,
      iOUThreshold: iOUThreshold,
      boxesLimit: boxesLimit,
      isTupleOutput: isTupleOutput,
      tupleIndex: tupleIndex,
    );
    addLabels(prediction);

    return prediction;
  }

  /// Returns a list of [ResultObjectDetection] for the given [imageAsBytes].
  ///
  /// The [minimumScore], [iOUThreshold], and [boxesLimit] parameters control the
  /// prediction quality. The [preProcessingMethod] parameter determines the
  /// method used for preprocessing the image.
  ///
  /// If [preProcessingMethod] is [PreProcessingMethod.imageLib], the image bytes
  /// are converted to a float buffer using [ImageUtilsIsolate.convertImageBytesToFloatBuffer]
  /// before making the prediction. Otherwise, the prediction is made directly
  /// using the image bytes.
  ///
  /// Returns a list of [ResultObjectDetection] where each result has a score
  /// greater than or equal to [minimumScore].
  Future<List<ResultObjectDetection>> getImagePredictionList(
    Uint8List imageAsBytes, {
    double minimumScore = 0.5,
    double iOUThreshold = 0.5,
    int boxesLimit = 10,
    PreProcessingMethod preProcessingMethod = PreProcessingMethod.imageLib,
    bool isTupleOutput = false,
    int tupleIndex = 0,
  }) async {
    if (preProcessingMethod == PreProcessingMethod.imageLib) {
      Uint8List data =
          await ImageUtilsIsolate.convertImageBytesToFloatBufferUInt8List(
              imageAsBytes, imageWidth, imageHeight, noMeanRGB, noSTDRGB);
      return (await ModelApi().getRawImagePredictionListObjectDetection(
        _index,
        data,
        minimumScore,
        iOUThreshold,
        boxesLimit,
        isTupleOutput,
        tupleIndex,
      ))
          .whereNotNull()
          .toList();
    }
    return (await ModelApi().getImagePredictionListObjectDetection(
      _index,
      imageAsBytes,
      null,
      null,
      null,
      minimumScore,
      iOUThreshold,
      boxesLimit,
      isTupleOutput,
      tupleIndex,
    ))
        .whereNotNull()
        .toList();
  }

  /// Performs object detection on an image as bytesList and returns a list of [ResultObjectDetection].
  ///
  /// Parameters:
  /// - [imageAsBytes]: The image as bytes in Uint8List format.
  /// - [imageWidth]: The width of the image.
  /// - [imageHeight]: The height of the image.
  /// - [minimumScore]: The minimum confidence score for a detected object to be included in the results. Default is 0.5.
  /// - [iOUThreshold]: The threshold for intersection over union (IOU) to filter out redundant bounding boxes. Default is 0.5.
  /// - [boxesLimit]: The maximum number of bounding boxes to return. Default is 10.
  /// - [preProcessingMethod]: The preprocessing method to apply to the image before object detection. Default is [PreProcessingMethod.imageLib].
  ///
  /// Returns:
  /// A list of [ResultObjectDetection] containing the detected objects and their bounding boxes.
  Future<List<ResultObjectDetection>> getImagePredictionListFromBytesList(
    List<Uint8List> imageAsBytesList,
    int imageWidth,
    int imageHeight, {
    double minimumScore = 0.5,
    double iOUThreshold = 0.5,
    int boxesLimit = 10,
    bool isTupleOutput = false,
    int tupleIndex = 0,
  }) async {
    final List<ResultObjectDetection> prediction =
        (await ModelApi().getImagePredictionListObjectDetection(
      _index,
      null,
      imageAsBytesList,
      imageWidth,
      imageHeight,
      minimumScore,
      iOUThreshold,
      boxesLimit,
      isTupleOutput,
      tupleIndex,
    ))
            .whereNotNull()
            .toList();

    return prediction;
  }

  /// Retrieves a list of [ResultObjectDetection] by predicting the objects in the given [cameraImage].
  /// The [rotation] parameter specifies the rotation of the camera image.
  /// The optional parameters [minimumScore], [iOUThreshold], [boxesLimit], [cameraPreProcessingMethod], and [preProcessingMethod]
  /// allow customization of the prediction process.
  Future<List<ResultObjectDetection>> getCameraImagePredictionList(
      CameraImage cameraImage, int rotation,
      {double minimumScore = 0.5,
      double iOUThreshold = 0.5,
      int boxesLimit = 10,
      CameraPreProcessingMethod cameraPreProcessingMethod =
          CameraPreProcessingMethod.imageLib,
      PreProcessingMethod preProcessingMethod =
          PreProcessingMethod.imageLib}) async {
    if (cameraPreProcessingMethod == CameraPreProcessingMethod.imageLib) {
      // Convert the camera image to bytes using ImageUtilsIsolate
      Uint8List? bytes =
          await ImageUtilsIsolate.convertCameraImageToBytes(cameraImage);
      if (bytes == null) {
        throw Exception("Unable to process image bytes");
      }
      // Get the image prediction list using the converted bytes
      return await getImagePredictionList(bytes,
          minimumScore: minimumScore,
          iOUThreshold: iOUThreshold,
          boxesLimit: boxesLimit,
          preProcessingMethod: preProcessingMethod);
    }
    // Get the image prediction list directly from the camera image planes
    return await getImagePredictionFromBytesList(
        cameraImage.planes.map((e) => e.bytes).toList(),
        cameraImage.width,
        cameraImage.height,
        minimumScore: minimumScore,
        iOUThreshold: iOUThreshold,
        boxesLimit: boxesLimit);
  }

  /// Retrieves a list of [ResultObjectDetection] with its assigned labels by predicting the objects in the given [cameraImage].
  /// The [rotation] parameter specifies the rotation of the camera image.
  /// The optional parameters [minimumScore], [iOUThreshold], [boxesLimit], [cameraPreProcessingMethod], and [preProcessingMethod]
  /// allow customization of the prediction process.
  Future<List<ResultObjectDetection>> getCameraImagePrediction(
      CameraImage cameraImage, int rotation,
      {double minimumScore = 0.5,
      double iOUThreshold = 0.5,
      int boxesLimit = 10,
      CameraPreProcessingMethod cameraPreProcessingMethod =
          CameraPreProcessingMethod.imageLib,
      PreProcessingMethod preProcessingMethod =
          PreProcessingMethod.imageLib}) async {
    final List<ResultObjectDetection> prediction =
        await getCameraImagePredictionList(cameraImage, rotation,
            minimumScore: minimumScore,
            iOUThreshold: iOUThreshold,
            boxesLimit: boxesLimit,
            cameraPreProcessingMethod: cameraPreProcessingMethod,
            preProcessingMethod: preProcessingMethod);
    addLabels(prediction);
    return prediction;
  }

  /// Renders a list of boxes on an image.
  ///
  /// The [image] parameter is the file representing the image.
  /// The [recognitions] parameter is a list of ResultObjectDetection objects containing information about the boxes to be rendered.
  /// The [boxesColor] parameter is the color of the boxes. If null, the color will be chosen based on the label.
  /// The [showPercentage] parameter determines whether to show the percentage value in the label.
  ///
  /// Returns a Widget that renders the image with the boxes.
  Widget renderBoxesOnImage(
      File image, List<ResultObjectDetection?> recognitions,
      {Color? boxesColor, bool showPercentage = true}) {
    return LayoutBuilder(builder: (context, constraints) {
      debugPrint(
          'Max height: ${constraints.maxHeight}, max width: ${constraints.maxWidth}');

      // Calculate the scaling factors for the boxes based on the layout constraints
      double factorX = constraints.maxWidth;
      double factorY = constraints.maxHeight;

      return Stack(
        children: [
          Positioned(
            left: 0,
            top: 0,
            width: factorX,
            height: factorY,
            child: Image.file(
              image,
              fit: BoxFit.fill,
            ),
          ),
          ...recognitions.map((re) {
            if (re == null) {
              return Container();
            }
            Color usedColor;
            if (boxesColor == null) {
              //change colors for each label
              usedColor = Colors.primaries[
                  ((re.className ?? re.classIndex.toString()).length +
                          (re.className ?? re.classIndex.toString())
                              .codeUnitAt(0) +
                          re.classIndex) %
                      Colors.primaries.length];
            } else {
              usedColor = boxesColor;
            }

            return Positioned(
              left: re.rect.left * factorX,
              top: re.rect.top * factorY - 20,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                mainAxisAlignment: MainAxisAlignment.start,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    height: 20,
                    alignment: Alignment.centerRight,
                    color: usedColor,
                    child: Text(
                      "${re.className ?? re.classIndex.toString()}_${showPercentage ? "${(re.score * 100).toStringAsFixed(2)}%" : ""}",
                    ),
                  ),
                  Container(
                    width: re.rect.width.toDouble() * factorX,
                    height: re.rect.height.toDouble() * factorY,
                    decoration: BoxDecoration(
                        border: Border.all(color: usedColor, width: 3),
                        borderRadius:
                            const BorderRadius.all(Radius.circular(2))),
                    child: Container(),
                  ),
                ],
              ),
            );
          }).toList()
        ],
      );
    });
  }
}
