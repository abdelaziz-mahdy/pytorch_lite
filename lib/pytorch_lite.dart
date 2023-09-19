import 'dart:async';
import 'dart:io';
import 'dart:math';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:pytorch_lite/classes/result_object_detection.dart';
import 'package:pytorch_lite/enums/model_type.dart';
import 'package:pytorch_lite/native_wrapper.dart';
import 'package:pytorch_lite/post_processor.dart';

export 'enums/dtype.dart';
export 'package:pytorch_lite/pigeon.dart';

const torchVisionNormMeanRGB = [0.485, 0.456, 0.406];
const torchVisionNormSTDRGB = [0.229, 0.224, 0.225];
 enum ObjectDetectionModelType {yolov5,yolov8  }
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
      String path, int imageWidth, int imageHeight,
      {String? labelPath}) async {
    String absPathModelPath = await _getAbsolutePath(path);
    int index = await ModelApi()
        .loadModel(absPathModelPath, null, imageWidth, imageHeight,null);
    List<String> labels = [];
    if (labelPath != null) {
      if (labelPath.endsWith(".txt")) {
        labels = await _getLabelsTxt(labelPath);
      } else {
        labels = await _getLabelsCsv(labelPath);
      }
    }

    return ClassificationModel(index, labels);
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

  ///Sets pytorch model path and returns Model
  static Future<ClassificationModel> loadClassificationModel(
      String path, int imageWidth, int imageHeight, int numberOfClasses,
      {String? labelPath}) async {
    String absPathModelPath = await _getAbsolutePath(path);

    int index = await PytorchFfi.loadModel(absPathModelPath);

    List<String> labels = [];
    if (labelPath != null) {
      if (labelPath.endsWith(".txt")) {
        labels = await _getLabelsTxt(labelPath);
      } else {
        labels = await _getLabelsCsv(labelPath);
      }
    }

    return ClassificationModel(
        index, labels, imageWidth, imageHeight, numberOfClasses);
  }

  ///Sets pytorch object detection model (path and lables) and returns Model
  static Future<ModelObjectDetection> loadObjectDetectionModel(
      String path, int numberOfClasses, int imageWidth, int imageHeight,
      {String? labelPath,
      ObjectDetectionModelType objectDetectionModelType =
          ObjectDetectionModelType.yolov5}) async {
    String absPathModelPath = await _getAbsolutePath(path);

    int index = await PytorchFfi.loadModel(absPathModelPath);
    // int index = await ModelApi().loadModel(absPathModelPath, numberOfClasses,
    //     imageWidth, imageHeight, objectDetectionModelType);
    List<String> labels = [];
    if (labelPath != null) {
      if (labelPath.endsWith(".txt")) {
        labels = await _getLabelsTxt(labelPath);
      } else {
        labels = await _getLabelsCsv(labelPath);
      }
    }
    return ModelObjectDetection(
        index,
        imageWidth,
        imageHeight,
        labels,
        PostProcessorObjectDetection(
            numberOfClasses, imageWidth, imageHeight, objectDetectionModelType),
        modelType: objectDetectionModelType);
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
  final int numberOfClasses;
  ClassificationModel(this._index, this.labels, this.imageWidth,
      this.imageHeight, this.numberOfClasses);

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

  List<double> getProbabilities(
    List<double> prediction,
  ) {
    List<double> predictionProbabilities = [];
    //Getting sum of exp
    double? sumExp;
    for (var element in prediction) {
      if (sumExp == null) {
        sumExp = exp(element);
      } else {
        sumExp = sumExp + exp(element);
      }
    }
    for (var element in prediction) {
      predictionProbabilities.add(exp(element) / sumExp!);
    }
    return predictionProbabilities;
  }

  ///predicts image but returns the raw net output
  Future<List<double>> getImagePredictionList(Uint8List imageAsBytes,
      {List<double> mean = torchVisionNormMeanRGB,
      List<double> std = torchVisionNormSTDRGB}) async {
    // Assert mean std
    assert(mean.length == 3, "Mean should have size of 3");
    assert(std.length == 3, "STD should have size of 3");

    return await PytorchFfi.imageModelInference(_index, imageAsBytes,
        imageHeight, imageWidth, mean, std, false, numberOfClasses);
  }

  ///predicts image and returns the supposed label belonging to it
  Future<String> getImagePrediction(Uint8List imageAsBytes,
      {List<double> mean = torchVisionNormMeanRGB,
      List<double> std = torchVisionNormSTDRGB}) async {
    // Assert mean std
    assert(mean.length == 3, "mean should have size of 3");
    assert(std.length == 3, "std should have size of 3");

    final List<double?> prediction =
        await getImagePredictionList(imageAsBytes, mean: mean, std: std);

    int maxScoreIndex = softMax(prediction);
    return labels[maxScoreIndex];
  }

  ///predicts image but returns the output as probabilities
  ///[image] takes the File of the image
  Future<List<double>?> getImagePredictionListProbabilities(
      Uint8List imageAsBytes,
      {List<double> mean = torchVisionNormMeanRGB,
      List<double> std = torchVisionNormSTDRGB}) async {
    // Assert mean std
    assert(mean.length == 3, "Mean should have size of 3");
    assert(std.length == 3, "STD should have size of 3");
    List<double> prediction =
        await getImagePredictionList(imageAsBytes, mean: mean, std: std);

    //Getting sum of exp
    return getProbabilities(prediction);
  }

  ///predicts image but returns the raw net output
  Future<List<double>> getCameraImagePredictionList(
      CameraImage cameraImage, int rotation,
      {List<double> mean = torchVisionNormMeanRGB,
      List<double> std = torchVisionNormSTDRGB}) async {
    // Assert mean std
    assert(mean.length == 3, "Mean should have size of 3");
    assert(std.length == 3, "STD should have size of 3");

    // On Android the image format is YUV and we get a buffer per channel,
    // in iOS the format is BGRA and we get a single buffer for all channels.
    // So the yBuffer variable on Android will be just the Y channel but on iOS it will be
    // the entire image
    var planes = cameraImage.planes;
    var yBuffer = planes[0].bytes;

    Uint8List? uBuffer;
    Uint8List? vBuffer;

    if (Platform.isAndroid) {
      uBuffer = planes[1].bytes;
      vBuffer = planes[2].bytes;
    }

    return await PytorchFfi.cameraImageModelInference(
        _index,
        yBuffer,
        uBuffer,
        vBuffer,
        rotation,
        imageHeight,
        imageWidth,
        cameraImage.height,
        cameraImage.width,
        mean,
        std,
        false,
        numberOfClasses);
  }

  ///predicts image and returns the supposed label belonging to it
  Future<String> getCameraImagePrediction(CameraImage cameraImage, int rotation,
      {List<double> mean = torchVisionNormMeanRGB,
      List<double> std = torchVisionNormSTDRGB}) async {
    // Assert mean std
    assert(mean.length == 3, "mean should have size of 3");
    assert(std.length == 3, "std should have size of 3");

    final List<double?> prediction = await getCameraImagePredictionList(
        cameraImage, rotation,
        mean: mean, std: std);

    int maxScoreIndex = softMax(prediction);
    return labels[maxScoreIndex];
  }

  ///predicts image but returns the output as probabilities
  ///[image] takes the File of the image
  Future<List<double>?> getCameraPredictionListProbabilities(
      CameraImage cameraImage, int rotation,
      {List<double> mean = torchVisionNormMeanRGB,
      List<double> std = torchVisionNormSTDRGB}) async {
    // Assert mean std
    assert(mean.length == 3, "Mean should have size of 3");
    assert(std.length == 3, "STD should have size of 3");
    final List<double> prediction = await getCameraImagePredictionList(
        cameraImage, rotation,
        mean: mean, std: std);

    //Getting sum of exp
    return getProbabilities(prediction);
  }
}

class ModelObjectDetection {
  final int _index;
  final int imageWidth;
  final int imageHeight;
  final List<String> labels;
  final ObjectDetectionModelType modelType;
  final PostProcessorObjectDetection postProcessorObjectDetection;
  ModelObjectDetection(this._index, this.imageWidth, this.imageHeight,
      this.labels, this.postProcessorObjectDetection,
      {this.modelType = ObjectDetectionModelType.yolov5});

  ///predicts image but returns the raw net output
  Future<List<ResultObjectDetection>> getImagePredictionList(
      Uint8List imageAsBytes,
      {double minimumScore = 0.5,
      double iOUThreshold = 0.5,
      int boxesLimit = 10,
      List<double> mean = noMeanRgb,
      List<double> std = noStdRgb}) async {
    List<ResultObjectDetection> prediction =
        await PytorchFfi.imageModelInferenceObjectDetection(
            _index,
            imageAsBytes,
            imageHeight,
            imageWidth,
            mean,
            std,
            modelType == ObjectDetectionModelType.yolov5,
            postProcessorObjectDetection.modelOutputLength,
            postProcessorObjectDetection);
    return prediction;
  }

  ///predicts image and returns the supposed label belonging to it
  Future<List<ResultObjectDetection>> getImagePrediction(Uint8List imageAsBytes,
      {double minimumScore = 0.5,
      double iOUThreshold = 0.5,
      int boxesLimit = 10,
      List<double> mean = noMeanRgb,
      List<double> std = noStdRgb}) async {
    List<ResultObjectDetection> prediction = await getImagePredictionList(
        imageAsBytes,
        minimumScore: minimumScore,
        iOUThreshold: iOUThreshold,
        boxesLimit: boxesLimit,
        mean: mean,
        std: std);

    for (var element in prediction) {
      element.className = labels[element.classIndex];
    }

    return prediction;
  }

  ///predicts image but returns the raw net output
  Future<List<ResultObjectDetection>> getCameraImagePredictionList(
      CameraImage cameraImage, int rotation,
      {double minimumScore = 0.5,
      double iOUThreshold = 0.5,
      int boxesLimit = 10,
      List<double> mean = noMeanRgb,
      List<double> std = noStdRgb}) async {
    // On Android the image format is YUV and we get a buffer per channel,
    // in iOS the format is BGRA and we get a single buffer for all channels.
    // So the yBuffer variable on Android will be just the Y channel but on iOS it will be
    // the entire image
    var planes = cameraImage.planes;
    var yBuffer = planes[0].bytes;

    Uint8List? uBuffer;
    Uint8List? vBuffer;

    if (Platform.isAndroid) {
      uBuffer = planes[1].bytes;
      vBuffer = planes[2].bytes;
    }

    List<ResultObjectDetection> prediction =
        await PytorchFfi.cameraImageModelInferenceObjectDetection(
            _index,
            yBuffer,
            uBuffer,
            vBuffer,
            rotation,
            imageHeight,
            imageWidth,
            cameraImage.height,
            cameraImage.width,
            mean,
            std,
            modelType == ObjectDetectionModelType.yolov5,
            postProcessorObjectDetection.modelOutputLength,
            postProcessorObjectDetection);

    return prediction;
  }

  ///predicts image and returns the supposed label belonging to it
  Future<List<ResultObjectDetection>> getCameraImagePrediction(
      CameraImage cameraImage, int rotation,
      {double minimumScore = 0.5,
      double iOUThreshold = 0.5,
      int boxesLimit = 10,
      List<double> mean = noMeanRgb,
      List<double> std = noStdRgb}) async {
    List<ResultObjectDetection> prediction = await getCameraImagePredictionList(
        cameraImage, rotation,
        minimumScore: minimumScore,
        iOUThreshold: iOUThreshold,
        boxesLimit: boxesLimit,
        mean: mean,
        std: std);

    for (var element in prediction) {
      element.className = labels[element.classIndex];
    }

    return prediction;
  }

  /*

   */
  Widget renderBoxesOnImage(
      File image, List<ResultObjectDetection?> recognitions,
      {Color? boxesColor, bool showPercentage = true}) {
    //if (_recognitions == null) return Cont;
    //if (_imageHeight == null || _imageWidth == null) return [];

    //double factorX = screen.width;
    //double factorY = _imageHeight / _imageWidth * screen.width;
    //boxesColor ??= Color.fromRGBO(37, 213, 253, 1.0);

    // print(recognitions.length);
    return LayoutBuilder(builder: (context, constraints) {
      debugPrint(
          'Max height: ${constraints.maxHeight}, max width: ${constraints.maxWidth}');
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

            // print({
            //   "left": re.rect.left.toDouble() * factorX,
            //   "top": re.rect.top.toDouble() * factorY,
            //   "width": re.rect.width.toDouble() * factorX,
            //   "height": re.rect.height.toDouble() * factorY,
            // });
            return Positioned(
              left: re.rect.left * factorX,
              top: re.rect.top * factorY - 20,
              //width: re.rect.width.toDouble(),
              //height: re.rect.height.toDouble(),

              //left: re?.rect.left.toDouble(),
              //top: re?.rect.top.toDouble(),
              //right: re.rect.right.toDouble(),
              //bottom: re.rect.bottom.toDouble(),
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
              /*
              Container(
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.all(Radius.circular(8.0)),
                  border: Border.all(
                    color: boxesColor!,
                    width: 2,
                  ),
                ),
                child: Text(
                  "${re.className ?? re.classIndex} ${(re.score * 100).toStringAsFixed(0)}%",
                  style: TextStyle(
                    background: Paint()..color = boxesColor!,
                    color: Colors.white,
                    fontSize: 12.0,
                  ),
                ),
              ),*/
            );
          }).toList()
        ],
      );
    });
  }
}
