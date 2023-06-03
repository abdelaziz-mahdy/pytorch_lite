import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:pytorch_lite/generated_bindings.dart';
import 'package:pytorch_lite/native_wrapper.dart';
import 'package:pytorch_lite/pigeon.dart';
import 'package:image/image.dart' as imageLib;

import 'native_locator.dart';

export 'enums/dtype.dart';
export 'utils.dart';
const torchVisionNormMeanRGB = [0.485, 0.456, 0.406];
const torchVisionNormSTDRGB = [0.229, 0.224, 0.225];

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
    int index = await PytorchFfi.loadModel(path);

    List<String> labels = [];
    if (labelPath != null) {
      if (labelPath.endsWith(".txt")) {
        labels = await _getLabelsTxt(labelPath);
      } else {
        labels = await _getLabelsCsv(labelPath);
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
    int index = await PytorchFfi.loadModel(path);

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
    return ModelObjectDetection(index, imageWidth, imageHeight, labels,
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
  ClassificationModel(
      this._index, this.labels, this.imageWidth, this.imageHeight);

  ///predicts image and returns the supposed label belonging to it
  Future<String> getImagePrediction(Uint8List imageAsBytes,
      {List<double> mean = torchVisionNormMeanRGB,
      List<double> std = torchVisionNormSTDRGB}) async {
    // Assert mean std
    assert(mean.length == 3, "mean should have size of 3");
    assert(std.length == 3, "std should have size of 3");

    final List<double?> prediction =
        await getImagePredictionList(imageAsBytes, mean: mean, std: std);

    double maxScore = double.negativeInfinity;
    int maxScoreIndex = -1;
    for (int i = 0; i < prediction.length; i++) {
      if (prediction[i]! > maxScore) {
        maxScore = prediction[i]!;
        maxScoreIndex = i;
      }
    }
    // free(dataPointer);
    return labels[maxScoreIndex];
  }

  ///predicts image but returns the raw net output
  Future<List<double>> getImagePredictionList(Uint8List imageAsBytes,
      {List<double> mean = torchVisionNormMeanRGB,
      List<double> std = torchVisionNormSTDRGB}) async {
    // Assert mean std
    assert(mean.length == 3, "Mean should have size of 3");
    assert(std.length == 3, "STD should have size of 3");

    return await PytorchFfi.imageModelInference(
        _index, imageAsBytes, imageHeight, imageWidth, mean, std);
  }

  ///predicts image but returns the output as probabilities
  ///[image] takes the File of the image
  Future<List<double?>?> getImagePredictionListProbabilities(
      Uint8List imageAsBytes,
      {List<double> mean = torchVisionNormMeanRGB,
      List<double> std = torchVisionNormSTDRGB}) async {
    // Assert mean std
    assert(mean.length == 3, "Mean should have size of 3");
    assert(std.length == 3, "STD should have size of 3");
    List<double> prediction =
        await getImagePredictionList(imageAsBytes, mean: mean, std: std);
    List<double?> predictionProbabilities = [];

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

  ///predicts image and returns the supposed label belonging to it
  Future<String> getImagePredictionFromBytesList(
      List<Uint8List> imageAsBytesList, int imageWidth, int imageHeight,
      {List<double> mean = torchVisionNormMeanRGB,
      List<double> std = torchVisionNormSTDRGB}) async {
    // Assert mean std
    assert(mean.length == 3, "mean should have size of 3");
    assert(std.length == 3, "std should have size of 3");

    final List<double> prediction = await getImagePredictionListFromBytesList(imageAsBytesList, imageWidth, imageHeight,mean: mean,std: std);

    double maxScore = double.negativeInfinity;
    int maxScoreIndex = -1;
    for (int i = 0; i < prediction!.length; i++) {
      if (prediction[i] > maxScore) {
        maxScore = prediction[i]!;
        maxScoreIndex = i;
      }
    }

    return labels[maxScoreIndex];
  }

  ///predicts image but returns the raw net output
  Future<List<double>> getImagePredictionListFromBytesList(
      List<Uint8List> imageAsBytesList, int imageWidth, int imageHeight,
      {List<double> mean = torchVisionNormMeanRGB,
      List<double> std = torchVisionNormSTDRGB}) async {
    // Assert mean std
    assert(mean.length == 3, "Mean should have size of 3");
    assert(std.length == 3, "STD should have size of 3");
    Uint8List combinedUint8List = Uint8List.fromList(
        imageAsBytesList.expand((Uint8List uint8List) => uint8List).toList());
    final List<double> prediction = await PytorchFfi.imageModelInference(
        _index, combinedUint8List, imageWidth, imageHeight, mean, std);

    return prediction;
  }

  ///predicts image but returns the output as probabilities
  ///[image] takes the File of the image
  Future<List<double>> getImagePredictionListProbabilitiesFromBytesList(
      List<Uint8List> imageAsBytesList, int imageWidth, int imageHeight,
      {List<double> mean = torchVisionNormMeanRGB,
      List<double> std = torchVisionNormSTDRGB}) async {
    // Assert mean std
    assert(mean.length == 3, "Mean should have size of 3");
    assert(std.length == 3, "STD should have size of 3");

    final List<double> prediction = await getImagePredictionListFromBytesList(imageAsBytesList, imageWidth, imageHeight,mean: mean,std: std);

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

  ///predicts image and returns the supposed label belonging to it
  Future<List<ResultObjectDetection?>> getImagePrediction(
      Uint8List imageAsBytes,
      {double minimumScore = 0.5,
      double iOUThreshold = 0.5,
      int boxesLimit = 10}) async {
    List<ResultObjectDetection?> prediction = await ModelApi()
        .getImagePredictionListObjectDetection(_index, imageAsBytes, null, null,
            null, minimumScore, iOUThreshold, boxesLimit);

    for (var element in prediction) {
      element?.className = labels[element.classIndex];
    }

    return prediction;
  }

  ///predicts image and returns the supposed label belonging to it
  Future<List<ResultObjectDetection?>> getImagePredictionFromBytesList(
      List<Uint8List> imageAsBytesList, int imageWidth, int imageHeight,
      {double minimumScore = 0.5,
      double iOUThreshold = 0.5,
      int boxesLimit = 10}) async {
    List<ResultObjectDetection?> prediction = await ModelApi()
        .getImagePredictionListObjectDetection(_index, null, imageAsBytesList,
            imageWidth, imageHeight, minimumScore, iOUThreshold, boxesLimit);

    for (var element in prediction) {
      element?.className = labels[element.classIndex];
    }

    return prediction;
  }

  ///predicts image but returns the raw net output
  Future<List<ResultObjectDetection?>> getImagePredictionList(
      Uint8List imageAsBytes,
      {double minimumScore = 0.5,
      double iOUThreshold = 0.5,
      int boxesLimit = 10}) async {
    final List<ResultObjectDetection?> prediction = await ModelApi()
        .getImagePredictionListObjectDetection(_index, imageAsBytes, null, null,
            null, minimumScore, iOUThreshold, boxesLimit);
    return prediction;
  }

  ///predicts image but returns the raw net output
  Future<List<ResultObjectDetection?>> getImagePredictionListFromBytesList(
      List<Uint8List> imageAsBytesList, int imageWidth, int imageHeight,
      {double minimumScore = 0.5,
      double iOUThreshold = 0.5,
      int boxesLimit = 10}) async {
    final List<ResultObjectDetection?> prediction = await ModelApi()
        .getImagePredictionListObjectDetection(_index, null, imageAsBytesList,
            imageWidth, imageHeight, minimumScore, iOUThreshold, boxesLimit);
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

/*
  ///predicts image and returns the supposed label belonging to it
  Future<String> getImagePrediction(
      File image, int width, int height, String labelPath,
      {List<double> mean = TORCHVISION_NORM_MEAN_RGB,
      List<double> std = TORCHVISION_NORM_STD_RGB}) async {
    // Assert mean std
    assert(mean.length == 3, "mean should have size of 3");
    assert(std.length == 3, "std should have size of 3");

    List<String> labels = [];
    if (labelPath.endsWith(".txt")) {
      labels = await _getLabelsTxt(labelPath);
    } else {
      labels = await _getLabelsCsv(labelPath);
    }

    List byteArray = image.readAsBytesSync();
    final List? prediction =
        await _channel.invokeListMethod("predictImage_ObjectDetection", {
      "index": _index,
      "image": byteArray,
      "width": width,
      "height": height,
      "mean": mean,
      "std": std
    });
    double maxScore = double.negativeInfinity;
    int maxScoreIndex = -1;
    for (int i = 0; i < prediction!.length; i++) {
      if (prediction[i] > maxScore) {
        maxScore = prediction[i];
        maxScoreIndex = i;
      }
    }
    return labels[maxScoreIndex];
  }

  ///predicts image but returns the raw net output
  Future<List?> getImagePredictionList(File image, int width, int height,
      {List<double> mean = TORCHVISION_NORM_MEAN_RGB,
      List<double> std = TORCHVISION_NORM_STD_RGB}) async {
    // Assert mean std
    assert(mean.length == 3, "Mean should have size of 3");
    assert(std.length == 3, "STD should have size of 3");
    final List? prediction =
        await _channel.invokeListMethod("predictImage_ObjectDetection", {
      "index": _index,
      "image": image.readAsBytesSync(),
      "width": width,
      "height": height,
      "mean": mean,
      "std": std
    });
    return prediction;
  }

 */
}
