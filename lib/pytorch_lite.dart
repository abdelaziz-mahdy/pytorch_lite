import 'dart:async';
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';

import 'enums/dtype.dart';

const TORCHVISION_NORM_MEAN_RGB = [0.485, 0.456, 0.406];
const TORCHVISION_NORM_STD_RGB = [0.229, 0.224, 0.225];

class PytorchLite {
  static const MethodChannel _channel = MethodChannel('pytorch_lite');

  static Future<String?> get platformVersion async {
    final String? version = await _channel.invokeMethod('getPlatformVersion');
    return version;
  }

  ///Sets pytorch model path and returns Model
  static Future<Model> loadModel(String path) async {
    String absPath = await _getAbsolutePath(path);
    int index = await _channel
        .invokeMethod("loadModel", {"absPath": absPath, "assetPath": path});
    return Model(index);
  }

  ///Sets pytorch model path and returns Model
  static Future<ModelObjectDetection> loadObjectDetectionModel(
      String path) async {
    String absPath = await _getAbsolutePath(path);
    int index = await _channel
        .invokeMethod("loadModel", {"absPath": absPath, "assetPath": path});
    return ModelObjectDetection(index);
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

class Model {
  static const MethodChannel _channel = const MethodChannel('pytorch_mobile');

  final int _index;

  Model(this._index);

  ///predicts abstract number input
  Future<List?> getPrediction(
      List<double> input, List<int> shape, DType dtype) async {
    final List? prediction = await _channel.invokeListMethod('predict', {
      "index": _index,
      "data": input,
      "shape": shape,
      "dtype": dtype.toString().split(".").last
    });
    return prediction;
  }

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
    final List? prediction = await _channel.invokeListMethod("predictImage", {
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
    final List? prediction = await _channel.invokeListMethod("predictImage", {
      "index": _index,
      "image": image.readAsBytesSync(),
      "width": width,
      "height": height,
      "mean": mean,
      "std": std
    });
    return prediction;
  }
}

class ModelObjectDetection {
  static const MethodChannel _channel = const MethodChannel('pytorch_mobile');

  final int _index;

  ModelObjectDetection(this._index);

  ///predicts abstract number input
  Future<List?> getPrediction(
      List<double> input, List<int> shape, DType dtype) async {
    final List? prediction =
        await _channel.invokeListMethod('predict_ObjectDetection', {
      "index": _index,
      "data": input,
      "shape": shape,
      "dtype": dtype.toString().split(".").last
    });
    return prediction;
  }

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
}
