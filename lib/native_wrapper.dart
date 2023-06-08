import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart';
import 'package:isolate_manager/isolate_manager.dart';

import 'package:pytorch_lite/generated_bindings.dart';
import 'package:image/image.dart';
import 'package:pytorch_lite/utils.dart';

import 'native_locator.dart';

/// The bindings to the native functions in [dylib].
final NativeLibrary _bindings = NativeLibrary(dylib);

class PytorchFfi {
  static bool initiated = false;
  static late IsolateManager loadModelManager;
  static late IsolateManager imageModelInferenceManager;

  static void init() {
    if (PytorchFfi.initiated) {
      return;
    }
    print("PytorchFfi initialization");
    PytorchFfi.initiated = true;
    PytorchFfi.loadModelManager =
        IsolateManager.create(_loadModel, concurrent: 3, isDebug: true);
    PytorchFfi.imageModelInferenceManager = IsolateManager.create(
        _imageModelInference,
        concurrent: 3,
        isDebug: true);
  }

  @pragma('vm:entry-point')
  static Future<int> loadModel(dynamic modelPath) async {
    PytorchFfi.init();
    return await loadModelManager.compute(modelPath);
  }

  @pragma('vm:entry-point')
  static Future<int> _loadModel(dynamic modelPath) async {
    Pointer<Utf8> data = (modelPath as String).toNativeUtf8();
    ModelLoadResult result = _bindings.load_ml_model(data);

    if (result.exception.toDartString().isNotEmpty) {
      throw Exception(result.exception.toDartString());
    }
    // calloc.free(data);
    // calloc.free(result.exception);
    return result.index;
  }

  static Future<List<double>> imageModelInference(
      int modelIndex,
      Uint8List imageAsBytes,
      int imageHeight,
      int imageWidth,
      List<double> mean,
      List<double> std,
      bool objectDetectionYolov5) async {
    PytorchFfi.init();

    return await imageModelInferenceManager.compute([
      modelIndex,
      imageAsBytes,
      imageHeight,
      imageWidth,
      mean,
      std,
      objectDetectionYolov5
    ]);
    // return _imageModelInference(
    //   [
    //   modelIndex,
    //   imageAsBytes,
    //   imageHeight,
    //   imageWidth,
    //   mean,
    //   std,
    //   objectDetectionYolov5
    // ]
    // );
  }

  @pragma('vm:entry-point')
  static List<double> _imageModelInference(dynamic values) {
    int modelIndex = values[0];
    Uint8List imageAsBytes = values[1];
    int imageHeight = values[2];
    int imageWidth = values[3];
    List<double> mean = values[4];
    List<double> std = values[5];
    bool objectDetection = values[6];

    Image? img = decodeImage(imageAsBytes);
    if (img == null) {
      throw Exception("Failed to decode image");
    }
    Image scaledImageBytes =
        copyResize(img, width: imageWidth, height: imageHeight);

    Pointer<UnsignedChar> dataPointer = convertUint8ListToPointerChar(
        ImageUtils.imageToUint8List(scaledImageBytes, mean, std));
    OutputData outputData = _bindings.image_model_inference(modelIndex,
        dataPointer, imageWidth, imageHeight, objectDetection ? 1 : 0);
    if (outputData.exception.toDartString().isNotEmpty) {
      throw Exception(outputData.exception.toDartString());
    }
    final List<double> prediction =
        outputData.values.asTypedList(outputData.length);
    // calloc.free(outputData.values);
    // calloc.free(outputData.exception);
    // calloc.free(dataPointer);

    return prediction;
  }
}
