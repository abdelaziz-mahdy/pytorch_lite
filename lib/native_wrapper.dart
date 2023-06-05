import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart';
import 'package:isolate_manager/isolate_manager.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
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
    PytorchFfi.loadModelManager = IsolateManager.create(_loadModel, isDebug: true);
    PytorchFfi.imageModelInferenceManager =
        IsolateManager.create(_imageModelInference, isDebug: true);
  }



  @pragma('vm:entry-point')
  static Future<int> loadModel(dynamic modelPath) async {
    PytorchFfi.init();
    return await loadModelManager.compute(modelPath);
  }

  @pragma('vm:entry-point')
  static Future<int> _loadModel(dynamic modelPath) async {

    ModelLoadResult result =
        _bindings.load_ml_model((modelPath as String).toNativeUtf8());

    if (result.exception.toDartString().isNotEmpty) {
      throw Exception(result.exception.toDartString());
    }
    return result.index;
  }

  static Future<List<double>> imageModelInference(
      int modelIndex,
      Uint8List imageAsBytes,
      int imageHeight,
      int imageWidth,
      List<double> mean,
      List<double> std) async {
    PytorchFfi.init();

    return await imageModelInferenceManager.compute(
        [modelIndex, imageAsBytes, imageHeight, imageWidth, mean, std]);
  }

  @pragma('vm:entry-point')
  static List<double> _imageModelInference(dynamic values) {
    int modelIndex = values[0];
    Uint8List imageAsBytes = values[1];
    int imageHeight = values[2];
    int imageWidth = values[3];
    List<double> mean = values[4];
    List<double> std = values[5];
    Image? img = decodeImage(imageAsBytes);
    if (img == null) {
      throw Exception("Failed to decode image");
    }
    Image scaledImageBytes =
        copyResize(img, width: imageWidth, height: imageHeight);

    Pointer<UnsignedChar> dataPointer = convertUint8ListToPointerChar(
        ImageUtils.imageToUint8List(scaledImageBytes, mean, std));
    OutputData outputData = _bindings.image_model_inference(
        modelIndex, dataPointer, imageWidth, imageHeight);
    if (outputData.exception.toDartString().isNotEmpty) {
      throw Exception(outputData.exception.toDartString());
    }
    final List<double> prediction =
        outputData.values.asTypedList(outputData.length);

    return prediction;
  }
}
