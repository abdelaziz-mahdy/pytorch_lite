import 'dart:ffi';
import 'dart:isolate';
import 'dart:typed_data';

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
        IsolateManager.create(_loadModel, concurrent: 2, isDebug: false);
    PytorchFfi.imageModelInferenceManager = IsolateManager.create(
        _imageModelInference,
        concurrent: 2,
        isDebug: false);
  }

  @pragma('vm:entry-point')
  static Future<int> loadModel(dynamic modelPath) async {
    PytorchFfi.init();
    return await loadModelManager.compute(modelPath);
  }

  @pragma('vm:entry-point')
  static Future<int> _loadModel(dynamic modelPath) async {
    Pointer<Utf8> data = (modelPath as String).toNativeUtf8(allocator: calloc);
    ModelLoadResult result = _bindings.load_ml_model(data);

    if (result.exception.toDartString().isNotEmpty) {
      throw Exception(result.exception.toDartString());
    }
    calloc.free(data);
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
      bool objectDetectionYolov5,
      int outputLength) async {
    PytorchFfi.init();

    return (await imageModelInferenceManager.compute([
      modelIndex,
      imageAsBytes,
      imageHeight,
      imageWidth,
      mean,
      std,
      objectDetectionYolov5,
      outputLength
    ]) as TransferableTypedData)
        .materialize()
        .asFloat32List()
        .toList();
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
  static TransferableTypedData _imageModelInference(dynamic values) {
    int modelIndex = values[0];
    Uint8List imageAsBytes = values[1];
    int imageHeight = values[2];
    int imageWidth = values[3];
    List<double> mean = values[4];
    List<double> std = values[5];
    bool objectDetection = values[6];
    int outputLength = values[7];

    var startTime = DateTime.now();
    Image? img = decodeImage(imageAsBytes);
    var endTime = DateTime.now();
    print(
        "decodeImage time: ${endTime.difference(startTime).inMilliseconds}ms");

    if (img == null) {
      throw Exception("Failed to decode image");
    }

    startTime = DateTime.now();
    Image scaledImageBytes =
        copyResize(img, width: imageWidth, height: imageHeight);
    endTime = DateTime.now();
    print("copyResize time: ${endTime.difference(startTime).inMilliseconds}ms");

// ImageUtils.imageToUint8List()
    startTime = DateTime.now();
    Uint8List convertedImage =
        ImageUtils.imageToUint8List(scaledImageBytes, mean, std);
    endTime = DateTime.now();
    print(
        "ImageUtils.imageToUint8List time: ${endTime.difference(startTime).inMilliseconds}ms");

// convertUint8ListToPointerChar()
    startTime = DateTime.now();
    Pointer<UnsignedChar> dataPointer =
        convertUint8ListToPointerChar(convertedImage);
    endTime = DateTime.now();
    print(
        "convertUint8ListToPointerChar time: ${endTime.difference(startTime).inMilliseconds}ms");

    Pointer<Float> output = calloc<Float>(outputLength + 1);

    startTime = DateTime.now();
    OutputData outputData = _bindings.image_model_inference(modelIndex,
        dataPointer, imageWidth, imageHeight, objectDetection ? 1 : 0, output);
    endTime = DateTime.now();
    print(
        "image_model_inference time: ${endTime.difference(startTime).inMilliseconds}ms");

    if (outputData.exception.toDartString().isNotEmpty) {
      throw Exception(outputData.exception.toDartString());
    }
    if (outputLength != outputData.length) {
      throw Exception(
          "output length does not match model length, please check model type and number of classes expected ${outputLength}, got ${outputData.length}");
    }

    startTime = DateTime.now();
    //to list is used to make a copy of the values
    // final List<double> prediction =
    //     (output.asTypedList(outputData.length)).toList();
    // endTime = DateTime.now();
    // print("toList time: ${endTime.difference(startTime).inMilliseconds}ms");


    startTime = DateTime.now();
    TransferableTypedData data =
        TransferableTypedData.fromList([Float32List.fromList(output.asTypedList(outputData.length))]);
    endTime = DateTime.now();
    print(
        "TransferableTypedData time: ${endTime.difference(startTime).inMilliseconds}ms");
    calloc.free(output);
    calloc.free(dataPointer);

    return data;
  }
}
