import 'dart:ffi';
import 'dart:io';
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
  static late IsolateManager cameraImageModelInferenceManager;

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
    PytorchFfi.cameraImageModelInferenceManager = IsolateManager.create(
        _cameraImageModelInference,
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

  static Future<List<double>> cameraImageModelInference(
    int modelIndex,
 Uint8List yBuffer, Uint8List? uBuffer, Uint8List? vBuffer,
    int rotation,
    int modelImageHeight,
    int modelImageWidth,
    int cameraImageHeight,
    int cameraImageWidth,
    List<double> mean,
    List<double> std,
    bool objectDetectionYolov5,
    int outputLength,
  ) async {
    PytorchFfi.init();

    return (await cameraImageModelInferenceManager.compute([
      modelIndex,
       yBuffer,  uBuffer,  vBuffer,
      rotation,
      modelImageHeight,
      modelImageWidth,
      cameraImageHeight,
      cameraImageWidth,
      mean,
      std,
      objectDetectionYolov5,
      outputLength,
    ]) as TransferableTypedData)
        .materialize()
        .asFloat32List()
        .toList();
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
  }

  @pragma('vm:entry-point')
  static TransferableTypedData _imageModelInference(dynamic values) {
    int modelIndex = values[0];
    Uint8List imageAsBytes = values[1];
    int imageHeight = values[2];
    int imageWidth = values[3];
    List<double> mean = values[4];
    List<double> std = values[5];
    bool objectDetectionYolov5 = values[6];
    int outputLength = values[7];

    var startTime = DateTime.now();
    var endTime = DateTime.now();

    startTime = DateTime.now();
    Pointer<UnsignedChar> dataPointer =
        convertUint8ListToPointerChar(imageAsBytes);
    endTime = DateTime.now();
    print(
        "convertUint8ListToPointerChar time: ${endTime.difference(startTime).inMilliseconds}ms");

    Pointer<Float> output = calloc<Float>(outputLength + 1);
    Pointer<Float> meanPointer = convertDoubleListToPointerFloat(mean);
    Pointer<Float> stdPointer = convertDoubleListToPointerFloat(std);

    startTime = DateTime.now();
    OutputData outputData = _bindings.image_model_inference(
        modelIndex,
        dataPointer,
        imageAsBytes.lengthInBytes,
        imageWidth,
        imageHeight,
        objectDetectionYolov5 ? 1 : 0,
        meanPointer,
        stdPointer,
        output);
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
    TransferableTypedData data = TransferableTypedData.fromList(
        [Float32List.fromList(output.asTypedList(outputData.length))]);
    endTime = DateTime.now();
    print(
        "TransferableTypedData time: ${endTime.difference(startTime).inMilliseconds}ms");
    calloc.free(output);
    calloc.free(dataPointer);
    calloc.free(meanPointer);
    calloc.free(stdPointer);

    return data;
  }

  @pragma('vm:entry-point')
  static TransferableTypedData _cameraImageModelInference(dynamic values) {
    int modelIndex = values[0];
    Uint8List yBuffer = values[1];
    Uint8List? uBuffer = values[2];
    Uint8List? vBuffer = values[3];
    int rotation = values[4];
    int modelImageHeight = values[5];
    int modelImageWidth = values[6];
    int cameraImageHeight = values[7];
    int cameraImageWidth = values[8];
    List<double> mean = values[9];
    List<double> std = values[10];
    bool objectDetectionYolov5 = values[11];
    int outputLength = values[12];

    var startTime = DateTime.now();
    var endTime = DateTime.now();
    
    var ySize = yBuffer.lengthInBytes;
    var uSize = uBuffer?.lengthInBytes ?? 0;
    var vSize = vBuffer?.lengthInBytes ?? 0;
    var totalSize = ySize + uSize + vSize;

    Pointer<Uint8> dataPointer = malloc.allocate<Uint8>(totalSize);

    // We always have at least 1 plane, on Android it si the yPlane on iOS its the rgba plane
    Uint8List _bytes = dataPointer.asTypedList(totalSize);
    _bytes.setAll(0, yBuffer);

    if (Platform.isAndroid) {
      // Swap u&v buffer for opencv
      _bytes.setAll(ySize, vBuffer!);
      _bytes.setAll(ySize + vSize, uBuffer!);
    }
    startTime = DateTime.now();


    endTime = DateTime.now();
    print(
        "convertUint8ListToPointerChar time: ${endTime.difference(startTime).inMilliseconds}ms");

    Pointer<Float> output = calloc<Float>(outputLength + 1);
    Pointer<Float> meanPointer = convertDoubleListToPointerFloat(mean);
    Pointer<Float> stdPointer = convertDoubleListToPointerFloat(std);

    startTime = DateTime.now();
    OutputData outputData = _bindings.camera_model_inference(
        modelIndex,
        dataPointer.cast<UnsignedChar>(),
        rotation,
        Platform.isAndroid?1:0,
        modelImageWidth,
        modelImageHeight,
        cameraImageHeight,
        cameraImageWidth,
        objectDetectionYolov5 ? 1 : 0,
        meanPointer,
        stdPointer,
        output);
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
    TransferableTypedData data = TransferableTypedData.fromList(
        [Float32List.fromList(output.asTypedList(outputData.length))]);
    endTime = DateTime.now();
    print(
        "TransferableTypedData time: ${endTime.difference(startTime).inMilliseconds}ms");
    calloc.free(output);
    calloc.free(dataPointer);
    calloc.free(meanPointer);
    calloc.free(stdPointer);

    return data;
  }
}
