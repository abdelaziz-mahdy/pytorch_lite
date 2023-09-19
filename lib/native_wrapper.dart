import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:computer/computer.dart';
import 'package:ffi/ffi.dart';

import 'package:pytorch_lite/generated_bindings.dart';
import 'package:pytorch_lite/post_processor.dart';
import 'package:pytorch_lite/utils.dart';

import 'classes/result_object_detection.dart';
import 'native_locator.dart';

/// The bindings to the native functions in [dylib].
final NativeLibrary _bindings = NativeLibrary(dylib);

class PytorchFfi {
  static bool initiated = false;
  static late Computer computer;

  static Future<void> init({int concurrent = 4}) async {
    if (PytorchFfi.initiated) {
      return;
    }
    print("PytorchFfi initialization");
    PytorchFfi.initiated = true;
    PytorchFfi.computer = Computer.create(); //Or Computer.shared()

    await PytorchFfi.computer.turnOn(
      workersCount: concurrent, // optional, default 2
      verbose: false, // optional, default false
    );
  }

  @pragma('vm:entry-point')
  static Future<int> loadModel(dynamic modelPath) async {
    await PytorchFfi.init();
    return await computer.compute(_loadModel, param: modelPath);
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
    Uint8List yBuffer,
    Uint8List? uBuffer,
    Uint8List? vBuffer,
    int rotation,
    int modelImageHeight,
    int modelImageWidth,
    int cameraImageHeight,
    int cameraImageWidth,
    List<double> mean,
    List<double> std,
    bool objectDetectionYoloV5,
    int outputLength,
  ) async {
    await PytorchFfi.init();
    TransferableTypedData transferableTypedData =
        await computer.compute(_cameraImageModelInference, param: [
      modelIndex,
      yBuffer,
      uBuffer,
      vBuffer,
      rotation,
      modelImageHeight,
      modelImageWidth,
      cameraImageHeight,
      cameraImageWidth,
      mean,
      std,
      objectDetectionYoloV5,
      outputLength,
    ]);
    var startTime = DateTime.now();
    List<double> data =
        transferableTypedData.materialize().asFloat32List().toList();
    var endTime = DateTime.now();
    print(
        "materialize output takes ${endTime.difference(startTime).inMilliseconds}ms");

    return data;
  }

  static Future<List<ResultObjectDetection>>
      cameraImageModelInferenceObjectDetection(
          int modelIndex,
          Uint8List yBuffer,
          Uint8List? uBuffer,
          Uint8List? vBuffer,
          int rotation,
          int modelImageHeight,
          int modelImageWidth,
          int cameraImageHeight,
          int cameraImageWidth,
          List<double> mean,
          List<double> std,
          bool objectDetectionYoloV5,
          int outputLength,
          PostProcessorObjectDetection postProcessorObjectDetection) async {
    await PytorchFfi.init();
    return await computer
        .compute(_cameraImageModelInferenceObjectDetection, param: [
      modelIndex,
      yBuffer,
      uBuffer,
      vBuffer,
      rotation,
      modelImageHeight,
      modelImageWidth,
      cameraImageHeight,
      cameraImageWidth,
      mean,
      std,
      objectDetectionYoloV5,
      outputLength,
      postProcessorObjectDetection
    ]);
  }

  static Future<List<double>> imageModelInference(
      int modelIndex,
      Uint8List imageAsBytes,
      int imageHeight,
      int imageWidth,
      List<double> mean,
      List<double> std,
      bool objectDetectionYoloV5,
      int outputLength) async {
    await PytorchFfi.init();

    return (await computer.compute(_imageModelInference, param: [
      modelIndex,
      imageAsBytes,
      imageHeight,
      imageWidth,
      mean,
      std,
      objectDetectionYoloV5,
      outputLength
    ]) as TransferableTypedData)
        .materialize()
        .asFloat32List()
        .toList();
  }

  static Future<List<ResultObjectDetection>> imageModelInferenceObjectDetection(
      int modelIndex,
      Uint8List imageAsBytes,
      int imageHeight,
      int imageWidth,
      List<double> mean,
      List<double> std,
      bool objectDetectionYoloV5,
      int outputLength,
      PostProcessorObjectDetection postProcessorObjectDetection) async {
    await PytorchFfi.init();

    return await computer.compute(_imageModelInferenceObjectDetection, param: [
      modelIndex,
      imageAsBytes,
      imageHeight,
      imageWidth,
      mean,
      std,
      objectDetectionYoloV5,
      outputLength,
      postProcessorObjectDetection
    ]);
  }

  @pragma('vm:entry-point')
  static TransferableTypedData _imageModelInference(dynamic values) {
    int modelIndex = values[0];
    Uint8List imageAsBytes = values[1];
    int imageHeight = values[2];
    int imageWidth = values[3];
    List<double> mean = values[4];
    List<double> std = values[5];
    bool objectDetectionYoloV5 = values[6];
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
        objectDetectionYoloV5 ? 1 : 0,
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
          "output length does not match model length, please check model type and number of classes expected $outputLength, got ${outputData.length}");
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
    bool objectDetectionYoloV5 = values[11];
    int outputLength = values[12];

    var startTime = DateTime.now();
    var endTime = DateTime.now();

    var ySize = yBuffer.lengthInBytes;
    var uSize = uBuffer?.lengthInBytes ?? 0;
    var vSize = vBuffer?.lengthInBytes ?? 0;
    var totalSize = ySize + uSize + vSize;

    Pointer<Uint8> dataPointer = malloc.allocate<Uint8>(totalSize);

    // We always have at least 1 plane, on Android it si the yPlane on iOS its the rgba plane
    Uint8List bytes = dataPointer.asTypedList(totalSize);
    bytes.setAll(0, yBuffer);

    if (Platform.isAndroid) {
      // Swap u&v buffer for opencv
      bytes.setAll(ySize, vBuffer!);
      bytes.setAll(ySize + vSize, uBuffer!);
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
        Platform.isAndroid ? 1 : 0,
        modelImageWidth,
        modelImageHeight,
        cameraImageHeight,
        cameraImageWidth,
        objectDetectionYoloV5 ? 1 : 0,
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
          "output length does not match model length, please check model type and number of classes expected $outputLength, got ${outputData.length}");
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
  static List<ResultObjectDetection> _cameraImageModelInferenceObjectDetection(
      dynamic values) {
    TransferableTypedData transferredData = _cameraImageModelInference(values);
    PostProcessorObjectDetection postProcessorObjectDetection =
        (values as List).last;
    List<double> data = transferredData.materialize().asFloat32List().toList();
    return postProcessorObjectDetection.outputsToNMSPredictions(data);
  }

  @pragma('vm:entry-point')
  static List<ResultObjectDetection> _imageModelInferenceObjectDetection(
      dynamic values) {
    TransferableTypedData transferredData = _imageModelInference(values);
    PostProcessorObjectDetection postProcessorObjectDetection =
        (values as List).last;
    List<double> data = transferredData.materialize().asFloat32List().toList();
    return postProcessorObjectDetection.outputsToNMSPredictions(data);
  }
}
