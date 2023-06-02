import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:pytorch_lite/generated_bindings.dart';
import 'package:image/image.dart';
import 'package:pytorch_lite/utils.dart';

import 'native_locator.dart';

/// The bindings to the native functions in [dylib].
final NativeLibrary _bindings = NativeLibrary(dylib);

class PytorchFfi {
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

  static Future<int> loadModel(String modelPath) async {
    String absPathModelPath = await _getAbsolutePath(modelPath);

    ModelLoadResult result =
        _bindings.load_ml_model(absPathModelPath.toNativeUtf8());

    if (result.exception.toDartString().isNotEmpty) {
      throw Exception(result.exception.toDartString());
    }
    return result.index;
  }

  static List<double> imageModelInference(
      int modelIndex,
      Uint8List imageAsBytes,
      int imageHeight,
      int imageWidth,
      List<double> mean,
      List<double> std) {
    Image? img = decodeImage(imageAsBytes);
    Image scaledImageBytes =
        copyResize(img!, width: imageWidth, height: imageHeight);

    Pointer<UnsignedChar> dataPointer = convertUint8ListToPointerChar(
        imageToUint8List(scaledImageBytes, mean, std));
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
