  import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:image/image.dart';

Pointer<Uint8> convertUint8ListToPointer(Uint8List data) {
    int length = data.length;
    Pointer<Uint8> dataPtr = calloc<Uint8>(length);

    for (int i = 0; i < length; i++) {
      dataPtr.elementAt(i).value = data[i];
    }

    return dataPtr;
  }

  Pointer<UnsignedChar> convertUint8ListToPointerChar(Uint8List data) {
    int length = data.length;
    Pointer<UnsignedChar> dataPtr = calloc<UnsignedChar>(length);

    for (int i = 0; i < length; i++) {
      dataPtr.elementAt(i).value = data[i];
    }

    return dataPtr;
  }

  Uint8List imageToUint8List(Image image, List<double> mean, List<double> std,
      {bool contiguous = true}) {
    var bytes = Float32List(1 * image.height * image.width * 3);
    var buffer = Float32List.view(bytes.buffer);

    if (contiguous) {
      int offset_g = image.height * image.width;
      int offset_b = 2 * image.height * image.width;
      int i = 0;
      for (var y = 0; y < image.height; y++) {
        for (var x = 0; x < image.width; x++) {
          Pixel pixel = image.getPixel(x, y);
          buffer[i] = ((pixel.r / 255) - mean[0]) / std[0];
          buffer[offset_g + i] = ((pixel.g / 255) - mean[1]) / std[1];
          buffer[offset_b + i] = ((pixel.b / 255) - mean[2]) / std[2];
          i++;
        }
      }
    } else {
      int i = 0;
      for (var y = 0; y < image.height; y++) {
        for (var x = 0; x < image.width; x++) {
          Pixel pixel = image.getPixel(x, y);
          buffer[i++] = ((pixel.r / 255) - mean[0]) / std[0];
          buffer[i++] = ((pixel.g / 255) - mean[1]) / std[1];
          buffer[i++] = ((pixel.b / 255) - mean[2]) / std[2];
        }
      }
    }

    return bytes.buffer.asUint8List();
  }

  Pointer<Float> convertListToPointer(List<double> floatList) {
    // Create a native array to hold the double values
    final nativeArray = calloc<Double>(floatList.length);

    // Copy the values from the list to the native array
    for (var i = 0; i < floatList.length; i++) {
      nativeArray[i] = floatList[i];
    }

    // Obtain the pointer to the native array
    final nativePointer = nativeArray.cast<Float>();

    return nativePointer;
  }