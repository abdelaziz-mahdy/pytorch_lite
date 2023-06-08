import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';
import 'package:camera/camera.dart';

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

class ImageUtils {
  static Uint8List imageToUint8List(
      Image image, List<double> mean, List<double> std,
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

  /// Converts a [CameraImage] in YUV420 format to [Image] in RGB format
  static Image? convertCameraImage(CameraImage cameraImage) {
    DateTime startTime = DateTime.now(); // Record the start time

    Image? image;
    if (cameraImage.format.group == ImageFormatGroup.yuv420) {
      image = convertYUV420ToImage(cameraImage);
    } else if (cameraImage.format.group == ImageFormatGroup.bgra8888) {
      image = convertBGRA8888ToImage(cameraImage);
    } else {
      return null;
    }

    // if (Platform.isAndroid) {
    //   image = copyRotate(image, angle: 90);
    // } else {
    //   image = copyRotate(image, angle: 270);
    // }

    DateTime endTime = DateTime.now(); // Record the end time
    int executionTime = endTime
        .difference(startTime)
        .inMilliseconds; // Calculate the execution time in milliseconds

    print("convertCameraImage: Execution time: $executionTime milliseconds");

    return image;
  }

  /// Converts a [CameraImage] in BGRA888 format to [Image] in RGB format
  static Image convertBGRA8888ToImage(CameraImage cameraImage) {
    Image img = Image.fromBytes(
      width: cameraImage.planes[0].width!,
      height: cameraImage.planes[0].height!,
      bytes: cameraImage.planes[0].bytes.buffer,
      order: ChannelOrder.bgra,
      // format: Format.bgra
    );
    return img;
  }

  /// Converts a [CameraImage] in YUV420 format to [Image] in RGB format
  static Image convertYUV420ToImage(CameraImage cameraImage) {
    final int width = cameraImage.width;
    final int height = cameraImage.height;

    final int uvRowStride = cameraImage.planes[1].bytesPerRow;
    final int? uvPixelStride = cameraImage.planes[1].bytesPerPixel;

    Image image = Image(width: width, height: height);
    Uint8List bytes = image.toUint8List();
    for (int w = 0; w < width; w++) {
      for (int h = 0; h < height; h++) {
        final int uvIndex =
            uvPixelStride! * (w / 2).floor() + uvRowStride * (h / 2).floor();
        final int index = h * width + w;

        final y = cameraImage.planes[0].bytes[index];
        final u = cameraImage.planes[1].bytes[uvIndex];
        final v = cameraImage.planes[2].bytes[uvIndex];

        if (image.data != null) {
          bytes[index] = ImageUtils.yuv2rgb(y, u, v);
        }
      }
    }
    image = Image.fromBytes(width: width, height: height, bytes: bytes.buffer);
    return image;
  }

  /// Convert a single YUV pixel to RGB
  static int yuv2rgb(int y, int u, int v) {
    // Convert yuv pixel to rgb
    int r = (y + v * 1436 / 1024 - 179).round();
    int g = (y - u * 46549 / 131072 + 44 - v * 93604 / 131072 + 91).round();
    int b = (y + u * 1814 / 1024 - 227).round();

    // Clipping RGB values to be inside boundaries [ 0 , 255 ]
    r = r.clamp(0, 255);
    g = g.clamp(0, 255);
    b = b.clamp(0, 255);

    return 0xff000000 |
        ((b << 16) & 0xff0000) |
        ((g << 8) & 0xff00) |
        (r & 0xff);
  }
}
