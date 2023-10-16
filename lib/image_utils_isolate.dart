import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:computer/computer.dart';
import 'package:image/image.dart';

class ImageUtilsIsolate {
  static late Computer computer;

  static bool initiated = false;

  static Future<void> init({int workersCount = 2}) async {
    if (ImageUtilsIsolate.initiated) {
      return;
    }
    print("ImageUtilsIsolate initialization");
    ImageUtilsIsolate.initiated = true;
    ImageUtilsIsolate.computer = Computer.create(); //Or Computer.shared()
    await ImageUtilsIsolate.computer.turnOn(
      workersCount: workersCount, // optional, default 2
      verbose: false, // optional, default false
    );
  }

  /// Converts a [CameraImage] in YUV420 format to [Image] in RGB format
  static Future<Image?> convertCameraImage(CameraImage cameraImage) async {
    Uint8List? bytes = (await convertCameraImageToBytes(cameraImage));
    if (bytes != null) {
      return decodeJpg(bytes);
    } else {
      return null;
    }
  }

  static TransferableTypedData? _convertCameraImageToBytes(dynamic values) {
    ImageFormatGroup imageFormatGroup = values[0];
    int? uvRowStride = values[1];
    int? uvPixelStride = values[2];
    List<Uint8List>? planes = values[3];
    int width = values[4];
    int height = values[5];
    Image? image;
    if (imageFormatGroup == ImageFormatGroup.yuv420) {
      image = convertYUV420ToImage(
          uvRowStride!, uvPixelStride!, planes!, width, height);
    } else if (imageFormatGroup == ImageFormatGroup.bgra8888) {
      image = convertBGRA8888ToImage(width, height, planes![0]);
    } else {
      image = null;
    }

    if (image != null) {
      if (Platform.isIOS) {
        // ios, default camera image is portrait view
        // rotate 270 to the view that top is on the left, bottom is on the right
        // image ^4.0.17 error here
        image = copyRotate(image, angle: 270);
      }
      if (Platform.isAndroid) {
        image = copyRotate(image, angle: 90);
      }
      return TransferableTypedData.fromList([encodeJpg(image)]);
    }
    return null;
  }

  static List<dynamic> _getParamsBasedOnType(CameraImage cameraImage) {
    if (cameraImage.format.group == ImageFormatGroup.yuv420) {
      return [
        cameraImage.format.group,
        cameraImage.planes[1].bytesPerRow,
        cameraImage.planes[1].bytesPerPixel ?? 0,
        cameraImage.planes.map((e) => e.bytes).toList(),
        cameraImage.width,
        cameraImage.height
      ];
    } else if (cameraImage.format.group == ImageFormatGroup.bgra8888) {
      return [
        cameraImage.format.group,
        null,
        null,
        cameraImage.planes.map((e) => e.bytes).toList(),
        cameraImage.width,
        cameraImage.height
      ];
    }
    // You can add more formats as needed
    return [];
  }

  /// Converts a [CameraImage] in YUV420 format to [Image] in RGB format
  static Future<Uint8List?> convertCameraImageToBytes(
      CameraImage cameraImage) async {
    await ImageUtilsIsolate.init();

    return (await ImageUtilsIsolate.computer.compute(_convertCameraImageToBytes,
                param: _getParamsBasedOnType(cameraImage))
            as TransferableTypedData?)
        ?.materialize()
        .asUint8List();
  }

  static Future<Float64List> convertImageBytesToFloatBuffer(
    Uint8List bytes,
    int width,
    int height,
    List<double> mean,
    List<double> std,
  ) async {
    await ImageUtilsIsolate.init();

    final float32List = (await ImageUtilsIsolate.computer.compute(
            _convertImageBytesToFloatBuffer,
            param: [bytes, width, height, mean, std]) as TransferableTypedData)
        .materialize()
        .asFloat32List();

    final float64List = Float64List(float32List.length);

    for (int i = 0; i < float32List.length; i++) {
      float64List[i] = float32List[i];
    }

    return float64List;
  }

  static Future<Uint8List> convertImageBytesToFloatBufferUInt8List(
    Uint8List bytes,
    int width,
    int height,
    List<double> mean,
    List<double> std,
  ) async {
    await ImageUtilsIsolate.init();

    return (await ImageUtilsIsolate.computer.compute(
            _convertImageBytesToFloatBuffer,
            param: [bytes, width, height, mean, std]) as TransferableTypedData)
        .materialize()
        .asUint8List();
  }

  static Float64List convertUInt8ListToFloat64List(Uint8List uint8List) {
    final float64List = Float64List(uint8List.length);

    for (int i = 0; i < uint8List.length; i++) {
      float64List[i] = uint8List[i].toDouble();
    }

    return float64List;
  }

  static TransferableTypedData _convertImageBytesToFloatBuffer(
      List<dynamic> params) {
    final bytes = params[0];
    final width = params[1];
    final height = params[2];
    final mean = params[3];
    final std = params[4];
    // Extract other variables from params as needed

    Image? img = decodeImage(bytes);
    if (img == null) {
      throw Exception("Unable to process image bytes");
    }
    Image scaledImageBytes = copyResize(img, width: width, height: height);

    return TransferableTypedData.fromList(
        [imageToUint8List(scaledImageBytes, mean, std)]);
  }

  static Uint8List imageToUint8List(
      Image image, List<double> mean, List<double> std,
      {bool contiguous = true}) {
    var bytes = Float32List(1 * image.height * image.width * 3);
    var buffer = Float32List.view(bytes.buffer);

    if (contiguous) {
      int offsetG = image.height * image.width;
      int offsetB = 2 * image.height * image.width;
      int i = 0;
      for (var y = 0; y < image.height; y++) {
        for (var x = 0; x < image.width; x++) {
          Pixel pixel = image.getPixel(x, y);
          buffer[i] = ((pixel.r / 255) - mean[0]) / std[0];
          buffer[offsetG + i] = ((pixel.g / 255) - mean[1]) / std[1];
          buffer[offsetB + i] = ((pixel.b / 255) - mean[2]) / std[2];
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

  static Image convertBGRA8888ToImage(int width, int height, Uint8List bytes) {
    return Image.fromBytes(
      width: width,
      height: height,
      bytes: bytes.buffer,
      order: ChannelOrder.bgra,
    );
  }

  static Image convertNV21ToImage(CameraImage image) {
    return Image.fromBytes(
      width: image.width,
      height: image.height,
      bytes: image.planes.first.bytes.buffer,
      order: ChannelOrder.bgra,
    );
  }

  static Image convertYUV420ToImage(int uvRowStride, int uvPixelStride,
      List<Uint8List> planes, int width, int height) {
    final img = Image(width: width, height: height);
    for (final p in img) {
      final x = p.x;
      final y = p.y;
      final uvIndex =
          uvPixelStride * (x / 2).floor() + uvRowStride * (y / 2).floor();
      final index = y * uvRowStride +
          x; // Use the row stride instead of the image width as some devices pad the image data, and in those cases the image width != bytesPerRow. Using width will give you a distored image.
      final yp = planes[0][index];
      final up = planes[1][uvIndex];
      final vp = planes[2][uvIndex];
      p.r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255).toInt();
      p.g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
          .round()
          .clamp(0, 255)
          .toInt();
      p.b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255).toInt();
    }

    return img;
  }
}
