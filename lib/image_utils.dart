import 'dart:io';
import 'dart:typed_data';
import 'package:camera/camera.dart';

import 'package:image/image.dart';

class ImageUtils {
  static Float64List imageToFloatBuffer(
      Image image, List<double> mean, List<double> std,
      {bool contiguous = true}) {
    var bytes = Float64List(1 * image.height * image.width * 3);
    var buffer = Float64List.view(bytes.buffer);

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

    return bytes;
  }

  /// Converts a [CameraImage] in YUV420 format to [Image] in RGB format
  static Image? convertCameraImage(CameraImage cameraImage) {
    if (cameraImage.format.group == ImageFormatGroup.yuv420) {
      return convertYUV420ToImage(cameraImage);
    } else if (cameraImage.format.group == ImageFormatGroup.bgra8888) {
      return convertBGRA8888ToImage(cameraImage);
    } else {
      return null;
    }
  }

  static Image convertBGRA8888ToImage(CameraImage image) {
    return Image.fromBytes(
      width: image.width,
      height: image.height,
      bytes: image.planes[0].bytes.buffer,
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

  static Image convertYUV420ToImage(CameraImage image) {
    final uvRowStride = image.planes[1].bytesPerRow;
    final uvPixelStride = image.planes[1].bytesPerPixel ?? 0;
    final img = Image(width: image.width, height: image.height);
    for (final p in img) {
      final x = p.x;
      final y = p.y;
      final uvIndex =
          uvPixelStride * (x / 2).floor() + uvRowStride * (y / 2).floor();
      final index = y * uvRowStride +
          x; // Use the row stride instead of the image width as some devices pad the image data, and in those cases the image width != bytesPerRow. Using width will give you a distored image.
      final yp = image.planes[0].bytes[index];
      final up = image.planes[1].bytes[uvIndex];
      final vp = image.planes[2].bytes[uvIndex];
      p.r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255).toInt();
      p.g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
          .round()
          .clamp(0, 255)
          .toInt();
      p.b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255).toInt();
    }

    return img;
  }

  static Image? processCameraImage(CameraImage cameraImage) {
    Image? image = ImageUtils.convertCameraImage(cameraImage);

    if (Platform.isIOS) {
      // ios, default camera image is portrait view
      // rotate 270 to the view that top is on the left, bottom is on the right
      // image ^4.0.17 error here
      image = copyRotate(image!, angle: 270);
    }
    if (Platform.isAndroid) {
      // ios, default camera image is portrait view
      // rotate 270 to the view that top is on the left, bottom is on the right
      // image ^4.0.17 error here
      image = copyRotate(image!, angle: 90);
    }

    return image;
    // processImage(inputImage);
  }
}
