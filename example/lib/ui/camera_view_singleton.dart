import 'dart:ui';

/// Singleton to record size related data
class CameraViewSingleton {
  static double ratio = 0;
  static Size screenSize = const Size(0, 0);
  static Size inputImageSize = const Size(0, 0);
  static Size get actualPreviewSize =>
      Size(screenSize.width, screenSize.width * ratio);
  static Size get actualPreviewSizeH {
    return Size(screenSize.width, screenSize.width * (ratio));
  }
}
