enum ObjectDetectionModelType {
  yolov5,
  yolov8,
}

enum PreProcessingMethod {
  /// Uses the imageLib library for preprocessing.
  /// This method is less performant but ensures the same results on Android and iOS.
  imageLib,

  /// Uses the native method for preprocessing.
  /// This method is faster but may result in small changes in the preprocessing results.
  native,
}

enum CameraPreProcessingMethod {
  /// Handles Android and iOS correctly.
  imageLib,

  /// Only tested on Android.
  byteList,
}
