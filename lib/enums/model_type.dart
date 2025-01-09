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

/// Enum representing the location of the model.
///
/// `asset` indicates that the model is located in the application's assets.
///
/// `path` indicates that the model is located at a specific file path.
enum ModelLocation {
  /// `asset` indicates that the model is located in the application's assets.
  asset,

  /// `path` indicates that the model is located at a specific file path.
  path
}

/// Enum representing the location of the labels.
///
/// `asset` indicates that the labels are located in the application's assets.
///
/// `path` indicates that the labels are located at a specific file path.
enum LabelsLocation {
  /// `asset` indicates that the model is located in the application's assets.
  asset,

  /// `path` indicates that the model is located at a specific file path.
  path
}
