import 'package:pytorch_lite/pigeon.dart';

import 'dart:convert';

extension ResultObjectDetectionExtension on ResultObjectDetection {
  // Convert object to map
  Map<String, dynamic> toMap() {
    return {
      'classIndex': classIndex,
      'className': className,
      'score': score,
      'rect': rect.toMap(),
    };
  }

  // Create object from map
  static ResultObjectDetection fromMap(Map<String, dynamic> map) {
    return ResultObjectDetection(
      classIndex: map['classIndex'],
      score: map['score'],
      rect: PyTorchRectExtension.fromMap(map['rect']),
    )..className = map['className'];
  }

  // Convert object to JSON string
  String toJson() {
    return jsonEncode(toMap());
  }

  // Create object from JSON string
  static ResultObjectDetection fromJson(String json) {
    return fromMap(jsonDecode(json));
  }
}

extension PyTorchRectExtension on PyTorchRect {
  // Convert object to map
  Map<String, dynamic> toMap() {
    return {
      'left': left,
      'top': top,
      'right': right,
      'bottom': bottom,
      'width': width,
      'height': height,
    };
  }

  // Create object from map
  static PyTorchRect fromMap(Map<String, dynamic> map) {
    return PyTorchRect(
      left: map['left'],
      top: map['top'],
      width: map['width'],
      height: map['height'],
      right: map['right'],
      bottom: map['bottom'],
    );
  }
}
