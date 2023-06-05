import 'package:pytorch_lite/classes/rect.dart';

class ResultObjectDetection {
  ResultObjectDetection({
    required this.classIndex,
    this.className,
    required this.score,
    required this.rect,
  });

  int classIndex;

  String? className;

  double score;

  PyTorchRect rect;


}