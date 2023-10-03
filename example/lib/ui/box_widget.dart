import 'package:flutter/material.dart';
import 'package:pytorch_lite/pigeon.dart';
import 'package:pytorch_lite/pytorch_lite.dart';
import 'package:pytorch_lite_example/ui/camera_view_singleton.dart';

/// Individual bounding box
class BoxWidget extends StatelessWidget {
  final ResultObjectDetection result;
  final Color? boxesColor;
  final bool showPercentage;
  const BoxWidget(
      {Key? key,
      required this.result,
      this.boxesColor,
      this.showPercentage = true})
      : super(key: key);
  @override
  Widget build(BuildContext context) {
    // Color for bounding box
    //print(MediaQuery.of(context).size);
    Color? usedColor;
    //Size screenSize = CameraViewSingleton.inputImageSize;
    Size screenSize = CameraViewSingleton.actualPreviewSizeH;
    //Size screenSize = MediaQuery.of(context).size;

    //print(screenSize);
    double factorX = screenSize.width;
    double factorY = screenSize.height;
    if (boxesColor == null) {
      //change colors for each label
      usedColor = Colors.primaries[
          ((result.className ?? result.classIndex.toString()).length +
                  (result.className ?? result.classIndex.toString())
                      .codeUnitAt(0) +
                  result.classIndex) %
              Colors.primaries.length];
    } else {
      usedColor = boxesColor;
    }

    return Positioned(
      left: result.rect.left * factorX,
      top: result.rect.top * factorY,
      width: result.rect.width * factorX,
      height: result.rect.height * factorY,

      //left: re?.rect.left.toDouble(),
      //top: re?.rect.top.toDouble(),
      //right: re.rect.right.toDouble(),
      //bottom: re.rect.bottom.toDouble(),
      child: Container(
        width: result.rect.width * factorX,
        height: result.rect.height * factorY,
        decoration: BoxDecoration(
            border: Border.all(color: usedColor!, width: 3),
            borderRadius: const BorderRadius.all(Radius.circular(2))),
        child: Align(
          alignment: Alignment.topLeft,
          child: FittedBox(
            child: Container(
              color: usedColor,
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: <Widget>[
                  Text(result.className ?? result.classIndex.toString()),
                  Text(" ${result.score.toStringAsFixed(2)}"),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
