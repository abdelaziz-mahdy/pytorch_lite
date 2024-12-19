import 'package:flutter/material.dart';
import 'package:pytorch_lite/pytorch_lite.dart';
import 'package:pytorch_lite_example/ui/box_widget.dart';

import 'ui/camera_view.dart';

/// [RunModelByCameraDemo] stacks [CameraView] and [BoxWidget]s with bottom sheet for stats
class RunModelByCameraDemo extends StatefulWidget {
  const RunModelByCameraDemo({Key? key}) : super(key: key);

  @override
  RunModelByCameraDemoState createState() => RunModelByCameraDemoState();
}

class RunModelByCameraDemoState extends State<RunModelByCameraDemo> {
  List<ResultObjectDetection>? results;
  Duration? objectDetectionInferenceTime;

  String? classification;
  Duration? classificationInferenceTime;

  /// Scaffold Key
  GlobalKey<ScaffoldState> scaffoldKey = GlobalKey();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      key: scaffoldKey,
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('Run model with Camera'),
      ),
      body: Stack(
        children: <Widget>[
          // Camera View
          CameraView(resultsCallback, resultsCallbackClassification),

          // Bounding boxes
          boundingBoxes2(results),

          // Heading
          // Align(
          //   alignment: Alignment.topLeft,
          //   child: Container(
          //     padding: EdgeInsets.only(top: 20),
          //     child: Text(
          //       'Object Detection Flutter',
          //       textAlign: TextAlign.left,
          //       style: TextStyle(
          //         fontSize: 28,
          //         fontWeight: FontWeight.bold,
          //         color: Colors.deepOrangeAccent.withOpacity(0.6),
          //       ),
          //     ),
          //   ),
          // ),

          //Bottom Sheet
          Align(
            alignment: Alignment.bottomCenter,
            child: DraggableScrollableSheet(
              initialChildSize: 0.4,
              minChildSize: 0.1,
              maxChildSize: 0.5,
              builder: (_, ScrollController scrollController) => Container(
                width: double.maxFinite,
                decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.9),
                    borderRadius: const BorderRadius.only(
                        topLeft: Radius.circular(24.0),
                        topRight: Radius.circular(24.0))),
                child: SingleChildScrollView(
                  controller: scrollController,
                  child: Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(Icons.keyboard_arrow_up,
                            size: 48, color: Colors.orange),
                        Padding(
                          padding: const EdgeInsets.all(8.0),
                          child: Column(
                            children: [
                              if (classification != null)
                                StatsRow('Classification:', '$classification'),
                              if (classificationInferenceTime != null)
                                StatsRow('Classification Inference time:',
                                    '${classificationInferenceTime?.inMilliseconds} ms'),
                              if (objectDetectionInferenceTime != null)
                                StatsRow('Object Detection Inference time:',
                                    '${objectDetectionInferenceTime?.inMilliseconds} ms'),
                            ],
                          ),
                        )
                      ],
                    ),
                  ),
                ),
              ),
            ),
          )
        ],
      ),
    );
  }

  /// Returns Stack of bounding boxes
  Widget boundingBoxes2(List<ResultObjectDetection>? results) {
    if (results == null) {
      return Container();
    }
    return Stack(
      children: results.map((e) => BoxWidget(result: e)).toList(),
    );
  }

  void resultsCallback(
      List<ResultObjectDetection> results, Duration inferenceTime) {
    if (!mounted) {
      return;
    }
    setState(() {
      this.results = results;
      objectDetectionInferenceTime = inferenceTime;
      for (var element in results) {
        print({
          "rect": {
            "left": element.rect.left,
            "top": element.rect.top,
            "width": element.rect.width,
            "height": element.rect.height,
            "right": element.rect.right,
            "bottom": element.rect.bottom,
          },
        });
      }
    });
  }

  void resultsCallbackClassification(
      String classification, Duration inferenceTime) {
    if (!mounted) {
      return;
    }
    setState(() {
      this.classification = classification;
      classificationInferenceTime = inferenceTime;
    });
  }

  // static const BOTTOM_SHEET_RADIUS = Radius.circular(24.0);
  // static const BORDER_RADIUS_BOTTOM_SHEET = BorderRadius.only(
  //     topLeft: BOTTOM_SHEET_RADIUS, topRight: BOTTOM_SHEET_RADIUS);
}

/// Row for one Stats field
class StatsRow extends StatelessWidget {
  final String title;
  final String value;

  const StatsRow(this.title, this.value, {Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8.0),
      child: Column(
        // mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            title,
            style: const TextStyle(fontWeight: FontWeight.bold),
          ),
          Text(value)
        ],
      ),
    );
  }
}
