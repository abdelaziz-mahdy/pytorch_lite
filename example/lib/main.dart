import 'package:flutter/material.dart';
import 'package:pytorch_lite/native_wrapper.dart';
import 'package:pytorch_lite/pytorch_lite.dart';
import 'package:pytorch_lite_example/run_model_by_camera_demo.dart';
import 'package:pytorch_lite_example/run_model_by_image_demo.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await PytorchFfi.init();
  String pathImageModel = "assets/models/model_classification.pt";
  //String pathCustomModel = "assets/models/custom_model.ptl";
  String pathObjectDetectionModel = "assets/models/yolov5s.torchscript";
  String pathObjectDetectionModelYolov8 = "assets/models/yolov8s.torchscript";
  try {
    await PytorchLite.loadClassificationModel(pathImageModel, 224, 224, 1000,
        labelPath: "assets/labels/label_classification_imageNet.txt");
    //_customModel = await PytorchLite.loadCustomModel(pathCustomModel);
    await PytorchLite.loadObjectDetectionModel(
        pathObjectDetectionModel, 80, 640, 640,
        labelPath: "assets/labels/labels_objectDetection_Coco.txt");
    await PytorchLite.loadObjectDetectionModel(
        pathObjectDetectionModelYolov8, 80, 640, 640,
        labelPath: "assets/labels/labels_objectDetection_Coco.txt",
        objectDetectionModelType: ObjectDetectionModelType.yolov8);
  } catch (e) {
    print("Error is $e");
  }
  runApp(const ChooseDemo());
}

class ChooseDemo extends StatefulWidget {
  const ChooseDemo({Key? key}) : super(key: key);

  @override
  State<ChooseDemo> createState() => _ChooseDemoState();
}

class _ChooseDemoState extends State<ChooseDemo> {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('Pytorch Mobile Example'),
        ),
        body: Builder(builder: (context) {
          return Center(
            child: Column(
              children: [
                TextButton(
                  onPressed: () => {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                          builder: (context) => const RunModelByCameraDemo()),
                    )
                  },
                  style: TextButton.styleFrom(
                    backgroundColor: Colors.blue,
                  ),
                  child: const Text(
                    "Run Model with Camera",
                    style: TextStyle(
                      color: Colors.white,
                    ),
                  ),
                ),
                TextButton(
                  onPressed: () => {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                          builder: (context) => const RunModelByImageDemo()),
                    )
                  },
                  style: TextButton.styleFrom(
                    backgroundColor: Colors.blue,
                  ),
                  child: const Text(
                    "Run Model with Image",
                    style: TextStyle(
                      color: Colors.white,
                    ),
                  ),
                )
              ],
            ),
          );
        }),
      ),
    );
  }
}
