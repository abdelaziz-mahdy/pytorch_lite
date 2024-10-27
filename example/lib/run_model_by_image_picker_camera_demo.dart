import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:pytorch_lite/pytorch_lite.dart';
import 'package:pytorch_lite_example/ui/box_widget.dart';

class RunModelByImagePickerCameraDemo extends StatefulWidget {
  const RunModelByImagePickerCameraDemo({Key? key}) : super(key: key);

  @override
  _RunModelByImagePickerCameraDemoState createState() =>
      _RunModelByImagePickerCameraDemoState();
}

class _RunModelByImagePickerCameraDemoState
    extends State<RunModelByImagePickerCameraDemo> {
  List<ResultObjectDetection>? objectDetectionResults;
  String? classificationResult;
  Duration? objectDetectionInferenceTime;
  Duration? classificationInferenceTime;
  File? _image;
  ModelObjectDetection? _objectModel;
  ClassificationModel? _imageModel;
  bool _isLoading = false; // Add loading state

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  Future loadModel() async {
    String pathImageModel = "assets/models/model_classification.pt";
    String pathObjectDetectionModel = "assets/models/yolov5s.torchscript";
    try {
      _imageModel = await PytorchLite.loadClassificationModel(
        pathImageModel, 224, 224, 1000, // Adjust as needed
        labelPath: "assets/labels/label_classification_imageNet.txt",
      );
      _objectModel = await PytorchLite.loadObjectDetectionModel(
        pathObjectDetectionModel,
        80,
        640,
        640,
        labelPath: "assets/labels/labels_objectDetection_Coco.txt",
      );
    } catch (e) {
      print("Error loading model: $e");
    }
  }

  Future runModels() async {
    setState(() => _isLoading = true);

    final ImagePicker picker = ImagePicker();
    final XFile? pickedImage =
        await picker.pickImage(source: ImageSource.camera);
    if (pickedImage == null) {
      setState(() => _isLoading = false);
      return;
    }

    File image = File(pickedImage.path);
    Uint8List imageBytes = await image.readAsBytes(); // Read bytes once

    // Run both models concurrently
    final results = await Future.wait([
      () async {
        Stopwatch stopwatch = Stopwatch()..start();
        try {
          return await _imageModel?.getImagePrediction(imageBytes);
        } catch (e) {
          print("Error during classification: $e");
          return null; // or handle the error as needed
        } finally {
          classificationInferenceTime = stopwatch.elapsed;
        }
      }(),
      () async {
        Stopwatch stopwatch = Stopwatch()..start();
        try {
          return await _objectModel?.getImagePrediction(
            imageBytes,
            minimumScore: 0.1,
            iOUThreshold: 0.3,
          );
        } catch (e) {
          print("Error during object detection: $e");
          return null; // or handle the error as needed
        } finally {
          objectDetectionInferenceTime = stopwatch.elapsed;
        }
      }(),
    ]);

    classificationResult = results[0] as String?;
    objectDetectionResults = results[1] as List<ResultObjectDetection>?;

    setState(() {
      _image = image;
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Run Models')),
      body: Center(
        child: _isLoading
            ? const CircularProgressIndicator() // Show loading indicator
            : Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  if (_image != null) ...[
                    SizedBox(
                      height: MediaQuery.sizeOf(context).height * 0.5,
                      child: Padding(
                        padding: const EdgeInsets.all(20),
                        child: _objectModel!.renderBoxesOnImage(
                            _image!, objectDetectionResults ?? []),
                      ),
                    ),
                    const SizedBox(height: 20),
                    Text(
                      "Classification Result: ${classificationResult ?? "N/A"}",
                      style: const TextStyle(fontSize: 16),
                    ),
                    Text(
                      "Classification Time: ${classificationInferenceTime?.inMilliseconds ?? "N/A"} ms",
                      style: const TextStyle(fontSize: 16),
                    ),
                    Text(
                      "Object Detection Time: ${objectDetectionInferenceTime?.inMilliseconds ?? "N/A"} ms",
                      style: const TextStyle(fontSize: 16),
                    ),
                    const SizedBox(height: 20),
                  ],
                  ElevatedButton(
                    onPressed: runModels,
                    child: const Text('Take Photo & Run Models'),
                  ),
                ],
              ),
      ),
    );
  }
}
