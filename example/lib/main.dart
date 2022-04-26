import 'package:flutter/material.dart';
import 'dart:async';

import 'package:flutter/services.dart';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:pytorch_lite/pigeon.dart';
import 'package:pytorch_lite/pytorch_lite.dart';

void main() => runApp(MyApp());

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  ClassificationModel? _imageModel;
  //CustomModel? _customModel;
  late ModelObjectDetection _objectModel;
  String? _imagePrediction;
  List? _prediction;
  File? _image;
  ImagePicker _picker = ImagePicker();
  bool objectDetection = false;
  List<ResultObjectDetection?> objDetect = [];
  @override
  void initState() {
    super.initState();
    loadModel();
  }

  //load your model
  Future loadModel() async {
    String pathImageModel = "assets/models/model.ptl";
    //String pathCustomModel = "assets/models/custom_model.ptl";
    String pathObjectDetectionModel = "assets/models/best (2).torchscript";
    try {
      _imageModel =
          await PytorchLite.loadClassificationModel(pathImageModel, 224, 224);
      //_customModel = await PytorchLite.loadCustomModel(pathCustomModel);
      _objectModel = await PytorchLite.loadObjectDetectionModel(
          pathObjectDetectionModel, 17, 640, 640);
    } on PlatformException {
      print("only supported for android and ios so far");
    }
  }

  //run an image model
  Future runImageModel() async {
    //pick a random image
    final PickedFile? image = await _picker.getImage(
        source: ImageSource.gallery, maxHeight: 512, maxWidth: 512);
    //get prediction
    //labels are 1000 random english words for show purposes
    _imagePrediction = await _imageModel!
        .getImagePrediction(File(image!.path), "assets/labels/labels_skin.csv");
    print(await _imageModel!.getImagePredictionList(
      File(image.path),
    ));

    List<ResultObjectDetection?> objDetect =
        await _objectModel.getImagePredictionList(File(image.path));
    objDetect.forEach((element) {
      print({
        "score": element?.score,
        "className": element?.className,
        "class": element?.classIndex,
        "rect": element?.rect,
      });
    });

    objDetect = await _objectModel.getImagePrediction(
        File(image.path), "assets/labels/labelmap.txt",
        minimumScore: 0.1, IOUThershold: 0.3);
    objDetect.forEach((element) {
      print({
        "score": element?.score,
        "className": element?.className,
        "class": element?.classIndex,
        "rect": {
          "left": element?.rect.left,
          "top": element?.rect.top,
          "width": element?.rect.width,
          "height": element?.rect.height,
        },
      });
    });
    setState(() {
      this.objDetect = objDetect;
      _image = File(image.path);
    });
  }

/*
  //run a custom model with number inputs
  Future runCustomModel() async {
    _prediction = await _customModel!
        .getPrediction([1, 2, 3, 4], [1, 2, 2], DType.float32);

    setState(() {});
  }
*/
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('Pytorch Mobile Example'),
        ),
        body: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Expanded(
              child: objDetect != null
                  ? _image == null
                      ? Text('No image selected.')
                      : _objectModel.renderBoxesOnImage(_image!, objDetect)
                  : _image == null
                      ? Text('No image selected.')
                      : Image.file(_image!),
            ),
            Center(
              child: Visibility(
                visible: _imagePrediction != null,
                child: Text("$_imagePrediction"),
              ),
            ),
            Center(
              child: TextButton(
                onPressed: runImageModel,
                child: Icon(
                  Icons.add_a_photo,
                  color: Colors.grey,
                ),
              ),
            ),
            /*
            TextButton(
              onPressed: runCustomModel,
              style: TextButton.styleFrom(
                backgroundColor: Colors.blue,
              ),
              child: Text(
                "Run custom model",
                style: TextStyle(
                  color: Colors.white,
                ),
              ),
            ),

             */
            Center(
              child: Visibility(
                visible: _prediction != null,
                child: Text(_prediction != null ? "${_prediction![0]}" : ""),
              ),
            )
          ],
        ),
      ),
    );
  }
}
