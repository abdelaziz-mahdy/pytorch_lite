
## 4.2.3
* fixed export of enums and classes

## 4.2.2
* upgrading pytorch android from 1.13.1 to 2.1.0
* ios is still LibTorch 1.13.0.1 since its the last one

## 4.2.1
* fix classification bug on android on native preprocessing
* Making model output dynamically calculated (yolov5,yolov8) to allow input sizes other than 640X640

## 4.2.0+2
* Fix formatting

## 4.2.0+1
* Adding license
* Adding docstring for all functions
## 4.2.0
* Fixing ios camera image decoding
* Converting package to use native methods instead of ffi (fixing ios)
* Adding parameters to allow choosing between imageLib or native preprocessing 
* Adding better integration testing 

## 4.0.0
* image processing is done using opencv to improve performance
* camera image decoding is done using opencv to improve performance
* camera images can be ran using getCameraImage* methods
* improving the performance of object detection by moving all the processing to isolate
* Fixed: CAMERA image boxes are shifted upwards in example
* Improving example
## 3.0.4
* improved speed of inference
* improve build speed on android
## 3.0.3
* windows extracting using tar instead of powershell since powershell is so slow
## 3.0.2
* improving cmake for windows extraction (thanks https://github.com/ZhaoXinZhang for pointing it out).
## 3.0.1
* fix android build on Windows.
## 3.0.0+1
* Converted to work on ffi (for improved performance, and ios support)
* Improved isolates to stop ui frame drops
* Added ImageUtilsIsolate to process camera images in isolate instead of ui thread
* Updated Camera example
* Better memory usage and freeing
* Fixed camera example usage
* Breaking changes
* getImagePredictionFromBytesList Removed, (check camera example for new usage)


## 2.0.5
* fixed dart analyses problems (renamed some variables to follow convention)
## 2.0.4
* removed unnecessary prints
## 2.0.3
* fixed yolov8 bad performance
## 2.0.2
* com.facebook.soloader:nativeloader from 0.8.0 to 0.10.5
## 2.0.1
* upgraded to pigeon 9.0.0
* adding yolov8 support, thanks to https://github.com/atanasko/android-demo-app
* updated camera,image to latest version in example
* converted to using pytorch_android version to 1.12
## 2.0.0
* breaking change: Rect is now PyTorchRect to avoid conflicts in ios (when it is implemented)
* upgraded to pigeon 4.2.0
## 1.1.2
* Updated pytorch_android_lite version to 1.12.2
## 1.1.1
* Updated pytorch_android_lite version to 1.11
## 1.1.0
* Added get prediction from bytes list to run on camera image
* Added example to camera image prediction (thanks to KingWu)
## 1.0.6
* Updated dependencies 
* Fixed release not working and Added to readme the solution 
## 1.0.5
* Made all functions take image bytes instead of image file (to avoid storing stream in storage)
## 1.0.4
* Fixed a bug (probabilities were wrong)
## 1.0.3
* Added getting classification as probabilities
## 1.0.2
* Initial release
## 1.0.1
* Making code run in background to avoid frame drops
## 1.0.0
* Made some optimizations
## 0.0.4
* implemented object detection native code
## 0.0.3
* Upgraded dependencies and made some optimizations 
## 0.0.2
* Used base code from https://pub.dev/packages/pytorch_mobile/changelog
## 0.0.1
* Setup pigeon messages
