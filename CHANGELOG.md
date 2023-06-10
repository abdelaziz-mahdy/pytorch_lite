## 3.0.0-alpha2
* Better memory usage and freeing
* Fixed yolov8 infinite width boxes
* Fixed camera example usage
## 3.0.0-alpha
* Converted to work on ffi (for improved performance, and ios support)
* ios support
* camera not working as expected yet

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
