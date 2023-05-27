#!/bin/bash

echo "flutter clean ..."
flutter clean
echo "Deleting .flutter-plugins ..."
rm -rf .flutter-plugins
echo "Deleting .packages ..."
rm -rf .packages
echo "Deleting .symlinks ..."
rm -rf ios/.symlinks/
echo "Deleting build/ ..."
rm -rf build/
echo "Deleting ios/Pods ..."
rm -rf ios/Pods
echo "Deleting ios/Podfile* ..."
rm ios/Podfile*
echo "Deleting .pub-cache ..."
rm -rf "${HOME}/.pub-cache/"
echo "Deleting pubspec.lock ..."
rm pubspec.lock
echo "Deleting ios/Runner.xcodeproj/project.xcworkspace ..."
rm -rf ios/Runner.xcodeproj/project.xcworkspace
echo "Deleting Library/Caches/CocoaPods ..."
rm -rf "${HOME}/Library/Caches/CocoaPods"
echo "Deleting DerivedData ..."
rm -rf "~/Library/Developer/Xcode/DerivedData"
echo "Running flutter packages get ..."
flutter packages get
echo "Done."