import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

Pointer<Uint8> convertUint8ListToPointer(Uint8List data) {
  int length = data.length;
  Pointer<Uint8> dataPtr = calloc<Uint8>(length);

  for (int i = 0; i < length; i++) {
    dataPtr.elementAt(i).value = data[i];
  }

  return dataPtr;
}

Pointer<UnsignedChar> convertUint8ListToPointerChar(Uint8List data) {
  final Pointer<Uint8> frameData = calloc<Uint8>(data.length);
  final pointerList = frameData.asTypedList(data.length);
  pointerList.setAll(0, data);

  return frameData.cast<UnsignedChar>();
}

Pointer<Float> convertDoubleListToPointerFloat(List<double> data) {
  final Pointer<Float> frameData = calloc<Float>(data.length);
  final pointerList = frameData.asTypedList(data.length);
  pointerList.setAll(0, data);

  return frameData;
}

Pointer<Float> convertListToPointer(List<double> floatList) {
  // Create a native array to hold the double values
  final nativeArray = calloc<Double>(floatList.length);

  // Copy the values from the list to the native array
  for (var i = 0; i < floatList.length; i++) {
    nativeArray[i] = floatList[i];
  }

  // Obtain the pointer to the native array
  final nativePointer = nativeArray.cast<Float>();

  return nativePointer;
}
