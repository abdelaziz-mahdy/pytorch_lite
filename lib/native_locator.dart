import 'dart:ffi';
import 'dart:io';
import 'package:path/path.dart' as p;

// /// A very short-lived native function.
// ///
// /// For very short-lived functions, it is fine to call them on the main isolate.
// /// They will block the Dart execution while running the native function, so
// /// only do this for native functions which are guaranteed to be short-lived.
// int sum(int a, int b) => _bindings.sum(a, b);

// /// A longer lived native function, which occupies the thread calling it.
// ///
// /// Do not call these kind of native functions in the main isolate. They will
// /// block Dart execution. This will cause dropped frames in Flutter applications.
// /// Instead, call these native functions on a separate isolate.
// ///
// /// Modify this to suit your own use case. Example use cases:
// ///
// /// 1. Reuse a single isolate for various different kinds of requests.
// /// 2. Use multiple helper isolates for parallel execution.
// Future<int> sumAsync(int a, int b) async {
//   final SendPort helperIsolateSendPort = await _helperIsolateSendPort;
//   final int requestId = _nextSumRequestId++;
//   final _SumRequest request = _SumRequest(requestId, a, b);
//   final Completer<int> completer = Completer<int>();
//   _sumRequests[requestId] = completer;
//   helperIsolateSendPort.send(request);
//   return completer.future;
// }

const String _libName = 'pytorch_lite';

/// The dynamic library in which the symbols for [FfigenAppBindings] can be found.
final DynamicLibrary dylib = () {
  // return DynamicLibrary.executable();
  // print(Directory.current.listSync());
  if (Platform.isMacOS || Platform.isIOS) {
    return DynamicLibrary.executable();
    // // Add from here...
    // if (Platform.environment.containsKey('FLUTTER_TEST')) {
    //   return DynamicLibrary.open('build/macos/Build/Products/Debug'
    //       '/$_libName/$_libName.framework/$_libName');
    // }
    // // // ...to here.
    // return DynamicLibrary.open('$_libName.framework/$_libName');
  }
  if (Platform.isAndroid || Platform.isLinux) {
    // Add from here...
    if (Platform.environment.containsKey('FLUTTER_TEST')) {
      return DynamicLibrary.open(
          'build/linux/x64/debug/bundle/lib/lib$_libName.so');
    }
    // ...to here.
    return DynamicLibrary.open('lib$_libName.so');
  }
  if (Platform.isWindows) {
    // Add from here...
    if (Platform.environment.containsKey('FLUTTER_TEST')) {
      return DynamicLibrary.open(p.canonicalize(
          p.join(r'build\windows\runner\Debug', '$_libName.dll')));
    }
    // ...to here.
    return DynamicLibrary.open('$_libName.dll');
  }
  throw UnsupportedError('Unknown platform: ${Platform.operatingSystem}');
}();
