import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'pytorch_lite_platform_interface.dart';

/// An implementation of [PytorchLitePlatform] that uses method channels.
class MethodChannelPytorchLite extends PytorchLitePlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('pytorch_lite');

  @override
  Future<String?> getPlatformVersion() async {
    final version =
        await methodChannel.invokeMethod<String>('getPlatformVersion');
    return version;
  }
}
