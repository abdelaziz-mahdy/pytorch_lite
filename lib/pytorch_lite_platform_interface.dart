import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'pytorch_lite_method_channel.dart';

abstract class PytorchLitePlatform extends PlatformInterface {
  /// Constructs a PytorchLitePlatform.
  PytorchLitePlatform() : super(token: _token);

  static final Object _token = Object();

  static PytorchLitePlatform _instance = MethodChannelPytorchLite();

  /// The default instance of [PytorchLitePlatform] to use.
  ///
  /// Defaults to [MethodChannelPytorchLite].
  static PytorchLitePlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [PytorchLitePlatform] when
  /// they register themselves.
  static set instance(PytorchLitePlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  Future<String?> getPlatformVersion() {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }
}
