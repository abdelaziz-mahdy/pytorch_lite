import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pytorch_lite/pytorch_lite.dart';

void main() {
  const MethodChannel channel = MethodChannel('pytorch_lite');

  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() {
    channel.setMockMethodCallHandler((MethodCall methodCall) async {
      return '42';
    });
  });

  tearDown(() {
    channel.setMockMethodCallHandler(null);
  });

  test('getPlatformVersion', () async {
    expect(await PytorchLite.platformVersion, '42');
  });
}
