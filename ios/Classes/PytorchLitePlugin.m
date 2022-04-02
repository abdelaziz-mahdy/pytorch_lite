#import "PytorchLitePlugin.h"
#if __has_include(<pytorch_lite/pytorch_lite-Swift.h>)
#import <pytorch_lite/pytorch_lite-Swift.h>
#else
// Support project import fallback if the generated compatibility header
// is not copied when this plugin is created as a library.
// https://forums.swift.org/t/swift-static-libraries-dont-copy-generated-objective-c-header/19816
#import "pytorch_lite-Swift.h"
#endif

@implementation PytorchLitePlugin
+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar>*)registrar {
  [SwiftPytorchLitePlugin registerWithRegistrar:registrar];
}
@end
