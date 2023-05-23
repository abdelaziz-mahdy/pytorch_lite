#import "PytorchLitePlugin.h"
#import "pigeon.h"


@interface PytorchLitePlugin () <ModelApi>

@property (nonatomic, assign) BOOL enable;

@end

@implementation PytorchLitePlugin
+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar>*)registrar {
  PytorchLitePlugin* instance = [[PytorchLitePlugin alloc] init];
  ModelApiSetup(registrar.messenger, instance);
}



- (void)getImagePredictionListIndex:(nonnull NSNumber *)index imageData:(nullable FlutterStandardTypedData *)imageData imageBytesList:(nullable NSArray<FlutterStandardTypedData *> *)imageBytesList imageWidthForBytesList:(nullable NSNumber *)imageWidthForBytesList imageHeightForBytesList:(nullable NSNumber *)imageHeightForBytesList mean:(nonnull NSArray<NSNumber *> *)mean std:(nonnull NSArray<NSNumber *> *)std completion:(nonnull void (^)(NSArray<NSNumber *> * _Nullable, FlutterError * _Nullable))completion {
    <#code#>
}

- (void)getImagePredictionListObjectDetectionIndex:(nonnull NSNumber *)index imageData:(nullable FlutterStandardTypedData *)imageData imageBytesList:(nullable NSArray<FlutterStandardTypedData *> *)imageBytesList imageWidthForBytesList:(nullable NSNumber *)imageWidthForBytesList imageHeightForBytesList:(nullable NSNumber *)imageHeightForBytesList minimumScore:(nonnull NSNumber *)minimumScore IOUThreshold:(nonnull NSNumber *)IOUThreshold boxesLimit:(nonnull NSNumber *)boxesLimit completion:(nonnull void (^)(NSArray<ResultObjectDetection *> * _Nullable, FlutterError * _Nullable))completion {
    <#code#>
}

- (void)getPredictionCustomIndex:(nonnull NSNumber *)index input:(nonnull NSArray<NSNumber *> *)input shape:(nonnull NSArray<NSNumber *> *)shape dtype:(nonnull NSString *)dtype completion:(nonnull void (^)(NSArray * _Nullable, FlutterError * _Nullable))completion {
    <#code#>
}

- (nullable NSNumber *)loadModelModelPath:(nonnull NSString *)modelPath numberOfClasses:(nullable NSNumber *)numberOfClasses imageWidth:(nullable NSNumber *)imageWidth imageHeight:(nullable NSNumber *)imageHeight objectDetectionModelType:(ObjectDetectionModelType)objectDetectionModelType error:(FlutterError * _Nullable __autoreleasing * _Nonnull)error {
    <#code#>
}

@end
