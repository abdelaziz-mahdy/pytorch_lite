#import "PytorchLitePlugin.h"
#import "pigeon.h"
#import "PrePostProcessor.h"
//#import "TorchModule.h"
//#import "UIImageExtension.h"
#import <LibTorch/LibTorch.h>

@interface PytorchLitePlugin () <ModelApi>

@property (nonatomic, strong) NSMutableArray<Module *> *modules;
@property (nonatomic, strong) NSMutableArray<PrePostProcessor *> *prePostProcessors;

@end

@implementation PytorchLitePlugin


+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar>*)registrar {
  PytorchLitePlugin* instance = [[PytorchLitePlugin alloc] init];
  ModelApiSetup(registrar.messenger, instance);
    instance.modules = [NSMutableArray array];
    instance.prePostProcessors = [NSMutableArray array];
}


- (void)getImagePredictionListIndex:(nonnull NSNumber *)index imageData:(nullable FlutterStandardTypedData *)imageData imageBytesList:(nullable NSArray<FlutterStandardTypedData *> *)imageBytesList imageWidthForBytesList:(nullable NSNumber *)imageWidthForBytesList imageHeightForBytesList:(nullable NSNumber *)imageHeightForBytesList mean:(nonnull NSArray<NSNumber *> *)mean std:(nonnull NSArray<NSNumber *> *)std completion:(nonnull void (^)(NSArray<NSNumber *> * _Nullable, FlutterError * _Nullable))completion {
    // <#code#>
}

- (void)getImagePredictionListObjectDetectionIndex:(nonnull NSNumber *)index imageData:(nullable FlutterStandardTypedData *)imageData imageBytesList:(nullable NSArray<FlutterStandardTypedData *> *)imageBytesList imageWidthForBytesList:(nullable NSNumber *)imageWidthForBytesList imageHeightForBytesList:(nullable NSNumber *)imageHeightForBytesList minimumScore:(nonnull NSNumber *)minimumScore IOUThreshold:(nonnull NSNumber *)IOUThreshold boxesLimit:(nonnull NSNumber *)boxesLimit completion:(nonnull void (^)(NSArray<ResultObjectDetection *> * _Nullable, FlutterError * _Nullable))completion {
    // <#code#>
}

- (void)getPredictionCustomIndex:(nonnull NSNumber *)index input:(nonnull NSArray<NSNumber *> *)input shape:(nonnull NSArray<NSNumber *> *)shape dtype:(nonnull NSString *)dtype completion:(nonnull void (^)(NSArray * _Nullable, FlutterError * _Nullable))completion {
    // <#code#>
}

- (nullable NSNumber *)loadModelModelPath:(nonnull NSString *)modelPath numberOfClasses:(nullable NSNumber *)numberOfClasses imageWidth:(nullable NSNumber *)imageWidth imageHeight:(nullable NSNumber *)imageHeight objectDetectionModelType:(ObjectDetectionModelType)objectDetectionModelType error:(FlutterError * _Nullable __autoreleasing * _Nonnull)error {
    
    NSInteger i = -1;
    @try {
        // [modules addObject:[LiteModuleLoader load:modelPath]];
        [modules addObject:[Module load:modelPath]];
        if (numberOfClasses != nil && imageWidth != nil && imageHeight != nil) {
            [prePostProcessors addObject:[[PrePostProcessor alloc] initWithNumberOfClasses:numberOfClasses.integerValue imageWidth:imageWidth.integerValue imageHeight:imageHeight.integerValue objectDetectionModelType:objectDetectionModelType]];
        } else {
            if (imageWidth != nil && imageHeight != nil) {
                [prePostProcessors addObject:[[PrePostProcessor alloc] initWithImageWidth:imageWidth.integerValue imageHeight:imageHeight.integerValue]];
            } else {
                [prePostProcessors addObject:[[PrePostProcessor alloc] init]];
            }
        }
        i = [modules count] - 1;
    } @catch (NSException *e) {
        NSLog(@"%@ is not a proper model: %@", modelPath, e);
        if (error != NULL) {
            *error = [FlutterError errorWithCode:@"ModelLoadingError" message:[NSString stringWithFormat:@"%@ is not a proper model", modelPath] details:e];
        }
    }

    return [NSNumber numberWithInteger:i];
}


@end
