#import "PytorchLitePlugin.h"
#import "pigeon.h"
#import "PrePostProcessor.h"
//#import "TorchModule.h"
//#import "UIImageExtension.h"
#import <LibTorch/LibTorch.h>
#include <vector>

@interface PytorchLitePlugin () <ModelApi>

@property (nonatomic, assign) std::vector<torch::jit::Module*> modulesVector;
@property (nonatomic, strong) NSMutableArray<PrePostProcessor *> *prePostProcessors;

@end

@implementation PytorchLitePlugin


+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar>*)registrar {
  PytorchLitePlugin* instance = [[PytorchLitePlugin alloc] init];
  ModelApiSetup(registrar.messenger, instance);
    // instance.modulesVector = [NSMutableArray array];
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
- (nullable NSNumber *)loadModelModelPath:(nonnull NSString *)modelPath numberOfClasses:(nullable NSNumber *)numberOfClasses imageWidth:(nullable NSNumber *)imageWidth imageHeight:(nullable NSNumber *)imageHeight isObjectDetection:(nullable NSNumber *)isObjectDetection objectDetectionModelType:(nullable NSNumber *)objectDetectionModelType error:(FlutterError * _Nullable __autoreleasing * _Nonnull)error {
    
    NSInteger i = -1;
    @try {
        torch::jit::Module *module = new torch::jit::Module(torch::jit::load(modelPath.UTF8String));
        self.modulesVector.push_back(module);
        
if (numberOfClasses != nil && imageWidth != nil && imageHeight != nil) {
            [ self.prePostProcessors addObject:[[PrePostProcessor alloc] initWithNumberOfClasses:numberOfClasses.integerValue imageWidth:imageWidth.integerValue imageHeight:imageHeight.integerValue objectDetectionModelType:objectDetectionModelType.integerValue]];
        } else {
            if (imageWidth != nil && imageHeight != nil) {
                [ self.prePostProcessors addObject:[[PrePostProcessor alloc] initWithImageWidth:imageWidth.integerValue imageHeight:imageHeight.integerValue]];
            } else {
                [ self.prePostProcessors addObject:[[PrePostProcessor alloc] init]];
            }
        }
        i = self.modulesVector.size() - 1;
    } @catch (NSException *e) {
        NSLog(@"%@ is not a proper model: %@", modelPath, e);
        if (error != NULL) {
            *error = [FlutterError errorWithCode:@"ModelLoadingError" message:[NSString stringWithFormat:@"%@ is not a proper model", modelPath] details:e];
        }
    }

    return [NSNumber numberWithInteger:i];
}


@end
