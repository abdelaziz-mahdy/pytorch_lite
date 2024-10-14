#import "PytorchLitePlugin.h"
#import "pigeon.h"
#import "PrePostProcessor.h"
//#import "TorchModule.h"
#import "helpers/UIImageExtension.h"
#import <LibTorch/LibTorch.h>
// #import <Libtorch-Lite/Libtorch-Lite.h>


@interface PytorchLitePlugin () <ModelApi>

@property (nonatomic, assign) std::vector<torch::jit::Module*> modulesVector;
@property (nonatomic, strong) NSMutableArray<PrePostProcessor *> *prePostProcessors;

@end

@implementation PytorchLitePlugin

+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar>*)registrar {
    PytorchLitePlugin* instance = [[PytorchLitePlugin alloc] init];
    SetUpModelApi(registrar.messenger, instance);
    instance.prePostProcessors = [NSMutableArray array];
}

- (void)getPredictionCustomIndex:(NSInteger)index input:(NSArray<NSNumber *> *)input shape:(NSArray<NSNumber *> *)shape dtype:(NSString *)dtype completion:(void (^)(NSArray<id> *_Nullable, FlutterError *_Nullable))completion {
    // Implement custom prediction logic here based on 'input', 'shape', and 'dtype'.
    // This is a placeholder, replace with your actual implementation.
    completion(nil, nil);
}


- (NSArray<NSNumber*>*)predictImage:(void*)imageBuffer withWidth:(int)width andHeight:(int)height atIndex:(NSInteger)moduleIndex isObjectDetection:(BOOL)isObjectDetection objectDetectionType:(NSInteger)objectDetectionType {
    try {
        torch::jit::Module* module = _modulesVector[moduleIndex];
        at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, height, width}, at::kFloat);

        torch::autograd::AutoGradMode guard(false);
        at::AutoNonVariableTypeMode non_var_type_mode(true);

        at::Tensor outputTensor;

        if (isObjectDetection) {
            if (objectDetectionType == 0) { // YOLO
                torch::jit::IValue outputTuple = module->forward({tensor}).toTuple();
                outputTensor = outputTuple.toTuple()->elements()[0].toTensor();
            } else { // SSD & other single output models
                outputTensor = module->forward({tensor}).toTensor();
            }
        } else { // Classification, Segmentation, etc.
            outputTensor = module->forward({tensor}).toTensor();
        }

        float *floatBuffer = outputTensor.data_ptr<float>();
        if (!floatBuffer) {
            return nil;
        }

        int prod = 1;
        for (int i = 0; i < outputTensor.sizes().size(); i++) {
            prod *= outputTensor.sizes().data()[i];
        }

        NSMutableArray<NSNumber*>* results = [[NSMutableArray<NSNumber*> alloc] init];
        for (int i = 0; i < prod; i++) {
            [results addObject: @(floatBuffer[i])];
        }

        return [results copy];

    } catch (const std::exception& e) {
        NSLog(@"%s", e.what());
        return nil;
    }
}




- (void)loadModelModelPath:(NSString *)modelPath numberOfClasses:(nullable NSNumber *)numberOfClasses imageWidth:(nullable NSNumber *)imageWidth imageHeight:(nullable NSNumber *)imageHeight objectDetectionModelType:(nullable NSNumber *)objectDetectionModelType completion:(void (^)(NSNumber *_Nullable, FlutterError *_Nullable))completion{
    NSInteger i = -1;
    try {
        torch::jit::Module *module = new torch::jit::Module(torch::jit::load(modelPath.UTF8String));
        _modulesVector.push_back(module);

if (numberOfClasses != nil && imageWidth != nil && imageHeight != nil) {
            [ self.prePostProcessors addObject:[[PrePostProcessor alloc] initWithNumberOfClasses:numberOfClasses.integerValue imageWidth:imageWidth.integerValue imageHeight:imageHeight.integerValue objectDetectionModelType:objectDetectionModelType.integerValue]];
        } else {
            if (imageWidth != nil && imageHeight != nil) {
                [ self.prePostProcessors addObject:[[PrePostProcessor alloc] initWithImageWidth:imageWidth.integerValue imageHeight:imageHeight.integerValue]];
            } else {
                [ self.prePostProcessors addObject:[[PrePostProcessor alloc] init]];
            }
        }
        i = _modulesVector.size() - 1;
        completion(@(i), nil);

    } catch (const std::exception& e) {
         NSLog(@"%@ is not a proper model: %s", modelPath, e.what());
        FlutterError *error = [FlutterError errorWithCode:@"ModelLoadingError" message:[NSString stringWithFormat:@"%@ is not a proper model", modelPath] details:@(e.what())];
        completion(nil, error);
    }
}



- (void)getImagePredictionListIndex:(NSInteger)index imageData:(nullable FlutterStandardTypedData *)imageData imageBytesList:(nullable NSArray<FlutterStandardTypedData *> *)imageBytesList imageWidthForBytesList:(nullable NSNumber *)imageWidthForBytesList imageHeightForBytesList:(nullable NSNumber *)imageHeightForBytesList mean:(NSArray<NSNumber *> *)mean std:(NSArray<NSNumber *> *)std completion:(void (^)(NSArray<NSNumber *> *_Nullable, FlutterError *_Nullable))completion {
    UIImage *bitmap = nil;
    PrePostProcessor *prePostProcessor = self.prePostProcessors[index];

    if (imageData) {
        bitmap = [UIImage imageWithData:imageData.data];
    } else {
        FlutterStandardTypedData *typedData = imageBytesList[0];
        bitmap = [UIImage imageWithData:typedData.data];
    }
    bitmap = [UIImageExtension resize:bitmap toWidth:prePostProcessor.mImageWidth toHeight:prePostProcessor.mImageHeight];

    float* input = [UIImageExtension normalize:bitmap withMean:mean withSTD:std];
    NSArray<NSNumber*> *results = [self predictImage:input withWidth:prePostProcessor.mImageWidth andHeight:prePostProcessor.mImageHeight atIndex:index isObjectDetection:FALSE objectDetectionType:0];

    if (results) {
        completion(results, nil);
    } else {
        FlutterError *error = [FlutterError errorWithCode:@"PREDICTION_ERROR" message:@"Prediction failed" details:nil];
        completion(nil, error);
    }
}



- (void)getImagePredictionListObjectDetectionIndex:(NSInteger)index imageData:(nullable FlutterStandardTypedData *)imageData imageBytesList:(nullable NSArray<FlutterStandardTypedData *> *)imageBytesList imageWidthForBytesList:(nullable NSNumber *)imageWidthForBytesList imageHeightForBytesList:(nullable NSNumber *)imageHeightForBytesList minimumScore:(double)minimumScore IOUThreshold:(double)IOUThreshold boxesLimit:(NSInteger)boxesLimit completion:(void (^)(NSArray<ResultObjectDetection *> *_Nullable, FlutterError *_Nullable))completion {
     UIImage *bitmap = nil;
    PrePostProcessor *prePostProcessor = self.prePostProcessors[index];
    prePostProcessor.mNmsLimit = boxesLimit;
    prePostProcessor.mScoreThreshold = minimumScore;
    prePostProcessor.mIOUThreshold = IOUThreshold;

    if (imageData) {
        bitmap = [UIImage imageWithData:imageData.data];
    } else {
        FlutterStandardTypedData *typedData = imageBytesList[0];
        bitmap = [UIImage imageWithData:typedData.data];
    }
    bitmap = [UIImageExtension resize:bitmap toWidth:prePostProcessor.mImageWidth toHeight:prePostProcessor.mImageHeight];
    float* input = [UIImageExtension normalize:bitmap withMean:prePostProcessor.NO_MEAN_RGB withSTD:prePostProcessor.NO_STD_RGB];
    NSArray<NSNumber*> *rawOutputs = [self predictImage:input withWidth:prePostProcessor.mImageWidth andHeight:prePostProcessor.mImageHeight atIndex:index isObjectDetection:TRUE objectDetectionType:prePostProcessor.mObjectDetectionModelType];

    NSMutableArray<ResultObjectDetection*> *results = [prePostProcessor outputsToNMSPredictions:rawOutputs];
 if (results) {
        completion(results, nil);
    } else {
        FlutterError *error = [FlutterError errorWithCode:@"PREDICTION_ERROR" message:@"Prediction failed" details:nil];
        completion(nil, error);
    }
}


- (void)getRawImagePredictionListIndex:(NSInteger)index imageData:(FlutterStandardTypedData *)imageData completion:(void (^)(NSArray<NSNumber *> *_Nullable, FlutterError *_Nullable))completion {
    PrePostProcessor *prePostProcessor = self.prePostProcessors[index];
     NSArray<NSNumber*> *results = [self predictImage:(float *)[imageData.data bytes] withWidth:prePostProcessor.mImageWidth andHeight:prePostProcessor.mImageHeight atIndex:index isObjectDetection:FALSE objectDetectionType:0];

    if (results) {
        completion(results, nil);
    } else {
        FlutterError *error = [FlutterError errorWithCode:@"PREDICTION_ERROR" message:@"Prediction failed" details:nil];
        completion(nil, error);
    }
}


- (void)getRawImagePredictionListObjectDetectionIndex:(NSInteger)index imageData:(FlutterStandardTypedData *)imageData minimumScore:(double)minimumScore IOUThreshold:(double)IOUThreshold boxesLimit:(NSInteger)boxesLimit completion:(void (^)(NSArray<ResultObjectDetection *> *_Nullable, FlutterError *_Nullable))completion {
    PrePostProcessor *prePostProcessor = self.prePostProcessors[index];
    prePostProcessor.mNmsLimit = boxesLimit;
    prePostProcessor.mScoreThreshold = minimumScore;
    prePostProcessor.mIOUThreshold = IOUThreshold;

    NSArray<NSNumber*> *rawOutputs = [self predictImage:(float *)[imageData.data bytes] withWidth:prePostProcessor.mImageWidth andHeight:prePostProcessor.mImageHeight atIndex:index isObjectDetection:TRUE objectDetectionType:prePostProcessor.mObjectDetectionModelType];

    NSMutableArray<ResultObjectDetection*> *results = [prePostProcessor outputsToNMSPredictions:rawOutputs];
    if (results) {
        completion(results, nil);
    } else {
        FlutterError *error = [FlutterError errorWithCode:@"PREDICTION_ERROR" message:@"Prediction failed" details:nil];
        completion(nil, error);
    }
}








- (void)dealloc {
    for (torch::jit::Module* module : _modulesVector) {
        delete module;
    }
    _modulesVector.clear();
}


@end
