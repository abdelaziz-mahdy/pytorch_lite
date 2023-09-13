// #import <Foundation/Foundation.h>
#import "pigeon.h"


@interface PrePostProcessor : NSObject

@property (nonatomic, strong) NSArray<NSString *> *mClasses;
@property (nonatomic, assign) float *NO_MEAN_RGB;
@property (nonatomic, assign) float *NO_STD_RGB;
@property (nonatomic, assign) int mNumberOfClasses;
@property (nonatomic, assign) int mOutputRow;
@property (nonatomic, assign) int mOutputColumn;
@property (nonatomic, assign) float mScoreThreshold;
@property (nonatomic, assign) float mIOUThreshold;
@property (nonatomic, assign) int mImageWidth;
@property (nonatomic, assign) int mImageHeight;
@property (nonatomic, assign) int mNmsLimit;
@property (nonatomic, assign) int mObjectDetectionModelType;

- (instancetype)init;
- (instancetype)initWithImageWidth:(int)imageWidth imageHeight:(int)imageHeight;
- (instancetype)initWithNumberOfClasses:(int)numberOfClasses imageWidth:(int)imageWidth imageHeight:(int)imageHeight objectDetectionModelType:(int)objectDetectionModelType;

+ (double)getFloatAsDouble:(float)fValue;
- (NSMutableArray<ResultObjectDetection *> *)nonMaxSuppression:(NSMutableArray<ResultObjectDetection *> *)boxes;
- (double)IOU:(PyTorchRect *)a boxB:(PyTorchRect *)b;
- (NSMutableArray<ResultObjectDetection *> *)outputsToNMSPredictionsYoloV8:(float *)outputs;
- (NSMutableArray<ResultObjectDetection *> *)outputsToNMSPredictionsYoloV5:(float *)outputs;
- (NSMutableArray<ResultObjectDetection *> *)outputsToNMSPredictions:(float *)outputs;

@end
