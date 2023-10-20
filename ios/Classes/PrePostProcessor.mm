

#import "pigeon.h"

@interface PrePostProcessor : NSObject

@property (nonatomic, strong) NSArray<NSString *> *mClasses;
@property (nonatomic, strong) NSArray<NSNumber *> *NO_MEAN_RGB;
@property (nonatomic, strong) NSArray<NSNumber *> *NO_STD_RGB;
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

@end

@implementation PrePostProcessor

- (instancetype)init {
    self = [super init];
    if (self) {
        _NO_MEAN_RGB = @[@0.0f, @0.0f, @0.0f];
        _NO_STD_RGB = @[@1.0f, @1.0f, @1.0f];

        _mNumberOfClasses = 17;
        // _mOutputRow = 25200;
        _mOutputColumn = _mNumberOfClasses + 5;
        _mScoreThreshold = 0.30f;
        _mIOUThreshold = 0.30f;
        _mImageWidth = 640;
        _mImageHeight = 640;
        _mNmsLimit = 15;
    }
    return self;
}

- (instancetype)initWithImageWidth:(int)imageWidth imageHeight:(int)imageHeight {
    self = [self init];
    if (self) {
        _mImageWidth = imageWidth;
        _mImageHeight = imageHeight;
    }
    return self;
}

- (instancetype)initWithNumberOfClasses:(int)numberOfClasses imageWidth:(int)imageWidth imageHeight:(int)imageHeight objectDetectionModelType:(int)objectDetectionModelType {
    self = [self init];
    if (self) {
        _mNumberOfClasses = numberOfClasses;
        _mImageWidth = imageWidth;
        _mImageHeight = imageHeight;
        _mObjectDetectionModelType = objectDetectionModelType;
        if (_mObjectDetectionModelType==0) {
            // _mOutputRow = 25200;
            _mOutputColumn = _mNumberOfClasses + 5;
        } else {
            // _mOutputRow = 8400;
            _mOutputColumn = _mNumberOfClasses + 4;
        }
    }
    return self;
}







  + (double)getFloatAsDouble:(float)fValue {
    return (double)fValue;
}

- (NSMutableArray<ResultObjectDetection *> *)nonMaxSuppression:(NSMutableArray<ResultObjectDetection *> *)boxes {
    // Sort the boxes by confidence scores, from high to low.
    [boxes sortUsingComparator:^NSComparisonResult(ResultObjectDetection *box1, ResultObjectDetection *box2) {
        return [box2.score compare:box1.score];
    }];
    
    NSMutableArray<ResultObjectDetection *> *selected = [NSMutableArray array];
    NSMutableArray<NSNumber *> *active = [NSMutableArray arrayWithCapacity:boxes.count];
    for (NSUInteger i = 0; i < boxes.count; i++) {
        [active addObject:@(YES)];
    }
    NSUInteger numActive = active.count;
    
    BOOL done = NO;
    for (NSUInteger i = 0; i < boxes.count && !done; i++) {
        if (active[i].boolValue) {
            ResultObjectDetection *boxA = boxes[i];
            [selected addObject:boxA];
            if (selected.count >= self.mNmsLimit) {
                break;
            }
            
            for (NSUInteger j = i + 1; j < boxes.count; j++) {
                if (active[j].boolValue) {
                    ResultObjectDetection *boxB = boxes[j];
                    if ([self IOU:boxA.rect boxB:boxB.rect] > self.mIOUThreshold) {
                        active[j] = @(NO);
                        numActive -= 1;
                        if (numActive <= 0) {
                            done = YES;
                            break;
                        }
                    }
                }
            }
        }
    }
    
    NSLog(@"PytorchLitePlugin result length after processing %lu", (unsigned long)selected.count);
    
    return selected;
}
- (double)IOU:(PyTorchRect *)a boxB:(PyTorchRect *)b {
    double areaA = ((a.right.doubleValue - a.left.doubleValue) * (a.bottom.doubleValue - a.top.doubleValue));
    if (areaA <= 0.0)
        return 0.0;

    double areaB = ((b.right.doubleValue - b.left.doubleValue) * (b.bottom.doubleValue - b.top.doubleValue));
    if (areaB <= 0.0)
        return 0.0;

    double intersectionMinX = MAX(a.left.doubleValue, b.left.doubleValue);
    double intersectionMinY = MAX(a.top.doubleValue, b.top.doubleValue);
    double intersectionMaxX = MIN(a.right.doubleValue, b.right.doubleValue);
    double intersectionMaxY = MIN(a.bottom.doubleValue, b.bottom.doubleValue);
    double intersectionArea = MAX(intersectionMaxY - intersectionMinY, 0) *
            MAX(intersectionMaxX - intersectionMinX, 0);
    return intersectionArea / (areaA + areaB - intersectionArea);
}

- (NSMutableArray<ResultObjectDetection *> *)outputsToNMSPredictionsYoloV8:(NSArray<NSNumber *> *)outputs {
    int mOutputRow = outputs.count / self.mOutputColumn; 
    NSLog(@"model mOutputRow is %d", mOutputRow); 
    NSMutableArray<ResultObjectDetection *> *results = [NSMutableArray array];

    for (int i = 0; i < mOutputRow; i++) {
        float x = [outputs[i] floatValue];
        float y = [outputs[mOutputRow + i] floatValue];
        float w = [outputs[2 * mOutputRow + i] floatValue];
        float h = [outputs[3 * mOutputRow + i] floatValue];

        float left = (x - w / 2);
        float top = (y - h / 2);
        float right = (x + w / 2);
        float bottom = (y + h / 2);

        float max = [outputs[4 * mOutputRow + i] floatValue];
        int cls = 0;
        for (int j = 4; j < self.mOutputColumn; j++) {
            float currentVal = [outputs[j * mOutputRow + i] floatValue];
            if (currentVal > max) {
                max = currentVal;
                cls = j - 4;
            }
        }

        if (max > self.mScoreThreshold) {
            PyTorchRect *rect = [PyTorchRect makeWithLeft:@([self.class getFloatAsDouble:left / self.mImageWidth])
                                                      top:@([self.class getFloatAsDouble:top / self.mImageHeight])
                                                    right:@([self.class getFloatAsDouble:right / self.mImageWidth])
                                                   bottom:@([self.class getFloatAsDouble:bottom / self.mImageHeight])
                                                    width:@([self.class getFloatAsDouble:w / self.mImageWidth])
                                                   height:@([self.class getFloatAsDouble:h / self.mImageHeight])];

            ResultObjectDetection *result = [ResultObjectDetection makeWithClassIndex:@(cls)
                                                                              className:nil
                                                                                 score:@([self.class getFloatAsDouble:max])
                                                                                  rect:rect];

            [results addObject:result];
        }
    }

    NSLog(@"PytorchLitePlugin result length before processing %lu", (unsigned long)results.count);
    return [self nonMaxSuppression:results];
}


- (NSMutableArray<ResultObjectDetection *> *)outputsToNMSPredictionsYolov5:(NSArray<NSNumber *> *)outputs {
    int mOutputRow = outputs.count / self.mOutputColumn; 
    NSLog(@"model mOutputRow is %d", mOutputRow); 
    NSMutableArray<ResultObjectDetection *> *results = [NSMutableArray array];
    for (int i = 0; i < mOutputRow; i++) {
 float score = [outputs[i*self.mOutputColumn + 4] floatValue];

    if (score > self.mScoreThreshold) {
        float x = [outputs[i * self.mOutputColumn] floatValue];
        float y = [outputs[i * self.mOutputColumn + 1] floatValue];
        float w = [outputs[i * self.mOutputColumn + 2] floatValue];
        float h = [outputs[i * self.mOutputColumn + 3] floatValue];

        float left = (x - w / 2);
        float top = (y - h / 2);
        float right = (x + w / 2);
        float bottom = (y + h / 2);

        float max = [outputs[i * self.mOutputColumn + 5] floatValue];
        int cls = 0;
        for (int j = 0; j < self.mOutputColumn - 5; j++) {
            float currentVal = [outputs[i * self.mOutputColumn + 5 + j] floatValue];
            if (currentVal > max) {
                max = currentVal;
                cls = j;
            }
        }

        PyTorchRect *rect = [PyTorchRect makeWithLeft:@([self.class getFloatAsDouble:left / self.mImageWidth])
                                                  top:@([self.class getFloatAsDouble:top / self.mImageHeight])
                                                right:@([self.class getFloatAsDouble:right / self.mImageWidth])
                                               bottom:@([self.class getFloatAsDouble:bottom / self.mImageHeight])
                                                width:@([self.class getFloatAsDouble:w / self.mImageWidth])
                                               height:@([self.class getFloatAsDouble:h / self.mImageHeight])];

        ResultObjectDetection *result = [ResultObjectDetection makeWithClassIndex:@(cls)
                                                                          className:nil
                                                                             score:@([self.class getFloatAsDouble:[outputs[i * self.mOutputColumn + 4] floatValue]])
                                                                              rect:rect];

        [results addObject:result];
        }
    }

    NSLog(@"PytorchLitePlugin result length before processing %lu", (unsigned long)results.count);
    return [self nonMaxSuppression:results];
}


- (NSMutableArray<ResultObjectDetection *> *)outputsToNMSPredictions:(NSArray<NSNumber *> *)outputs {
    if (self.mObjectDetectionModelType == 0) {
        return [self outputsToNMSPredictionsYolov5:outputs];
    } else {
        return [self outputsToNMSPredictionsYoloV8:outputs];
    }
}

@end
