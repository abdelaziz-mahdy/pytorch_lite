// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package com.zezo357.pytorch_lite;

import android.util.Log;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class PrePostProcessor {
    static String[] mClasses;
    // for yolov5 model, no need to apply MEAN and STD
    float[] NO_MEAN_RGB = new float[] { 0.0f, 0.0f, 0.0f };
    float[] NO_STD_RGB = new float[] { 1.0f, 1.0f, 1.0f };
    int mNumberOfClasses = 17;
    // model output is of size 25200*(num_of_class+5)
    // int mOutputRow = 25200; // as decided by the YOLOv5 model for input image of size 640*640
    int mOutputColumn = (mNumberOfClasses + 5); // left, top, right, bottom, score and 80 class probability
    float mScoreThreshold = 0.30f; // score above which a detection is generated
    float mIOUThreshold = 0.30f; // IOU thershold
    int mImageWidth = 640;
    int mImageHeight = 640;
    int mNmsLimit = 15;
    int mObjectDetectionModelType;

    PrePostProcessor() {
    }

    PrePostProcessor(int imageWidth, int imageHeight) {
        mImageWidth = imageWidth;
        mImageHeight = imageHeight;
    }

    PrePostProcessor(int numberOfClasses, int imageWidth, int imageHeight,
           int objectDetectionModelType) {
        mNumberOfClasses = numberOfClasses;
        mImageWidth = imageWidth;
        mImageHeight = imageHeight;
        mObjectDetectionModelType = objectDetectionModelType;
        if (objectDetectionModelType == 0) {
            // mOutputRow = 25200; // as decided by the YOLOv5 model for input image of size 640*640
            mOutputColumn = (mNumberOfClasses + 5); // left, top, right, bottom, score and 80 class probability
        } else {
            // mOutputRow = 8400; // as decided by the YOLOv5 model for input image of size 640*640
            mOutputColumn = (mNumberOfClasses + 4); // left, top, right, bottom, score and 80 class probability
        }

    }

    // The two methods nonMaxSuppression and IOU below are ported from
    // https://github.com/hollance/YOLO-CoreML-MPSNNGraph/blob/master/Common/Helpers.swift

    public static Double getFloatAsDouble(Float fValue) {
        return Double.valueOf(fValue);
    }

    /**
     * Removes bounding boxes that overlap too much with other boxes that have
     * a higher score.
     * - Parameters:
     * - boxes: an array of bounding boxes and their scores
     * - limit: the maximum number of boxes that will be selected
     * - threshold: used to decide whether boxes overlap too much
     */
    ArrayList<Pigeon.ResultObjectDetection> nonMaxSuppression(ArrayList<Pigeon.ResultObjectDetection> boxes) {
        // Log.i("PytorchLitePlugin","first score before sorting
        // "+boxes.get(0).getScore());
        // Do an argsort on the confidence scores, from high to low.
        Collections.sort(boxes,
                (o2, o1) -> o1.getScore().compareTo(o2.getScore()));
        // Log.i("PytorchLitePlugin","first score after sorting
        // "+boxes.get(0).getScore());
        ArrayList<Pigeon.ResultObjectDetection> selected = new ArrayList<>();
        boolean[] active = new boolean[boxes.size()];
        Arrays.fill(active, true);
        int numActive = active.length;

        // The algorithm is simple: Start with the box that has the highest score.
        // Remove any remaining boxes that overlap it more than the given threshold
        // amount. If there are any boxes left (i.e. these did not overlap with any
        // previous boxes), then repeat this procedure, until no more boxes remain
        // or the limit has been reached.
        boolean done = false;
        for (int i = 0; i < boxes.size() && !done; i++) {
            if (active[i]) {
                Pigeon.ResultObjectDetection boxA = boxes.get(i);
                selected.add(boxA);
                if (selected.size() >= mNmsLimit)
                    break;

                for (int j = i + 1; j < boxes.size(); j++) {
                    if (active[j]) {
                        Pigeon.ResultObjectDetection boxB = boxes.get(j);
                        if (IOU(boxA.getRect(), boxB.getRect()) > mIOUThreshold) {
                            active[j] = false;
                            numActive -= 1;
                            if (numActive <= 0) {
                                done = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
        Log.i("PytorchLitePlugin", "result length after processing " + selected.size());

        return selected;
    }

    /**
     * Computes intersection-over-union overlap between two bounding boxes.
     */
    Double IOU(Pigeon.PyTorchRect a, Pigeon.PyTorchRect b) {
        Double areaA = ((a.getRight() - a.getLeft()) * (a.getBottom() - a.getTop()));
        if (areaA <= 0.0)
            return 0.0;

        Double areaB = ((b.getRight() - b.getLeft()) * (b.getBottom() - b.getTop()));
        if (areaB <= 0.0)
            return 0.0;

        Double intersectionMinX = Math.max(a.getLeft(), b.getLeft());
        Double intersectionMinY = Math.max(a.getTop(), b.getTop());
        Double intersectionMaxX = Math.min(a.getRight(), b.getRight());
        Double intersectionMaxY = Math.min(a.getBottom(), b.getBottom());
        Double intersectionArea = Math.max(intersectionMaxY - intersectionMinY, 0) *
                Math.max(intersectionMaxX - intersectionMinX, 0);
        return intersectionArea / (areaA + areaB - intersectionArea);
    }

    ArrayList<Pigeon.ResultObjectDetection> outputsToNMSPredictionsYoloV8(float[] outputs) {
int mOutputRow = outputs.length/mOutputColumn;
                Log.i("PytorchLitePlugin", "model mOutputRow is " + mOutputRow);

        ArrayList<Pigeon.ResultObjectDetection> results = new ArrayList<>();
        for (int i = 0; i < mOutputRow; i++) {

            float x = outputs[i];
            float y = outputs[mOutputRow + i];
            float w = outputs[2 * mOutputRow + i];
            float h = outputs[3 * mOutputRow + i];

            float left = (x - w / 2);
            float top = (y - h / 2);
            float right = (x + w / 2);
            float bottom = (y + h / 2);

            float max = outputs[4 * mOutputRow + i];
            int cls = 0;
            for (int j = 4; j < mOutputColumn; j++) {
                if (outputs[j * mOutputRow + i] > max) {
                    max = outputs[j * mOutputRow + i];
                    cls = j - 4;
                }
            }

            if (max > mScoreThreshold) {
                Pigeon.PyTorchRect rect = new Pigeon.PyTorchRect.Builder().setLeft(
                        getFloatAsDouble(left / mImageWidth)).setTop(
                                getFloatAsDouble(top / mImageHeight))
                        .setWidth(
                                getFloatAsDouble(w / mImageWidth))
                        .setHeight(
                                getFloatAsDouble(h / mImageHeight))
                        .setBottom(
                                getFloatAsDouble(bottom / mImageHeight))
                        .setRight(
                                getFloatAsDouble(right / mImageWidth))
                        .build();
                Pigeon.ResultObjectDetection result = new Pigeon.ResultObjectDetection.Builder()
                        .setClassIndex((long) cls).setScore(getFloatAsDouble(max)).setRect(rect).build();

                results.add(result);

            }
        }

        Log.i("PytorchLitePlugin", "result length before processing " + results.size());
        return nonMaxSuppression(results);
    }

    ArrayList<Pigeon.ResultObjectDetection> outputsToNMSPredictionsYolov5(float[] outputs) {
        int mOutputRow = outputs.length/mOutputColumn;
        Log.i("PytorchLitePlugin", "model mOutputRow is " + mOutputRow);
        ArrayList<Pigeon.ResultObjectDetection> results = new ArrayList<>();
        for (int i = 0; i < mOutputRow; i++) {
            // Log.i("PytorchLitePlugin","0:"+outputs[i* mOutputColumn]+"1");
            if (outputs[i * mOutputColumn + 4] > mScoreThreshold) {
                float x = outputs[i * mOutputColumn];
                float y = outputs[i * mOutputColumn + 1];
                float w = outputs[i * mOutputColumn + 2];
                float h = outputs[i * mOutputColumn + 3];

                float left = (x - w / 2);
                float top = (y - h / 2);
                float right = (x + w / 2);
                float bottom = (y + h / 2);

                // Log.i("PytorchLitePlugin","i* mOutputColumn +4="+outputs[i* mOutputColumn
                // +4]+",outputs[i* mOutputColumn +5] "+outputs[i* mOutputColumn +5]);
                float max = outputs[i * mOutputColumn + 5];
                int cls = 0;
                for (int j = 0; j < mOutputColumn - 5; j++) {
                    if (outputs[i * mOutputColumn + 5 + j] > max) {
                        max = outputs[i * mOutputColumn + 5 + j];
                        cls = j;
                    }
                }

                Pigeon.PyTorchRect rect = new Pigeon.PyTorchRect.Builder().setLeft(
                        getFloatAsDouble(left / mImageWidth)).setTop(
                                getFloatAsDouble(top / mImageHeight))
                        .setWidth(
                                getFloatAsDouble(w / mImageWidth))
                        .setHeight(
                                getFloatAsDouble(h / mImageHeight))
                        .setBottom(
                                getFloatAsDouble(bottom / mImageHeight))
                        .setRight(
                                getFloatAsDouble(right / mImageWidth))
                        .build();
                Pigeon.ResultObjectDetection result = new Pigeon.ResultObjectDetection.Builder()
                        .setClassIndex((long) cls).setScore(getFloatAsDouble(outputs[i*mOutputColumn+4]))
                        .setRect(rect).build();

                results.add(result);

            }
        }

        Log.i("PytorchLitePlugin", "result length before processing " + results.size());
        return nonMaxSuppression(results);
    }

    ArrayList<Pigeon.ResultObjectDetection> outputsToNMSPredictions(float[] outputs) {
        // passed on model return value
        if (mObjectDetectionModelType == 0) {
            return outputsToNMSPredictionsYolov5(outputs);
        } else {
            return outputsToNMSPredictionsYoloV8(outputs);
        }
    }

}