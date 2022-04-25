package com.zezo789.pytorch_lite;

import androidx.annotation.NonNull;

import io.flutter.embedding.engine.plugins.FlutterPlugin;

import io.flutter.plugin.common.BinaryMessenger;
import io.flutter.view.TextureRegistry;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import android.content.Context;

import android.util.Log;

import org.pytorch.DType;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.pytorch.LiteModuleLoader;
import java.util.ArrayList;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

/** PytorchLitePlugin */
public class PytorchLitePlugin implements FlutterPlugin, Pigeon.ModelApi {

  private static final String TAG = "PytorchLitePlugin";
  ArrayList<Module> modules = new ArrayList<>();
  ArrayList<PrePostProcessor> prePostProcessors = new ArrayList<>();

  private FlutterState flutterState;

  @Override
  public void onAttachedToEngine(FlutterPluginBinding binding) {
    this.flutterState =
            new FlutterState(
                    binding.getApplicationContext(),
                    binding.getBinaryMessenger(),
                    binding.getTextureRegistry());
    flutterState.startListening(this, binding.getBinaryMessenger());
  }

  @Override
  public void onDetachedFromEngine(FlutterPluginBinding binding) {
    if (flutterState == null) {
      Log.wtf(TAG, "Detached from the engine before registering to it.");
    }
    flutterState.stopListening(binding.getBinaryMessenger());
    flutterState = null;
  }






  @Override
  public Long loadModel(String modelPath, Long numberOfClasses, Long imageWidth, Long imageHeight) {
    int i=-1;
    try {
      modules.add(LiteModuleLoader.load(modelPath));
      if (numberOfClasses != null) {
        prePostProcessors.add(new PrePostProcessor(numberOfClasses.intValue(),imageWidth.intValue(),imageHeight.intValue()));
      }else{
        prePostProcessors.add(new PrePostProcessor());
      }
      i= (modules.size() - 1);
    } catch (Exception e) {
      Log.e(TAG, modelPath + " is not a proper model", e);
    }

    return (long) i;
  }

  @java.lang.Override
  public void getPredictionCustom(Long index, List<Double> input, List<Long> shape, String dtype, Pigeon.Result<List<Object>> result) {
    Module module = null;
    Double[] dataFormatted = new Double[input.size()];
    Integer[] shapeFormatted = new Integer[shape.size()];
    DType dtypeEnum = null;

    try{
      module = modules.get(index.intValue());
      dtypeEnum = DType.valueOf(dtype.toUpperCase());


      for (int i = 0; i < dataFormatted.length; i++) {
        dataFormatted[i] = input.get(i);
      }



        Log.e(TAG, String.valueOf(shape));

      /*
      for (int i = 0; i < shapeFormatted.length; i++) {
        Log.e(TAG, shape.get(i).toString());
        shapeFormatted[i] =(Integer) shape.get(i).intValue();
      }*/

    }catch(Exception e){
      Log.e(TAG, "error parsing arguments", e);
    }


    //prepare input tensor
    final Tensor inputTensor = getInputTensor(dtypeEnum, dataFormatted, shapeFormatted);

    //run model
    Tensor outputTensor = null;
    try {
      outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
      //result.success(outputTensor.getDataAsFloatArray());
    }catch(RuntimeException e){
      Log.e("PyTorchMobile", "Your input type " + dtypeEnum.toString().toLowerCase()  + " (" + Convert.dtypeAsPrimitive(dtypeEnum.toString()) +") " + "does not match with model input type",e);
      result.error(e);
    }

    successResult(result, dtypeEnum, outputTensor);
  }

  @Override
  public void getImagePredictionList(@NonNull Long index, @NonNull byte[] imageData, @NonNull Long width, @NonNull Long height, @NonNull List<Double> mean, @NonNull List<Double> std, Pigeon.Result<List<Double>> result) {
    Module imageModule = null;
    Bitmap bitmap = null;
    float[] meanFormatted=new float[mean.size()];
    float[] stdFormatted=new float[std.size()];
    try {

      imageModule = modules.get(index.intValue());



      bitmap = BitmapFactory.decodeByteArray(imageData,0,imageData.length);

      bitmap = Bitmap.createScaledBitmap(bitmap, width.intValue(), height.intValue(), false);


      for (int i = 0; i < meanFormatted.length; i++) {
        meanFormatted[i] = mean.get(i).floatValue();
      }

      for (int i = 0; i < stdFormatted.length; i++) {
        stdFormatted[i] = std.get(i).floatValue();
      }


    }catch (Exception e){
      Log.e(TAG, "error reading image", e);
    }

    try {
      final Tensor imageInputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, meanFormatted, stdFormatted);

      final Tensor imageOutputTensor = imageModule.forward(IValue.from(imageInputTensor)).toTensor();

      // getting tensor content as java array of doubles
      float[] scores = imageOutputTensor.getDataAsFloatArray();

      Double[] scoresDouble=new Double[scores.length];
      for (int i = 0; i < scoresDouble.length; i++) {

        scoresDouble[i] = Double.valueOf(Float.valueOf(scores[i]).toString());
      }
      result.success(Arrays.asList(scoresDouble));
    }catch (Exception e){
      Log.e(TAG, "error classifying image", e);
    }
  }

  @Override
  public void getImagePredictionListObjectDetection(Long index, byte[] imageData, Double minimumScore, Double IOUThreshold, Long boxesLimit, Pigeon.Result<List<Pigeon.ResultObjectDetection>> result) {
    Module imageModule = null;
    PrePostProcessor prePostProcessor = null;
    Bitmap bitmap = null;

    try {

      imageModule = modules.get(index.intValue());

      prePostProcessor = prePostProcessors.get(index.intValue());
      prePostProcessor.mNmsLimit=boxesLimit.intValue();
      prePostProcessor.mScoreThreshold=minimumScore.floatValue();
      prePostProcessor.mIOUThreshold=IOUThreshold.floatValue();

      bitmap = BitmapFactory.decodeByteArray(imageData,0,imageData.length);

      bitmap = Bitmap.createScaledBitmap(bitmap,prePostProcessor.mImageWidth, prePostProcessor.mImageHeight, false);





    }catch (Exception e){
      Log.e(TAG, "error reading image", e);
    }

    try {
      final Tensor imageInputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, prePostProcessor.NO_MEAN_RGB, prePostProcessor.NO_STD_RGB);

      IValue[] outputTuple  = imageModule.forward(IValue.from(imageInputTensor)).toTuple();

      final Tensor outputTensor = outputTuple[0].toTensor();
      final float[] outputs = outputTensor.getDataAsFloatArray();



      final ArrayList<Pigeon.ResultObjectDetection> results = prePostProcessor.outputsToNMSPredictions(outputs);

      result.success(results);
    }catch (Exception e){
      Log.e(TAG, "error classifying image", e);
    }

  }













  //returns input tensor depending on dtype
  private Tensor getInputTensor(DType dtype, Double[] data, Integer[] shape){
    switch (dtype){
      case FLOAT32:
        return Tensor.fromBlob(Convert.toFloatPrimitives(data), Convert.toPrimitives(shape));
      case FLOAT64:
        return  Tensor.fromBlob(Convert.toDoublePrimitives(data), Convert.toPrimitives(shape));
      case INT32:
        return Tensor.fromBlob(Convert.toIntegerPrimitives(data), Convert.toPrimitives(shape));
      case INT64:
        return Tensor.fromBlob(Convert.toLongPrimitives(data), Convert.toPrimitives(shape));
      case INT8:
        return Tensor.fromBlob(Convert.toBytePrimitives(data), Convert.toPrimitives(shape));
      case UINT8:
        return Tensor.fromBlobUnsigned(Convert.toBytePrimitives(data), Convert.toPrimitives(shape));
      default:
        return null;
    }
  }

  //gets tensor depending on dtype and creates list of it, which is being returned
  private void successResult(Pigeon.Result result, DType dtype, Tensor outputTensor){
    switch (dtype){
      case FLOAT32:
        ArrayList<Float> outputListFloat = new ArrayList<>();
        for(float f : outputTensor.getDataAsFloatArray()){
          outputListFloat.add(f);
        }
        result.success(outputListFloat);
        break;
      case FLOAT64:
        ArrayList<Double> outputListDouble = new ArrayList<>();
        for(double d : outputTensor.getDataAsDoubleArray()){
          outputListDouble.add(d);
        }
        result.success(outputListDouble);
        break;
      case INT32:
        ArrayList<Integer> outputListInteger = new ArrayList<>();
        for(int i : outputTensor.getDataAsIntArray()){
          outputListInteger.add(i);
        }
        result.success(outputListInteger);
        break;
      case INT64:
        ArrayList<Long> outputListLong = new ArrayList<>();
        for(long l : outputTensor.getDataAsLongArray()){
          outputListLong.add(l);
        }
        result.success(outputListLong);
        break;
      case INT8:
        ArrayList<Byte> outputListByte = new ArrayList<>();
        for(byte b : outputTensor.getDataAsByteArray()){
          outputListByte.add(b);
        }
        result.success(outputListByte);
        break;
      case UINT8:
        ArrayList<Byte> outputListUByte = new ArrayList<>();
        for(byte ub : outputTensor.getDataAsUnsignedByteArray()){
          outputListUByte.add(ub);
        }
        result.success(outputListUByte);
        break;
      default:
        result.success(null);
        break;
    }
  }

  private static final class FlutterState {
    private final Context applicationContext;
    private final BinaryMessenger binaryMessenger;
    private final TextureRegistry textureRegistry;

    FlutterState(Context applicationContext,
                 BinaryMessenger messenger,
                 TextureRegistry textureRegistry) {
      this.applicationContext = applicationContext;
      this.binaryMessenger = messenger;
      this.textureRegistry = textureRegistry;
    }


    void startListening(PytorchLitePlugin methodCallHandler, BinaryMessenger messenger) {
      Pigeon.ModelApi.setup(messenger, methodCallHandler);
    }

    void stopListening(BinaryMessenger messenger) {
      Pigeon.ModelApi.setup(messenger, null);
    }
  }
}
