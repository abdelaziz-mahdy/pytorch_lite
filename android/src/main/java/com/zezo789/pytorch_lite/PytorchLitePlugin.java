package com.zezo789.pytorch_lite;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

import io.flutter.embedding.engine.plugins.FlutterPlugin;
import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;
import io.flutter.plugin.common.MethodChannel.MethodCallHandler;
import io.flutter.plugin.common.MethodChannel.Result;
import io.flutter.plugin.common.BinaryMessenger;
import io.flutter.plugin.common.EventChannel;
import io.flutter.view.TextureRegistry;
import java.util.List;
import android.content.Context;
import android.net.Uri;
import android.os.Build;
import android.util.Log;
import android.util.LongSparseArray;



/** PytorchLitePlugin */
public class PytorchLitePlugin implements FlutterPlugin, Pigeon.ModelApi {

  private static final String TAG = "PytorchLitePlugin";
  List<Module> modules = new List<>();

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
  public Long loadModel(String modelPath, String labelsPath) {
    int i=-1;

    try {
      modules.add(LiteModuleLoader.load(modelPath));
      i= (modules.size() - 1);
    } catch (Exception e) {
      Log.e(TAG, modelPath + " is not a proper model", e);
    }

    long l=i;
    return l;
  }

  @java.lang.Override
  public void getPredictionCustom(Long index, List<Double> input, List<Long> shape, String dtype, Pigeon.Result<List<Object>> result) {
    Module module = null;
    //Double[] data = null;
    DType dtypeEnum = null;

    try{
      module = modules.get(index.intValue());
      dtypeEnum = DType.valueOf(dtype.toUpperCase());

    }catch(Exception e){
      Log.e(TAG, "error parsing arguments", e);
    }

    //prepare input tensor
    final Tensor inputTensor = getInputTensor(dtypeEnum, input, shape);

    //run model
    Tensor outputTensor = null;
    try {
      outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
      result.success(outputTensor.getDataAsFloatArray());
    }catch(RuntimeException e){
      Log.e("PyTorchMobile", "Your input type " + dtypeEnum.toString().toLowerCase()  + " (" + Convert.dtypeAsPrimitive(dtypeEnum.toString()) +") " + "does not match with model input type",e);
      result.error(e);
    }

    //successResult(result, dtype, outputTensor);
  }

  @java.lang.Override
  public void getImagePrediction(Long index, String imagePath, Long width, Long height, List<Double> mean, List<Double> std, Pigeon.Result<String> result) {


    float maxScore = -Float.MAX_VALUE;
    int maxScoreIdx = -1;
    for (int i = 0; i < scores.length; i++) {
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        maxScoreIdx = i;
      }
    }

    String className = ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];

  }

  @java.lang.Override
  public void getImagePredictionList(Long index, String imagePath, Long width, Long height, List<Double> mean, List<Double> std, Pigeon.Result<List<Object>> result) {

  }

  @java.lang.Override
  public void getImagePredictionObjectDetection(Long index, String imagePath, Long width, Long height, List<Double> mean, List<Double> std, Pigeon.Result<String> result) {

  }

  @java.lang.Override
  public void getImagePredictionListObjectDetection(Long index, String imagePath, Long width, Long height, List<Double> mean, List<Double> std, Pigeon.Result<List<Object>> result) {

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
