package com.zezo357.pytorch_lite;

import androidx.annotation.NonNull;

import io.flutter.embedding.engine.plugins.FlutterPlugin;
import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;
import io.flutter.plugin.common.MethodChannel.MethodCallHandler;
import io.flutter.plugin.common.MethodChannel.Result;

/** PytorchLitePlugin */
public class PytorchLitePlugin implements FlutterPlugin, MethodCallHandler {
  /// The MethodChannel that will the communication between Flutter and native Android
  ///
  /// This local reference serves to register the plugin with the Flutter Engine and unregister it
  /// when the Flutter Engine is detached from the Activity
  private MethodChannel channel;

  @Override
  public void onAttachedToEngine(@NonNull FlutterPluginBinding flutterPluginBinding) {
    channel = new MethodChannel(flutterPluginBinding.getBinaryMessenger(), "pytorch_lite");
    channel.setMethodCallHandler(this);
  }

  @Override
  public void onMethodCall(@NonNull MethodCall call, @NonNull Result result) {
    if (call.method.equals("getPlatformVersion")) {
      result.success("Android " + android.os.Build.VERSION.RELEASE);
    } else {
      result.notImplemented();
    }
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
    public Long loadModel(String modelPath, Long numberOfClasses, Long imageWidth, Long imageHeight,
                          Long objectDetectionModelType) {
        int i = -1;
        try {
            // modules.add(LiteModuleLoader.load(modelPath));
            modules.add(Module.load(modelPath));
            if (numberOfClasses != null && imageWidth != null && imageHeight != null) {
                prePostProcessors.add(new PrePostProcessor(numberOfClasses.intValue(), imageWidth.intValue(),
                        imageHeight.intValue(), objectDetectionModelType.intValue()));
            } else {
                if (imageWidth != null && imageHeight != null) {
                    prePostProcessors.add(new PrePostProcessor(imageWidth.intValue(), imageHeight.intValue()));
                } else {
                    prePostProcessors.add(new PrePostProcessor());
                }
            }
            i = (modules.size() - 1);
        } catch (Exception e) {
            Log.e(TAG, modelPath + " is not a proper model", e);
        }

        return (long) i;
    }


  

    @RequiresApi(api = Build.VERSION_CODES.N)
    @java.lang.Override
    public void getPredictionCustom(Long index, List<Double> input, List<Long> shape, String dtype,
            Pigeon.Result<List<Object>> result) {
        Module module = null;
        Double[] data = new Double[input.size()];
        DType dtype_enum = null;
        long[] shape_primitve = null;
        try {
            module = modules.get(index.intValue());

            dtype_enum = DType.valueOf(dtype.toUpperCase());

            Log.i(TAG, "parsed dtype_enum");

            // Long[] l = shape.toArray(new Long[0]);
            // long[] l = ArrayUtils.toPrimitive(l);
            // long[] result

            shape_primitve = shape.stream().mapToLong(l -> l).toArray();

            Log.i(TAG, "parsed shape_formmatted");
            data = input.toArray(new Double[0]);
            Log.i(TAG, "parsed data");

        } catch (Exception e) {
            Log.e(TAG, "error parsing arguments", e);
        }

        // prepare input tensor
        final Tensor inputTensor = getInputTensor(dtype_enum, data, shape_primitve);

        // run model
        Tensor outputTensor = null;
        try {
            outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        } catch (RuntimeException e) {
            Log.e(TAG, "Your input type " + dtype_enum.toString().toLowerCase() + " (" + Convert.dtypeAsPrimitive(dtype)
                    + ") " + "does not match with model input type", e);
            result.success(null);
        }
        result.success(Collections.singletonList(outputTensor));
    }

    @Override
    public void getImagePredictionList(Long index, byte[] imageData, List<byte[]> imageBytesList,
            Long imageWidthForBytesList, Long imageHeightForBytesList, List<Double> mean, List<Double> std,
            Pigeon.Result<List<Double>> result) {
        Module imageModule = null;
        Bitmap bitmap = null;
        PrePostProcessor prePostProcessor = null;
        float[] meanFormatted = new float[mean.size()];
        float[] stdFormatted = new float[std.size()];
        try {

            imageModule = modules.get(index.intValue());

            prePostProcessor = prePostProcessors.get(index.intValue());
            if (imageData != null) {
                bitmap = BitmapFactory.decodeByteArray(imageData, 0, imageData.length);
            } else {
                bitmap = getBitmapFromBytesList(imageBytesList, imageWidthForBytesList.intValue(),
                        imageHeightForBytesList.intValue());
            }
            Matrix matrix = new Matrix();
            matrix.postRotate(90.0f);
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
            bitmap = Bitmap.createScaledBitmap(bitmap, prePostProcessor.mImageWidth, prePostProcessor.mImageHeight,
                    false);

            for (int i = 0; i < meanFormatted.length; i++) {
                meanFormatted[i] = mean.get(i).floatValue();
            }

            for (int i = 0; i < stdFormatted.length; i++) {
                stdFormatted[i] = std.get(i).floatValue();
            }

        } catch (Exception e) {
            Log.e(TAG, "error reading image", e);
        }

        try {
            final Tensor imageInputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, meanFormatted, stdFormatted);

            final Tensor imageOutputTensor = imageModule.forward(IValue.from(imageInputTensor)).toTensor();

            // getting tensor content as java array of doubles
            float[] scores = imageOutputTensor.getDataAsFloatArray();

            Double[] scoresDouble = new Double[scores.length];
            for (int i = 0; i < scoresDouble.length; i++) {

                scoresDouble[i] = Double.valueOf(Float.valueOf(scores[i]).toString());
            }
            result.success(Arrays.asList(scoresDouble));
        } catch (Exception e) {
            Log.e(TAG, "error classifying image", e);
        }
    }

    @Override
    public void getImagePredictionListObjectDetection(Long index, byte[] imageData, List<byte[]> imageBytesList,
            Long imageWidthForBytesList, Long imageHeightForBytesList, Double minimumScore, Double IOUThreshold,
            Long boxesLimit, Pigeon.Result<List<Pigeon.ResultObjectDetection>> result) {
        Module imageModule = null;
        PrePostProcessor prePostProcessor = null;
        Bitmap bitmap = null;

        try {

            imageModule = modules.get(index.intValue());

            prePostProcessor = prePostProcessors.get(index.intValue());
            prePostProcessor.mNmsLimit = boxesLimit.intValue();
            prePostProcessor.mScoreThreshold = minimumScore.floatValue();
            prePostProcessor.mIOUThreshold = IOUThreshold.floatValue();

            if (imageData != null) {
                bitmap = BitmapFactory.decodeByteArray(imageData, 0, imageData.length);
            } else {
                bitmap = getBitmapFromBytesList(imageBytesList, imageWidthForBytesList.intValue(),
                        imageHeightForBytesList.intValue());
                Matrix matrix = new Matrix();
                matrix.postRotate(90.0f);
                bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
            }

            bitmap = Bitmap.createScaledBitmap(bitmap, prePostProcessor.mImageWidth, prePostProcessor.mImageHeight,
                    false);

        } catch (Exception e) {
            Log.e(TAG, "error reading image", e);
        }

        try {

            final Tensor imageInputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, prePostProcessor.NO_MEAN_RGB,
                    prePostProcessor.NO_STD_RGB);
            Tensor outputTensor = null;
            if (prePostProcessor.mObjectDetectionModelType == 0) {
                IValue[] outputTuple = imageModule.forward(IValue.from(imageInputTensor)).toTuple();
                outputTensor = outputTuple[0].toTensor();
            } else {
                outputTensor = imageModule.forward(IValue.from(imageInputTensor)).toTensor();
            }

            final float[] outputs = outputTensor.getDataAsFloatArray();

            final ArrayList<Pigeon.ResultObjectDetection> results = prePostProcessor.outputsToNMSPredictions(outputs);

            result.success(results);
        } catch (Exception e) {
            Log.e(TAG, "error classifying image", e);
        }
    }

    Bitmap getBitmapFromBytesList(List<byte[]> bytesList, int imageWidth, int imageHeight) throws IOException {
        ByteBuffer yBuffer = ByteBuffer.wrap(bytesList.get(0));
        ByteBuffer uBuffer = ByteBuffer.wrap(bytesList.get(1));
        ByteBuffer vBuffer = ByteBuffer.wrap(bytesList.get(2));

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, imageWidth, imageHeight, null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }
    /*
     * public Allocation renderScriptNV21ToRGBA888(Context context, int width, int
     * height, byte[] nv21) {
     * // https://stackoverflow.com/a/36409748
     * RenderScript rs = RenderScript.create(context);
     * ScriptIntrinsicYuvToRGB yuvToRgbIntrinsic =
     * ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs));
     * 
     * Type.Builder yuvType = new Type.Builder(rs,
     * Element.U8(rs)).setX(nv21.length);
     * Allocation in = Allocation.createTyped(rs, yuvType.create(),
     * Allocation.USAGE_SCRIPT);
     * 
     * Type.Builder rgbaType = new Type.Builder(rs,
     * Element.RGBA_8888(rs)).setX(width).setY(height);
     * Allocation out = Allocation.createTyped(rs, rgbaType.create(),
     * Allocation.USAGE_SCRIPT);
     * 
     * in.copyFrom(nv21);
     * 
     * yuvToRgbIntrinsic.setInput(in);
     * yuvToRgbIntrinsic.forEach(out);
     * return out;
     * }
     */

    // returns input tensor depending on dtype
    private Tensor getInputTensor(DType dtype, Double[] data, long[] shape) {
        switch (dtype) {
            case FLOAT32:
                return Tensor.fromBlob(Convert.toFloatPrimitives(data), shape);
            case FLOAT64:
                return Tensor.fromBlob(Convert.toDoublePrimitives(data), shape);
            case INT32:
                return Tensor.fromBlob(Convert.toIntegerPrimitives(data), shape);
            case INT64:
                return Tensor.fromBlob(Convert.toLongPrimitives(data), shape);
            case INT8:
                return Tensor.fromBlob(Convert.toBytePrimitives(data), shape);
            case UINT8:
                return Tensor.fromBlobUnsigned(Convert.toBytePrimitives(data), shape);
            default:
                return null;
        }
    }

    // gets tensor depending on dtype and creates list of it, which is being
    // returned
    private void successResult(Pigeon.Result result, DType dtype, Tensor outputTensor) {
        switch (dtype) {
            case FLOAT32:
                ArrayList<Float> outputListFloat = new ArrayList<>();
                for (float f : outputTensor.getDataAsFloatArray()) {
                    outputListFloat.add(f);
                }
                result.success(outputListFloat);
                break;
            case FLOAT64:
                ArrayList<Double> outputListDouble = new ArrayList<>();
                for (double d : outputTensor.getDataAsDoubleArray()) {
                    outputListDouble.add(d);
                }
                result.success(outputListDouble);
                break;
            case INT32:
                ArrayList<Integer> outputListInteger = new ArrayList<>();
                for (int i : outputTensor.getDataAsIntArray()) {
                    outputListInteger.add(i);
                }
                result.success(outputListInteger);
                break;
            case INT64:
                ArrayList<Long> outputListLong = new ArrayList<>();
                for (long l : outputTensor.getDataAsLongArray()) {
                    outputListLong.add(l);
                }
                result.success(outputListLong);
                break;
            case INT8:
                ArrayList<Byte> outputListByte = new ArrayList<>();
                for (byte b : outputTensor.getDataAsByteArray()) {
                    outputListByte.add(b);
                }
                result.success(outputListByte);
                break;
            case UINT8:
                ArrayList<Byte> outputListUByte = new ArrayList<>();
                for (byte ub : outputTensor.getDataAsUnsignedByteArray()) {
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
