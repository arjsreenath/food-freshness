package com.example.freshnessapp;  // <-- change to your package

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import java.io.File;
import android.os.Bundle;
import android.util.Size;
import android.widget.Button;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Map;
import java.util.HashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final int REQ_CAMERA = 10;
    private static final int INPUT_SIZE = 224;

    private PreviewView previewView;
    private ImageCapture imageCapture;
    private Interpreter interpreter;
    private org.tensorflow.lite.DataType modelInputType;
    private int inputWidth = 224;   // will be overwritten by real model shape
    private int inputHeight = 224;  // will be overwritten by real model shape
    private String[] labelNames;

    private final ExecutorService cameraExecutor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);

        // ask camera permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQ_CAMERA);
        } else {
            startCamera();
        }

        // load model + labels
        try {
            interpreter = new Interpreter(loadModel("model.tflite"), new Interpreter.Options());
            labelNames = loadLabels("class_indices.json");

            // Detect model input details
            int[] inputShape = interpreter.getInputTensor(0).shape(); // [1, H, W, 3]
            org.tensorflow.lite.DataType inputType = interpreter.getInputTensor(0).dataType();

            // Save for later
            modelInputType = inputType;
            inputHeight = inputShape[1];
            inputWidth  = inputShape[2];
        } catch (IOException | JSONException e) {
            throw new RuntimeException("Failed to load model/labels", e);
        }

        Button btn = findViewById(R.id.btnPredict);
        TextView tv = findViewById(R.id.tvResult);
        btn.setOnClickListener(v -> captureAndPredict(result -> runOnUiThread(() -> tv.setText(result))));
    }

    /** Start CameraX preview + capture */
    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                imageCapture = new ImageCapture.Builder()
                        .setTargetResolution(new Size(1280, 720))
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                        .build();

                CameraSelector selector = CameraSelector.DEFAULT_BACK_CAMERA;

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, selector, preview, imageCapture);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private interface ResultCallback { void onResult(String text); }

    /** Take a photo frame and run the model */
    private void captureAndPredict(@NonNull ResultCallback cb) {
        if (imageCapture == null) { cb.onResult("Camera not ready"); return; }

        // Save to app-specific external files dir (no storage permission needed)
        File outFile = new File(getExternalFilesDir(null),
                "capture_" + System.currentTimeMillis() + ".jpg");

        ImageCapture.OutputFileOptions opts =
                new ImageCapture.OutputFileOptions.Builder(outFile).build();

        imageCapture.takePicture(opts, cameraExecutor, new ImageCapture.OnImageSavedCallback() {
            @Override public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {
                // Decode JPEG -> Bitmap
                BitmapFactory.Options o = new BitmapFactory.Options();
                o.inPreferredConfig = Bitmap.Config.ARGB_8888;
                Bitmap bmp = BitmapFactory.decodeFile(outFile.getAbsolutePath(), o);

                // (Optional) rotate if needed — many back cameras are 90°
                // bmp = rotateIfRequired(bmp); // uncomment and implement if your image is rotated

                String result = runInference(bmp);
                runOnUiThread(() -> cb.onResult(result));
            }

            @Override public void onError(@NonNull ImageCaptureException exception) {
                runOnUiThread(() -> cb.onResult("Capture error: " + exception.getMessage()));
            }
        });
    }


    /** Resize → TensorImage → run TFLite → pick best class */
    private String runInference(Bitmap bitmap) {
        // 1) Resize input; normalize only if model INPUT is float32
        ImageProcessor.Builder b = new ImageProcessor.Builder()
                .add(new ResizeOp(inputHeight, inputWidth, ResizeOp.ResizeMethod.BILINEAR));

        boolean needsFloatNorm = (modelInputType == DataType.FLOAT32);
        if (needsFloatNorm) {
            // Map [0,255] -> [0,1] for float models
            b.add(new NormalizeOp(0f, 255f));
        }
        ImageProcessor processor = b.build();

        // 2) Make a TensorImage of the correct INPUT type
        TensorImage ti = new TensorImage(modelInputType);
        ti.load(bitmap);
        ti = processor.process(ti);

        // 3) Prepare OUTPUT buffer using the model's true output tensor
        final int outIndex = 0;
        int[] outShape = interpreter.getOutputTensor(outIndex).shape();   // [1, numClasses]
        DataType outType = interpreter.getOutputTensor(outIndex).dataType();
        TensorBuffer output = TensorBuffer.createFixedSize(outShape, outType);

        // 4) Run inference
        interpreter.run(ti.getBuffer(), output.getBuffer().rewind());

        // 5) Read scores as floats
        float[] probs;
        if (outType == DataType.UINT8 || outType == DataType.INT8) {
            // Proper dequantization using scale/zeroPoint
            org.tensorflow.lite.Tensor outTensor = interpreter.getOutputTensor(outIndex);
            org.tensorflow.lite.Tensor.QuantizationParams qp = outTensor.quantizationParams();
            float scale = qp.getScale();          // e.g., ~0.0039 for 8-bit probs
            int zeroPoint = qp.getZeroPoint();    // often 0 or 128

            java.nio.ByteBuffer bb = output.getBuffer();
            bb.rewind();
            int n = outShape[outShape.length - 1];
            probs = new float[n];
            for (int i = 0; i < n; i++) {
                int q = bb.get() & 0xFF;                 // unsigned
                float v = (q - zeroPoint) * scale;       // dequantize to float
                probs[i] = v;
            }
            // Optional but recommended: turn logits into probabilities
            probs = softmax(probs);
        } else {
            // FLOAT32 output already; may already be probs or logits
            probs = output.getFloatArray();
            // If these are logits, uncomment:
            // probs = softmax(probs);
        }

        // 6) Argmax
        int bestIdx = 0;
        float best = -Float.MAX_VALUE;
        for (int i = 0; i < probs.length; i++) {
            if (probs[i] > best) { best = probs[i]; bestIdx = i; }
        }

        String label = labelNames[bestIdx];
        // show confidence as 0–1; multiply by 100 if you prefer %
        return "Prediction: " + label + "  (conf: " + String.format("%.2f", best) + ")";
    }

    // --- helper: softmax ---
    private float[] softmax(float[] x) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : x) if (v > max) max = v;
        float sum = 0f;
        float[] e = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            e[i] = (float) Math.exp(x[i] - max);
            sum += e[i];
        }
        for (int i = 0; i < x.length; i++) e[i] /= (sum == 0f ? 1f : sum);
        return e;
    }



    /** Memory-map model from assets */
    private MappedByteBuffer loadModel(String assetName) throws IOException {
        AssetFileDescriptor afd = getAssets().openFd(assetName);
        FileInputStream fis = new FileInputStream(afd.getFileDescriptor());
        FileChannel channel = fis.getChannel();
        long startOffset = afd.getStartOffset();
        long declaredLength = afd.getDeclaredLength();
        return channel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /** Read labels from class_indices.json and order by index */
    private String[] loadLabels(String assetJson) throws IOException, JSONException {
        BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open(assetJson)));
        StringBuilder sb = new StringBuilder();
        String line;
        while ((line = br.readLine()) != null) sb.append(line);
        br.close();

        JSONObject obj = new JSONObject(sb.toString());

        // obj.names() returns a JSONArray of keys, works on all Android versions
        JSONArray keys = obj.names();
        if (keys == null) return new String[0];

        String[] labels = new String[keys.length()];
        for (int i = 0; i < keys.length(); i++) {
            String key = keys.getString(i);   // "fresh" or "rotten"
            int idx = obj.getInt(key);        // numeric index
            labels[idx] = key;                // place label in correct index
        }
        return labels;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] perms, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, perms, grantResults);
        if (requestCode == REQ_CAMERA && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        }
    }
}
