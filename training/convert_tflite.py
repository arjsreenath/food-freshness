import tensorflow as tf, numpy as np, os, glob, PIL.Image as Image, json

model = tf.keras.models.load_model('model_freshness.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations=[tf.lite.Optimize.DEFAULT]

# representative dataset for full-integer quant (faster on phone)
def rep_data():
    imgs = glob.glob('../dataset/*/*')[:150]
    for p in imgs:
        try:
            img = Image.open(p).convert('RGB').resize((224,224))
            arr = (np.array(img)/255.0).astype(np.float32)
            yield [arr[np.newaxis,...]]
        except: pass

converter.representative_dataset = rep_data
converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type=tf.uint8
converter.inference_output_type=tf.uint8

tfl = converter.convert()
open('model_freshness_int8.tflite','wb').write(tfl)
print("Saved model_freshness_int8.tflite")

