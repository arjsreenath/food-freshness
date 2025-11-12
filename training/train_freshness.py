import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

IMG_SIZE=(224,224); BATCH=32; DATA_DIR="../dataset"

gen = ImageDataGenerator(rescale=1./255, rotation_range=15,
                         width_shift_range=0.08, height_shift_range=0.08,
                         brightness_range=(0.7,1.2), horizontal_flip=True,
                         validation_split=0.15)

train = gen.flow_from_directory(DATA_DIR, target_size=IMG_SIZE,
                                batch_size=BATCH, subset='training', class_mode='categorical')
val   = gen.flow_from_directory(DATA_DIR, target_size=IMG_SIZE,
                                batch_size=BATCH, subset='validation', class_mode='categorical')

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE,3), pooling='avg')
base.trainable = False
inp = tf.keras.Input(shape=(*IMG_SIZE,3))
x = base(inp, training=False)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
out = layers.Dense(train.num_classes, activation='softmax')(x)
model = models.Model(inp, out)
model.compile(optimizer=optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

cbs=[callbacks.ReduceLROnPlateau('val_loss',0.5,3), callbacks.EarlyStopping('val_loss',patience=6,restore_best_weights=True)]
model.fit(train, validation_data=val, epochs=12, callbacks=cbs)

base.trainable = True
for layer in base.layers[:-50]:
    layer.trainable=False
model.compile(optimizer=optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train, validation_data=val, epochs=6, callbacks=cbs)

model.save('model_freshness.h5')
with open('class_indices.json','w') as f: json.dump(train.class_indices,f)
print("Saved model_freshness.h5 and class_indices.json")

