import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications import MobileNetV2
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D,GlobalAveragePooling2D
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.optimizers import Adam

img_rows, img_cols = 224, 224
batch_size = 32
epochs = 20
LR = 1e-04
classes = 2

train_data = "/Users/dipit/Face Mask/New Masks Dataset/Train"
val_data = "/Users/dipit/Face Mask/New Masks Dataset/Validation"
test_data = "/Users/dipit/Face Mask/New Masks Dataset/Test"

train_datagen = ImageDataGenerator(rescale = 1./255,
                                    rotation_range = 20,
                                    zoom_range = 0.3,
                                    height_shift_range = 0.2,
                                    width_shift_range = 0.2,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode = "nearest"
                                    )
val_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_data,
                                                    target_size=(img_rows, img_cols),
                                                    batch_size = batch_size,
                                                    shuffle = True,
                                                    class_mode = "categorical")


val_generator = val_datagen.flow_from_directory(val_data,
                                                target_size = (img_rows, img_cols),
                                                batch_size = batch_size,
                                                shuffle = False,
                                                class_mode = "categorical"
                                                )

test_generator = test_datagen.flow_from_directory(test_data,
                                                  target_size = (img_rows, img_cols),
                                                  batch_size = batch_size,
                                                  shuffle = False,
                                                  class_mode = "categorical")

base_model = MobileNetV2(
                        weights="imagenet",
                        include_top = False,
                        input_shape = (img_rows, img_cols, 3)
                        )

for layer in base_model.layers:
    layer.trainable = False
    
head_model = base_model.output  
head_model = GlobalAveragePooling2D()(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(classes, activation="softmax")(head_model)

model = Model(inputs = base_model.input, outputs=head_model)
model.summary()

opt = Adam(lr = LR, decay = LR/epochs)
model.compile(loss="categorical_crossentropy",
             optimizer = opt,
             metrics=['acc'])

history = model.fit(train_generator,
                    epochs = epochs,
                    steps_per_epoch=train_generator.n//train_generator.batch_size,
                    validation_data = val_generator,
                    validation_steps = val_generator.n//val_generator.batch_size)

model.save("FaceMaskDetectionModel_1.h5")

print("Validation Evaluation")
scores = model.evaluate(val_generator,steps=val_generator.n//val_generator.batch_size, verbose=1)
print("Validation Accuracy: %.3f Validation Loss: %.3f"%(scores[1]*100, scores[0]))

print("Test Evaluation")
test_scores = model.evaluate(test_generator,steps=test_generator.n//test_generator.batch_size, verbose=1)
print("Test Accuracy: %.3f Test Loss: %.3f"%(scores[1]*100, scores[0]))

print("Testing on Single Input")
IMG = load_img("/Users/dipit/Face Mask/New Masks Dataset/Test/Mask/2260.png",
                target_size=(img_rows, img_cols))
plt.imshow(IMG)
plt.show()
img = img_to_array(IMG)
img = img/255
img = np.expand_dims(img,axis=0)
print(img.shape)

CLASSES = {
    'TEST':['Mask','Non Mask']
}

classifier = load_model("FaceMaskDetectionModel_1.h5")
predictions = classifier.predict(img)
predictions_c = np.argmax(predictions,axis=1)   

predicted_class = CLASSES['TEST'][predictions_c[0]]
print("I think it is {}ed image".format(predicted_class.lower()))










































