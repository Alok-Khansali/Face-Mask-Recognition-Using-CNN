# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4   	# initial learning rate - learning rate is less then loss will be calculated correctly(0.00001)
EPOCHS = 20
BS = 32

# Directory shows where the dataset is present
DIRECTORY = r"E:\Face-Mask-Detection-master\dataset"
# categories of data set
CATEGORIES = ["with_mask", "without_mask"]

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")


# appending all image arrays in datalist
data = []
# append all the data related to dataset which is with_mast and without_mask
labels = []


# looping through the dataset("CATEGORIES")
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):              					# listdir - list down all the images in particular directory
    	img_path = os.path.join(path, img)    					# joining the path to corresponding image
    	image = load_img(img_path, target_size=(224, 224))   	# load_img - loads the image path, target size is height and width 															  of the image
    	image = img_to_array(image)            					# saving the image to a varable 'image'
    	image = preprocess_input(image)        					# preprocess input is coming from mobileNet

    	data.append(image)
    	labels.append(category)

# data is in numerical value but labels are still alphabetical values - so converting them
# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)  								# converting dataset to numerical values like 0 and 1
labels = to_categorical(labels)


# converting the data to numpy arrays 
data = np.array(data, dtype="float32")
labels = np.array(labels)

# train test split - 20% to testing and 80% for training the model
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)


# construct the training image generator for data augmentation using ImageDataGenerator - used for creating many images using single images but with different properties
aug = ImageDataGenerator( 
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
# input tensor shape of image(rgb = 3)
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)  			# relu is goto activation function for non linear use Cases
headModel = Dropout(0.5)(headModel)                   			# dropout is used for avoiding the overfitting of the model
headModel = Dense(2, activation="softmax")(headModel) 			# softmax function is probability based function (2 describes with 																	  mask and without mask)


# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)


# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process because they just a replacement for CNN
for layer in baseModel.layers:
	layer.trainable = False


# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])


# train the head of the network
print("[INFO] training head...")
# fitting the model
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))


# serialize the model to disk
print("[INFO] saving mask detector model...")

# An H5 file is a data file saved in the Hierarchical Data Format (HDF). It contains multidimensional arrays of scientific data. 
model.save("mask_detector.model", save_format="h5")


# plot the training loss and accuracy using matplotlib
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="center right")
plt.savefig("plot.png")  										# saving the image using matplotlib