from cv_utils.callbacks import TrainingMonitor
from cv_utils.nn import MiniVGGNet
from cv_utils.preprocessing import AspectAwarePreprocessor
from cv_utils.preprocessing import ImageToArrayPreprocessor
from cv_utils.datasets import SimpleDatasetLoader
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import os

dataset_folder = "flowers17/images/"
weights = "/artifacts/minivggnet.hdf5";
jsonPath = "/artifacts/flowers.json"
picpath = "/artifacts/graph.jpg"

classNames = list(os.listdir(dataset_folder));
imagePaths = [];
for _class in classNames:
    new_folder_name = dataset_folder + _class + '/';
    imagePaths.extend([new_folder_name + name for name in os.listdir(new_folder_name)]);



aap = AspectAwarePreprocessor(64, 64);
iap = ImageToArrayPreprocessor();

sdl = SimpleDatasetLoader(preprocessors=[aap, iap]);
(data, labels) = sdl.load(imagePaths, verbose=500);
data = data.astype("float")/255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state =42);

trainY = LabelBinarizer().fit_transform(trainY);
testY = LabelBinarizer().fit_transform(testY);

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest");

print("[INFO] compiling model")


opt = SGD(lr = 0.05);
model = MiniVGGNet.build(height = 64, width = 64, depth = 3, classes = len(classNames));
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"]);
checkpoint = ModelCheckpoint(weights, monitor = "val_loss", save_best_only=True, verbose =1 );



callbacks = [TrainingMonitor(picpath, jsonPath=jsonPath)];

print("[INFO] training network ...");
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY), steps_per_epoch=len(trainX)//32, callbacks = callbacks, epochs = 100, verbose = 1);

print("[INFO] evaluating network ... ");
predictions = model.predict(testX, batch_size = 32);
print(classification_report(testY.argmax(axis = 1), predictions.argmax(axis = 1), target_names=classNames));

