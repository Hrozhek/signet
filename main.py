import os

import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

np.random.seed(42)

def load_data(dir, test_size=0.2, verbose=True, load_grayscale=True):
    """
        Loads the data into a dataframe.

        Arguments:
            dir: str
            test_size: float
        Returns:
            (x_train, y_train, x_test, y_test, x_val, y_val, df)
    """
    features = []
    features_forged = []
    features_real = []
    features_dict = {}
    labels = []  # forged: 0 and real: 1
    mode = "rgb"
    if load_grayscale:
        mode = "grayscale"

    for folder in os.listdir(dir):
        # forged images
        if folder == '.DS_Store' or folder == '.ipynb_checkpoints':
            continue
        print("Searching folder {}".format(folder))
        for sub in os.listdir(dir + "/" + folder + "/forge"):
            f = dir + "/" + folder + "/forge/" + sub
            img = load_img(f, color_mode=mode, target_size=(150, 150))
            features.append(img_to_array(img))
            features_dict[sub] = (img, 0)
            features_forged.append(img)
            if verbose:
                print("Adding {} with label 0".format(f))
            labels.append(0)  # forged
        # real images

        for sub in os.listdir(dir + "/" + folder + "/real"):
            f = dir + "/" + folder + "/real/" + sub
            img = load_img(f, color_mode=mode, target_size=(150, 150))
            features.append(img_to_array(img))
            features_dict[sub] = (img, 1)
            features_real.append(img)
            if verbose:
                print("Adding {} with label 1".format(f))
            labels.append(1)  # real

    features = np.array(features)
    labels = np.array(labels)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

    print("Generated data.")
    return features, labels, features_forged, features_real, features_dict, x_train, x_test, y_train, y_test, x_val, y_val


def convert_label_to_text(label=0):
    """
        Convert label into text

        Arguments:
            label: int
        Returns:
            str: The mapping
    """
    return "Forged" if label == 0 else "Real"


features, labels, features_forged, features_real, features_dict, x_train, x_test, y_train, y_test, x_val, y_val = load_data("F:\\Desktop\\test_set\\Dataset_Signature_Final\\Dataset",
    verbose=False, load_grayscale=False)

f, axarr = plt.subplots(3,3)
axarr[0,0].imshow(features[0]/255.)
axarr[0,0].text(2, 2, labels[0], bbox={'facecolor': 'white', 'pad': 3})
axarr[0,1].imshow(features[1]/255.)
axarr[0,1].text(2, 2, labels[1], bbox={'facecolor': 'white', 'pad': 3})
axarr[0,2].imshow(features[2]/255.)
axarr[0,2].text(2, 2, labels[2], bbox={'facecolor': 'white', 'pad': 3})
axarr[1,0].imshow(features[300]/255.)
axarr[1,0].text(2, 2, labels[300], bbox={'facecolor': 'white', 'pad': 3})
axarr[1,1].imshow(features[400]/255.)
axarr[1,1].text(2, 2, labels[400], bbox={'facecolor': 'white', 'pad': 3})
axarr[1,2].imshow(features[512]/255.)
axarr[1,2].text(2, 2, labels[512], bbox={'facecolor': 'white', 'pad': 3})
axarr[2,0].imshow(features[6]/255.)
axarr[2,0].text(2, 2, labels[6], bbox={'facecolor': 'white', 'pad': 3})
axarr[2,1].imshow(features[200]/255.)
axarr[2,1].text(2, 2, labels[200], bbox={'facecolor': 'white', 'pad': 3})
axarr[2,2].imshow(features[100]/255.)
axarr[2,2].text(2, 2, labels[100], bbox={'facecolor': 'white', 'pad': 3})
plt.show()

print ("Distribution: {}".format(np.bincount(labels)))

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3,3), input_shape=(150,150,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2))
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2))
model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2))
model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy, metrics=['acc'])

model.summary()

x_train /= 255.
x_val /= 255.
x_test /= 255.

history = model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))

def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

plot_history(history)

# loading Inception
model2 = tf.keras.applications.InceptionV3(include_top=False, input_shape=(150,150,3))
# freezing layers
for layer in model2.layers:
    layer.trainable=False
# getting mixed7 layer
l = model2.get_layer("mixed7")
print ("mixed7 shape: {}".format(l.output_shape))

x = tf.keras.layers.Flatten()(l.output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(.5)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
net = tf.keras.Model(model2.input, x)
net.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy, metrics=['acc'])

h2 = net.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5)

plot_history(h2)

preds = net.predict(x_test)
pred_labels = []

# threshold is 0.5
for p in preds:
    if p >= 0.5:
        pred_labels.append(1)
    else:
        pred_labels.append(0)
pred_labels = np.array(pred_labels)

print ("Accuracy on test set: {}".format(accuracy_score(y_test, pred_labels)))
