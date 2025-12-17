import numpy as np
import soundfile as sf
import librosa
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import matplotlib.pyplot as plt

# Sampling rate. I use librosa.load
# to convert everything to the same sampling rate
SR = 16000
# The samples were supposed to be 5-6 seconds long
# so I doubt there will be samples less than 4 seconds
# long
DURATION = 4
# Length of the vector when the audio is processed
# to pyhton
TARGET_LEN = SR*DURATION


def preprocess(in_path, out_path):
    # wav conversion and normalization
    audio, _ = librosa.load(in_path, sr=SR, mono=True)

    if len(audio) < TARGET_LEN:
        audio = np.pad(audio, (0, TARGET_LEN - len(audio)))
    else:
        audio = audio[:TARGET_LEN]

    audio = audio / np.max(np.abs(audio) + 1e-9)

    sf.write(out_path, audio, SR)

def feature_extraction(path):
    """
    Here we process the audio files. We can change
    which operations we do to it, and what kind of
    information is included in the array that is used
    in the classification
    """
    # Load the audio
    audio, _ = librosa.load(path, sr=SR, mono=True)

    # Make the audio 4 seconds long so every
    # sample is the same length
    if len(audio) < TARGET_LEN:
        # If the sample is less than 4s,  pad it with 0s
        audio = np.pad(audio, (0, TARGET_LEN - len(audio)))
    else:
        # Otherwise clip it from the beginning
        audio = audio[:TARGET_LEN]

    # Normalize the audio to 0-1
    audio = audio / np.max(audio)

    # Take the short time fourier transform
    stft = librosa.stft(y=audio)
    # The result can be complex, lets take the absolute value
    # and make the matrix into a vector
    stft = np.abs(stft).flatten()

    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=40)
    mfcc = mfcc.flatten()

    final = np.r_[stft, mfcc]

    return final

def process_data(path1, path2, label1, label2):
    """
    This function processes the data from two folders
    to an array with the data as vectors, and an array with
    the same indeces with the labels for the corresponding
    vectors
    """
    print("Starting processing")
    files1 = os.listdir(path1)
    files2 = os.listdir(path2)

    # Use lists at this point to make it easier to
    # process any type and size of data
    data = list()
    labels = list()

    n = 0
    for file in files1:
        # Add the data to the data list
        data.append(feature_extraction(f"{path1}/{file}"))
        # Add the label to label list
        labels.append(label1)
        if n % 10 == 0: print(f"{n} processed")
        n += 1

    for file in files2:
        data.append(feature_extraction(f"{path2}/{file}"))
        labels.append(label2)
        if n % 10 == 0: print(f"{n} processed")
        n += 1

    # Convert the list to numpy array
    data = np.array(data)
    labels = np.array(labels)

    return data, labels

def classify(x_train, y_train, x_test, n_neighbors):
    # Create classifier instance
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    print("Starting classifying")
    # Feed the data to it
    knn.fit(x_train, y_train)
    print("Start predicting")
    # Do the classifying
    y_pred = knn.predict(x_test)
    return y_pred

def test_k(x_train, x_valid, y_train, y_valid):

    ks = range(1, 20, 2)
    accuracys = list()

    for k in ks:
        print(f"k = {k}:")
        y_pred = classify(x_train, y_train, x_valid, k)
        accuracy = accuracy_score(y_valid, y_pred)
        accuracys.append(accuracy)
        print(f"Accuracy = {accuracy}:")

    plt.figure(figsize=(8, 5))
    plt.plot(ks, accuracys, marker='o')
    plt.title('K vs Validation Accuracy')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Validation Accuracy')
    plt.xticks(ks)
    plt.grid(True)
    plt.show()

def main():
    # Define the paths to the audio files and
    # the labels for them
    tram_train_path = "tram_train"
    car_train_path = "car_train"
    tram_validation_path = "tram_validation"
    car_validation_path = "car_validation"
    tram_test_path = "tram_test"
    car_test_path = "car_test"

    tram_label = "tram"
    car_label = "car"

    # Process the information
    x_train, y_train = process_data(tram_train_path, car_train_path, tram_label, car_label)
    x_validation, y_validation = process_data(tram_validation_path, car_validation_path, tram_label, car_label)

    test_k(x_train, x_validation, y_train, y_validation)


if __name__ == "__main__":
    main()
