import numpy as np
import soundfile as sf
import librosa
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
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
    # Load audio
    audio, _ = librosa.load(path, sr=SR, mono=True)

    # Pad or truncate
    if len(audio) < TARGET_LEN:
        audio = np.pad(audio, (0, TARGET_LEN - len(audio)))
    else:
        audio = audio[:TARGET_LEN]

    # Normalize safely
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(audio)

    # STFT magnitude (log scale)
    stft = librosa.stft(y=audio)
    stft_features = np.abs(stft).flatten()

    # MFCC mean & std
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=40)
    mfcc_features = mfcc.flatten()

    # Combine features
    final_features = np.r_[stft_features, mfcc_features]

    return final_features

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

    k_values = range(1, 25, 2)
    accuracys = list()
    tram_precisions = list()
    car_precisions = list()
    tram_recalls = list()
    car_recalls = list()

    for k in k_values:
        print(f"k = {k}:")
        y_pred = classify(x_train, y_train, x_valid, k)
        accuracy, tram_precision, car_precision, tram_recall, car_recall = score(y_valid, y_pred)
        accuracys.append(accuracy)
        print(f"Accuracy = {accuracy}:")
        tram_precisions.append(tram_precision)
        print(f"Tram precision = {tram_precision}:")
        car_precisions.append(car_precision)
        print(f"Car precision = {car_precision}:")
        tram_recalls.append(tram_recall)
        print(f"Tram recall = {tram_recall}:")
        car_recalls.append(car_recall)
        print(f"Car recall = {car_recall}:")

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracys, marker='o')
    plt.title('K vs Validation Accuracy')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Validation Accuracy')
    plt.xticks(k_values)
    plt.grid(True)
    plt.savefig("results/k_accuracy.png")
    plt.show()

    fig, axs = plt.subplots(2, 1, sharex=True)

    # --- Precision plot ---
    axs[0].plot(k_values, tram_precisions, marker='o', label='Tram Precision')
    axs[0].plot(k_values, car_precisions, marker='s', label='Car Precision')
    axs[0].set_ylabel('Precision')
    axs[0].set_title('Precision vs k')
    axs[0].legend()
    axs[0].grid(True)

    # --- Recall plot ---
    axs[1].plot(k_values, tram_recalls, marker='o', label='Tram Recall')
    axs[1].plot(k_values, car_recalls, marker='s', label='Car Recall')
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('Recall')
    axs[1].set_title('Recall vs k')
    axs[1].legend()
    axs[1].grid(True)

    plt.xticks(k_values)
    plt.tight_layout()
    plt.savefig("results/precision_recall_k.png")
    plt.show()


def score(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    tram_precision = precision_score(y_test, y_pred, pos_label="tram")
    car_precision = precision_score(y_test, y_pred, pos_label="car")
    tram_recall = recall_score(y_test, y_pred, pos_label="tram")
    car_recall = recall_score(y_test, y_pred, pos_label="car")

    return accuracy, tram_precision, car_precision, tram_recall, car_recall

def final_test(x_train, y_train, x_test, y_test):
    y_pred = classify(x_train, y_train, x_test, 7)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Final Accuracy: {accuracy:.4f}")

    # 2️⃣ Classification report (precision, recall, f1-score per class)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    # 3️⃣ Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=["tram", "car"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["tram", "car"])

    # 4️⃣ Plot nicely
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)
    plt.title("Confusion Matrix - Tram vs Car")
    plt.savefig("results/confusion_matrix.png")
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
    x_test, y_test = process_data(tram_test_path, car_test_path, tram_label, car_label)

    #test_k(x_train, x_validation, y_train, y_validation)

    x_train = np.r_[x_train, x_validation]
    y_train = np.r_[y_train, y_validation]

    final_test(x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    main()
