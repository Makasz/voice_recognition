import numpy as np
import scipy.io.wavfile as wav
import scipy.fftpack as fft
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sn
import pandas as pd

THRESHOLD = 320
INPUT_DIR = "./" + str(sys.argv[1]) + "/"
error_matrix = np.zeros((2,2))
lista = []


def max_spect(spectrum):
    a = np.copy(spectrum)
    for x in range(2, 5):
        r_len = len(spectrum[::x])
        a[:r_len] *= spectrum[::x]
    # plt.plot(range(len(a[40:r_len])), a[40:r_len])
    # plt.savefig("wykres_hps.pdf")od
    return np.argmax(a[40:r_len])


def hps(audio, sample_rate):
    audio_len = len(audio)
    part = np.hamming(len(audio)) * audio
    wave_spect = np.log(np.abs(fft.rfft(part)))
    # plt.plot(range(len(wave_spect)), wave_spect)
    # plt.savefig("wykres_013M_spect.pdf")
    return max_spect(wave_spect) * sample_rate / audio_len


def error_matrix_creator(filename, is_man):
    if "M" in filename and is_man:
        error_matrix[1, 1] += 1
    if "M" in filename and not is_man:
        error_matrix[0, 1] += 1
    if "K" in filename and is_man:
        error_matrix[1, 0] += 1
    if "K" in filename and not is_man:
        error_matrix[0, 0] += 1
    return 0


def display(error_matrix):
    print("Trafność: " + str((error_matrix[1, 1] + error_matrix[0, 0]) * 100 / (
    error_matrix[1, 1] + error_matrix[0, 0] + error_matrix[0, 1] + error_matrix[1, 0])) + "%")
    error_matrix = error_matrix.astype(int)
    df_cm = pd.DataFrame(error_matrix, index=[i for i in "KM"], columns=[i for i in "KM"])
    sn.set(font_scale=1.4)
    ax = sn.heatmap(df_cm, annot=True, center=np.sum(error_matrix) / 4, cbar=False)
    ax.set_xlabel('Stan rzeczywisty')
    ax.set_ylabel('Stan przewidziany ')
    plt.savefig('matrix.pdf')
    plt.show()
    return 0


for file in os.listdir(INPUT_DIR):
    sample_rate, audio = wav.read(INPUT_DIR + file)
    audio = audio.astype(float)
    audio -= np.mean(audio)
    if len(audio.shape) == 2:
        audio = (audio[:,0] + audio[:,1]) / 2
    calc_data = hps(audio, sample_rate)
    if "K" in file:
        s_type = "Kobieta  "
    else:
        s_type = "Mężczyzna"

    error_matrix_creator(file, calc_data < THRESHOLD)

    print("Plik: " + file + "  Źródło: " + s_type)

display(error_matrix)
