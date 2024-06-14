import pandas as pd
import librosa
from urllib.request import urlretrieve
import os
from pathlib import Path
import libfmp.b
import joblib


class Download_EpidemicDataset():
    def __init__(self, dataset_csv_file, writePath):

        self.dataPath = dataset_csv_file
        self.writePath = writePath
        self.pd_frames = pd.read_csv(self.dataPath)

    def download(self, audio_url, index, text):
        response = urlretrieve(audio_url, 'temp.mp3')
        waveform, sample_rate = librosa.load('temp.mp3')

        if waveform.shape[0] > sample_rate * 11:
            waveform = waveform[:sample_rate * 11]

        subdirectory = str(int(index / 10000))
        path = Path(f"{self.writePath}/{subdirectory}")
        path.mkdir(parents=True, exist_ok=True)

        fileName = os.path.join(self.writePath, subdirectory, format(index, '07d') + '.flac')
        #sf.write(fileName, waveform, sample_rate)
        libfmp.b.b_audio.write_audio(fileName, waveform, sample_rate)

        txt_file = os.path.join(self.writePath, subdirectory, format(index, '07d') + '.txt')
        with open(txt_file, 'w') as f:
            f.write(text)

    def down_load_i(self, index):
        audio_url = self.pd_frames.iloc[index, 0]
        text = self.pd_frames.iloc[index, 3]
        if type(text) is not str:  # ?? Not sure if this work
            text = self.pd_frames.iloc[index, 2]
            if text[:14] == 'the sounds of ': text = text[14:]  # delete 'the sounds of '

        self.download(audio_url, index, text)
        try:
            self.download(audio_url, index, text)
        except:
            print(index, '--------------------------------------------\n\n')
        if index % 100 == 0:
            print("good @ index=", index)


    def download_all(self, start, end):
        assert end <= len(self.pd_frames)

        joblib.Parallel(n_jobs=1, verbose=10)(
            joblib.delayed(self.down_load_i)(i) for i
            in range(start, end)
            )


downloader = Download_EpidemicDataset('../Epidemic_all_debiased.csv', 'EpidemicDataset')
downloader.download_all(20300, 70800)
