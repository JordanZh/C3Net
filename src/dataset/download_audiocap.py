import os
import joblib
import pandas as pd
import json

class Downloader:
    """
    This class implements the download of the AudioSet dataset.
    It only downloads the audio files according to the provided list of labels and associated timestamps.
    """
    def __init__(self,
                 download_root_path: str,
                 n_jobs: int = 1,
                 download_csv: str = 'AudioCap/test.csv',
                 ):
        """
        This method initializes the class.
        :param download_root_path: root path of the dataset
        :param n_jobs: number of parallel jobs
        :param download_csv: type of download (test.csv, val.csv, train.csv)
        """
        # Set the parameters
        self.download_root_path = download_root_path
        self.n_jobs = n_jobs
        self.download_csv = download_csv
        self.file_format = None
        self.quality = None
        self.caption_dict = {}

        # Create the path
        os.makedirs(self.download_root_path, exist_ok=True)

    def download(
            self,
            file_format: str = 'vorbis',
            quality: int = 5,
    ):
        """
        This method downloads the dataset using the provided parameters.
        :param file_format: format of the audio file (vorbis, mp3, m4a, wav), default is vorbis
        :param quality: quality of the audio file (0: best, 10: worst), default is 5
        """
        self.file_format = file_format
        self.quality = quality
        self.caption_dict = {}

        # Load the metadata
        metadata = pd.read_csv(
            self.download_csv,
            sep=',',
            skiprows=1,
            header=None,
            names=['audiocap_id', 'youtube_id', 'start_time', 'caption']
        )

        # remove " in the caption
        metadata['caption'] = metadata['caption'].apply(lambda x: x.replace('"', ''))
        metadata = metadata.reset_index(drop=True)

        print(f'Downloading {len(metadata)} files...')

        # Download the dataset
        joblib.Parallel(n_jobs=self.n_jobs, verbose=10)(
            joblib.delayed(self.download_file)(i, metadata.loc[i, 'audiocap_id'], metadata.loc[i, 'youtube_id'],
                                               metadata.loc[i, 'start_time'], metadata.loc[i, 'caption']) for i
            in range(23500, len(metadata))
        )

        for index in range(0, len(metadata)):
            file_path_wo_suffix = os.path.join(self.download_root_path, str(index//10000), str(index))
            file_path_w_suffix = file_path_wo_suffix + f'.{self.file_format}'
            youtube_id = metadata.loc[index, 'youtube_id']
            start_time = metadata.loc[index, 'start_time']
            caption = metadata.loc[index, 'caption']
            self.caption_dict[file_path_w_suffix] = {'info': f'AudioCap_youtubeId={youtube_id}_startTime={start_time}', 'caption': caption}

        with open(f"{self.download_root_path}/caption.json", "w") as outfile:
            outfile.write(json.dumps(self.caption_dict, indent=4))

        print('Done.')

    def download_file(
            self,
            index,
            audiocap_id: str,
            youtube_id: str,
            start_time: float,
            caption: str,
    ):
        """
        This method downloads a single file. It only download the audio file at __16__(q=5)kHz.
        If a file is associated to multiple labels, it will be stored multiple times.
        """

        file_path_wo_suffix = os.path.join(self.download_root_path, str(index//10000), str(index))
        file_path_w_suffix = file_path_wo_suffix + f'.{self.file_format}'
        os.makedirs(os.path.join(self.download_root_path, str(index//10000)), exist_ok=True)

        os.system(
            f'yt-dlp -x --audio-format {self.file_format} --audio-quality {self.quality} --output "{file_path_wo_suffix}" --postprocessor-args "-ss {start_time} -to {start_time+10}" https://www.youtube.com/watch?v={youtube_id}')

        return


if __name__ == '__main__':
    d = Downloader(download_root_path='AudioCap/train', n_jobs=8, download_csv='AudioCap/train.csv')
    d.download(file_format='wav', quality=2)
