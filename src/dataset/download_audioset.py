import os
import joblib
import pandas as pd
import json
import hashlib

class Downloader:
    """
    This class implements the download of the AudioSet dataset.
    It only downloads the audio files according to the provided list of labels and associated timestamps.
    """
    def __init__(self,
                 download_root_path: str,
                 n_jobs: int = 1,
                 download_caption_csv: str = 'AudioSet/unbalanced_train_segments.csv',
                 download_time_csv: str = 'AudioSet/unbalanced_train_segments.csv',
                 ):
        """
        This method initializes the class.
        :param download_root_path: root path of the dataset
        :param n_jobs: number of parallel jobs
        :param download_caption_csv: type of download (test.csv, val.csv, train.csv)
        :param download_time_csv: type of download (test.csv, val.csv, train.csv)
        """
        # Set the parameters
        self.download_root_path = download_root_path
        self.n_jobs = n_jobs
        self.download_caption_csv = download_caption_csv
        self.download_time_csv = download_time_csv
        self.file_format = None
        self.quality = None
        self.caption_dict = {}
        # Create the path
        os.makedirs(self.download_root_path, exist_ok=True)

    def hash_str2long(self, string, mod=10**8):
        return int(hashlib.sha1(string.encode("utf-8")).hexdigest(), 16) % mod
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
        caption_dataframe = pd.read_csv(
            self.download_caption_csv,
            sep=',',
            skiprows=2,
            header=None,
            names=['youtube_id', 'caption', 'caption_t5']
        )
        # remove " in the caption
        caption_dataframe['caption'] = caption_dataframe['caption'].apply(lambda x: x.replace('"', '')[14:])
        caption_dataframe['caption_t5'] = caption_dataframe['caption_t5'].apply(lambda x: str(x).replace('"', ''))
        caption_dataframe = caption_dataframe.reset_index(drop=True)
        print(f'Downloading {len(caption_dataframe)} files...')
        #print(metadata.loc[3232])
        #input()

        time_dataframe = pd.read_csv(
            self.download_time_csv,
            sep=', ',  # Here's a different
            skiprows=3,
            header=None,
            names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels']
        )
        time_dataframe = time_dataframe.reset_index(drop=True)

        # hash_map = np.zeros(10**8)
        mapping_YTID2time = {}
        for i in range(0, len(time_dataframe)):
            YTID = time_dataframe.loc[i, 'YTID']
            start_seconds = time_dataframe.loc[i, 'start_seconds']
            # hash_ind = self.hash_str2long(YTID)
            mapping_YTID2time[YTID] = start_seconds

        # Download the dataset
        joblib.Parallel(n_jobs=self.n_jobs, verbose=10)(
            joblib.delayed(self.download_file)(i, caption_dataframe.loc[i, 'youtube_id'],
                                               [caption_dataframe.loc[i, 'caption'], caption_dataframe.loc[i, 'caption_t5']],
                                               start_time=mapping_YTID2time[caption_dataframe.loc[i, 'youtube_id']]
                                               if caption_dataframe.loc[i, 'youtube_id'] in mapping_YTID2time
                                               else -1) for i
            in range(390400, len(caption_dataframe))
        )

        for index in range(0, len(caption_dataframe)):
            file_path_wo_suffix = os.path.join(self.download_root_path, str(index//10000), str(index))
            file_path_w_suffix = file_path_wo_suffix + f'.{self.file_format}'
            youtube_id = caption_dataframe.loc[index, 'youtube_id']
            start_time = mapping_YTID2time[youtube_id]
            caption = [caption_dataframe.loc[index, 'caption'], caption_dataframe.loc[index, 'caption_t5']]
            self.caption_dict[file_path_w_suffix] = {'info': f'AudioCap_youtubeId={youtube_id}_startTime={start_time}', 'caption': caption}

        with open(f"{self.download_root_path}/caption.json", "w") as outfile:
            outfile.write(json.dumps(self.caption_dict, indent=4))

        print('Done.')

    def download_file(
            self,
            index,
            youtube_id: str,
            caption: list,
            start_time: float
    ):
        """
        This method downloads a single file. It only download the audio file at __16__(q=5)kHz.
        If a file is associated to multiple labels, it will be stored multiple times.
        """
        # if youtube_id not in self.mapping_YTID2time:
        #    print(index, "--------------------\n\n")
        #    return
        # start_time = self.mapping_YTID2time[youtube_id]

        if start_time < 0:
            return

        file_path_wo_suffix = os.path.join(self.download_root_path, str(index//10000), str(index))
        os.makedirs(os.path.join(self.download_root_path, str(index//10000)), exist_ok=True)

        os.system(
            f'yt-dlp -x --audio-format {self.file_format} --audio-quality {self.quality} --output "{file_path_wo_suffix}" --postprocessor-args "-ss {start_time} -to {start_time+10}" https://www.youtube.com/watch?v={youtube_id}')
        with open(f'{file_path_wo_suffix}.txt', 'w') as f:
            f.write(str(caption))

        return


if __name__ == '__main__':
    d = Downloader(download_root_path='AudioSet/unbalanced_train', n_jobs=8,
                   download_caption_csv='AudioSet/audioset_unbalanced_train.csv',
                   download_time_csv='AudioSet/unbalanced_train_segments.csv')
    d.download(file_format='wav', quality=2)












