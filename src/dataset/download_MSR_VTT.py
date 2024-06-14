import os
import joblib
import json
import csv
from os import listdir
from os.path import isfile, join


class Download_Video_MSR_VTT():
    def __init__(self, video_dict, write_path, config):
        self.video_dict = video_dict
        self.write_path = write_path
        self.n_jobs = config['n_jobs']
        self.config = config
        self.csv_list = []

    def fix(self):
        files = [f for f in listdir('MSR_VTT/videos') if isfile(join('MSR_VTT/videos', f))]
        for f in files:
            os.system(f'ffmpeg -i video.ts -c copy MSR_VTT/videos/{f}')
            os.system(f'ffmpeg')

    def download_one(self, video_item):
        video_url = video_item['url']
        start = video_item['start time']
        duration = video_item['end time'] - video_item['start time']
        #if duration > 10:
        #    start_skip = (duration - 10) // 2
        #    start = start + start_skip
        #    duration = 10
        this_video_id = video_item['video_id']

        reply = os.system(
            f"ffmpeg -ss {start} -i $(yt-dlp -f 22 -g '{video_url}') -t {duration} -c copy {self.write_path}/{this_video_id}.mp4")


    def download_all(self):
        joblib.Parallel(n_jobs=self.n_jobs, verbose=10)(
            joblib.delayed(self.download_one)(video_item) for video_item in self.video_dict
        )

    def generate_csv_for_videomae(self):
        csv_list = []

        for filename in os.listdir(self.write_path):
            file_path_from_dataset = os.path.join(self.write_path, filename)
            # checking if it is a file
            if os.path.isfile(file_path_from_dataset):
                if os.path.getsize(file_path_from_dataset) > 50000:
                    if filename[5: -4] != '3584' and filename[5: -4] != '5190' and filename[5: -4] != '1453':
                        if filename[5: -4] not in ['3227', '6785']:  # '2994', '5807', '6763', '920', '1331',
                            csv_list.append([f"dataset/{file_path_from_dataset}", filename[5: -4]])

        with open('MSR_VTT/dataset_helper_for_videoMae.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=' ')
            writer.writerows(csv_list)


config = {'n_jobs': 8,
          }

f = open('MSR_VTT/train_val_videodatainfo.json')
data = json.load(f)
caption_dict = {}
for item in data['sentences']:
    video_id = item['video_id']
    caption = item['caption']
    if video_id in caption_dict:
        caption_dict[video_id].append(caption)
    else:
        caption_dict[video_id] = [caption]

with open("MSR_VTT/train_val_caption.json", "w") as outfile:
    json.dump(caption_dict, outfile)

downloader = Download_Video_MSR_VTT(data['videos'], 'MSR_VTT/video22', config)
#downloader.download_all()
downloader.generate_csv_for_videomae()
#os.system("ffmpeg -ss 0 -i $(yt-dlp -f 18 -g 'https://www.youtube.com/watch?v=9lZi22qLlEo') -t 10 -c copy 9.mp4")
