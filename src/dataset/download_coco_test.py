import os
from zipfile import ZipFile, BadZipFile

os.system('wget http://images.cocodataset.org/zips/val2017.zip -O ./COCO_val/coco_val2017.zip')
os.system('wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O ./COCO_val/coco_ann2017.zip')


def extract_zip_file(extract_path):
    try:
        with ZipFile(extract_path+".zip") as zfile:
            zfile.extractall(extract_path)
        # remove zipfile
        zfileTOremove=f"{extract_path}"+".zip"
        if os.path.isfile(zfileTOremove):
            os.remove(zfileTOremove)
        else:
            print("Error: %s file not found" % zfileTOremove)
    except BadZipFile as e:
        print("Error:", e)


extract_val_path = "./COCO_val/coco_val2017"
extract_ann_path = "./COCO_val/coco_ann2017"
extract_zip_file(extract_val_path)
extract_zip_file(extract_ann_path)
