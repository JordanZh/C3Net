from huggingface_hub import hf_hub_download
import tarfile
import os

for i in range(90):
  hf_hub_download(repo_id="Meranti/CLAP_freesound", filename=f"freesound_no_overlap/test/{i}.tar", repo_type="dataset", local_dir="FreeSound_test")

def extract_zip_file(extract_path):
      with tarfile.open(f"{extract_path}.tar") as tfile:
          tfile.extractall(extract_path)
      # remove zipfile
      tarfileTOremove=f"{extract_path}"+".tar"
      if os.path.isfile(tarfileTOremove):
          os.remove(tarfileTOremove)
      else:
          print("Error: %s file not found" % tarfileTOremove)

extract_train_path = "./FreeSound_test/freesound_no_overlap/test/"

for i in range(90):
  extract_zip_file(f"{extract_train_path}{i}")
