import shutil
import os

path_json = ".\\dataset\\labelme_json\\"
path_mask = ".\\dataset\\cv2_mask\\"

json_file = os.listdir(path_json)
for file in json_file:
    fileid = file.split("_")[0]
    print("++++++++++++++ fileid = " + fileid)
    shutil.copyfile(path_json + fileid + "_json\\label.png", 
        path_mask + fileid + ".png")
