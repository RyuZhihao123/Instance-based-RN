import os
import shutil
path = '.\\maskrcnn\\dataset\\json\\'  # path是你存放json的路径
path_labelme = ".\\maskrcnn\\dataset\\labelme_json\\"

print("start....")
json_file = os.listdir(path)
for file in json_file:
    print("--------------------------------------\n"+ path + file+"\n")
    os.system("labelme_json_to_dataset.exe %s"%(path + file))

    fileid = file.split(".")[0]
    print("++++++++++++++ fileid = " + fileid)
    shutil.move( path + fileid +"_json", path_labelme + fileid +"_json")