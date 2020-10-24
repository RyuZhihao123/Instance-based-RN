import json
import cv2,base64

class WRITE_JSON:

    @staticmethod
    def GetShapes(points,type):
        shapes = []

        for i in range(len(points)):

            curShape = {"label": type+ str(i+1),
                        "points": points[i],
                        "group_id": 0,
                        "shape_type": "polygon",
                        "flags": {}}
            shapes.append(curShape)

        return shapes

    @staticmethod
    def SAVE_TO_FILE( target_json_file, image, image_name, points = [],  itype = 'bar'):

        string_image = base64.b64encode(cv2.imencode('.png', image)[1]).decode()

        data =  { 'version' : "4.5.6", 'flags' : {},
                   'shapes' : WRITE_JSON.GetShapes(points, itype),
                   'imagePath' : "{}.png".format(image_name),
                   'imageData' : string_image }


        with open(target_json_file,'w',encoding='utf-8') as file:
            file.write(json.dumps(data,indent=2))
