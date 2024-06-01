from PIL import Image
from icecream import ic 
import os 


IMAGE_PATH = "datasets/model-inputs/train/d1_251_png.rf.8361fa86a1f73533c98702783fde724e.jpg"
MAIN_PATH = "datasets/model-inputs-jn/val/"
OUTPUT_PATH = "datasets/576a/model-inputs-jn/val/"


with Image.open(IMAGE_PATH) as image:
    width, height = image.size
    ic(width,height)

for c,filename in enumerate(os.listdir(os.path.join(MAIN_PATH))):
    with Image.open(os.path.join(MAIN_PATH,filename)) as image:
        width, height = image.size
        ic(filename,width,height)

        # d3_20220729_115942_jpg.rf.a2ba069f0a1f952feee909cc6940c678.jpg
        #if filename == "d3_20220729_121143_jpg.rf.f6278befb4cd0216bee87d13c6a79875.jpg":
        #if filename == "d3_20220729_115942_jpg.rf.a2ba069f0a1f952feee909cc6940c678.jpg":

        image.thumbnail((576,576), Image.LANCZOS)
        #image = image.resize((576,576))

        op = os.path.join(OUTPUT_PATH,filename)
        image.save(op)

    # with Image.open(IMAGE_PATH) as image:
    #     xres, yres = image.info['dpi']
    #     ic(width,height)
