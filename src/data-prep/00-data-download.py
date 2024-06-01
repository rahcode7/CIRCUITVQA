#!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="roboflow_api_key")
project = rf.workspace("development-tohnm").project("cghd-full-supplemented")
dataset = project.version(13).download("yolov8")

