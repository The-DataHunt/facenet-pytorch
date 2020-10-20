import logging
from PIL import Image
import requests
import json
import io
import os
from FaceDetector import FaceDetector
import torch
from ts.torch_handler.base_handler import BaseHandler
import sys

class FacenetHandler(BaseHandler):
    def __init__(self, *args, **kwargs):
        self.model = None
        self.initialized = False
        self.device = "cpu" 
        self.model_dir = ""

    def initialize(self, ctx):
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.model_dir = model_dir
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        onet_path = os.path.join(model_dir, "onet.pt")
        pnet_path = os.path.join(model_dir, "pnet.pt")
        rnet_path = os.path.join(model_dir, "rnet.pt")

        # Read model definition file
        model_def_path = os.path.join(model_dir, "FaceDetector.py")

        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model def file")

        # Load trained weights
        self.model = FaceDetector(device=self.device)
        self.model.onet.load_state_dict(torch.load(onet_path, map_location=self.device))
        self.model.pnet.load_state_dict(torch.load(pnet_path, map_location=self.device))
        self.model.rnet.load_state_dict(torch.load(rnet_path, map_location=self.device))
        self.model.eval()

        self.initialized = True

    def preprocess_one_image(self, req, i):
        json_dict = req

        img_url    = json_dict['imgurl'] if 'imgurl' in json_dict else ''
        #image_bytes = requests.files['file'] if 'file' in requests.files else None
        image_bytes = json_dict['file'] if 'file' in json_dict else None
        req_image = ('req_image' in json_dict)

        if img_url:
            response = requests.get(img_url.decode('utf-8'))
            org_img = Image.open(io.BytesIO(response.content))
        else :
            #image_bytes = image_bytes.read()
            org_img = Image.open(io.BytesIO(image_bytes))
        
        org_img = org_img.convert('RGB')
        return org_img

    def preprocess(self, reqs):
        images = [self.preprocess_one_image(req,i) for i,req in enumerate(reqs)]
        return images

    def inference(self, x):
        with torch.no_grad():
            outs = [self.model(d)[0] for d in x]
        return outs

    def postprocess(self, preds):
        output = []
        for pred in preds:
            # each pred is a 2d-array containing the list of coords
            if pred is not None : output.append({ "dets" : [ list(coords.astype(str)) for coords in pred ]})
            else: output.append({"dets" : []})
        return output

_service = FacenetHandler()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None
    
    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data