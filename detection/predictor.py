# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import atexit
import torch

from detectron2.engine.defaults import DefaultPredictor

class VisualizationDemo(object):
    def __init__(self, cfg):
        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        with torch.no_grad():
            predictions = self.predictor(image)
        return predictions 