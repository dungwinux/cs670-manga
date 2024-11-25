import os
import sys
import torch
import pathlib

from fastai.vision import *
# Required for loading models
root = pathlib.Path(__file__).parents[0].resolve()
sys.path.insert(0, str(root / 'Manga-Text-Segmentation/code'))
# print(root)
# print(sys.path)

import dataset as dataset

class _MTS:
    m_learner: List[Learner] = [None] * 5
    def __init__(self, model_path: str = 'models'):
        self.m_learner = [load_learner(model_path, f'fold.{_}.-.final.refined.model.2.pkl') for _ in range(5)]
    
    def process(self, path_to_image_input: Union[Path, str]):
        im = open_image(path_to_image_input)
        pred = [self.m_learner[_].predict(im) for _ in range(5)]
        return (im, pred)

MTS = _MTS()

def colorizePrediction(prediction, truth):
    prediction, truth = prediction[0], truth[0]
    colorized = torch.zeros(4, prediction.shape[0], prediction.shape[1]).int()
    r, g, b, a = colorized[:]
    
    fn = (truth >= 1) & (truth <= 5) & (truth != 3) & (prediction == 0)
    tp = ((truth >= 1) & (truth <= 5)) & (prediction >= 1)
    fp = (truth == 0) & (prediction >= 1)
    
    r[fp] = 255
    r[fn] = g[fn] = b[fn] = 255
    g[tp] = 255

    a[:, :] = 128
    a[tp | fn | fp] = 255

    return colorized

