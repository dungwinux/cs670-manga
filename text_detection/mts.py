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
    m_learner: List[Learner] = [None]
    model_count: int = 5
    def __init__(self, cwd = pathlib.Path(__file__).parents[1].resolve(), model_count: int = 5, model_load_bitmask: int = -1):
        model_path = 'models'
        self.model_count = model_count
        # Batch prediction does not work because the base model does not support it
        # input_set = ImageList.from_folder(cwd / 'input_test', extensions=['.jpg'], recurse=True)
        self.m_learner = {_: load_learner(str(root / model_path), f'fold.{_}.-.final.refined.model.2.pkl') for _ in range(model_count) if (1 << _) & model_load_bitmask}
        print(f"Loaded {len(self.m_learner)} out of {self.model_count} models")

    def process(self, path_to_image_input: Union[Path, str]):
        im = open_image(path_to_image_input)
        pred = [learner.predict(im) for (_, learner) in self.m_learner.items()]
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
