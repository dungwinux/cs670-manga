import cv2
import algorithm as al
import processing as proc
import json
from pathlib import Path

class TextDetection:
    def __init__(self, cwd):
        self.cwd = Path(cwd)
        self.input_base = cwd / 'inputs'
        self.output_base1 = cwd / 'detboxes'
        self.output_base2 = cwd / 'detmasks'
        self.cache_base = cwd / 'mts_caches'
        self.cv2_model = al.Text_ER()
        self.mts_level = 2
    def execute(self, input_relpath, *, use_mts_cache=False):
        img_path = self.input_base / input_relpath
        if not img_path.exists():
            print(f"Cannot find {img_path}. ({self.input_base=}, {input_relpath=})")
            return
        img_name = str(img_path)
        output1 = self.output_base1 / input_relpath
        output2 = self.output_base2 / input_relpath
        output1.parent.mkdir(parents=True, exist_ok=True)
        output2.parent.mkdir(parents=True, exist_ok=True)
        result = proc.text_detection(input_relpath, self.input_base, mts_level=self.mts_level, cv2_model=self.cv2_model, use_cache=self.cache_base if use_mts_cache else None)
        output = {
            "absolute_path": img_name,
            "path": str(img_path.relative_to(self.cwd)),
            "coordinates": result.bbs
        }
        with open(output1.with_suffix('.json'), 'w') as fd:
            json.dump(output, fd)
        # mask, grayscale, should have the size equivalent to the original image, nd.array with shape (m, n) (**NOT (m, n, 3)**)
        cv2.imwrite(output2.with_suffix('.png'), result.mask)
