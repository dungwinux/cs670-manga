import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import SemanticInpaintingModel, MangaInpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .svae import ScreenVAE
import torch.nn.functional as F
from .morphology import Dilation2d, Erosion2d


class MangaInpaintor():
    def __init__(self, config):
        self.config = config

        self.semantic_inpaint_model = SemanticInpaintingModel(config).to(config.DEVICE)
        self.manga_inpaint_model = MangaInpaintingModel(config).to(config.DEVICE)
        self.svae_model = ScreenVAE().to(config.DEVICE)

        self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_LINE_FLIST, config.TEST_MASK_FLIST, augment=False, training=False)
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

    def load(self):
        print('Loading models...')
        self.semantic_inpaint_model.load()
        self.manga_inpaint_model.load()

    def test(self):
        self.semantic_inpaint_model.eval()
        self.manga_inpaint_model.eval()

        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:
            name = self.test_dataset.load_name(index)
            images, lines, masks = self.cuda(*items[:3])
            h, w = items[3]
            index += 1

            # Print input dimensions
            print(f"Processing: {name}")
            print(f"Original Input Dimensions: Images {images.shape}, Lines {lines.shape}, Masks {masks.shape}")

            dilate = Dilation2d(1, 1, 3, soft_max=False)
            masks = dilate(masks, iterations=2)
            manga_masked = (images * (1 - masks)) + masks
            lines_masked = (lines * (1 - masks)) + masks

            screen_masked = self.svae_model(manga_masked, lines_masked, rep=True)
            screen0 = self.svae_model(images, lines, rep=True)

            manga_masked = (images * (1 - masks)) + masks
            lines_masked = (lines * (1 - masks)) + masks

            screenl, linesl, masksl = self.semantic_inpaint_model.test(screen_masked, lines_masked, masks)

            screen = screenl[-1]
            lines = linesl[-1]
            screen_decode = screen_masked * (1 - masks) + screen * masks
            lines_decode = lines_masked * (1 - masks) + lines * masks

            outputs = self.manga_inpaint_model(images, torch.cat([screen, lines], 1), masks)
            outputs_merged = (outputs * masks) + (images * (1 - masks))
            outputs_merged_l = (outputs_merged + 1) * (lines + 1) / 2 - 1

            # Print output dimensions
            print(f"Output Dimensions: {outputs_merged.shape}")

            # Ensure dimensions match original input before saving
            if outputs_merged.shape[2] >= h and outputs_merged.shape[3] >= w:
                self.save_images(outputs_merged[:, :, :h, :w], name, '')
                print(f"Saved output for {name}")
            else:
                print(f"Error: Output dimensions {outputs_merged.shape} are smaller than input dimensions ({h}, {w}). Skipping file.")

            torch.cuda.empty_cache()

        print('\nEnd test....')

    def save_images(self, img, name, fld_name):
        output = self.postprocess(img)[0]
        print(f"Postprocessed Image Dimensions: {output.shape}")
        os.makedirs(os.path.join(self.results_path, fld_name), exist_ok=True)
        path = os.path.join(self.results_path, fld_name, name)

        # Print saved file path and output dimensions
        print(f"Saving image at: {path}, Image Dimensions: {output.shape}")
        imsave(output, path)


    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        # img = img * 255.0
        img = img * 127.5+127.5
        img = img.permute(0, 2, 3, 1)
        return img.int()