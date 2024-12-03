import glob
from pathlib import Path
import algorithm as al
from tqdm import tqdm
import cv2
import processing as proc
from visualize import visualize_contours, visualize_cv2
import random

# model = al.Stub()
model = al.Text_ER()
# model.er1 = cv2.text.createERFilterNM1(model.erc1, 16, 0.0005, 0.5, 0.4, True, 0.05)
model.er1 = cv2.text.createERFilterNM1(model.erc1, 16, 0.0005, 0.7, 0.25, True, 0.05)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
def run(img_name, in_base, out_base, *, noshow=True):
    p = Path(img_name).relative_to(in_base)
    filename = p.name
    rel_path = p.parent
    out_dir = out_base / rel_path
    # print(img_name, '=>', out_dir)
    # mask, bbs, mtsout, filtered, cv_mask
    result  = proc.text_detection(img_name, mts_level=1, cv2_model=model, group_kernel=kernel)
    img = cv2.imread(img_name)
    if not noshow:
        visualize_cv2(result.mts_img)
        visualize_cv2(result.filtered_img)
        visualize_cv2(result.cv2_det)
        visualize_cv2(result.mask)
    bb_visual = visualize_contours(result.bbs, img, noshow=noshow)
    
    postfix_1 = "_txtdet"
    postfix_2 = "_txtmsk"
    # Don't save yet since it will crash
    # out_dir.mkdir(parents=True, exist_ok=True)
    # z1 = out_dir / (p.stem + postfix_1 + p.suffix)
    # z2 = out_dir / (p.stem + postfix_2 + p.suffix)
    # cv2.imwrite(z1, bb_visual)
    # cv2.imwrite(z2, mask)

cwd = Path('/home/saratoga/cs670-manga/')
input_base = str(cwd / 'inputs')
output_base = str(cwd / 'outputs')
root = str(cwd)
test_set = glob.glob(input_base + '/**/*.jpg', recursive=True)

sample = random.sample(test_set, 5)
for test in tqdm(sample):
    print(test)
    run(test, input_base, output_base, noshow=True)
# run `scalene perftest.py`