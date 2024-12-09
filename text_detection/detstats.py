import zipfile
import tarfile
from pathlib import Path
import cv2
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import sys
from skimage.segmentation import watershed
from cucim.skimage.feature import peak_local_max
from skimage.color import label2rgb
import skimage
import cupy as cp

class GTMask:
    ad: zipfile.ZipFile
    def __init__(self, path='post-processed.zip'):
        gtmask_arc = path
        if not zipfile.is_zipfile(gtmask_arc):
            raise f'Invalid zip file {gtmask_arc}!'
        self.ad = zipfile.ZipFile(gtmask_arc, 'r') # Read-only!
    def read_raw(self, relpath):
        actual_path = str(Path('post-processed') / relpath)
        return self.ad.open(actual_path)
    def read_mask(self, relpath):
        fd = self.read_raw(Path(relpath).with_suffix('.png'))
        raw_bytes = np.array(bytearray(fd.read()), dtype=np.uint8)
        fd.close()
        raw = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)
        mask_like = cv2.bitwise_not(raw)
        mask = np.where(mask_like != 0, 255, mask_like)
        return mask
    def namelist(self):
        return [str(Path(member.filename).relative_to('post-processed')) for member in self.ad.infolist() if not member.is_dir()]

class DetArcMask:
    ad: tarfile.TarFile
    def __init__(self, filename):
        if not tarfile.is_tarfile(filename):
            raise f'Invalid tar file {filename}!'
        self.ad = tarfile.open(filename, 'r:*')
    def read_raw(self, relpath):
        actual_path = str(Path('detmasks') / relpath)
        return self.ad.extractfile(actual_path)
    def read_mask(self, relpath):
        fd = self.read_raw(Path(relpath).with_suffix('.png'))
        raw_bytes = np.array(bytearray(fd.read()), dtype=np.uint8)
        fd.close()
        mask = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)
        return mask
    def namelist(self):
        return [str(Path(member.name).relative_to('detmasks')) for member in self.ad.getmembers() if member.isfile()]


class DetMask:
    base: Path
    def __init__(self, cwd):
        if not Path(cwd).exists():
            raise f'Cannot find path {cwd}!'
        self.base = Path(cwd).resolve()
    def read_raw(self, relpath):
        actual_path = str(self.base / 'detmasks' / relpath)
        return open(actual_path, 'rb')
    def read_mask(self, relpath):
        fd = self.read_raw(Path(relpath).with_suffix('.png'))
        raw_bytes = np.array(bytearray(fd.read()), dtype=np.uint8)
        fd.close()
        mask = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)
        return mask
    def namelist(self):
        return [str(path.relative_to(self.base / 'detmasks')) for path in sorted(self.base.rglob('*')) if path.is_file() and not path.is_dir()]
    

class MTSCacheMask:
    def __init__(self, basepath):
        if not Path(basepath).exists():
            raise f'Cannot find path {basepath}!'
        self.base = Path(basepath)
        self.model_ids = []
        for item in self.base.glob('*'):
            if item.is_dir():
                self.model_ids.append(item.name)
        self.model_ids.sort()

    def read_raw(self, relpath, model_id=None):
        sel_model_id = model_id if model_id else self.model_ids[0]
        actual_path = str(self.base / sel_model_id / relpath)
        return open(actual_path, 'rb')
    def read_mask(self, relpath, model_id=None):
        fd = self.read_raw(Path(relpath).with_suffix('.png'), model_id=None)
        raw_bytes = np.array(bytearray(fd.read()), dtype=np.uint8)
        fd.close()
        mask = cv2.imdecode(raw_bytes, cv2.IMREAD_GRAYSCALE)
        return mask
    def read_mask_all_variants(self, relpath):
        return [self.read_mask(relpath, model) for model in self.model_ids]

def issue1():
    sample = Path('TsubasaNoKioku/000.jpg')
    sample_img: Image.Image = Image.open(str(Path('../inputs') / sample))
    # sample_img.show()
    gt = GTMask()
    our = DetArcMask('../det_v1er.tar.gz')
    mts = MTSCacheMask('../mts_caches')

    gt_mask = gt.read_mask(sample)
    # Image.fromarray(gt_mask).show()
    our_mask = our.read_mask(sample)
    # Image.fromarray(our_mask).show()
    mask3s = mts.read_mask_all_variants(sample)
    # [Image.fromarray(x).show() for x in mask3s]

    w = sample_img.width
    h = sample_img.height // 2
    cmp = Image.new('RGB', (w, h * 3))
    cmp.paste(sample_img.crop((0, 0, w, h)))
    # Image.fromarray(gt_mask).show()
    cmp.paste(Image.fromarray(gt_mask).crop((0, 0, w, h)), (0, h))
    cmp.paste(Image.fromarray(our_mask).crop((0, 0, w, h)), (0, h * 2))
    small: Image.Image = cmp.resize((w//4, h * 3 //4))
    fnt = ImageFont.truetype('ubuntu/Ubuntu-B.ttf', 24)
    d = ImageDraw.Draw(small)
    fill = (0, 100, 210)
    d.text((0, 0), "Sample", font=fnt, stroke_fill=fill, stroke_width=1)
    d.text((0, h // 4), "Mask dataset", font=fnt, stroke_fill=fill, stroke_width=1)
    d.text((0, h // 2), "Our result", font=fnt, stroke_fill=fill, stroke_width=1)
    # [cmp.paste(x.crop(0, 0, w // 2, h)) for x in mask3s]
    small.show()
    # small.save('cmp.jpg')

def compare_vertical():
    gt = GTMask()
    mts = MTSCacheMask('../mts_caches')
    er = DetArcMask('../det_v1er.tar.gz')
    swt = DetArcMask('../det_v1swt.tar.gz')
    east = DetArcMask('../det_v1east.tar.gz')
    db = DetArcMask('../det_v1db.tar.gz')
    # new = DetMask('../')
    # sample = Path('ARMS/003.jpg')
    sample = Path(sys.argv[-1])
    sample_img: Image.Image = Image.open(str(Path('../inputs') / sample))
    w = sample_img.width
    h = sample_img.height
    crop_sz = (0, 0, w, h)
    gt_mask = gt.read_mask(sample)
    mts_masks = mts.read_mask_all_variants(sample)
    er_mask = er.read_mask(sample)
    swt_mask = swt.read_mask(sample)
    east_mask = east.read_mask(sample)
    db_mask = db.read_mask(sample)


    cmp = Image.new('RGB', (w * 5, h * 3))
    cmp.paste(sample_img.crop(crop_sz))
    cmp.paste(Image.fromarray(gt_mask).crop(crop_sz), (w, 0))
    for i, mask in enumerate(mts_masks):
        cmp.paste(Image.fromarray(mask).crop(crop_sz), (w * i, h))
    cmp.paste(Image.fromarray(er_mask).crop(crop_sz), (0, h * 2))
    cmp.paste(Image.fromarray(swt_mask).crop(crop_sz), (w, h * 2))
    cmp.paste(Image.fromarray(east_mask).crop(crop_sz), (w * 2, h * 2))
    cmp.paste(Image.fromarray(db_mask).crop(crop_sz), (w * 3, h * 2))
    cmp.show()

def breakdown(img):
    # Watershed first (from scikit-iamge docs)
    smoothed = skimage.filters.gaussian(img, 1)
    peak_idx = peak_local_max(cp.asarray(smoothed), min_distance=4).get()
    # peak_idx = peak_local_max(smoothed, min_distance=4)
    peak_mask = np.zeros_like(smoothed, dtype=bool)
    peak_mask[tuple(peak_idx.T)] = True
    peak_mask = skimage.morphology.dilation(peak_mask, skimage.morphology.ellipse(3, 3))
    label_peaks, total_peaks = skimage.measure.label(peak_mask, return_num=True)
    smoothed_inv = np.max(smoothed) - smoothed
    phase1 = watershed(smoothed_inv, label_peaks, watershed_line=True)
    new_markers = (phase1 == 0) * (total_peaks + 1) + label_peaks
    edges = skimage.filters.gaussian(img, 4)
    edges = skimage.filters.sobel(edges)
    final = watershed(edges, new_markers, watershed_line=False)
    final[final == (total_peaks + 1)] = 0
    return final

def breakdown_color(img, original):
    final = breakdown(img)
    final_color = label2rgb(final, image=original)
    return (final, final_color)

def rem0(x):
    return np.delete(x, np.where(x == 0))

def connected_stats(gt, det):
    gt_breakdown = breakdown(gt)
    det_breakdown = breakdown(det)
    return connected_stats_raw(gt_breakdown, det_breakdown)

def connected_stats_raw(gt_breakdown, det_breakdown):
    m = rem0(np.unique(gt_breakdown)).shape[0]
    if m == 0:
        return None

    det_w_gtmask = det_breakdown[gt_breakdown != 0]
    det_overlap_labels0 = np.unique(det_w_gtmask)
    # det_overlap_labels = rem0(det_overlap_labels0)
    # det_overlap = det_breakdown[np.isin(det_breakdown, det_overlap_labels)]
    gt_w_detmask = gt_breakdown[det_breakdown != 0]
    gt_overlap_labels0 = np.unique(gt_w_detmask)
    gt_overlap_labels = rem0(gt_overlap_labels0)
    # gt_overlap = gt_breakdown[np.isin(gt_breakdown, gt_overlap_labels)]
    # det_non_overlap = det_breakdown[det_breakdown == 0 | np.isin(det_breakdown, det_overlap_labels)]

    tp = gt_overlap_labels.shape[0]
    fp = np.unique(det_breakdown).shape[0] - det_overlap_labels0.shape[0]
    fn = m - gt_overlap_labels.shape[0]

    def get_quality(label):
        detj_w_mask = det_breakdown[gt_breakdown == label]
        detj_overlap_labels0 = np.unique(detj_w_mask)
        detj_overlap_labels = rem0(detj_overlap_labels0)
        common_area = np.count_nonzero(detj_w_mask != 0)
        detj_area = np.count_nonzero(np.isin(det_breakdown, detj_overlap_labels))
        label_area = np.count_nonzero(gt_breakdown == label)
        # Accuracy
        acc = np.float64(common_area) / detj_area
        # Coverage
        cov = np.float64(common_area) / label_area
        return np.array([acc, cov], dtype=np.float64)
    
    quantity_precision = np.float64(tp) / (tp + fp)
    quantity_recall = np.float64(tp) / m

    acc_cov = np.sum([get_quality(l) for l in gt_overlap_labels], axis=0)
    # print(acc_cov.shape, acc_cov.dtype, tp)
    quality_rnp = np.divide(acc_cov, tp, out=np.zeros_like(acc_cov), where=tp!=0)
    quality_precision = quality_rnp[0]
    quality_recall = quality_rnp[1]
    quality_f1 = (quality_precision * quality_recall * 2) / (quality_recall + quality_precision)
    # global_precision = quantity_precision * quality_precision
    # global_recall = quantity_recall * quality_recall
    global_precision = acc_cov[0] / (tp + fp)
    global_recall = acc_cov[1] / m
    global_f1 = (global_precision * global_recall * 2) / (global_recall + global_precision)

    return ((tp, fp, fn), (quantity_recall, quantity_precision), (quality_recall, quality_precision, quality_f1), (global_recall, global_precision, global_f1))

def pixelarea_stats(gt, det):
    tp = np.count_nonzero((gt != 0) & (det != 0))
    fp = np.count_nonzero((gt == 0) & (det != 0))
    fn = np.count_nonzero((gt != 0) & (det == 0))
    precision = np.float64(tp) / (tp + fp) if (tp+fp) != 0 else 0
    recall = np.float64(tp) / (tp + fn) if (tp+fn) != 0 else 0
    f1 = (precision * recall * 2) / (precision + recall) if (precision + recall) != 0 else 0
    return ((tp, fp, fn), (recall, precision, f1))


# def _compute_stats(gt, mask, file):
#     # print('Processing', file)
#     # mask = output.read_mask(file)
#     gt_mask = gt.read_mask(file)[:mask.shape[0], :mask.shape[1]]
#     cc, quant, qual, glob = connected_stats(gt_mask, mask)
#     pa, pixel = pixelarea_stats(gt_mask, mask)
#     return np.array([*quant, *qual, *glob, *pixel])

# def compute():
#     from tqdm import tqdm
#     cwd = Path('/home/saratoga/cs670-manga')
#     input_folder = cwd / 'inputs'
#     gt = GTMask(cwd / 'text_detection/post-processed.zip')
#     methods = ('db', 'er', 'swt', 'stub')
#     det_results = {method: DetArcMask(cwd / f'det_v2{method}.tar.gz') for method in methods}
#     file_list = list(gt.namelist())
#     for method, output in det_results.items():
#         total = np.zeros((11,))
#         print(f'Checking {method}')
#         for f in tqdm(file_list, miniters=1):
#             total += _compute_stats(gt, output.read_mask(f), f)
#         print(f'{method} result: {total}, {total / len(file_list)}')
