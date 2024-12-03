from shapely import Polygon, intersection, normalize, simplify, make_valid
from shapely.errors import GEOSException
import numpy as np
import mts
import cv2
import algorithm as al
from dataclasses import astuple, dataclass

# points: (n, 1, 2)
# Polygon: (n + 1, 2)


def convert_polygon_to_points(p, contour=False):
    pts = np.array([x for x in p.exterior.coords], dtype=np.int32)
    # Why tho?
    if contour:
        pts = pts.reshape((-1, 1, 2))
    # print(pts)
    return pts[:-1]

def convert_points_to_polygons(pts, contour=False):
    if contour:
        pts = np.squeeze(pts, 1)
    assert pts.shape[0] >= 3, f"Sth is wrong: {pts.shape=} {pts=}"
    try:
        return fix_invalid_polygon(Polygon(pts.astype(np.int32)))
    except ValueError as e:
        print(f"{pts=}")
        raise

def convert_polygons_to_pointslist(ps):
    return [convert_polygon_to_points(p) for p in ps]

def convert_pointslist_to_polygons(ptsl):
    # chars_fix = []
    # for char in ptsl:
    #     assert char.shape[0] >= 3, f"{char=}, {char.shape=}, {ptsl}"
    #     try:
    #         p = Polygon(char)
    #     except ValueError:
    #         print(f"Incompatible dim: {char=}")
    #         raise
    #     fix_invalid_polygon(p)
    return [p for pts in ptsl for p in convert_points_to_polygons(pts)]

def condense(bbs):
    while True:
        condensed = []
        for bb in bbs:
            # Somehow this is still counted as Polygon, so we must filter
            if not bb.area > 0:
                continue
            if not bb.is_valid:
                print(f"WARN invalid polygon; Ignoring {bb=}")
                continue
            has_merged = False
            for i in range(len(condensed)):
                c: Polygon = condensed[i]
                # There's a bug when doing intersects, so we must use overlaps to make sure
                if c.equals(bb):
                    has_merged = True
                    break
                elif bb.overlaps(c) or bb.intersects(c) or c.equals_exact(bb, 1):
                    inter = bb.intersection(c)
                    if not inter.area > 0:
                        continue
                    condensed[i] = normalize(c.union(bb))
                    # Double-check, just to be sure
                    assert condensed[i].geom_type == "Polygon", "{} intersecs {} seems impossible {}".format(c, bb, condensed[i])
                    has_merged = True
                    break
            if not has_merged:
                condensed.append(bb)
                
        if len(bbs) == len(condensed):
            break
        bbs = condensed
    return condensed

def convert_mask_to_points(mask, contour=False):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        # Reshape to (n, 2) array
        if not contour:
            cnt = np.squeeze(cnt, 1)
        polygons.append(cnt)
    return polygons

def dup_check(chars, isPoly=False):
    ccs = chars if isPoly else convert_pointslist_to_polygons(chars)
    for i, char in enumerate(ccs):
        dup = list([i])
        for j, char_2 in enumerate(ccs):
            if i != j and char.equals_exact(char_2, tolerance=0.5):
                dup.append(j) 
        assert len(dup) == 1, "There are {} duplicate of {} at index {}: {}".format(len(dup), char, dup, [ccs[i] for i in dup])

def mts_process(image_path, *, layer_agreement=2):
    model = mts.MTS
    
    im, pred = model.process(image_path)
    pred_col = np.array([p[0].px for p in pred])
    total = pred_col.sum(axis=0)
    layer_agreement = min(layer_agreement, len(pred))
    # print(f"{im.shape=}, {total.shape=}, {pred_col.shape=}")
    total[total < layer_agreement] = 0.
    total[total >= layer_agreement] = 1.
    # NOTE: For some unknown reasons, the output of MTS is slightly bigger than input
    # We can try fixing this by cropping
    total = total[..., :im.shape[-2], :im.shape[-1]]
    # print(f"{im.shape=}, {total.shape=}, {pred_col.shape=}")
    return (total, im)

def cv2_process(image_path, method):
    """Process using OpenCV algorithms. Will condense by default (TODO: remove?)"""
    im = cv2.imread(image_path)
    chars, chains, sep = method.process(im)
    # polys = convert_pointslist_to_polygons(chars)
    # polys = [p.simplify(0.1) for p in polys]
    # chars_fix = [x for char in chars for x in fix_invalid_polygon()]
    chars_fix = convert_pointslist_to_polygons(chars)
    total = condense(chars_fix)
    # total = polys
    # print(total)
    return (convert_polygons_to_pointslist(total), im)

def text_grouping(image: cv2.typing.MatLike, *, _kernel = None) -> cv2.typing.MatLike:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # This is very good
    kernel = _kernel if _kernel is not None else cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))
    # kernel2 = np.zeros((64, 32, 3), dtype=np.uint8)
    # custom_shape = np.array([(0, 0), (31, 0), (63, 31), (31, 31)], dtype=np.int32)
    # cv2.drawContours(kernel2, [custom_shape], -1, (255, 255, 255), thickness=cv2.FILLED)
    # kernel2 = cv2.cvtColor(kernel2, cv2.COLOR_RGB2GRAY) // 255
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    # dilation = cv2.dilate(dilation, kernel2, iterations=1)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()
    bbs = [cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.int32) for cnt in contours]
    return bbs

def fix_invalid_polygon(p1):
    # print(f"{p1.is_valid}")
    if p1.is_valid:
        # print("No need to fix")
        return [p1]
    p2 = make_valid(p1)
    # print(f"{p2}")
    if p2.geom_type == 'Polygon':
        return [p2]
    if p2.geom_type == 'MultiPolygon':
        # print("Fixed by splitting")
        return [*p2.geoms]
    elif p2.geom_type == 'GeometryCollection':
        return [x for x in p2.geoms if x.geom_type == 'Polygon']
    elif p2.geom_type == 'MultiLineString':
        return []
    elif p2.geom_type == 'LineString':
        return []
    else:
        assert False, (f"Don't know how to fix {p2.geom_type}: {p2}")
    return []

@dataclass
class TextDetectionResult:
    # Mask
    mask: cv2.typing.MatLike
    # List of bounding boxes
    bbs: list
    # MTS
    mts_img: cv2.typing.MatLike
    # Filtered
    filtered_img: cv2.typing.MatLike
    cv2_det: cv2.typing.MatLike

    def __init__(self, *, mask, bbs, mts_img, filtered_img, cv2_det):
        self.mask = mask
        self.bbs = bbs
        self.mts_img = mts_img
        self.filtered_img = filtered_img
        self.cv2_det = cv2_det
    def __iter__(self):
        return iter(astuple(self))

def text_detection(image_path, *, cv2_model, mts_level=2, group_kernel=None):
    # First, we use MTS to find the base mask of text
    mask, _ = mts_process(image_path, layer_agreement=mts_level)
    # We will build bb based on 
    # i don't remember why i had to cut
    mask_np = mask[0].astype(np.uint8) * 255
    # print(mask_np.shape)
    polys_cv2 = convert_mask_to_points(mask_np)
    # print(polys_cv2)
    # visualize_al(polys_cv2, cv2.imread(sample))

    # Post-Processing 1: We use CV2 in algorithm to find SFX
    polys = convert_pointslist_to_polygons([p for p in polys_cv2 if len(p) >= 3])
    # Remove noises
    im_area = mask_np.shape[0] * mask_np.shape[0]
    polys = [normalize(p) for p in polys if p.area / im_area > 0]
    # polys = [p if p.is_valid else simplify(p, 3) for p in polys]
    # visualize_al(convert_polygons_to_pointslist(polys), cv2.imread(sample))

    if len(polys) == 0:
        print("WARN filter is too strict. Receive 0 results from CV2")

    chars, img = cv2_process(image_path, cv2_model)

    # dup_check(chars)

    marks = []
    for char in chars:
        # We are dropping anything beyond 10-edges
        # if len(char) > 8:
        #     continue
        overlaps = 0
        boxes = convert_points_to_polygons(char)
        for p in polys:
            overlap_area = 0.
            for box in boxes:
                if p.overlaps(box) or p.intersects(box):
                    try:
                        inter = normalize(p.intersection(box))
                        # print(f"{inter.area=} {p.area=}")
                        overlap_area += inter.area
                    except GEOSException as e:
                        # print(f"{p=}, {box=}")
                        print(p)
                        print(box)
                        raise e
            if overlap_area / p.area > 0.05:
                overlaps += 1
                # else:
                #     print(f"Does not overlap: {p} {box}")
        if overlaps > 0:
            marks.append(char)
    print(f"Found {len(marks)} possible overlaps out of {len(chars)}") 
    # visualize_cv2(mask_np)
    new_mask = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2RGB)
    # Since all contours are disjoints, we can do in one line
    new_draw = cv2.drawContours(new_mask, marks, -1, color=(255, 255, 255), thickness=cv2.FILLED)
    # for i in range(len(marks)):
    #     new_draw = cv2.drawContours(new_mask, marks, i, color=(i, 255, i), thickness=cv2.FILLED)
    # visualize_cv2(new_draw)
    final_mask = cv2.cvtColor(new_draw, cv2.COLOR_RGB2GRAY)

    # print(f"{img.shape=}, {final_mask.shape=}")
    img[final_mask == 0] = (255, 255, 255)

    char_vis = np.zeros_like(img)
    cv2.drawContours(char_vis, chars, -1, color=(255, 255, 255), thickness=cv2.FILLED)
    
    return TextDetectionResult(mask=final_mask, bbs=text_grouping(img, _kernel=group_kernel), mts_img=mask_np, filtered_img=img, cv2_det=cv2.cvtColor(char_vis, cv2.COLOR_RGB2GRAY))
