from shapely import Polygon, intersection, normalize
import numpy as np
import mts
import cv2
import algorithm as al

def convert_polygon_to_points(p):
    pts = np.array([x for x in p.exterior.coords], dtype=np.int32)
    pts.reshape((-1, 1, 2))
    # print(pts)
    return pts[:-1]

def convert_points_to_polygon(pts):
    return Polygon(pts)

def convert_polygons_to_pointslist(ps):
    return [convert_polygon_to_points(p) for p in ps]

def convert_pointslist_to_polygons(ptsl):
    return [convert_points_to_polygon(pts) for pts in ptsl]

def condense(bbs):
    while True:
        condensed = []
        for bb in bbs:
            # Somehow this is still counted as Polygon, so we must filter
            if not bb.area > 0:
                continue
            if not bb.is_valid:
                print("WARN invalid polygon {}".format(bb))
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

def convert_mask_to_points(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        # Reshape to (n, 2) array
        p = contour.reshape((-1, 2))
        polygons.append(p)
    return polygons

def dup_check(chars, isPoly=False):
    ccs = chars if isPoly else convert_pointslist_to_polygons(chars)
    for i, char in enumerate(ccs):
        dup = list([i])
        for j, char_2 in enumerate(ccs):
            if i != j and char.equals_exact(char_2, tolerance=0.5):
                dup.append(j) 
        assert len(dup) == 1, "There are {} duplicate of {} at index {}: {}".format(len(dup), char, dup, [ccs[i] for i in dup])

def mts_process(image_path):
    model = mts.MTS
    layer_agreement = 3
    im, pred = model.process(image_path)
    total = sum(p[0].px for p in pred)
    total[total < layer_agreement] = 0.
    total[total >= layer_agreement] = 1.
    return (total, im)

def cv2_process(image_path, method):
    """Process using OpenCV algorithms. Will condense by default (TODO: remove?)"""
    im = cv2.imread(image_path)
    chars, chains, sep = method.process(im)
    polys = convert_pointslist_to_polygons(chars)
    # polys = [p.simplify(0.1) for p in polys]
    total = condense(polys)
    # total = polys
    # print(total)
    return (convert_polygons_to_pointslist(total), im)

def text_detection(image_path, cv2_model):
    # First, we use MTS to find the base mask of text
    mask, _ = mts_process(image_path)
    # We will build bb based on 
    mask_np = np.array(mask[0], dtype=np.uint8) * 255
    # print(mask_np.shape)
    polys_cv2 = convert_mask_to_points(mask_np)
    # print(polys_cv2)
    # visualize_al(polys_cv2, cv2.imread(sample))

    # Post-Processing 1: We use TextER in algorithm to find SFX
    polys = convert_pointslist_to_polygons([p for p in polys_cv2 if len(p) > 3])
    # Remove noises
    polys = [p for p in polys if p.area > 20]
    # visualize_al(convert_polygons_to_pointslist(polys), cv2.imread(sample))

    # model.er1 = cv2.text.createERFilterNM1(model.erc1, 16, 0.00005, 0.7, 0.25, True, 0.05)
    chars, _ = cv2_process(image_path, cv2_model)

    # dup_check(chars)

    marks = []
    for char in chars:
        # We are dropping anything beyond 10-edges
        if len(char) > 10:
            continue
        overlaps = 0
        box = convert_points_to_polygon(char)
        for i in range(len(polys)):
            p = polys[i]
            if p.overlaps(box) or p.intersects(box):
                inter = normalize(p.intersection(box))
                if inter.area / p.area > 0.4:
                    overlaps += 1
        if overlaps > 0:
            marks.append(char)
    # visualize_cv2(mask_np)
    new_mask = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2RGB)
    # Since all contours are disjoints, we can do in one line
    new_draw = cv2.drawContours(new_mask, marks, -1, color=(255, 255, 255), thickness=cv2.FILLED)
    # for i in range(len(marks)):
    #     new_draw = cv2.drawContours(new_mask, marks, i, color=(i, 255, i), thickness=cv2.FILLED)
    # visualize_cv2(new_draw)
    final_mask = cv2.cvtColor(new_draw, cv2.COLOR_RGB2GRAY)

    return final_mask
