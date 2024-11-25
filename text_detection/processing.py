from shapely import Polygon, intersection, normalize
import numpy as np


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