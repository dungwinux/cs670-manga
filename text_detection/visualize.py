import cv2
import mts
from matplotlib import pyplot as plt

def visualize_al(poly, background: cv2.typing.MatLike):
    bg = background.copy()
    # print(chains.shape)
    # print(np.array(poly).shape)
    cv2.polylines(bg, poly, True, (0, 255, 0), 2)
    return visualize_cv2(bg)

def visualize_cv2(img: cv2.typing.MatLike):
    # print(chains.shape)
    # print(np.array(poly).shape)
    # plt.clf()
    plt.figure(figsize=(22, 22))
    plt.imshow(img)
    return img

def visualize_mts(pred, background):
    mts.torch.zeros(4, pred.shape[0], pred.shape[1]).int()
    segment = mts.ImageSegment(pred)
    background.show(y = segment, figsize=(22, 22), alpha=0.8)
    return segment

def visualize_contours(contours, img: cv2.typing.MatLike, *, noshow=False):
    out = img.copy()
    for cnt in contours:
        cv2.drawContours(out, [cnt], 0, (0, 255, 0), 2)
    return out if noshow else visualize_cv2(out)
