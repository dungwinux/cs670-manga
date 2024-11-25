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
    plt.clf()
    plt.figure(figsize=(22, 22))
    plt.imshow(img)
    return plt

def visualize_mts(pred, background):
    mts.torch.zeros(4, pred.shape[0], pred.shape[1]).int()
    background.show(y = mts.ImageSegment(pred), figsize=(22, 22), alpha=0.8)
