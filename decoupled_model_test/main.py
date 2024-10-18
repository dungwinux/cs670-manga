import custom_logger
import os
import torch
import random
import numpy as np
import cv2
import time

logger = custom_logger.Logger(print_to_console=True)

def load_jit_model(path, device):
    if os.path.exists(path):
        model_path = path
    else:
        logger.log(f"Model not found at: {path}")
        return None

    logger.log(f"Loading model from: {model_path}")
    try:
        model = torch.jit.load(model_path, map_location="cpu").to(device)
    except Exception as e:
        logger.log(f"Failed to load model: {e}")
        return None
    model.eval()
    return model

#Logging options (ALL, NONE)
def forward_to_models(image, mask, line_model, inpaintor_model, device, logging="ALL"):
        """
        image: [H, W, C] RGB
        mask: [H, W, 1]
        return: BGR IMAGE
        """
        cur_res = None
        #ive put the logging options here so that we don't have to check it for every logging call
        if logging == "ALL":
            logger.log("Forwarding image to models. Device: {}. Logging Mode: ALL".format(device))
            seed = 42 #arbitrary seed
            random.seed(seed)
            np.random.seed(seed)
            logger.log(f"seed: {seed}")
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray_img = torch.from_numpy(
                gray_img[np.newaxis, np.newaxis, :, :].astype(np.float32)
            ).to(device)
            start = time.time()
            lines = line_model(gray_img)
            torch.cuda.empty_cache()
            lines = torch.clamp(lines, 0, 255)
            logger.log(f"line_model time: {time.time() - start}")

            mask = torch.from_numpy(mask[np.newaxis, :, :, :]).to(device)
            mask = mask.permute(0, 3, 1, 2)
            mask = torch.where(mask > 0.5, 1.0, 0.0)
            noise = torch.randn_like(mask)
            ones = torch.ones_like(mask)

            gray_img = gray_img / 255 * 2 - 1.0
            lines = lines / 255 * 2 - 1.0

            start = time.time()
            inpainted_image = inpaintor_model(gray_img, lines, mask, noise, ones)
            logger.log(f"image_inpaintor_model time: {time.time() - start}")

            cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = (cur_res * 127.5 + 127.5).astype(np.uint8)
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_GRAY2BGR)
        elif logging == "NONE":
            seed = 42 #arbitrary seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray_img = torch.from_numpy(
                gray_img[np.newaxis, np.newaxis, :, :].astype(np.float32)
            ).to(device)
            start = time.time()
            lines = line_model(gray_img)
            torch.cuda.empty_cache()
            lines = torch.clamp(lines, 0, 255)
            mask = torch.from_numpy(mask[np.newaxis, :, :, :]).to(device)
            mask = mask.permute(0, 3, 1, 2)
            mask = torch.where(mask > 0.5, 1.0, 0.0)
            noise = torch.randn_like(mask)
            ones = torch.ones_like(mask)

            gray_img = gray_img / 255 * 2 - 1.0
            lines = lines / 255 * 2 - 1.0

            start = time.time()
            inpainted_image = inpaintor_model(gray_img, lines, mask, noise, ones)
            cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = (cur_res * 127.5 + 127.5).astype(np.uint8)
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_GRAY2BGR)
        return cur_res

def __main__():
    erika_path = "models/line_model.jit"
    inpaintor_path = "models/manga_inpaintor.jit"
    test_img = cv2.imread("test.png")

    erika = load_jit_model(erika_path, "cpu")
    inpaintor = load_jit_model(inpaintor_path, "cpu")

    if erika is None or inpaintor is None:
        logger.log("Failed to load models")
        return
    
    #Create a mask in the center of the image, 30% of the image size
    h, w, _ = test_img.shape
    mask = np.zeros((h, w, 1))
    mask[h//3:h//3*2, w//3:w//3*2] = 1

    #show a preview of the mask as a black square
    preview = test_img.copy()
    preview[mask[:, :, 0] == 1] = [0, 0, 0]
    if False:
        cv2.imshow("Mask", preview)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()  # Close the window
        exit()

    res = forward_to_models(test_img, mask, erika, inpaintor, "cpu", logging="ALL")

    cv2.imshow("Result", res)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()  # Close the window



if __name__ == "__main__":
    __main__()


