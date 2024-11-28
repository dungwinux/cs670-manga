import os
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from PIL import Image
import fnmatch
import cv2
from test import inpaint
import time

import numpy as np

#torch.set_printoptions(precision=10)


class _bn_relu_conv(nn.Module):
    def __init__(self, in_filters, nb_filters, fw, fh, subsample=1):
        super(_bn_relu_conv, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_filters, eps=1e-3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_filters, nb_filters, (fw, fh), stride=subsample, padding=(fw//2, fh//2), padding_mode='zeros')
        )

    def forward(self, x):
        return self.model(x)

        # the following are for debugs
        print("****", np.max(x.cpu().numpy()), np.min(x.cpu().numpy()), np.mean(x.cpu().numpy()), np.std(x.cpu().numpy()), x.shape)
        for i,layer in enumerate(self.model):
            if i != 2:
                x = layer(x)
            else:
                x = layer(x)
                #x = nn.functional.pad(x, (1, 1, 1, 1), mode='constant', value=0)
            print("____", np.max(x.cpu().numpy()), np.min(x.cpu().numpy()), np.mean(x.cpu().numpy()), np.std(x.cpu().numpy()), x.shape)
            print(x[0])
        return x


class _u_bn_relu_conv(nn.Module):
    def __init__(self, in_filters, nb_filters, fw, fh, subsample=1):
        super(_u_bn_relu_conv, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_filters, eps=1e-3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_filters, nb_filters, (fw, fh), stride=subsample, padding=(fw//2, fh//2)),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        return self.model(x)



class _shortcut(nn.Module):
    def __init__(self, in_filters, nb_filters, subsample=1):
        super(_shortcut, self).__init__()
        self.process = False
        self.model = None
        if in_filters != nb_filters or subsample != 1:
            self.process = True
            self.model = nn.Sequential(
                    nn.Conv2d(in_filters, nb_filters, (1, 1), stride=subsample)
                )

    def forward(self, x, y):
        #print(x.size(), y.size(), self.process)
        if self.process:
            y0 = self.model(x)
            #print("merge+", torch.max(y0+y), torch.min(y0+y),torch.mean(y0+y), torch.std(y0+y), y0.shape)
            return y0 + y
        else:
            #print("merge", torch.max(x+y), torch.min(x+y),torch.mean(x+y), torch.std(x+y), y.shape)
            return x + y

class _u_shortcut(nn.Module):
    def __init__(self, in_filters, nb_filters, subsample):
        super(_u_shortcut, self).__init__()
        self.process = False
        self.model = None
        if in_filters != nb_filters:
            self.process = True
            self.model = nn.Sequential(
                nn.Conv2d(in_filters, nb_filters, (1, 1), stride=subsample, padding_mode='zeros'),
                nn.Upsample(scale_factor=2, mode='nearest')
            )

    def forward(self, x, y):
        if self.process:
            return self.model(x) + y
        else:
            return x + y


class basic_block(nn.Module):
    def __init__(self, in_filters, nb_filters, init_subsample=1):
        super(basic_block, self).__init__()
        self.conv1 = _bn_relu_conv(in_filters, nb_filters, 3, 3, subsample=init_subsample)
        self.residual = _bn_relu_conv(nb_filters, nb_filters, 3, 3)
        self.shortcut = _shortcut(in_filters, nb_filters, subsample=init_subsample)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.residual(x1)
        return self.shortcut(x, x2)

class _u_basic_block(nn.Module):
    def __init__(self, in_filters, nb_filters, init_subsample=1):
        super(_u_basic_block, self).__init__()
        self.conv1 = _u_bn_relu_conv(in_filters, nb_filters, 3, 3, subsample=init_subsample)
        self.residual = _bn_relu_conv(nb_filters, nb_filters, 3, 3)
        self.shortcut = _u_shortcut(in_filters, nb_filters, subsample=init_subsample)

    def forward(self, x):
        y = self.residual(self.conv1(x))
        return self.shortcut(x, y)


class _residual_block(nn.Module):
    def __init__(self, in_filters, nb_filters, repetitions, is_first_layer=False):
        super(_residual_block, self).__init__()
        layers = []
        for i in range(repetitions):
            init_subsample = 1
            if i == repetitions - 1 and not is_first_layer:
                init_subsample = 2
            if i == 0:
                l = basic_block(in_filters=in_filters, nb_filters=nb_filters, init_subsample=init_subsample)
            else:
                l = basic_block(in_filters=nb_filters, nb_filters=nb_filters, init_subsample=init_subsample)
            layers.append(l)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class _upsampling_residual_block(nn.Module):
    def __init__(self, in_filters, nb_filters, repetitions):
        super(_upsampling_residual_block, self).__init__()
        layers = []
        for i in range(repetitions):
            l = None
            if i == 0: 
                l = _u_basic_block(in_filters=in_filters, nb_filters=nb_filters)#(input)
            else:
                l = basic_block(in_filters=nb_filters, nb_filters=nb_filters)#(input)
            layers.append(l)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class res_skip(nn.Module):

    def __init__(self):
        super(res_skip, self).__init__()
        self.block0 = _residual_block(in_filters=1, nb_filters=24, repetitions=2, is_first_layer=True)#(input)
        self.block1 = _residual_block(in_filters=24, nb_filters=48, repetitions=3)#(block0)
        self.block2 = _residual_block(in_filters=48, nb_filters=96, repetitions=5)#(block1)
        self.block3 = _residual_block(in_filters=96, nb_filters=192, repetitions=7)#(block2)
        self.block4 = _residual_block(in_filters=192, nb_filters=384, repetitions=12)#(block3)
        
        self.block5 = _upsampling_residual_block(in_filters=384, nb_filters=192, repetitions=7)#(block4)
        self.res1 = _shortcut(in_filters=192, nb_filters=192)#(block3, block5, subsample=(1,1))

        self.block6 = _upsampling_residual_block(in_filters=192, nb_filters=96, repetitions=5)#(res1)
        self.res2 = _shortcut(in_filters=96, nb_filters=96)#(block2, block6, subsample=(1,1))

        self.block7 = _upsampling_residual_block(in_filters=96, nb_filters=48, repetitions=3)#(res2)
        self.res3 = _shortcut(in_filters=48, nb_filters=48)#(block1, block7, subsample=(1,1))

        self.block8 = _upsampling_residual_block(in_filters=48, nb_filters=24, repetitions=2)#(res3)
        self.res4 = _shortcut(in_filters=24, nb_filters=24)#(block0,block8, subsample=(1,1))

        self.block9 = _residual_block(in_filters=24, nb_filters=16, repetitions=2, is_first_layer=True)#(res4)
        self.conv15 = _bn_relu_conv(in_filters=16, nb_filters=1, fh=1, fw=1, subsample=1)#(block7)

    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        x5 = self.block5(x4)
        res1 = self.res1(x3, x5)

        x6 = self.block6(res1)
        res2 = self.res2(x2, x6)

        x7 = self.block7(res2)
        res3 = self.res3(x1, x7)

        x8 = self.block8(res3)
        res4 = self.res4(x0, x8)

        x9 = self.block9(res4)
        y = self.conv15(x9)

        return y

class MyDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def get_class_label(self, image_name):
        # your method here
        head, tail = os.path.split(image_name)
        #print(tail)
        return tail
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path)
        y = self.get_class_label(image_path.split('/')[-1])
        if self.transform is not None:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.image_paths)

def loadImages(folder):
    imgs = []
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, '*'):
            matches.append(os.path.join(root, filename))
   
    return matches

if __name__ == "__main__":
    model = res_skip()
    model.load_state_dict(torch.load('erika.pth'))
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        print("Using CUDA")
        model.cuda()
    else:
        print("Using CPU")
        model.cpu()
    model.eval()
    keep_image_file_structure = True #Use this to keep the same folder structure as the input images in the output folder. Also expects Masks to be in the same folder structure as the input images
    clear_output_folder = True #Clear the output folder before starting the process
    clear_lines_folder = False #Clear the lines folder before starting the process
    run_line_model = False #Run the line model before running the inpainting model
    input_image_folder = "./inputs" #sys.argv[1]
    line_image_output_folder = "./lines" #sys.argv[2]
    output_folder = "./outputs" #sys.argv[3]
    filelists = loadImages(input_image_folder)
    #print(filelists)

    if clear_lines_folder:
        #Delete everything in line_image_output_folder and 
        for root, dirnames, filenames in os.walk(line_image_output_folder):
            for filename in filenames:
                os.unlink(os.path.join(root, filename))
            for dirname in dirnames:
                os.rmdir(os.path.join(root, dirname))
        print("Cleared line folder")
    if clear_output_folder:
        #Delete everything in output_folder
        for root, dirnames, filenames in os.walk(output_folder):
            for filename in filenames:
                os.unlink(os.path.join(root, filename))
            for dirname in dirnames:
                os.rmdir(os.path.join(root, dirname))
        print("Cleared output folder")

    if keep_image_file_structure:
        #Create all intermediate folders in line_image_output_folder
        for root, dirnames, filenames in os.walk(input_image_folder):
            for dirname in dirnames:
                relative_path = os.path.relpath(os.path.join(root, dirname), input_image_folder)
                output_dir = os.path.join(line_image_output_folder, relative_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    print("Created folder: ", output_dir)
                else:
                    print("WARNING: Folder already exists: ", output_dir)
                    print("Images in this folder will be overwritten if they have the same name")

    if not os.path.exists(line_image_output_folder):
        os.makedirs(line_image_output_folder)

    with torch.no_grad():
        for imname in filelists:
            if not run_line_model:
                break
            src = cv2.imread(imname,cv2.IMREAD_GRAYSCALE)
            head, tail = os.path.split(imname)
            
            # start_time = time.time()
            rows = int(np.ceil(src.shape[0]/16))*16
            cols = int(np.ceil(src.shape[1]/16))*16
            # print(f"Resizing time: {time.time() - start_time} seconds")

            # start_time = time.time()
            patch = np.ones((1,1,rows,cols),dtype="float32")
            patch[0,0,0:src.shape[0],0:src.shape[1]] = src
            # print(f"Patch creation time: {time.time() - start_time} seconds")

            # start_time = time.time()
            if is_cuda: 
                tensor = torch.from_numpy(patch).cuda()
            else:
                tensor = torch.from_numpy(patch).cpu()
            # print(f"Data transfer time: {time.time() - start_time} seconds")

            # start_time = time.time()
            y = model(tensor)
            # print(f"Model inference time: {time.time() - start_time} seconds")

            # start_time = time.time()
            yc = y.cpu().numpy()[0,0,:,:]
            yc[yc>255] = 255
            yc[yc<0] = 0
            # print(f"Post-processing time: {time.time() - start_time} seconds")

            # print(imname, torch.max(y), torch.min(y))

            head, tail = os.path.split(imname)
            #tail = tail.replace(".jpg", "_line_image.jpg").replace(".png", "_line_image.png")
            if keep_image_file_structure:
                #right now imname is input_image_folder + relative_path + filename
                #we need to replace input_image_folder with line_image_output_folder
                #replace the first occurrence of input_image_folder with line_image_output_folder
                new_head = head.replace(input_image_folder, line_image_output_folder, 1)
                cv2.imwrite(new_head+"/"+tail.replace(".jpg",".png"),yc[0:src.shape[0],0:src.shape[1]])
            else:
                cv2.imwrite(line_image_output_folder+"/"+tail.replace(".jpg",".png"),yc[0:src.shape[0],0:src.shape[1]])
    
    print("Done creating line images")
    line_model_info = {
        'input_folder': input_image_folder,
        'mask_folder': "./masks",
        'line_folder': line_image_output_folder,
        'output_folder': output_folder,
        'model': None,
        'inpaint_model_location': './checkpoints/mangainpaintor',
    }
    inpaint(None, line_model_info)

    
