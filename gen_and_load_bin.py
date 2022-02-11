import os

import cv2
import numpy as np
import pickle
import mxnet as mx
from tqdm import tqdm

LFW_PATH = "/home/funzin/DISK1/lfw.bin"
MASK_DIR = "/home/funzin/Documents/masked_lfw/masked_lfw_unfold"
# original LFW image size (112,112,3)

def load_bin(path, image_size):
    """ code from insightface/verification.py"""

    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  #py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  #py3

    # with open(path, 'rb') as f:
    #     bins, issame_list = pickle.load(f, encoding='bytes')

    FLIP = False

    # with open("issame.txt","wb") as f:
    #     pickle.dump(issame_list,f)

    save_path = "test"

    if FLIP:
        save_path += "_flip"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pairfile = open(os.path.join(save_path,'lfw_pair.txt'), 'w')

        data_list = []
        for flip in [0, 1]:
            data = mx.nd.empty(
                (len(issame_list) * 2, 3, image_size, image_size))
            data_list.append(data)
        for i in tqdm(range(len(issame_list) * 2)):
            _bin = bins[i]
            img = mx.image.imdecode(_bin) #(112,112,3)
            if img.shape[1] != image_size:
                img = mx.image.resize_short(img, image_size) #(160,160,3)
            img = mx.nd.transpose(img, axes=(2, 0, 1)) #(3,160,160)
            for flip in [0, 1]:
                if flip == 1:
                    img = mx.ndarray.flip(data=img, axis=2)
                if i%2 == 0:
                    label = 1 if issame_list[i//2] == True else -1
                    pairfile.write(str(i)+'_0.jpg'+' '+str(i)+'_1.jpg'+' '+str(label)+'\n')
                cv2.imwrite(os.path.join(save_path,str(i)+"_"+str(flip)+".png"),mx.nd.transpose(img,axes=(1,2,0)).asnumpy())
                data_list[flip][i][:] = img
            if i % 1000 == 0:
                print('loading bin', i)
        print(data_list[0].shape)
        return (data_list, issame_list)

    else:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pairfile = open(os.path.join(save_path,'lfw_pair.txt'), 'w')

        for idx in tqdm(range(len(bins))):
            _bin = bins[idx]
            img = mx.image.imdecode(_bin).asnumpy()
            img2 = cv2.resize(img,(image_size,image_size))
            img3 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_path, str(idx+1).zfill(5)+'.jpg'), img3)
            if idx % 2 == 0:
                label = 1 if issame_list[idx//2] == True else -1
                pairfile.write(str(idx+1).zfill(5) + '.jpg' + ' ' + str(idx+2).zfill(5) +'.jpg' + ' ' + str(label) + '\n')

def generate_bin(data_dir):

    # with  open(os.path.join(data_dir,'lfw_pair.txt'), 'r') as f:
    #     pairs = [x.strip() for x in f.readlines()]
    
    # with open("issame.txt","rb") as f:
    #     pairs = pickle.load(f)

    files = os.listdir(data_dir)
    files.sort()

    pairs = [True] * int((len(files)/2))

    imgbytes = []
    for file in files:
        if not file.endswith((".png",".jpg",".jpeg")):
            continue
        image = open(os.path.join(data_dir,file),"rb")
        f = image.read()
        imgbytes.append(f)

    final = (imgbytes,pairs)
    with open("test.bin","wb") as f:
        pickle.dump(final,f)

def main():
    # load_bin(LFW_PATH, 160)
    generate_bin(MASK_DIR)
    load_bin("test.bin", 160)
if __name__ == "__main__":
    main()