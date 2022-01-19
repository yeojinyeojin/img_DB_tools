from logging import error
import os
import argparse

from PIL import Image
import cv2
import pickle
import mxnet as mx
from tqdm import tqdm

'''
For train dataset, insightface provide a mxnet .rec file, just install a mxnet-cpu for extract images
'''

def load_mx_rec(rec_path,save_path):

    # imgrec = mx.recordio.MXIndexedRecordIO(os.path.join(rec_path, 'train.idx'), os.path.join(rec_path, 'train.rec'), 'r')
    imgrec = mx.recordio.MXRecordIO(os.path.join(rec_path, 'train.rec'), 'r')
    img_info = imgrec.read()
    header,_ = mx.recordio.unpack(img_info)
    # max_idx = int(header.label[0])
    # for idx in tqdm(range(1,max_idx)):
    for idx in range(99999999999999):
        img_info = imgrec.read()
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #img = Image.fromarray(img)
        label_path = os.path.join(save_path, "id"+str(label))
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        #img.save(os.path.join(label_path, str(idx).zfill(8) + '.jpg'), quality=95)
        # print(os.path.join(label_path, "id"+str(label)+"_"+str(idx) + '.jpg'))
        cv2.imwrite(os.path.join(label_path, "id"+str(label)+"_"+str(idx) + '.jpg'), img)
    # print(label)


def load_image_from_bin(bin_path, save_dir):
    name = os.path.splitext(os.path.basename(bin_path))[0]
    save_path = os.path.join(save_dir,name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file = open(os.path.join(save_dir, '../', 'lfw_pair.txt'), 'w')
    
    bins, issame_list = pickle.load(open(bin_path, 'rb'), encoding='bytes')
    for idx in tqdm(range(len(bins))):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_path, str(idx+1).zfill(5)+'.jpg'), img)
        if idx % 2 == 0:
            label = 1 if issame_list[idx//2] == True else -1
            file.write(str(idx+1).zfill(5) + '.jpg' + ' ' + str(idx+2).zfill(5) +'.jpg' + ' ' + str(label) + '\n')

def load_image_from_bin_2(bin_path, save_dir):
    image_size = [112,112]
    name = os.path.splitext(os.path.basename(bin_path))[0]
    save_path = os.path.join(save_dir,name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with open(bin_path, 'rb') as f:
        bins, issame_list = pickle.load(f, encoding='bytes')  # py3
    data_list = []
    for flip in [0, 1]:
        data = mx.nd.empty(
            (len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for i in tqdm(range(len(issame_list) * 2)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        
        # cv2.imwrite(os.path.join(save_path,str(i)+".jpg"),img.asnumpy())
        # img = mx.nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            cv2.imwrite(os.path.join(save_path,str(flip)+".jpg"),img.asnumpy())
            # data_list[flip][i][:] = img

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--rec', type=str, help='Path to rec file')
    parser.add_argument('--bin', type=str, help='Path to bin file')
    parser.add_argument('--o', type=str, default='.', help='Path to output directory')

    return parser.parse_args()

if __name__ == '__main__':
    # #bin_path = 'D:/face_data_emore/faces_webface_112x112/lfw.bin'
    # #save_dir = 'D:/face_data_emore/faces_webface_112x112/lfw'
    # rec_path = '/home/funzin/Documents/eval/faces_webface_112x112'
    # output_path = './data/casia'
    # load_mx_rec(rec_path,output_path)
    # #load_image_from_bin(bin_path, save_dir)

    args = parse_arguments()
    args.bin = 'lfw.bin'
    args.o = 'tmptmp'

    if args.rec is not None:
        load_mx_rec(args.rec,args.o)
    elif args.bin is not None:
        load_image_from_bin(args.bin,args.o)
    else:
        raise ValueError('Type either rec or bin to proceed')
