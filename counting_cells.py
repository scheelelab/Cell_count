# -*- coding: utf-8 -*-
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import glob
#import imageio.v3 as iio
import skimage.color
import skimage.filters
#import seaborn as sns
from kneebow.rotor import Rotor
import pandas as pd



parser = argparse.ArgumentParser()
parser.add_argument("-path_to_imgs", type=str, required=True)
parser.add_argument("-output_name", type=str, required=False, default="Detected_dots.pdf")
parser.add_argument("-extra_polish", action='store_true', default=False, required=False)
parser.add_argument("-this_size", type=int, default=300, required=False)
args = parser.parse_args()

print(args.path_to_imgs)
print(args.output_name)
print(args.extra_polish)
print(args.this_size)


def getting_cc_and_sizes(thrs_blur):
# Finding sizes of connected components
    cc = cv2.connectedComponentsWithStats(np.expand_dims(np.asarray((thrs_blur/np.max(thrs_blur))*255, dtype=np.uint8),axis=-1))
    sizes = cc[-2][:,-1]
    # Finding the best cut-off 
    dat = np.sort(sizes[1:]).reshape((-1,1))
    indx = np.arange(len(dat)).reshape((-1,1))
    data = np.concatenate([indx, dat],axis=-1)
    
    
    #@misc{kneebow,
    #  title={ {kneebow}: Knee or elbow detection for curves},
    #  author={Unterholzner Georg},
    #  year={2019},
    #  howpublished={\url{https://github.com/georg-un/kneebow}},
    #}
    rotor = Rotor()
    rotor.fit_rotate( data )
    elbow_index = rotor.get_elbow_index()
    this_size = dat[elbow_index+1]
    #print("\t",elbow_index, this_size)
    return sizes, this_size, cc
    #return sizes, cc



def remove_noise(cc, sizes, this_size, count_=None):
    z = cc[1].copy()
    if count_ == None:
        count_ = 0
    else:
        count_ = count_
        
    for j,i in enumerate(sizes):
        if i < 10: # manually setting a threshold for small dots --> noise
            z[ z == j] = 0
        elif i > this_size:
            z[ z == j] = 0
        else:
            count_ += 1
    z[ z > 0] = 255
    #plt.imshow(z,cmap="gray")
    #plt.pause(0.01)

    return count_, z


def getting_markings(thrs_blur, cc, this_size, sizes):
    z1 = cc[1].copy()
    for j,i in enumerate(sizes):
        if i <= this_size: 
            z1[ z1 == j] = 0
    z1[ z1 > 0] = 1
    weird_markings = z1*thrs_blur
    #plt.imshow(weird_markings,cmap="gray")
    #plt.pause(0.01)
    
    t1 = skimage.filters.threshold_multiotsu(weird_markings, classes=4)
    w = weird_markings.copy()
    w[w < t1[-1]] = 0
    #plt.imshow(w,cmap="gray")
    #plt.pause(0.01)
    return w


def collected_in_one(thrs_blur, extra_polish, this_size):
    #this_size = 500
    sizes, elb_size, cc = getting_cc_and_sizes(thrs_blur)
    count_, z = remove_noise(cc, sizes, np.min([elb_size,this_size]), count_=None)
    if extra_polish:
        w = getting_markings(thrs_blur, cc, np.min([elb_size,this_size]), sizes)
        sizes1, elb_size, cc1 = getting_cc_and_sizes(w)
        print("\tbefore", count_)
        count_, z1 = remove_noise(cc1, sizes1, np.min([elb_size,this_size]), count_=count_)
        return count_, z, z1
    else:
        return count_, z, 0



if __name__ == "__main__":
    
    path = args.path_to_imgs
    img_names = glob.glob1(path, "*")

    print_img = []
    for img in img_names:
        print(img)
        img = cv2.imread(path + img, cv2.IMREAD_GRAYSCALE)
        if not args.extra_polish:
            _, threshold = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
            cc = cv2.connectedComponentsWithStats(np.expand_dims(threshold,axis=-1))
            im = img.copy()
            im[cc[1] == 1] = 0
            #t = skimage.filters.threshold_multiotsu(im, classes=4)
        else:
            im = img.copy()
            #t = skimage.filters.threshold_multiotsu(im, classes=4)
        #blurred_image = im.copy()#skimage.filters.gaussian(im, sigma=1.0)
        #plt.imshow(blurred_image, cmap="gray")
        #plt.pause(0.01)
    
        #if args.extra_polish:
        #    t = skimage.filters.threshold_multiotsu(im, classes=3)
        #else:
        #    t = skimage.filters.threshold_multiotsu(im, classes=4)
        t = skimage.filters.threshold_multiotsu(im, classes=4)
        thrs_ = im.copy()
        thrs_[thrs_ < t[-1]] = 0
        
        count_, z, z1 = collected_in_one(thrs_, args.extra_polish, args.this_size)
        print("\t",count_)
        #plt.imshow(z+z1, cmap="gray")
        #plt.pause(0.01)

        print_img.append([img, z+z1, count_])
    
    #plt.rcParams["figure.figsize"] = [10, 10]
    plt.rcParams["figure.autolayout"] = True
    matplotlib.use('Agg')
    figs = []
    for j,i in enumerate(print_img):
        fig, axs = plt.subplots(2)
        fig.set_size_inches(30, 30)
        #fig.set_dpi(500)
        #fig.f
        axs[0].imshow(i[0], cmap="gray")
        axs[0].set_title(img_names[j])
        axs[1].imshow(i[1], cmap="gray")
        axs[1].set_title("Dots found: {}".format(i[2]))
        axs[0].get_xaxis().set_visible(False)
        axs[0].get_yaxis().set_visible(False)
        axs[1].get_xaxis().set_visible(False)
        axs[1].get_yaxis().set_visible(False)
        figs.append(fig)
        plt.close()
        plt.pause(0.01)
    
    
    filename = args.output_name
    
    co = [i[2] for i in print_img]
    dct = {"Sample name": img_names, "Cell count":co}
    pd.DataFrame(dct).to_excel(filename+".xlsx", index=None)
    
    if filename[-4:] != ".pdf":
        filename += ".pdf"
    
    pp = PdfPages(filename)
    for fig in figs:
        fig.savefig(pp, format='pdf', bbox_inches='tight')
    pp.close()

    
