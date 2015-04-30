# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:12:34 2015

@author: Ying
"""

import Image
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def get_img():
    #Get the input file from user
    while True :
        filename=raw_input('Input filename of the image : ')
        try:
            with open(filename) as file:
             output = Image.open(filename)
             break
        except IOError as e:
            print "Unable to open file"
    return (output, filename)

def get_n_samples(width,height):

    while True :
        n = raw_input('Input number of samples : ');
        n = int(n)    # cast rmin value to integer
        

        if(n > width*height):    # check rmin and rmax condition
            print "Number of samples should be less than number of pixels"
        else:
            break
    return (n)    
    
def form_data_mat(width,height):
    #get the coordinates and ones for forming data matrix
    coordinates = [[x, y] for x in xrange(height) for y in xrange(width)]
    
    xy = (np.asarray(coordinates)).T
    
    ones = np.asarray([1]*width*height)
    #compute the data matrix
    datamat = np.matrix([xy[0],xy[1],ones]) 
    datamat = datamat.T
    
    return datamat
    
def form_cov_mat(width, height):
    coordinates = [[x, y] for x in xrange(height) for y in xrange(width)]
    xy = (np.asarray(coordinates)).T
    ones = np.asarray([1]*width*height)
    xk = xy[0]
    yk = xy[1]
    xkyk = xk*yk
    
    covmat = np.matrix([ones,xk,yk,xkyk]) 
    covmat = covmat.T
    
    return covmat
    
def get_para_vec(datamat,z):
    paravec = np.matrix((datamat.T * datamat).I * datamat.T)*z
    
    return paravec
    
    
def compute_i_r(datamat,w,imgl1D):
    #compute the 1D illumination image and convert it to 2D
    imgi1D = np.reshape((datamat * w),(-1,1))
    imgi2D = np.reshape(imgi1D, (-1,width))

    #compute the 1D reflex image and convert it to 2D
    imgr1D = np.exp(imgl1D - imgi1D)
    imgr2D = np.reshape(imgr1D, (-1,width))
    
    return (imgi2D,imgr2D)
    
def show_result(original, compensation, illumination, mode):
    
    plt.subplot(131),plt.title('Original image'),plt.imshow(img, cmap = 'gray')
    plt.subplot(132),plt.title('%s Compensation'%(mode)),plt.imshow(linearR, cmap = 'gray')
    plt.subplot(133),plt.title('%s Illumination'%(mode)),plt.imshow(linearI, cmap = 'gray')
##================================================

#Initialization
img, filename = get_img()
filename_split = filename.split(".")

#get the size of the image
width = img.size[0]
height = img.size[1]
#get the number of samples
n = get_n_samples(width,height)

#compute the log of image and convert the 2D image array to 1D
imgl = np.log(np.array(img)+0.1)
imgl1D = np.reshape(imgl,(-1,1))

datamat = np.matrix(form_data_mat(width,height))
covmat = form_cov_mat(width,height)

#compute the parameter vector w
z = np.matrix(imgl1D)
linearpara = get_para_vec(datamat,z)
bilinearpara = get_para_vec(covmat,z)

linearI,linearR = compute_i_r(datamat,linearpara,imgl1D)
bilinearI,bilinearR = compute_i_r(covmat,bilinearpara,imgl1D)

#show the result
plt.figure(1)
show_result(img, linearR, linearI, "Linear")
plt.figure(2)
show_result(img, bilinearR, bilinearI, "Bilinear")

#construct the coordinates for 3D plotting
Xs = np.arange(0, width, 1)
Ys = np.arange(0, height, 1)
Xs, Ys = np.meshgrid(Xs, Ys)

#plot the 3D intensity
fig = plt.figure(3)
ax = fig.add_subplot(131, projection='3d')
ax.plot_wireframe(Xs,Ys,linearR,rstride=25, cstride=25)

ax1 = fig.add_subplot(132, projection='3d')
ax1.plot_wireframe(Xs,Ys,img,rstride=25, cstride=25)

ax2 = fig.add_subplot(133, projection='3d')
ax2.plot_wireframe(Xs,Ys,linearI,rstride=25, cstride=25)

##================================================
randrow = np.random.randint(datamat.shape[0],size=n)
sampleddata = datamat[randrow]
sampledcov = covmat[randrow]
sampledZ = z[randrow]

linearpara_s = get_para_vec(sampleddata,sampledZ)
bilinearpara_s = get_para_vec(sampledcov,sampledZ)

sampledI,sampledR = compute_i_r(datamat,linearpara_s,imgl1D)
sampledIb,sampledRb = compute_i_r(covmat,bilinearpara_s,imgl1D)

#show the result
plt.figure(3)
show_result(img, sampledR, sampledI, "Linear(Sampled)")
plt.figure(4)
show_result(img, sampledRb, sampledIb, "Bilinear(Sampled)")

fig = plt.figure(6)
ax = fig.add_subplot(131, projection='3d')
ax.plot_wireframe(Xs,Ys,sampledR,rstride=25, cstride=25)

ax1 = fig.add_subplot(132, projection='3d')
ax1.plot_wireframe(Xs,Ys,img,rstride=25, cstride=25)

ax2 = fig.add_subplot(133, projection='3d')
ax2.plot_wireframe(Xs,Ys,sampledI,rstride=25, cstride=25)



