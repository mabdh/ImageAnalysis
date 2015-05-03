__author__ = 'muhammadabduh'

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import getopt

def get_img():
    #   Get the input file from user
    #   input : None
    #   output: output (PIL image file)
    #           filename (File name : string)
    while True:
        filename = raw_input('Input filename of the image : ')
        try:
            with open(filename) as file:
                output = Image.open(filename)
                break
        except IOError as e:
            print e
    return (output, filename)

def plot_histogram(img, img2):
    #   Plot histogram of 2 images
    #   input : img (PIL image file)
    #           img2 (PIL image file)
    #   output: None
    a = np.asarray(img).ravel()
    a2 = np.asarray(img2).ravel()
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].hist(a, bins=256)
    axarr[1].hist(a2, bins=256)
    plt.title("Image Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

def plot_graph(a_v, b_v, L):
    #   Plot the quantization graph
    #   input : a_v (quantization interval)
    #           b_v (quantization result)
    #           L (quantization level : integer)
    #   output: None
    plt.figure(2)
    for i in range(0, L+1):
        a_v[i] = i * (256/L)
    b_v = np.append(0, b_v)
    plt.step(a_v[0:L+1], b_v)
    plt.axis([0, 256, 0, 256], pickradius=256/L)
    plt.xticks(a_v)
    plt.yticks(a_v)
    plt.show()

def mse(a_v, b_v, p_x, L):
    #   Mean squared quantization error function
    #   input : a_v (quantization interval)
    #           b_v (quantization result)
    #           p_x (probability density function)
    #           L (quantization level : integer)
    #   output: err (error : float)
    err_el = 0
    for v in range(L):
        x = np.linspace(a_v[v], a_v[v+1], int(a_v[v+1])-int(a_v[v])).astype(dtype='int16')
        bv = np.ones(np.size(x))
        bv = bv * b_v[v]
        px = p_x[int(a_v[v]):int(a_v[v+1])]
        e = np.sum(np.multiply(np.power(np.subtract(x, bv), 2), px))
        err_el = np.append(err_el, e)
    err = np.sum(err_el)
    return err

def calculate_lloyd_max(a_v, b_v, img, L, iteration):
    #   Perform Lloyd-Max algorithm
    #   input : a_v (quantization interval)
    #           b_v (quantization result)
    #           img (PIL image file)
    #           L (quantization level : integer)
    #           iteration (number of iteration : integer)
    #   output : b_v (quantization result)
    #           it_count (iteration until minimum error : integer)
    hx = img.histogram()
    hx = np.array(hx)
    sum_hx = np.sum(hx)
    p_x = hx.astype(dtype='f')/sum_hx

    for i in range(0, L+1):
        a_v[i] = i * (256/L)

    for i in range(0, L):
        b_v[i] = i * (256/L) + (256/(2*L))

    # Error calculation
    err_prev = mse(a_v, b_v, p_x, L)
    it_count = 1
    # while True:
    for j in range(0, iteration-1):
        err = 0
        for v in range(1, L):
            a_v[v] = 0.5*(b_v[v] + b_v[v-1])

        for v in range(0, L):
            num = 0.0
            denum = 0.0
            for av in range(int(a_v[v]), int(a_v[v+1])):
                num += av * p_x[av]
                denum += p_x[av]
            if denum == 0:
                denum = 0.0001
            b_v[v] = num/denum

        # Error calculation
        err = mse(a_v, b_v, p_x, L)
        it_count += 1
        # print "Error\t\t:\t" + str(err)
        if abs(err-err_prev) == 0:
            print "Error\t:\t" + str(err)
            break
        else:
            err_prev = err
    print "Interval : " + str(a_v.astype(dtype='int16'))
    return (b_v.astype(dtype='int16'), it_count)

def implement_quantization(b_v, img, L):
    #   Create a new image file that is already quantized
    #   input : b_v (quantization result)
    #           img (PIL image file)
    #           L (quantization level : integer)
    #   output: res (array of image)
    print "implementing quantization..."
    img_array = np.array(img)
    width, height = img.size
    res = np.zeros(shape=(width, height), dtype=np.int)
    for i in range(0, width):
        for j in range(0, height):
            for k in range(L):
                threshold_min = k*256/L
                threshold_max = (k+1)*256/L
                if img_array[i][j] > threshold_min:
                    if img_array[i][j] < threshold_max :
                        res[i][j] = b_v[k]

    image = Image.fromarray(np.uint8(res))
    # image.show()
    # img.show()
    return res

def img_to_file(img, filename):
    #   Convert and save array of image to the image file
    #   input : img (array of image)
    #           filename (file name : string)
    #   output : output (Image file)
    print "writing image to file..."
    output = Image.fromarray(img.astype(np.uint8))
    output.save(filename)
    print filename + " is created"
    return output

def main(argv):
    helpstr = """Task1.py -l <QuantizationLevel> -i <NumberOfIteration> [optional -q <ImplementQuantization>]\n\nThe maximum iteration value is the number until iteration is stopped (the error is minimized)\nImplement quantization will create new image based on quantization, show the histogram comparation, and save to file\n\nExample :\nTask1.py -l 8 -i 10 -q true"""
    # parse command line
    iteration = None
    q_level = None
    implement = None
    implementtrigger = 0
    try:
        opts, args = getopt.getopt(argv, "l:i:q:")
    except getopt.GetoptError:
        print helpstr
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print helpstr
            sys.exit()
        elif opt == "-l":
            q_level = int(arg)
        elif opt == "-i":
            iteration = int(arg)
        elif opt == "-q":
            implement = arg
    if q_level == None:
        print helpstr
        sys.exit(2)
    if iteration == None:
        print helpstr
        sys.exit(2)
    if implement == "true":
        implementtrigger = 1
    # iteration = 2
    # q_level = 8
    img, filename = get_img()
    filename_split = filename.split(".")

    a_v = np.zeros(q_level+1)
    b_v = np.zeros(q_level)
    quantization_value, it_count = calculate_lloyd_max(a_v, b_v, img, q_level, iteration)
    print "Quantization value : " + str(quantization_value)
    print "Iteration : " + str(it_count)
    plot_graph(a_v, quantization_value, q_level)

    if implementtrigger == 1:
        res_image = implement_quantization(quantization_value, img, q_level)
        new_filename = filename_split[0] + "-task1." + filename_split[1]    # create filename
        result = img_to_file(res_image, new_filename)
        plot_histogram(img, result)

if __name__ == "__main__":
    main(sys.argv[1:])