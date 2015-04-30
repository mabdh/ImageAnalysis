__author__ = 'muhammadabduh'

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import getopt

def get_img():
    #Get the input file from user
    while True :
        filename='bauckhage-gamma-2.png'#raw_input('Input filename of the image : ')
        try:
            with open(filename) as file:
             output = Image.open(filename)
             break
        except IOError as e:
            print "Unable to open file"
    return (output, filename)

def plot_graph(a_v, b_v, L):
    for i in range(0, L+1):
        a_v[i] = i * (256/L)
    plt.step(a_v[1:L+1], b_v.astype(np.int16))
    plt.axis([0, 256, 0, 256], pickradius=256/L)
    plt.xticks(a_v)
    plt.yticks(a_v)
    plt.show()

def calculate_lloyd_max(a_v, b_v, img, L, iteration):
    # initialize
    a = np.asarray(img)
    hx = img.histogram()
    hx = np.array(hx).astype(dtype='f')
    sum_hx = np.sum(hx)
    px = hx/sum_hx

    for i in range(0, L+1):
        a_v[i] = i * (256/L)

    b_v = np.zeros(L)
    for i in range(0, L):
        b_v[i] = i * (256/L) + (256/(2*L))
    print a_v
    print b_v

    # Error calculation
    # err_prev = 0.0
    # for v in range(0, L):
    #         for av in range(int(a_v[v]), int(a_v[v+1])):
    #             err_prev += np.power((av - b_v[v]), 2) * px[av]

    # while True:
    for j in range(0, iteration):
        err = 0
        for v in range(1, L):
            a_v[v] = 0.5*(b_v[v] + b_v[v-1])
        print a_v

        for v in range(0, L):
            num = 0.0
            denum = 0.0
            for av in range(int(a_v[v]), int(a_v[v+1])):
                num += av * px[av]
                denum += px[av]
            if denum == 0:
                denum = 0.0001
            b_v[v] = num/denum
        print b_v

        # Error calculation
        # for v in range(1, L):
        #     for av in range(int(a_v[v-1]), int(a_v[v])):
        #         err += np.power((av - b_v[v-1]), 2) * px[av]
        # print err
        # print err_prev
        # if abs(err-err_prev) < epsilon:
        #     break
        # else:
        #     err_prev = err
    return b_v.astype(dtype='int16')

# def implement_quantization(b_v, img, L):
#     img_array = np.asarray(img)
#     width, height = img.size
#     res = np.zeros(shape=(width, height), dtype=np.int)
#     for i in range(0, width):
#         for j in range(0, height):
#             for k in range(L):
#                 threshold_min = k*256/L
#                 threshold_max = (k+1)*256/L
#                 if img_array[i][j] > threshold_min:
#                     if img_array[i][j] < threshold_max :
#                         res[i][j] = b_v[k]
#     image = Image.fromarray(np.uint8(img_array))
#     image.show()

#Defining the Main Function
def main(argv):
    helpstr = """Task1.py -l <QuantizationLevel> -i <NumberOfIteration>"""
    # parse command line
    iteration = None
    q_level = None
    try:
        opts, args = getopt.getopt(argv, "l:i:")
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

    if q_level == None:
        print helpstr
        sys.exit(2)
    if iteration == None:
        print helpstr
        sys.exit(2)

    img, filename = get_img()
    filename_split = filename.split(".")

    a_v = np.zeros(q_level+1)
    b_v = np.zeros(q_level+1)
    quantization_value = calculate_lloyd_max(a_v, b_v, img, q_level, iteration)
    print quantization_value
    plot_graph(a_v, quantization_value, q_level)
    # implement_quantization(quantization_value, img, q_level)
if __name__ == "__main__":
    main(sys.argv[1:])