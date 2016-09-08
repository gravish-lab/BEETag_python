

import matplotlib
matplotlib.use('qt4agg')


import BEEtag as BT

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


if __name__ == '__main__':

    print('testing now')

    im = np.array(Image.open('./data/scaleExample.png'))

    a = BT.BEEtag(visualize=0)
    print(a)

    a.set_image(im)
    a.find_valid_regions()
    a.find_square_regions()

    a.transform_to_grid()

    a.undistort_squares()

#    a.get_codes()


#
    # drawing stuff
    plt.figure(1)
    plt.imshow(a.im)

    plt.figure(2)
    plt.imshow(a.im_gray)

    plt.figure(3)
    plt.imshow(a.BW_Label)

    plt.figure(4)
    for kk, (reg, x, y) in enumerate(zip(a.regions, a.contour_x_points, a.contour_y_points)):
        plt.subplot(len(a.regions) / 2 + 1, 2, kk+1)
        plt.imshow(reg.filled_image, interpolation='nearest')
        plt.title(len(x))
        plt.plot(x, y, 'o-')

    a.draw_possible_regions()

    a.draw_quad_regions()


    plt.draw()
    plt.show(block=False)





