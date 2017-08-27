


# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import scipy.io as sio
# from PIL import Image
# import beeTag as bt

import matplotlib
matplotlib.use('qt5agg')

import matplotlib.pyplot as plt

import cv2 as cv # written for opencv3
import numpy as np


from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage.measure import regionprops, label

#TODO Not implemented yet:
#TODO 1) Robust track
#TODO 2)

class BEEtag:
    '''
    locates optical tags and spits out regionprops info for all tags

    Input form is locateCodes(im, varargin)

    Required input:

    'im' is an image containing tags, can be rgb or grayscale - currently not
    supported to directly input


    Optional inputs include:

    'col_mode' - determines whether to show gray (1)  or bw (0) image, 2 is
      rgb, anything else (i.e. 3) plots on whatever background is already plotted

    'threshold' - thresholding value to turn grayscale into a binary image,
      ranges from 0 to 1, default (None) is to calculate threshold value automatically

    'vis' - whether or not to visualize results, 0 being no visualization, 1
      being visualization. Default is visualization

    'sizeThresh' - one element vector sets the mimimum size threshold, two
      element vector sets the minimum and maximum size threshold. Only really
      helps to clean out noise - start with a low number at first!
      Default is a minimum threshold of 100

    'robustTrack' - whether or not to identify binary values for tracking codes
      from black and white binary image, or to track over a range of values from
      an original grayscale image with intelligent thresholding. The latter
      produces more false positives, and it is recommended to only use this in
      conjunction with a pre-specificed list of tags for tracking. Adding size
      restrictions on valied tags is also recommended. When using this option,
      you must specify a grayscale image to take the pixel values from (can
      be the same as 'im');

    'tagList'- option to add list of pre-specified valid tags to track. The
      taglist should be a vector of tag numbers that are actually in im.
      Output from any other tags found in the picture is ignored

    'threshMode' - options for black-white thresholding. Default is 0, which
      uses supplied threshold and above techniques. Alternative option is
      Bradley local adaptive thresholding, which helps account for local
      variation in illumination within the image.

    'bradleyFilterSize' - two element vector defining the X and Y
      (respectively) size of locally adaptive filter. Only supply when
      'threshMode' is 1 (using adaptive thresholding).

    'bradleyThreshold' - black-white threshold value after local filtering.
      Default value is 3, lower values produce darker images, and vice versa.



    Outputs are:
    Area: area of tag in pixel:

    Centroid: X and Y coordinates of tag center

    Bounding Box: Boundig region of image containing tag

    corners: Coordinates of four calculated corner points of tag

    code: 25 bit binary code read from tag

    number: original identification number of tag

    frontX: X coordiante (in pixels) of tag "front"

    frontY: Y coordinate (in pixels) of tag "front"

    '''


    def __init__(self, col_mode = 1, threshold = None, visualize = True,
                 region_size = (500, 3000), robust_track = False, tag_list = [],
                 localized_threshold_params = (41, 0),
                 length_epsilon = 0.2,
                 parallel_epsilon = 15,
                 contour_epsilon = 0.1):
        '''
        Default initialization
        '''

        self.col_mode = col_mode  # color images
        self.visualize = visualize
        self.localized_thresh_mode = (threshold == None)
        self.robust_track = robust_track
        self.tag_list = tag_list
        self.localized_thresh_blocksize = localized_threshold_params[0]
        self.localized_thresh_offset = localized_threshold_params[1]
        self.region_size = region_size

        self.im = None
        self.im_gray = None
        self.BW_Label = None
        self.regions = None

        self.sc = 100 # for unwarping
        self.length_epsilon = length_epsilon
        self.parallel_epsilon = parallel_epsilon
        self.contour_epsilon = contour_epsilon

        # for plotting
        self.plots_corner_size = 10

        if threshold is None:
            self.auto_threshold = True
            self.threshold = 0
        else:
            self.auto_threshold = False
            self.threshold = threshold

    def __str__(self):
        '''
        Print out diagnostics
        '''
        outstr = 'threshold = ' + str(self.threshold) + '\r\n' + \
                 'threshold type = ' + str(self.auto_threshold) + '\r\n' + \
                 'col mode = ' + str(self.col_mode) + '\r\n' + \
                 'size tag_list = ' + str(len(self.tag_list)) + '\r\n' + \
                 'robust_track = ' + str(self.robust_track)

        return outstr


    def locate_codes(self, im):
        '''
        Function to locade codes in the providec image
        '''

    def set_image(self, im):
        self.im = im
        self.im_gray = cv.cvtColor(self.im, cv.COLOR_RGB2GRAY)

    def threshold_image(self):
        '''
        Convert image to grayscale for thresholding
        Using either global threshold or local adaptive threshold
        '''

        # check for global or local thresholding
        if self.localized_thresh_mode:
            self.im_bw = cv.adaptiveThreshold(self.im_gray,
                                              1,
                                              cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv.THRESH_BINARY,
                                              self.localized_thresh_blocksize,
                                              self.localized_thresh_offset)

        else:
            # Otsu's method
            self.im_bw  = cv.threshold(self.im_gray,
                                          0,
                                          1,
                                          cv.THRESH_BINARY+cv.THRESH_OTSU)

        self.BW_Label = label(self.im_bw, neighbors=8, background=0)

    def find_valid_regions(self):
        '''
        Segment the black and white image into connected components.
         Ignore component regions beyond specifications
        '''
        #TODO Need to convert to opencv based processing
        regions_tmp = regionprops(self.BW_Label)

        # chained comparisons!!!!!
        self.regions = [reg for reg in regions_tmp if self.region_size[0] < reg['Area'] < self.region_size[1]]

    def find_square_regions(self):
        '''
        documentation from opencv
        http://docs.opencv.org/3.0-last-rst/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#approxpolydp

        '''

        def unit_vector(vector):
            """
            Returns the unit vector of the vector.
            """
            return vector / np.linalg.norm(vector)

        def angle_between(v1, v2):
            """
            Returns the angle in radians between vectors 'v1' and 'v2'::
            """
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        self.contour_x_points = [] # mainly for debugging
        self.contour_y_points = []

        self.square_x_points = []
        self.square_y_points = []

        self.square_regions = []

        self.number_squares = 0

        for reg in self.regions:

            # pull out the contours
            imout, contours, hierarchy = cv.findContours(np.uint8(reg.filled_image),
                                           cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

            # Best approximation of a polygon for the shape (not sure about 0 here need to chck TODO)
            tmp = cv.approxPolyDP(contours[0], self.contour_epsilon * cv.arcLength(contours[0], True), True)

            x = [pt[0][0] for pt in tmp]
            y = [pt[0][1] for pt in tmp]

            self.contour_x_points.append(x)  # convert out of opencv's annoying nested list
            self.contour_y_points.append(y)

            # 4 points in a square
            if len(x) != 4:
                continue

            # check if square
            # if square these points will be parallel, and all the same length approximately
            # for x, y in zip(self.contour_x_points[-1], self.contour_y_points[-1]):

            # first check lengths
            lengths = [np.sqrt((x[k % 4] - x[(k + 1) % 4]) ** 2 + (y[k % 4] - y[(k + 1) % 4]) ** 2) for k in np.arange(1, 5)]
            error = np.abs(1-lengths/np.mean(lengths))

            OK_lengths = all([err < self.length_epsilon for err in error])

            # next check paralelism
            dots = [angle_between(np.array([x[k + 1], y[k + 1]]) - np.array([x[k], y[k]]),
                                  np.array([x[(k + 2) % 4], y[(k + 2) % 4]]) -
                                  np.array([x[(k + 2 + 1) % 4], y[(k + 2 + 1) % 4]])) for k in range(2)]


            OK_parallel = all([dot < self.parallel_epsilon for dot in dots])

            if OK_lengths and OK_parallel:

                self.square_x_points.append(x)  # convert out of opencv's annoying nested list
                self.square_y_points.append(y)

                self.square_regions.append(reg)

                self.number_squares += 1

    def transform_to_grid(self):
        '''
        Transform the accepted quad regions to a perfect grid

        kk = index of square to transform
        '''

        dest_grid = np.float32([[0, 0], [0, 1], [1, 1], [1, 0]])*self.sc

        self.undistorted_im = []
        self.undistorted_bw_im = []
        self.transformation_matrix = []

        for sq_x, sq_y, reg in zip(self.square_x_points, self.square_y_points, self.square_regions):

            source_grid = np.float32([[x, y] for x,y in zip(sq_x, sq_y)])
            M = cv.getPerspectiveTransform(source_grid, dest_grid)
            self.transformation_matrix.append(M)

            imm = self.im_from_bbox(self.im, reg.bbox)
            imm_bw = self.im_from_bbox(self.im_bw, reg.bbox)

            imm_transformed = cv.warpPerspective(np.array(imm, dtype='uint8'), M, (self.sc, self.sc))
            imm_transformed_bw = cv.warpPerspective(np.array(imm_bw, dtype='uint8'), M, (self.sc, self.sc))

            self.undistorted_im.append(imm_transformed)
            self.undistorted_bw_im.append(imm_transformed_bw)

    def get_codes(self):

        def checkOrs25(imc):
            """
            Checks for valid 5x5 code pattern in each of the 4 possible configurations
            Returns a code and the orientation
            """
            check = []

            # checks all 4 possible orientations of the tag.
            # checks the checksum column and stores the value of whether valid tag or not in in check list
            for cc in range(4):
                imcr = np.rot90(imc, cc)
                check.append(checkCodes25(imcr))  # TODO must return bool

            check = np.array(check)

            # there can only be one valid orientation of a correct tag, so there should only be 1 entry that is
            # non zero in the check array. If the sum is 0 or greater than 1 we know it is invalid

            codesFinal = None
            orienation = None

            if sum(check) == 1:
                orienation = np.where(check)[0]
                codesFinal = np.rot90(imc, orienation)

            return codesFinal, orienation

        def checkCodes25(imc):
            """
            Checks that code is valid by checking error bits in last 2 columns of 5x5 pattern. Error bits are as follows

            column 4: row bits 1-3 are parity of columns 1-3, row bit 4 is parity of top 3 rows in 3 columns of code.
            (parity of upper left 3x3 code matrix). row bit 5 is parity of bottom 2 rows of 3 columns (bottom left
            2x3 code matrix).

            column 5: reverse of column 4.

            column 5 must equal reverse of column 4 for pass
            """

            im = imc[:, 0:3]
            check_column1 = imc[:, 3].tolist()
            check_column2 = imc[:, 4].tolist()
            check_column2.reverse()

            for first_three_bits in range(3):
                if (np.sum(im[:, first_three_bits]) % 2) != check_column1[first_three_bits]:
                    return False

            if (np.sum(im[:3, :]) % 2) != check_column1[3]:
                return False

            if (np.sum(im[3:, :]) % 2) != check_column1[4]:
                return False

            if any([c1 != c2 for c1, c2 in zip(check_column1, check_column2)]):
                return False

            # if we got thorough all the checks, return true
            return True

        code_locations = np.array([1.5 / 7, 2.5 / 7, 3.5 / 7, 4.5 / 7, 5.5 / 7]) * self.sc
        y_coord, x_coord = np.meshgrid(code_locations, code_locations)
        x_coord = np.reshape(x_coord, 25)
        y_coord = np.reshape(y_coord, 25)

        averaging_width = round((self.sc / 7) / 4)

        self.codes = []
        self.tag_id = []

        for code_im in self.undistorted_bw_im:
            tmp_code = []

            # pull out the binary code values
            for x,y in zip(x_coord, y_coord):

                xrange = np.int32(round(x) + np.array([-averaging_width, averaging_width]))
                yrange = np.int32(round(y) + np.array([-averaging_width, averaging_width]))

                tmp_code.append(np.mean(code_im[yrange[0]:yrange[1], xrange[0]:xrange[1]]) > .5)

                if self.visualize == True:
                    plt.clf()
                    self.draw_undistorted(color=False)
                    plt.plot(x, y, 'ow')
                    plt.draw()
                    plt.gca().invert_yaxis()
                    plt.title('This point is ' +  str(tmp_code[-1]) + ' press any button to continue')
                    plt.waitforbuttonpress()

            # convert back to 5x5 array
            tmp_code = np.reshape(tmp_code, (5,5)).T
            # do the heavy lifting to check the code
            final_code, valid_code = checkOrs25(tmp_code)

            if valid_code is not None:
                self.codes.append(final_code)
                self.tag_id.append(self.code_to_id(final_code))
            else:
                self.tag_id.append([])
                self.codes.append([])

    def code_to_id(self, code):
        """
        reshape so that is a 1D array columns first
        """
        code = np.uint8(np.ravel(code.T))
        return int(str().join(str(x) for x in code[0:15]), 2)

    def draw_undistorted(self, color = True):

        plot_w_dimension = np.ceil(np.sqrt(self.number_squares))
        plot_h_dimension = np.ceil(self.number_squares / plot_w_dimension)

        if color:
            ims = self.undistorted_im
        else:
            ims = self.undistorted_bw_im

        for kk, (reg, x, y) in enumerate(zip(ims, self.square_x_points, self.square_y_points)):
            plt.subplot(plot_h_dimension, plot_w_dimension, kk + 1)
            plt.imshow(reg, interpolation='nearest')
            [plt.plot([(self.sc/7) * (k+1), (self.sc/7) * (k + 1)], [0, self.sc], 'r') for k in range(7)]
            [plt.plot([0, self.sc],[(self.sc / 7) * (k + 1), (self.sc / 7) * (k + 1)], 'r') for k in range(7)]
            plt.axis([0, self.sc, 0, self.sc])
            plt.gca().invert_yaxis()

    def draw_possible_regions(self):

        plt.figure()

        for kk, (reg, x, y) in enumerate(zip(self.regions, self.contour_x_points, self.contour_y_points)):
            plt.subplot(len(self.regions) / 2 + 1, 2, kk + 1)
            plt.imshow(reg.filled_image, interpolation='nearest')
            plt.title(len(x))
            plt.plot(x, y, 'o-')
            plt.show(block = False)

    def draw_code_regions(self):

        plt.imshow(self.im, interpolation='nearest')

        for x,y,reg, tag_id_curr in zip(self.square_x_points, self.square_y_points,
                           self.square_regions, self.tag_id):
            if tag_id_curr != []:
                plt.plot(np.array(x) + reg.bbox[1], np.array(y) + reg.bbox[0],'o-')
                plt.gca().annotate(str(tag_id_curr), xy=(reg.bbox[1] + 10, reg.bbox[0] + 10),
                                   fontsize = 14, color = 'w')


    def draw_quad_regions(self):

        plt.imshow(self.im, interpolation='nearest')

        if len(self.tag_id) == len(self.square_y_points):
            for x, y, reg, tag_id_curr in zip(self.square_x_points, self.square_y_points,
                                              self.square_regions, self.tag_id):
                plt.plot(np.array(x) + reg.bbox[1], np.array(y) + reg.bbox[0], 'o-')
                plt.gca().annotate(str(tag_id_curr), xy=(reg.bbox[1] + 10, reg.bbox[0] + 10),
                                   fontsize=14, fontcolor='w')

        else:
            for x, y, reg in zip(self.square_x_points, self.square_y_points, self.square_regions):
                plt.plot(np.array(x) + reg.bbox[1], np.array(y) + reg.bbox[0], 'o-')

    def im_from_bbox(self, im_in, bbox):

        return im_in[bbox[0]:bbox[2], bbox[1]:bbox[3]]




if __name__ == '__main__':

    import BEEtag as BT

    import numpy as np
    from PIL import Image




    print('testing now')

    im = np.array(Image.open('./data/scaleExample.png'))

    a = BT.BEEtag(visualize=0)
    print(a)

    a.set_image(im)
    a.threshold_image()
    a.find_valid_regions()
    a.find_square_regions()

    a.transform_to_grid()


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





