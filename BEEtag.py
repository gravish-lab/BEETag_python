


# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import scipy.io as sio
# from PIL import Image
# import beeTag as bt

import cv2  # written for opencv3
import numpy as np

import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage.measure import regionprops, label


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
                 region_size = (50, 1000), robust_track = False, tag_list = [],
                 localized_thresh_mode = False, localized_threshold_params= (15, 0)):
        '''
        Default initialization
        '''

        self.col_mode = col_mode  # color images
        self.visualize = visualize
        self.localized_thresh_mode = localized_thresh_mode
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
        self.im_gray = rgb2gray(self.im)
        self.threshold_image()

    def threshold_image(self):
        '''
        Convert image to grayscale for thresholding
        Using either global threshold or local adaptive threshold
        '''

        # check for global or local thresholding
        if self.localized_thresh_mode == True:
            self.im_bw = threshold_adaptive(self.im_gray,
                                            block_size = self.localized_thresh_blocksize,
                                            offset = self.localized_thresh_offset)
        else:
            self.threshold = threshold_otsu(self.im_gray)
            self.im_bw = self.im_gray > self.threshold

    def find_valid_regions(self):
        '''
        Segment the black and white image into connected components.
         Ignore component regions beyond specifications
        '''

        self.BW_Label = label(self.im_bw, neighbors=8, background=0)
        regions_tmp = regionprops(self.BW_Label)

        self.regions = [reg for reg in regions_tmp if reg['Area'] > self.region_size[0] and
                                                      reg['Area'] < self.region_size[1]]

    def find_square_regions(self):
        '''
        documentation from opencv
        http://docs.opencv.org/3.0-last-rst/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#approxpolydp

        '''

        contour_x_points = [] # mainly for debugging
        contour_y_points = []

        square_x_points = []
        square_y_points = []

        square_regions = []

        for reg in self.regions:

            # pull out the contours
            imout, contours, hierarchy = cv2.findContours(np.array(reg.filled_image, dtype='uint8'),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Best approximation of a polygon for the shape (not sure about 0 here need to chck TODO)
            tmp = cv2.approxPolyDP(contours[0], 0.01 * cv2.arcLength(contours[0], True), True)

            contour_x_points.append([pt[0][0] for pt in tmp])  # convert out of opencv's annoying nested list
            contour_y_points.append([pt[0][1] for pt in tmp])

            # check if square
            if len(tmp) == 4:
                square_x_points.append([pt[0][0] for pt in tmp])  # convert out of opencv's annoying nested list
                square_y_points.append([pt[0][1] for pt in tmp])

                square_regions.append(reg)

        self.contour_x_points = contour_x_points
        self.contour_y_points = contour_y_points

        self.square_x_points = square_x_points
        self.square_y_points = square_y_points

        self.square_regions = square_regions


    def transform_to_grid(self):
        '''
        Transform the accepted quad regions to a perfect grid

        kk = index of square to transform
        '''

        dest_grid = np.float32([[0, 0], [0, 1], [1, 1], [1, 0]])*self.sc
        M = []
        for sq_x, sq_y in zip(self.square_x_points, self.square_y_points):

            source_grid = np.float32([[x, y] for x,y in zip(sq_x, sq_y)])
            M.append(cv2.getPerspectiveTransform(source_grid, dest_grid))

        self.transformation_matrix = M

    def undistort_squares(self):

        undistorted_im = []
        undistorted_bw_im = []

        for M, x, y, reg in zip(self.transformation_matrix,
                                self.square_x_points,
                                self.square_y_points,
                                self.square_regions):

            imm = self.im_from_bbox(self.im, reg.bbox)
            imm_bw = self.im_from_bbox(self.im_bw, reg.bbox)

            imm_transformed = cv2.warpPerspective(np.array(imm, dtype='uint8'), M, (self.sc, self.sc))
            imm_transformed_bw = cv2.warpPerspective(np.array(imm_bw, dtype='uint8'), M, (self.sc, self.sc))

            undistorted_im.append(imm_transformed)
            undistorted_bw_im.append(imm_transformed_bw)

            if self.visualize:
                plt.clf()
                plt.subplot(2,2,1)
                plt.imshow(imm, interpolation='nearest')

                plt.subplot(2,2,2)
                plt.imshow(imm_transformed, interpolation='nearest')

                plt.subplot(2,2,3)
                plt.imshow(imm_bw, interpolation='nearest')

                plt.subplot(2, 2, 4)
                plt.imshow(imm_transformed_bw, interpolation='nearest')

                plt.draw()
                plt.show()

        self.undistorted_im = undistorted_im
        self.undistorted_bw_im = undistorted_bw_im

    def get_codes(self):

        code_locations = np.array([5.5/7, 4.5/7, 3.5/7, 2.5/7, 1.5/7])*self.sc
        x_coord, y_coord = np.meshgrid(code_locations, code_locations)
        x_coord = np.reshape(x_coord, 25)
        y_coord = np.reshape(y_coord, 25)

        codes = []
        for code_im in self.undistorted_bw_im:
            tmp_code = []

            for x,y in zip(x_coord, y_coord):

                tmp_code.append(code_im[round(x),round(y)] == True)

            codes.append(tmp_code)

        self.id_codes = codes





    def draw_possible_regions(self):

        plt.figure()

        for kk, (reg, x, y) in enumerate(zip(self.regions, self.contour_x_points, self.contour_y_points)):
            plt.subplot(len(self.regions) / 2 + 1, 2, kk + 1)
            plt.imshow(reg.filled_image, interpolation='nearest')
            plt.title(len(x))
            plt.plot(x, y, 'o-')
            plt.show(block = False)

    def draw_quad_regions(self):

        plt.figure()
        plt.imshow(self.im, interpolation='nearest')

        for x,y,reg in zip(self.square_x_points, self.square_y_points, self.square_regions):
            plt.plot(np.array(x) + reg.bbox[1], np.array(y) + reg.bbox[0],'o-')


    def im_from_bbox(self, im_in, bbox):

        return im_in[bbox[0]:bbox[2], bbox[1]:bbox[3]]

