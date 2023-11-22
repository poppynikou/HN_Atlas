import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import time
from scipy.interpolate import Rbf, interp1d, pchip_interpolate, CubicSpline
import nibabel as nib 
from natsort import natsorted 
from sklearn.preprocessing import normalize

'''
from: https://github.com/MaciejPMarciniak/curvature/blob/master/SimpleCurvature.py
'''


class Curvature:
    """
    Class for computing curvature of ordered list of points on a plane
    """
    def __init__(self, trace, interpolation_function):

        self.trace = np.array(trace)
        self.interpolation_function = interpolation_function
        self.curvature = None

    @staticmethod
    def _get_twice_triangle_area(a, b, c):

        if np.all(a == b) or np.all(b == c) or np.all(c == a):
            exit('CURVATURE:\nAt least two points are at the same position')

        twice_triangle_area = (b[0] - a[0])*(c[1] - a[1]) - (b[1]-a[1]) * (c[0]-a[0])

        if twice_triangle_area == 0:
            warnings.warn('Collinear consecutive points found: '
                          '\na: {}\t b: {}\t c: {}'.format(a, b, c))

        return twice_triangle_area

    def _get_menger_curvature(self, a, b, c):

        menger_curvature = (2 * self._get_twice_triangle_area(a, b, c) /
                            (np.linalg.norm(a - b) * np.linalg.norm(b - c) * np.linalg.norm(c - a)))
        return menger_curvature

    def calculate_curvature(self, interpolation_target_n=500):
        self.trace = interpolate_trace(self.trace, self.interpolation_function, interpolation_target_n)

        self.curvature = np.zeros(len(self.trace) - 2)
        for point_index in range(len(self.curvature)):
            triplet = self.trace[point_index:point_index+3]
            self.curvature[point_index-1] = self._get_menger_curvature(*triplet)
        return self.curvature

    def plot_curvature(self):
        fig, _ = plt.subplots(figsize=(8, 7))
        _.plot(self.trace[1:-1, 0], self.curvature, 'r-', lw=2)
        _.set_title('Corresponding Menger\'s curvature'.format(len(self.curvature)))
        plt.show()
        fig.savefig(os.path.join('images', 'Curvature.png'))
        return _


class GradientCurvature:

    def __init__(self, trace, interpolation_function, plot_derivatives=True):
        self.trace = trace
        self.plot_derivatives = plot_derivatives
        self.interpolation_function = interpolation_function
        self.curvature = None

    def _get_gradients(self):
        self.x_trace = [x[0] for x in self.trace]
        self.y_trace = [y[1] for y in self.trace]

        x_prime = np.gradient(self.x_trace)
        y_prime = np.gradient(self.y_trace)
        x_bis = np.gradient(x_prime)
        y_bis = np.gradient(y_prime)

        if self.plot_derivatives:
            plt.subplot(411)
            plt.plot(self.y_trace, label='y')
            plt.title('Function')

            plt.subplot(412)
            # plt.plot(x_prime, label='x\'')
            plt.plot(y_prime, label='y\'')
            plt.title('First spatial derivative')
            plt.legend()
            plt.subplot(413)
            # plt.plot(x_bis, label='x\'\'')
            plt.plot(y_bis, label='y\'\'')
            plt.title('Second spatial derivative')
            plt.legend()

        return x_prime, y_prime, x_bis, y_bis

    def calculate_curvature(self, interpolation_target_n=500):

        self.trace = interpolate_trace(self.trace, self.interpolation_function, interpolation_target_n)
        x_prime, y_prime, x_bis, y_bis = self._get_gradients()
        curvature = x_prime * y_bis / ((x_prime ** 2 + y_prime ** 2) ** (3/2)) - \
            y_prime * x_bis / ((x_prime ** 2 + y_prime ** 2) ** (3/2))  # Numerical trick to get accurate values
        self.curvature = curvature
        return curvature


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


def interpolate_trace(trace, interpolation_function, target_n=500):

    n_trace_points = len(trace)
    x_points = [_x[0] for _x in trace]
    y_points = [_y[1] for _y in trace]

    positions = np.arange(n_trace_points)  # strictly monotonic, number of points in single trace
    interpolation_base = np.linspace(0, n_trace_points-1, target_n+1)

    x_interpolated, y_interpolated = interpolation_function(x_points, y_points, positions, interpolation_base)

    interpolated_trace = np.array([[x, y] for x, y in zip(x_interpolated, y_interpolated)])

    return interpolated_trace


def rbf_interpolation(x, y, positions, interpolation_base):

    # Radial basis function interpolation 'quintic': r**5 where r is the distance from the next point
    # Smoothing is set to length of the input data

    rbf_x = Rbf(positions, x, smooth=len(positions), function='quintic')
    rbf_y = Rbf(positions, y, smooth=len(positions), function='quintic')

    #print('Interpolation')
    #print(len(positions))
    #print(rbf_x.norm)
    #print('------------')
    # Interpolate based on the RBF model
    x_interpolated = rbf_x(interpolation_base)
    y_interpolated = rbf_y(interpolation_base)

    return x_interpolated, y_interpolated


def interp1d_interpolation(x, y, positions, interpolation_base, interpolation_kind='cubic'):

    interp1d_x = interp1d(positions, x, kind=interpolation_kind)
    interp1d_y = interp1d(positions, y, kind=interpolation_kind)

    x_interpolated = interp1d_x(interpolation_base)
    y_interpolated = interp1d_y(interpolation_base)

    return x_interpolated, y_interpolated


def pchip_interpolation(x, y, positions, interpolation_base):

    #print(positions)
    #print(interpolation_base)

    pchip_x = pchip_interpolate(positions, x, interpolation_base)
    pchip_y = pchip_interpolate(positions, y, interpolation_base)

    return pchip_x, pchip_y




def calc_max_spinal_cord_curvature(path_to_spinal_cord):
      
    img_obj, _, img_header = get_image_objects(path_to_spinal_cord)
    img_dims = img_header['dim'][1:4]

    # find the coordinates of the middle of the cord 
    # assuming small changes in the coronal 
    middleCordCoords = {'Y': [], 'Z':[]}
    for z in range(0, img_dims[2]):
        sum_of_slice = np.sum(np.sum(img_obj[:,:,z])) 
        if sum_of_slice !=0:
            img_slice = img_obj[:,:,z]
            indexes = np.nonzero(img_slice)
            middleIndex = int((len(indexes) - 1)/2)
            middleCordCoords['Y'].append(indexes[1][middleIndex])
            middleCordCoords['Z'].append(z)
    
    middleCordCoords['Y'] = normalize([middleCordCoords['Y']], norm='max')
    middleCordCoords['Z'] = normalize([middleCordCoords['Z']], norm='max')

    k = 1  # Resolution
    xy = list(zip(middleCordCoords['Z'],middleCordCoords['Y']))  # list of points in 2D space

    plt.scatter(middleCordCoords['Z'],middleCordCoords['Y'])
    plt.ylim([0,1])
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    '''
    ifunc = interp1d_interpolation
    curv1 = GradientCurvature(trace=xy, interpolation_function=ifunc)
    curv1.calculate_curvature()

    curv2 = Curvature(trace=xy, interpolation_function=ifunc)
    #start = time.time()
    curv2.calculate_curvature()

    plt.subplot(414)
    plt.plot(range(2, len(curv2.curvature)+2), curv2.curvature, 'd-', label='Menger curvature')
    plt.plot(curv1.curvature, 'g.-', label='Gradient curvature')
    plt.legend()
    plt.show()
    '''
    
    
def get_image_objects(Img_path):
    
        """Get the key information from the nifti image. 
        Function reads in the nifti image and returns the image voxel value, the affine and the header. 

        :param Img_path: path to the image to read in 
        :type Img_path: str
        :return img_obj: voxel values stored in the image 
        :rtype img_obj: array
        :return img_affine: affine matrix associated with the image
        :rtype img_affine: Matrix
        :return img_header: header information associated with the image
        :rtype img_header: struct
        """
        
        img = nib.load(Img_path)
        img_obj = img.get_fdata()
        img_affine = img.affine
        img_header = img.header

        return img_obj, img_affine, img_header

def calc_body_volume(path_to_body):
    
    img_obj, _, img_header = get_image_objects(path_to_body)
    voxel_volume = img_header['pixdim'][1] * img_header['pixdim'][2] * img_header['pixdim'][3]
    del img_header

    total_voxels = np.sum(np.sum(np.sum(img_obj,0),0),0)

    total_volume = total_voxels * voxel_volume

    return total_volume

def calc_integral_limits(path_to_vertabrae_segms):

    path_to_C1 = path_to_vertabrae_segms + '/vertebrae_C1.nii.gz'
    path_to_T2 = path_to_vertabrae_segms + '/vertebrae_T2.nii.gz'

    C1_obj,_,header = get_image_objects(path_to_C1)
    T2_obj,_,_ = get_image_objects(path_to_T2)
    slice_thickness = header['pixdim'][3]

    nonzero_C1 = np.nonzero(C1_obj)
    nonzero_T2 = np.nonzero(T2_obj)

    maximum = max(nonzero_C1[2])
    minimum = min(nonzero_T2[2])

    return maximum, minimum, slice_thickness



if __name__ == '__main__':

    path_to_data = 'D:/HNSCC/ARCHIVE/2023_10_08'
    folders = os.listdir(path_to_data)
    folders = natsorted(folders)


    text_file = 'C:/Users/poppy/Documents/HN_Atlas/HNSCC/HNSCC_Characteristics.txt'
    f = open(text_file, "a")
    f.write('Patient, Body_Volume Max_Curvature Height\n')
        
    for folder in folders:
        print(folder)

        path_to_body = path_to_data + '/'+ folder + '/NIFTI_LIMBUS/BIN_body.nii.gz'
        #Volume = calc_body_volume(path_to_body)
        
        path_to_spinal_cord = path_to_data + '/'+ folder + '/NIFTI_LIMBUS/BIN_spinalcord.nii.gz'
        #curvature = calc_max_spinal_cord_curvature(path_to_spinal_cord)

        path_to_vertabrae_segms = path_to_data + '/'+ folder + '/NIFTI_TOTALSEG/'
        height = calc_height_of_patient(path_to_vertabrae_segms)

        #f.write(str(folder) + ' ' + str(Volume) + ' ' + str(curvature) + ' ' + str(height) +  '\n')


