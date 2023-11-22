import numpy as np 
import nibabel as nib 
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import os
from natsort import natsorted

def calc_body_volume(path_to_body, path_to_vertabrae_segms):
    
    img_obj, _, img_header = get_image_objects(path_to_body)
    voxel_volume = img_header['pixdim'][1] * img_header['pixdim'][2] * img_header['pixdim'][3]
    del img_header

    path_to_C1 = path_to_vertabrae_segms + '/vertebrae_C1.nii.gz'
    path_to_T2 = path_to_vertabrae_segms + '/vertebrae_T2.nii.gz'

    C1_obj,_,header = get_image_objects(path_to_C1)
    T2_obj,_,_ = get_image_objects(path_to_T2)

    nonzero_C1 = max(np.nonzero(C1_obj)[2])
    nonzero_T2 = min(np.nonzero(T2_obj)[2])

    total_voxels = np.sum(np.sum(np.sum(img_obj[:,:,nonzero_T2:nonzero_C1],0),0),0)

    total_volume = total_voxels * voxel_volume

    return total_volume


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


def fit_polynomial(path_to_spinal_cord):
      
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
    
    p = np.polyfit(middleCordCoords['Z'][0], middleCordCoords['Y'][0], deg=3)
    #minima = -p[1]/2*p[0]
    #print(p)
    '''
    plt.scatter(middleCordCoords['Z'],middleCordCoords['Y'])
    x = np.arange(0,1.1,0.1)
    x2 = x**2
    x3 = x**3
    plt.plot(x, p[0]*x3+p[1]*x2+p[2]*x+p[3], color = 'red')
    #plt.ylim([0,1])
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    '''

    return p

def calc_minima(polynomial_coefficients):

    a = 3*polynomial_coefficients[0]
    b = 2*polynomial_coefficients[1]
    c = polynomial_coefficients[2]

    #calculate discriminant
    discriminant = np.sqrt(np.square(b)- (4*a*c))

    #calc solutions
    minima1 = ((-b)+discriminant)/(2*a)
    minima2 = ((-b)-discriminant)/(2*a)  
    minimas = [minima1, minima2]

    # calc second derivative
    second_derivatives = []
    for minima in minimas:
        sec_der = (6*a*minima) + (2*b)
        second_derivatives.append(sec_der)

    # find minima
    for index, second_vericative in enumerate(second_derivatives):
        if second_vericative>0:
            return minimas[index]
        else:
            pass

def calc_height_of_patient(path_to_vertabrae_segms, parameters):
    
    path_to_C1 = path_to_vertabrae_segms + '/vertebrae_C1.nii.gz'
    path_to_T2 = path_to_vertabrae_segms + '/vertebrae_T2.nii.gz'

    C1_obj,_,header = get_image_objects(path_to_C1)
    T2_obj,_,_ = get_image_objects(path_to_T2)
    slice_thickness = header['pixdim'][3]

    nonzero_C1 = np.nonzero(C1_obj)
    nonzero_T2 = np.nonzero(T2_obj)

    arclength = calc_cubic_arc_length(parameters)

    height = (max(nonzero_C1[2]) - min(nonzero_T2[2])) * slice_thickness * arclength
    
    return height


def calc_cubic_arc_length(parameters):
    '''
    Numerical integration of arc length of cubic polynomial.
    Integration between 0 and 1. 
    '''
    a = 0
    b = 1
    n = 100
    h = (b - a) / (n - 1)
    x = np.linspace(a, b, n)

    coef1 =  parameters[0]
    coef2 = parameters[1]
    coef3 = parameters[2]

    f = np.sqrt(1+ (3*coef1*x**2 + 2*coef2*x + coef3)**2)

    I_simp = (h/3) * (f[0] + 2*sum(f[:n-2:2]) + 4*sum(f[1:n-1:2]) + f[n-1])

    return I_simp


if __name__ == '__main__':

    path_to_data = 'D:/HNSCC/ARCHIVE/2023_10_08'
    folders = os.listdir(path_to_data)
    folders = natsorted(folders)


    text_file = 'C:/Users/poppy/Documents/HN_Atlas/HNSCC/HNSCC_Characteristics.txt'
    f = open(text_file, "a")
    f.write('Patient,Volume,Height,a,b,c,minima\n')
        
    for folder in folders:

        path_to_vertabrae_segms = path_to_data + '/'+ folder + '/NIFTI_TOTALSEG/'
        path_to_body = path_to_data + '/'+ folder + '/NIFTI_LIMBUS/BIN_body.nii.gz'
        Volume = calc_body_volume(path_to_body, path_to_vertabrae_segms)
        
        path_to_spinal_cord = path_to_data + '/'+ folder + '/NIFTI_LIMBUS/BIN_spinalcord.nii.gz'
        quadratic_fitting_params = fit_polynomial(path_to_spinal_cord)
        minima_loc = calc_minima(quadratic_fitting_params)

        height = calc_height_of_patient(path_to_vertabrae_segms, quadratic_fitting_params)

        #dont write the intercept will always be 1 roughly
        f.write(str(folder) + ',' + str(Volume) + ',' + str(height)+',' + str(quadratic_fitting_params[0]) + ',' + str(quadratic_fitting_params[1]) + ',' + str(quadratic_fitting_params[2]) + ',' + str(minima_loc) + '\n')

