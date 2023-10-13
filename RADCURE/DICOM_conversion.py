from utils import * 



#### --- USER INPUTS --- ####
'''
Base_path: string. folder in which the HN folders are stored.
hn_contournames_path: string. path where the excel file is stored which contains the contours you want to convert and their associated contour options. 
'''

Base_path = 'X:/manifest-1685648397515/RADCURE'
hn_contournames_path = 'Contour_Naming.xlsx'

# get the contour names and the associations in a usable format 
contour_names, associations = get_contour_names_and_associations(hn_contournames_path)

#do the conversion 
DICOM_CONVERT(Base_path, contour_names, associations)

# check the dicom conversions have worked 
check_DICOM_conversions(Base_path)