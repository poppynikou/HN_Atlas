Groupwise Class
==================================

Groupwise Class which inherits from Data Class. Contains methods to postprocess the images after alignments.
Contains methods to align patients, and generate bash scripts to run on cluster for deformable registrations.

The groupwise registration algorithm is an itterative process. 

The code assumes that the data is stored in a folder 'BATCH_X'. 
All initial images and masks must be stored in a folder 'Iteration_0'.
The code will then generate the remaining folders. 

set_initial_ref_patient
-------------------------------------
.. automodule:: Classes.Groupwise.set_initial_ref_patient
    :members:
    :undoc-members:
    :inherited-members: 
    
.. _set_initial_ref_patient:


set_itteration_no
-------------------------------------
.. automodule:: Classes.Groupwise.set_itteration_no
    :members:
    :undoc-members:
    :inherited-members: 
    
.. _set_itteration_no:


AffineAlignment
-------------------------------------
.. automodule:: Classes.Groupwise.AffineAlignment
    :members:
    :undoc-members:
    :inherited-members: 
    
.. _AffineAlignment:

Generate_Bash_Scripts
-------------------------------------
.. automodule:: Classes.Groupwise.Generate_Bash_Scripts
    :members:
    :undoc-members:
    :inherited-members: 
    
.. _Generate_Bash_Scripts:


calc_average_transformation
-------------------------------------
.. automodule:: Classes.Groupwise.calc_average_transformation
    :members:
    :undoc-members:
    :inherited-members: 
    
.. _calc_average_transformation:

calc_average_transformation_affine
-------------------------------------
.. automodule:: Classes.Groupwise.calc_average_transformation_affine
    :members:
    :undoc-members:
    :inherited-members: 
    
.. _calc_average_transformation_affine:



calc_average_transformation_def
-------------------------------------
.. automodule:: Classes.Groupwise.calc_average_transformation_def
    :members:
    :undoc-members:
    :inherited-members: 
    
.. _calc_average_transformation_def:


calc_inv_average_transformation
-------------------------------------
.. automodule:: Classes.Groupwise.calc_inv_average_transformation
    :members:
    :undoc-members:
    :inherited-members: 
    
.. _calc_inv_average_transformation:


calc_composition
-------------------------------------
.. automodule:: Classes.Groupwise.calc_composition
    :members:
    :undoc-members:
    :inherited-members: 
    
.. _calc_composition:

resample_imgs
-------------------------------------
.. automodule:: Classes.Groupwise.resample_imgs
    :members:
    :undoc-members:
    :inherited-members: 
    
.. _resample_imgs:


calc_average_image
-------------------------------------
.. automodule:: Classes.Groupwise.calc_average_image
    :members:
    :undoc-members:
    :inherited-members: 
    
.. _calc_average_image:


create_multichannel_imgs
-------------------------------------
.. automodule:: Classes.Groupwise.create_multichannel_imgs
    :members:
    :undoc-members:
    :inherited-members: 
    
.. _create_multichannel_imgs:

