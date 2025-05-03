# HCMTask3
Code for HCM Group Research Project 
(Please look at this file as a code file. If not, formatting gets messed up)

Hello, 

This is the code I used for the Hypertrophic Cardiomyopathy (HCM) Group research project Task 3. I have included images of the main code at the appendix of the report but I thought having a git repository for the entire code and files would be better. 

When opening up the notebook, the working directory should be the "src" file as the code might not work with other working directories. The folder with all the patient data should be beside the src file:

project/
├── src/
│   └── working directory here
├── HCMR_100_1 (patient data)/
│   └── HCMR_001_0001 (patient #1)
│       └── HCMR_001_0001_Model_Frame_000.txt
│   └── HCMR_001_0002 (patient #2)

However, since there was one file with an error, the remaining 99 patient frames have already been extracted and placed inside "src" in "ed_models".

The main code is "KritisCode.ipynb" which is in the format of a Python Notebook for ease of reading and running the codes. The code calls other files throughout the notebook file such as "main.py" and "biv_plots.py" for visualisation, which were premade files given to us at the start of the project, but only "KritisCode.ipynb" needs to be run, the other files just have to be present in the src directory. 

Thank you! :)
