# HCMTask3
Code for HCM Group Research Project 

Hello, 

This is the code I used for the Hypertrophic Cardiomyopathy (HCM) Group research project Task 3. I have included images of the main code at the appendix of the report but I thought having a git repository for the entire code and files would be better. 

When opening up the notebook, the working directory should be the "src" file as the code might not work with other working directories. The folder with all the patient data should be beside the src file:

project/

├── src/

│   └── working directory here

├── HCMR_100_1 (patient data)/

│   └── HCMR_001_0001 

│   └── HCMR_001_0002 

However, since there was one file with an error, the remaining 99 patient frames have already been extracted and placed inside "src" in "ed_models".

The main code is "KritisCode.ipynb" which is in the format of a Python Notebook for ease of reading and running the codes. The code calls other files throughout the notebook file such as "main.py" and "biv_plots.py" for visualisation, which were premade files given to us at the start of the project, but only "KritisCode.ipynb" needs to be run, the other files just have to be present in the src directory. 

When setting up the environment in terminal, its the same commands as was given to us:

conda create --name biv-me python=3.11

conda activate biv-me

pip install -r requirements.txt

I have also included the results that I have gotten from this code to show you what results I am working with for the report (under "results").

Thank you! :)
