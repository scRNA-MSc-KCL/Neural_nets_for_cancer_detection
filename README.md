Prior to running of the models in contained in this folder the appropriate datasets must be downloaded from their source files. 

These can found in the following locations

Dataset1;
https://zenodo.org/record/3357167#.YTIZLrBKjIU
files to download;
 Combined_10x_CelSeq2_5cl_data.csv88.5 MB
 Labels.csv37.6 kB

Dataset2;
files can be obtained from Synapse
https://www.synapse.org/#!Synapse:syn21041850
or by cloning a giy hub repository
git clone http://github.com/krasnowlab/hlca

Dataset3;
Data can be downloaded from GEO https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131508

Interdataset Performance
https://zenodo.org/record/3357167#.YTIZLrBKjIU
10x_5cl_data.csv91.5 MB
Labels.csv
CelSeq2_5cl_data.csv14.9 MB
Labels.csv


Preprocessing.py must first be run to pre process the datasets. File location must be changed to the approriate folder on the users local device.

Files can then be run in any order. 

Output of the files is either printed to the terminal or automatically downloaded to the local machine

Files included in this folder not created by the author are the rbflayer.py, initializer and k_means initializer.py

In order to run the CNN files the DeepInsight Github repository must be cloned using the following command

pip -q install git+git://github.com/alok-ai-lab/DeepInsight.git#egg=DeepInsight
