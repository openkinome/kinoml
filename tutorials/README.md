How to use the tutorials folder
==============================
This tutorial folder contains two subfolders:



* **getting_started**: this folder contains four jupyter notebook tutorials that give the user a general overview of KinoML potential usage and capabilities.

    * **getting_started_with_kinoml**: this notebook aims to give a brief overview of KinoML capabilities. This notebook is divided into three parts that show how to use KinoML to: (1) filter and obtain the desired data from an external data source, (2) featurize this data to make it ML readable and (3) train and evaluate a ML model on the featurized data obtain from the previous steps. 

    * **kinoml_object_model**: this notebook aims to guide the user through the KinoML object model, showing how to access each object.

    * **OpenEye_structural_featurizer_showcase**: this notebook displays all the OpenEye-based structural modeling featurizers implemented in KinoML and how to use each of them.

    * **Schrodinger_structural_featurizer_showcase**: this notebook introduces the structural modeling featurizers implemented in KinoML that use the molecular modeling capabilities from the Schrodinger Suite to prepare protein structures and to dock small molecules into their binding sites.



* **experiments**:  this folder contains four separate structure-based experiments to predict ligand binding affinity to the EGFR kinase. The aim of these notebook are to showcase how to use KinoML to conduct experiments end-to-end, from obtaining the data from the database to training and evaluating a ML model to predict ligand binding affinity. Note that if the user wants to run this notebooks with their own data, they can do so by adjusting the neccesary parameters within the notebooks. All experiments are divided into two parts:

    1. **Featurize the data set**: obtaining the data set and featurize it with the featurization pipeline of choice.

    2. **Run the experiment**: the ML model of choice, implemented in the `kinoml.ml` class is trained and evaluated.


Please note that the order in which the different notebooks are displayed here is the recommended order for running them, providing a more comprehensive understanding of KinoML.

⚠️ You will need a valid OpenEye License for the tutorials to work. For the Schrodinger featurizers tutorial (`Schrodinger_structural_featurizer_showcase.ipynb`) you will also need a Schrodinger License!

