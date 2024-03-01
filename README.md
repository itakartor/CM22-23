# CM22-23
This repository contains a Python didactic purpose implementation of two methods: The **Conjugate Gradient** and **MINRES**, used to assess the MIN-COST flow problem described in project #21 of "Computational Mathematics for learning and data analysis".
The Linear Min-Cost Flow Problems used for the study of the two methods are generated by generators available at [link](https://commalab.di.unipi.it/datasets/mcf/)

## Software
**Language:**

Python >=3.10

**External Modules:**

- Numpy >= 1.24.2
- matplotlib >= 3.6.3

## Folders
 
* deprecated -> contains old files written during the implementation of the project
* generators -> contains the folders of generators and the files used for creating the graphs
* outputMatrix -> contains logs file generated by the test executions
* scriptGenGraph -> This contains files used for creating a script in Python to build+run a C program. NOTE: This assumes you have gcc installed. Tested only in unix systems
* testAnalyze -> Folder contains the incidence matrix used by the test
* IncidenceMatrix.py -> Contains the classes and methods definitions for creating the Object representing an Incidence Matrix. Using  the .dmx file created by the generetors
* Minres.py -> contains the implementation of the MINRES method defined in the function: **def custom_minres(A:np.ndarray, b:np.ndarray , m_dimension:int, x0:np.ndarray = None, tol:float = 1e-5, maxiter:int = None):** and other definitions of support functions (some used only in a test phase)
* configs.py -> contains some of the constants variable definitions
* conjugateGradient.py -> contains the implementation of the Conjugate Method using a Class named ConjugateGradient built passing an Incidence Matrix and a D matrix during the initialization. The method is applied using the function: **start_cg(self, inNumIteration:int=0,inTol:float = 1e-5):**
* test.py -> Run the test using the three types of graph chosen in the REPORT ('Complete','Grid','RMF') contained in the folder testanalyze and show in runtime the graph generated with the data collected by the application of Conjugate Gradient and MINRES methods.
* util.py -> contains support functions used to assess the results and run the tests.

## Run the project
Just execute the test.py file that is already configured to run the project, starting by the three graphs contained in testAnalyze folder and a random D matrix generated runtime. The execution will show in runtime the graphs generated by the test. 

