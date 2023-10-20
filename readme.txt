Comparative Analysis between Infomap and Label Propagation for Community Detection - CRC 2023/2024, Group 25:

ComparativeAnalysis.py generates statistics regarding the execution of the algorithms and the quality of their
partition and outputs it into report.txt and stdout.
Please note that in the main function edgeListModels describes the paths of the graphs to be analysed as lists of
edges and infoMapArgumentsList and labelPropagationArgumentsList, as the name suggests, refer to the parameters
to pass to the Infomap and Label Propagation algorithms. 
Please also note that n, tau1, tau2, mu and average_degree also in main refer to the parameters for the
artificial graphs generated through the LFR_benchmark_graph function.
These variables make up the degree of customization of the project. The values used by default are the 
ones described in the report. 

How to run:

In order to install the necessary dependencies from the main directory run:

> pip3 install -r requirements.txt

Following this, to run the project from the main directory simply run the main python file as follows:

> python3 ComparativeAnalysis.py mode

Where 'mode' is 0 for the data's graphs and 1 for the LFR graphs visualization

Important Remarks:

If hardware resources are lacking creating the LFR graphs might be very time consuming.
(See line 323 in ComparativeAnalysis.py)
Please note that the LFR graphs for testing were only created in our PCs after several trials, doing so
with the specified parameters is a very resourse intensive task possibly stalling in some parameter combination