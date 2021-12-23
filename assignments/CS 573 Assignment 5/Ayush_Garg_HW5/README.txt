Name - Ayush Garg
PUID - 0033794803

Python version used for this is 3.9.1
Dependencies required - numpy, pandas, matplotlib, multiprocessing 

                                                Instructions to run code

For Q1 - for exploring the dataset
Command to be entered on terminal: python exploration.py
Output: Displays two graphs, one for visualising the data point from each class and other to visualize 1000 randomly selected points

For Q2.1 - for running KMeans
Command to be entered on terminal: python kmeans.py digits-embedding.csv 10
Output: WC-SSD: 1489650.532
        SC: 0.40
        NMI: 0.359

For Q2.2.1 - for analysing KMeans for different values of K
Command to be entered on terminal: python kmeans_analysis_1.py
Output: Provides the output plots for WC-SSD and SC for different values of K

For Q2.2.3 - for analysing KMeans sensitivity to starting points  (takes approx 35 minutes)
Command to be entered on terminal: python kmeans_analysis_3.py
Output: Provides 6 plots corresponding to WC-SSD and SC for each of the 3 datasets

For Q2.2.4 - for analysing NMI values and visualisation results
Command to be entered on terminal: python kmeans_analysis_4.py
Output: Dataset1 NMI: 0.346
        Dataset2 NMI: 0.455
        Dataset3 NMI: 0.491

        Plots showing the visualisation are also generated and get saved in the current folder.
        
For Q3.2 - 3.4 - for plotting dendrograms for different linkages and analysing their values of WC-SSD and SC for different K values
Command to be entered on terminal: python hierarchical.py
Output: Plot showing the dendrograms for all 3 linkages as well as plots for WC-SSD and SC values for variation in K values. They get saved in the current folder.

For Q3.6 - for computing NMI values for each of the dendrogram
Command to be entered on terminal: python hierarchical.py
Output: NMI_single: 0.37476587930890864
        NMI_complete: 0.4096988613663228
        NMI_average: 0.404185471150106
