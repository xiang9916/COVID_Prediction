
import warnings

from codes import DataDownloading, DataExtraction, DataVisualization

warnings.filterwarnings('ignore')

display = "\
    [0] all\n\
    [1] Download COVID-19 data from github\n\
    [2] Extract sequences from downloaded data\n\
    [3] Visualize Cases Data\n\
    choose an option:\
"

option = input(display)

if option == '0' or option == '1':
    DataDownloading.download_raw()
if option == '0' or option == '2':
    DataExtraction.cumulative()
    DataExtraction.new()
if option == '0' or option == '3':
    print('Clustering ...')
    DataVisualization.clustering()
    print('Exporting New Cases Plot ...')
    DataVisualization.view_new_cases()
    print('Exporting Cumulative Cases Plot ...')
    DataVisualization.view_cumulative_cases()
