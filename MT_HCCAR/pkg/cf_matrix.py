import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def make_confusion_matrix(cf,
                          cot_value,
                          group_names=True,
                          categories='auto',
                          count=True,
                          percent=True,
                          COT_mean=True,
                          COT_median=True,
                          COT_range=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='viridis',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.

    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    if COT_mean:
        cot_mean_text = ['' for i in [range(cf.shape[0]+1)]]
        for i in range(cf.shape[0]-1):
            cot_mean_text.append ("M: {0:.2}\n".format(np.mean(cot_value[i])))
        cot_mean_text.append('')
        for i in range(cf.shape[0]-1):
            cot_mean_text.append ("M: {0:.2}\n".format(np.mean(cot_value[i+2])))
    else:
        cot_mean_text = blanks

    if COT_median:
        cot_median_text = ['' for i in range(cf.shape[0]+1)]
        for i in range(cf.shape[0]-1):
            cot_median_text.append ("Mdn: {0:.2}\n".format(np.median(cot_value[i])))
        cot_median_text.append('')
        for i in range(cf.shape[0]-1):
            cot_median_text.append ("M: {0:.2}\n".format(np.median(cot_value[i+2])))
    else:
        cot_median_text = blanks

    if COT_range:
        cot_range_text = ['' for i in range(cf.shape[0]+1)]
        for i in range(cf.shape[0]-1):
            cot_range_text.append("{0:.2}\n".format(np.max(cot_value[i]) - np.min(cot_value[i])))
        cot_range_text.append('')
        for i in range(cf.shape[0]-1):
            cot_range_text.append ("M: {0:.2}\n".format(np.max(cot_value[i]) - np.min(cot_value[i])))
    else:
        cot_range_text = blanks


    box_labels = [f"{v2}{v3}{v4}{v5}{v6}".strip() for v2, v3, v4, v5, v6 in zip(group_counts,group_percentages, cot_mean_text, cot_median_text, cot_range_text)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)