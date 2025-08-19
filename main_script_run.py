# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:45:47 2024

@author:       Klest Dedja
@institution:  KU Leuven
"""
import os, sys
import gc
import numpy as np
import pandas as pd
import shap
import pickle
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont # stitch images together, write text etc.
from IPython.display import display

# sklearn utils
from sklearn.model_selection import train_test_split, KFold 

# sksurv utils
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored as c_index

# from utilities import SDT_to_dict_interval as SDT_to_dict, tree_list_to_dict_model
from utilities import SurvivalModelConverter, predict_hazard_function
from utilities import auto_rename_fields, adjust_tick_label_size
from utilities import format_timedelta, format_SHAP_values


if __name__ == '__main__':
    
    DPI_RES = 180

    root_folder = os.getcwd()
    
    X = pd.read_csv(os.path.join(root_folder, 'FLChain-single-event-imputed', 'data.csv'))#[:700]
    # X['flc_ratio'] = X['kappa']/X['lambda']
    X.rename(columns={"sample_yr": "sample_year"})
    y = pd.read_csv(os.path.join(root_folder, 'FLChain-single-event-imputed', 'targets.csv')).to_records(index=False)#[:700]
    y = auto_rename_fields(y)
    
    
    if np.max(y['time']) > 100:
        y_time_years = y['time']#.astype(np.float64)
        y_time_years = y_time_years / 365        
        y['time'] = y_time_years
    
    general_figs_folder = os.path.join(root_folder, 'figures')
    interval_figs_folder = os.path.join(root_folder, 'figures', 'interval-plots')
    
    
    if len(X) <= 700:
        general_figs_folder = os.path.join(root_folder, 'figures', 'drafts')
        interval_figs_folder = os.path.join(root_folder, 'figures', 'interval-plots', 'drafts')
    
    clf = RandomSurvivalForest(n_estimators=100, min_samples_split=10,
                                n_jobs=5, random_state=0)
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    test_perf = c_index(y_test['event'], y_test['time'], y_pred)[0]
    print("Test performance: {:.4f}".format(test_perf))

    unique_times = clf.unique_times_

    y_train_surv = clf.predict_survival_function(X_train, return_array=True)
    y_train_surv = pd.DataFrame(y_train_surv, columns=unique_times)
    y_pred_surv = clf.predict_survival_function(X_test, return_array=True)
    y_pred_surv = pd.DataFrame(y_pred_surv, columns=unique_times)

    idx_plot = 1 #idx = 36 for size = 700. idx =1 for full data 
    FONTSIZE = 14
    
    
    y_survs = clf.predict_survival_function(X_test, return_array=True)
    y_surv = clf.predict_survival_function(X_test)[idx_plot].y
    y_hazard = clf.predict_cumulative_hazard_function(X_test)[idx_plot].y
    
    
    dy_hazard = predict_hazard_function(clf, X_test, event_times='auto')[idx_plot]
    
    
    from utilities import rolling_kernel
    
    y_surv_smooth = rolling_kernel(y_surv, kernel_size=20)
    y_hazard_smooth = rolling_kernel(y_hazard, kernel_size=20)
    dy_hazard_smooth = rolling_kernel(dy_hazard, kernel_size=50)

    
    plt.figure()
    plt.title("Survival function $S(t)$", fontsize=FONTSIZE+2)
    # plt.plot(unique_times, y_surv)
    plt.plot(unique_times*365, y_surv_smooth, lw=2)
    plt.xlabel("time $t$", fontsize=FONTSIZE)
    plt.xlim(0, 5020)    
    plt.ylabel("$S(t)$", fontsize=FONTSIZE)
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(os.path.dirname(root_folder), 'Thesis-material', 'survival-curve-ex.pdf'))
    plt.show()
    
    
    plt.figure()
    plt.title("Cum. Hazard function $\Lambda(t)$", fontsize=FONTSIZE+2)
    # plt.plot(unique_times, y_hazard)
    plt.plot(unique_times*365, y_hazard_smooth, lw=2)
    plt.xlabel("time $t$", fontsize=FONTSIZE)
    plt.xlim(0, 5020)
    plt.ylabel("$\Lambda(t)$", fontsize=FONTSIZE)
    plt.savefig(os.path.join(os.path.dirname(root_folder), 'Thesis-material', 'cum-hazard-curve-ex.pdf'))
    plt.show()
    

    plt.figure()
    plt.title("Hazard function $\lambda(t)$", fontsize=FONTSIZE+2)
    # plt.plot(unique_times, dy_hazard)
    plt.plot(unique_times*365, 100*dy_hazard_smooth, lw=2)
    plt.xlabel("time $t$", fontsize=FONTSIZE)
    plt.xlim(0, 5020)    
    plt.ylabel("$100 \ \lambda(t)$", fontsize=FONTSIZE)
    plt.savefig(os.path.join(os.path.dirname(root_folder), 'Thesis-material', 'hazard-curve-ex.pdf'))
    plt.show()
    
    
    #%%

    
    unique_times = clf.unique_times_
    interval_shap_values = {}

    convert_all = SurvivalModelConverter(clf_obj=clf, T_start=0, T_end=max(unique_times)+1)
    
    clf_dict = []
    tree_dicts = [convert_all.surv_tree_to_dict(idx=i, output_format='probability') 
                  for i in range(len(clf.estimators_))]
    clf_dict = convert_all.tree_list_to_dict_model(tree_list=tree_dicts,
                                               learning_weight=1/len(tree_dicts))

    explainer = shap.TreeExplainer(model=clf_dict, data=None,
                                   model_output="raw",
                                   feature_perturbation='tree_path_dependent')
    
    shap_values = explainer(X_test, check_additivity=True)    
    shap_values = format_SHAP_values(shap_values, clf, X_test)
    
    global_plt_name = 'Global_SHAP.pdf'
    
    plt.figure()
    plt.title("Global explanation", size=16)
    shap.summary_plot(shap_values, max_display=10, 
                      alpha=0.7, show=False)
    plt.xlabel('SHAP value: impact on output', size=14)
    plt.savefig(os.path.join(general_figs_folder, global_plt_name),
                bbox_inches='tight', dpi=DPI_RES)
    plt.show()
    
    '''LOCAL EXPLANATIONS HERE: split explanations in time intervals
    (loop over such intervals, binarise outputs, store resulting SHAP values)
    '''
    # split timeline in intervals and explain each segment
    # time_intervals = [0, 1720, 3440, 5160] #in days
    time_intervals = [0, 3.5, 7, 10.5, 14] #in years (this gives nice plots, but very long)
    time_intervals = [0, 5, 10, 14] #in years
        
    for i, t_i in enumerate(range(len(time_intervals)-1)):
        
        T_start = time_intervals[t_i]
        T_end = time_intervals[t_i+1]
        
        convert_interv = SurvivalModelConverter(clf_obj=clf, T_start=T_start, T_end=T_end)
        
        clf_interv = []
        tree_intervs = [convert_interv.surv_tree_to_dict(idx=i, output_format='auto') 
                      for i in range(len(clf.estimators_))]
        clf_interv = convert_interv.tree_list_to_dict_model(tree_list=tree_intervs,
                                                     learning_weight=1/len(tree_intervs))
        

        explainer = shap.TreeExplainer(model=clf_interv, data=None,
                                       model_output="raw",
                                       feature_perturbation='tree_path_dependent')
        
        shap_values_int = explainer(X_test, check_additivity=True)
        shap_values_int = format_SHAP_values(shap_values_int, clf, X_test)            
        interval_shap_values[f'{str(T_start)}-{str(T_end)}'] = shap_values_int
        
        
    ## dump trained model dictionary, predictions, and computed SHAP values:
    data_to_save = {
        "clf_dict": clf_dict,
        "unique_times": unique_times,
        "interval_shap_values": interval_shap_values,
        "y_train_surv": y_train_surv,
        "y_pred_surv": y_pred_surv,
        "dpi": DPI_RES
    }
    
    filename = f'saved_data_{len(X)}.pkl'
    
    with open(filename, 'wb') as file:
        pickle.dump(data_to_save, file)  
            
        
        #%% now iterate over single samples to be explained (for each interval)
        
    load_size = len(X)
        
    with open(f'saved_data_{load_size}.pkl', 'rb') as f:
        data = pickle.load(f)
        clf_dict = data['clf_dict']
        unique_times = data['unique_times']
        interval_shap_values = data['interval_shap_values']
        y_train_surv = data['y_train_surv']
        y_pred_surv = data['y_pred_surv']
        DPI_RES = data['dpi']
    
    N = 12 if load_size > 700 else 3
    
    for i in range(N):
        
        t0_local_explains = datetime.now()
        
        y_pred_pop = y_train_surv.mean(axis=0) #sample from training data
        y_pred_pop_med = np.percentile(y_train_surv.values, q=50, axis=0)
        y_pred_pop_low = np.percentile(y_train_surv.values, q=25, axis=0)
        y_pred_pop_high= np.percentile(y_train_surv.values, q=75, axis=0)
        
        y_pred_i = y_pred_surv.iloc[i,:] # sample from TEST (unseen) data
        
        ''' survival plot here, dump as .mpl file'''
        
        plt.figure() # figsize will is overwritten after reloading, 
        plt.suptitle('Predicted survival curve', size=round(8*(DPI_RES/72)), y=0.98)
        plt.step(y_pred_i.index, y_pred_i.values, where="post", label='$S(t)$',
                 lw=2.4, color='purple')
        plt.step(y_pred_pop.index, y_pred_pop.values, where="post", label='population',
                 lw=1.6, color= 'forestgreen')

        plt.fill_between(unique_times, y_pred_pop_low, y_pred_pop_high, alpha=0.3,
                         label='P25-P75', color='forestgreen')

        for t in time_intervals:
            plt.axvline(x=t, color='k', linestyle='--', linewidth=1, alpha=0.7)
        plt.xlabel('time', fontsize=round(8*(DPI_RES/72)))
        plt.xlim([0, None])
        plt.xticks(fontsize=round(8*(DPI_RES/72)))
        plt.ylim([0.4, 1.0])
        plt.ylabel('Survival over time', fontsize=round(8*(DPI_RES/72)))
        plt.yticks(np.arange(0.2, 1.15, 0.2), None, fontsize=round(8*(DPI_RES/72)))
        plt.yticks(np.arange(0.1, 1, 0.2), None, minor=True)
        plt.legend(fontsize=round(7*(DPI_RES/72))) #loc='auto' or 'upper right'
        with open(f'temp_plot_surv_{i}.mpl', 'wb') as file:
            pickle.dump(plt.gcf(), file)
        if i == 7: # extra plot for Thesis manuscript
            plt.legend(fontsize=round(6*(DPI_RES/72))) #loc='auto' or 'upper right'
            plt.savefig(os.path.join(general_figs_folder, 'survival-curves', f'survival_curve_idx_extra{i}.pdf'),
                        bbox_inches='tight', dpi=DPI_RES)
        plt.legend(fontsize=round(7*(DPI_RES/72))) #loc='auto' or 'upper right'
        plt.savefig(os.path.join(general_figs_folder, 'survival-curves', f'survival_curve_idx{i}.png'),
                    bbox_inches='tight', dpi=DPI_RES)
        # plt.show()
        plt.close()
        
        
        ''' local SHAP plot here: computed over entire interval '''
        local_plt_name = f'Local_SHAP_idx{i}.pdf'
        
        fig = shap.plots.waterfall(shap_values[i], max_display=10, 
                                   base_fontsize=16, show=False)
        fig.set_size_inches((5, 5))
        plt.title("Output explanation", fontsize=round(8*(DPI_RES/72)))
        plt.savefig(os.path.join(general_figs_folder, 'local-SHAP', local_plt_name), bbox_inches='tight')
        plt.savefig(f'temp_plot_{i}_full.png', bbox_inches='tight', dpi=DPI_RES)
        # plt.show()
        plt.close()  # Close the figure to free up memory
        
        ''' given local instance, iterate through time intervals 
        (stored in previous dictionary) '''
        
        for key, value in interval_shap_values.items():
            
            T_start, T_end = [float(s) for s in key.split('-')]
            index_T_end = np.argmax(unique_times > T_end) - 1

            local_interv_plt_name = f'Local_SHAP_idx{i}_T{key}.pdf'
            combo_local_plt_name_pdf = f'Time-SHAP_idx{i}_combined.pdf'
            combo_local_plt_name_png = f'Time-SHAP_idx{i}_combined.png'

            
            shap_values_use = interval_shap_values[key][i]
            
            ## conditional SHAP can contain NaN values:
            if np.isnan(shap_values_use.values).sum() > 0:
                shap_values_use.values[np.isnan(shap_values_use.values)] = 0
                warnings.warn('NaN values were found when computing interval-specific SHAP values,\
                possibly, the event is estimated to happen before the queried time interval [{T_start}-{T_end}]')
    

            ### TODOs:
                # - rethink the probability outputs: rescale them? They are not very intuititve atm
                # - change notation e.g. E(f(X)) and similar
    
            fig = shap.plots.waterfall(shap_values_use, max_display=10, 
                                       base_fontsize=16, show=False)
            single_plotwidth = max(3.4, 7 - len(interval_shap_values))
            fig.set_size_inches((single_plotwidth, 7))
            plt.title(f"Output explanation, interval [{key}]   ", fontsize=round(8*(DPI_RES/72)))

            plt.savefig(os.path.join(interval_figs_folder, local_interv_plt_name),
                        bbox_inches='tight', dpi=DPI_RES)
            plt.savefig(f'temp_plot_{i}_{key}.png',
                        bbox_inches='tight', dpi=DPI_RES)
            # plt.show()
            plt.close(fig)  # Close the figure to free up memory
            
            print(f'TIME INTERVAl: {key}')
            print('Local prediction auto: {:.4f}'.format(shap_values_use.base_values+\
                                                         shap_values_use.values.sum()))
                
            print('Population pred autom.: {:.4f}'.format(shap_values_use.base_values))
            print('Population pred manual: {:.4f}'.format(1-y_pred_pop.iloc[index_T_end]))
            
            
            # interval loop closed, now load images and paste them one next ot each other

        '''Here manipulate and paste the images one next to each other '''
        
        with open(f'temp_plot_surv_{i}.mpl', 'rb') as file:
            fig = pickle.load(file)
        
        # Update the figure size
        # fit survival plot and local SHAP plot over the entire interval
        width_pad_prop = 0.05
        tot_width = single_plotwidth*(len(interval_shap_values)-1) 
        fig.set_size_inches((1-2*width_pad_prop)*tot_width, 5.3, forward=True)
        plt.tight_layout()
        plt.savefig(f'temp_plot_surv_{i}.png', bbox_inches='tight', dpi=DPI_RES)
        # plt.show()
        plt.close()
        
        # Load the saved images and find the total width and height for the combined image
        surv_image = Image.open(f'temp_plot_surv_{i}.png')
        local_image = Image.open(f'temp_plot_{i}_full.png')
        images = [Image.open(f'temp_plot_{i}_{key}.png') for key, val in interval_shap_values.items()]
        # images = [img.rotate(270, expand=True) for img in images]
        
        # collect widths and heights, magnitude is pixel-wise, not in inches as in mpl
        widths, heights = zip(*(i.size for i in images))
        
        y_pad = 10 # needed not to cut off the survival curve plot title
        y_pad_intrarow = 100 # padding between top row and bottom row
        x_pad_intrarow = -70 if i == 7 else -25 # for the Thesis picture, it still fits
        
        combo_height = max(heights) + surv_image.size[1] + y_pad + y_pad_intrarow
        combo_width = sum(widths) + x_pad_intrarow*(len(widths)-1) # N-1 gaps 
        
        # Create a new image with the appropriate size to contain all the plots
        combo_image = Image.new('RGB', (combo_width, combo_height), color=(255, 255, 255))
        
        # Paste the matplotlib survival curve on top, add some padding on the x_axis
        # bear in mind, the unit of measure is in pixels so the 2 vars must be integers
        pos_left =  (combo_width-surv_image.size[0]-local_image.size[0])//2 - 70*len(widths) + 70
        pos_right = (combo_width+surv_image.size[0]-local_image.size[0])//2 + 70*len(widths) - 70
        
        combo_image.paste(surv_image, (pos_left, y_pad))
        # Paste the overall local SHAP next
        combo_image.paste(local_image, (pos_right, y_pad))

        # Paste each of the other images below the matplotlib image
        x_offset = 0
        y_offset = surv_image.size[1] + y_pad + y_pad_intrarow  # Pasting below the survival curve image
        for img in images:
            combo_image.paste(img, (x_offset, y_offset))
            x_offset += img.size[0]  # Update the x_offset by the width of the current image
            x_offset += x_pad_intrarow # And add the x padding

        # combo_image.save(os.path.join(general_figs_folder, combo_local_plt_name))
        # combo_image.show()
        local_image.close()
        surv_image.close()
        
        
        '''add title image on top of the current collage'''
        
        title_text = "Time-SHAP explanation" # for sample instance i={i}
        font_size = round(28*(DPI_RES/72))  # Adjust title size. Scale is relative to dpi=72
        font = ImageFont.truetype("arial.ttf", font_size) #insert correct font path here
        # font = ImageFont.truetype("DejaVuSans.ttf", font_size) #insert correct font path here

        # Determine the size required for the title text
        draw = ImageDraw.Draw(Image.new('RGB', (10, 10)))  # Temp image for calculating text size
        text_width, text_height = draw.textsize(title_text, font=font)
        title_image = Image.new('RGB', (combo_width, text_height + 10), color=(255, 255, 255))  # Added padding
        # Initialize drawing context
        draw = ImageDraw.Draw(title_image)
        # Calculate text position (centered)
        text_x = (title_image.width - text_width) / 2
        text_y = (title_image.height - text_height) / 2
        # Draw the text
        draw.text((text_x, text_y), title_text, fill="black", font=font)
        
        # Create a new image with a height that includes both the title and the combo images
        final_image = Image.new('RGB', (combo_width, title_image.height + combo_height), color=(255, 255, 255))        
        # Paste the combo image below the title first
        final_image.paste(combo_image, (0, title_image.height-10))
        # Now paste the title above it
        final_image.paste(title_image, (0, 0))
        
        final_image.save(os.path.join(general_figs_folder, 'combo-plots', combo_local_plt_name_pdf),
                         dpi=(DPI_RES, DPI_RES)) # overwrite combo_image
        final_image.save(os.path.join(general_figs_folder, 'combo-plots', combo_local_plt_name_png),
                         dpi=(DPI_RES, DPI_RES)) # overwrite combo_image

        display(final_image)
                
        # display(combo_image)

        # Clean up the temporary images. Explicit garbage collection is necessary
        gc.collect()
        os.remove(f'temp_plot_surv_{i}.png')
        os.remove(f'temp_plot_surv_{i}.mpl')
        os.remove(f'temp_plot_{i}_full.png')

        for key, value in interval_shap_values.items():        
            os.remove(f'temp_plot_{i}_{key}.png')
            
        t1_local_explains = datetime.now()
        time_local_explains = format_timedelta(t1_local_explains -t0_local_explains,'mm:ss:ms')
        print("Plotted time-SHAP expalantions in: {}".format(time_local_explains))
