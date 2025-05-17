import scipy.io as scio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from PIL import Image as im
import utils

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase


class CalciumSignal():
    """
    Class for handling calcium signal data
    """

    def __init__(self, path, std=1.5, indices:list=None): 
        """
        Initialize an instance of the CalciumSignal class.

        Parameters:
        path (str): The path to the data file.
        std (float): The standard deviation threshold for peak detection.
        indices (list, optional): A list of specific indices to select.
        """
        data_all = scio.loadmat(path) 
        data = data_all['analysis']
        self.mask = data['L'][0,0]  
        self.name = path.split('/')[-1] 
        self.image = data['image'][0,0]  
        self.fr = pd.DataFrame(data['F_cell'][0,0])  
        self.locs, self.centers = self._get_locs(self.mask)  
        if indices is not None:
            self.fr = self.fr.loc[indices]  
            self.locs = [self.locs[i] for i in indices]  
            self.centers = [self.centers[i] for i in indices]  
        # 
        self.regu_fr = utils.regu(self.fr.copy())  
        self.peak = utils.peak_std(self.regu_fr.copy(),p=std)  
        self.with_peak = self.peak.loc[~(self.peak==0).all(axis=1)].index  

    def _get_locs(self, mask):
        """
        Private method to get locations and centers from the mask.

        Parameters:
        mask (np.ndarray): The mask array indicating regions of interest.

        Returns:
        tuple: A tuple containing lists of locations and centers.
        """
        locs = []
        centers = [] 
        for i in range(0,mask.max()): 
            loc = np.where(mask==i+1)
            locs.append(loc)
            center = [np.mean(loc[1]),np.mean(loc[0])]
            centers.append(center)
        return locs, centers 
         
    def plot_mask(self, link = True, peak_only = False, mask = True, center = False, save=None, show=False): 
        """
        Plot the mask with optional linking and centering.

        Parameters:
        link (bool): Whether to plot links between regions.
        peak_only (bool): Whether to plot only regions with peaks.
        mask (bool): Whether to plot the mask.
        center (bool): Whether to plot the centers of regions.
        save (str): The directory path to save the plot.
        show (bool): Whether to display the plot.
        """
        pic = im.fromarray(self.image[:,:,1])
        plt.figure(figsize=(10,10))
        plt.imshow(pic, cmap='gray')
        if mask: 
            self.mask_map(peak_only=peak_only)
        if link: 
            self.link_map(peak_only=peak_only)
        if center:
            self.center_map(peak_only=peak_only, annot=False)

        if show:    
            plt.show()
        if save is not None:
            plt.savefig(os.path.join(save, self.name) + '_mask.svg')
    
    def plot_center(self, peak_only = False, show=False, save=None):
        """
        Plot the centers of regions.

        Parameters:
        peak_only (bool): Whether to plot only centers of regions with peaks.
        show (bool): Whether to display the plot.
        save (str): The directory path to save the plot.
        """
        plt.figure(figsize=(7,7)) 
        colors = ['grey', 'pink', 'violet', 'purple']
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=len(colors))
        self.link_map(peak_only=peak_only)
        self.center_map(peak_only=peak_only)
        cbar = ColorbarBase(plt.gca().inset_axes([1.0, 0.1, 0.05, 0.8]), orientation='vertical', cmap=cmap)
        cbar.set_label('Correlation')
        tick_positions = [0.25, 0.5, 0.75]
        tick_labels = ['0.3', '0.4', '0.5']
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)
        plt.axis('off')
        if show:
            plt.show()
        if save is not None:
            plt.savefig(os.path.join(save, self.name) + '_center.svg')
        plt.close()

    def _corr_map(self, peak_only = False): 
        """
        Private method to calculate the correlation map.

        Parameters:
        peak_only (bool): Whether to calculate the correlation map only for regions with peaks.

        Returns:
        pd.DataFrame: The correlation matrix.
        """
        if peak_only == True: 
            fr_t = self.fr.loc[self.with_peak].T
        else: 
            fr_t = self.fr.T
        corr_fr = fr_t.corr()
        return corr_fr
    
    def plot_heat(self, annot = False, peak_only = False, show = False, save = None):
        """
        Plot a heatmap of the correlation matrix.

        Parameters:
        annot (bool): Whether to annotate the heatmap.
        peak_only (bool): Whether to plot the heatmap only for regions with peaks.
        show (bool): Whether to display the plot.
        save (str): The directory path to save the plot.
        """
        corr_matrix = self._corr_map(peak_only = peak_only)
        try: 
            sns.clustermap(corr_matrix,
                       annot=annot,
                       metric='correlation',
                       vmax=1,
                       vmin=-1,
                       cmap='coolwarm'
                       )
        except: 
            raise ValueError('Error, Pass')
        if show:
            plt.show()
        if save is not None: 
            plt.savefig(os.path.join(save, self.name) + '.svg')
        n = len(corr_matrix)
        return (np.array(corr_matrix).mean() * n - 1)/(n-1)
    
    def plot_firing(self, peak_only = False, show = False, save = None):
        """
        Plot the firing pattern as a heatmap.

        Parameters:
        peak_only (bool): Whether to plot the heatmap only for regions with peaks.
        show (bool): Whether to display the plot.
        save (str): The directory path to save the plot.
        """
        if peak_only: 
            regu_fr = self.regu_fr.loc[self.with_peak]
        else: 
            regu_fr = self.regu_fr
        # plt.imshow(regu_fr)
        # sns.clustermap(regu_fr, vmax = 1.5, vmin=0, col_cluster=False, col_linkage=False, cmap='viridis')
        sns.heatmap(regu_fr, vmax = 1.5, vmin=0, cmap='viridis')
        if show:
            plt.show()  
        if save is not None:
            plt.savefig(os.path.join(save, self.name) + '_firing.svg')
    
    def plot_curve(self, idx:int, show:str= False, save:bool= None):
        """
        Plot the firing curve for a specific index.

        Parameters:
        idx (int): The index of the region to plot.
        show (bool): Whether to display the plot.
        save (str): The directory path to save the plot.
        """
        assert idx in self.regu_fr.index, 'The index is not in the peak region'
        plt.figure(figsize=(10,5))
        plt.plot(self.regu_fr.loc[idx], label='Original')
        plt.plot(self.peak.loc[idx], label='Peak')
        plt.legend()
        plt.ylim(-0.1, 1.5)
        plt.title(f'Firing curve of cell #{idx} in {self.name}')
        if show:
            plt.show()
        if save is not None:
            plt.savefig(os.path.join(save, self.name) + f'_curve_{idx}.svg')
        plt.close()

        
    def mask_map(self, peak_only = False):
        """
        Plot the mask map.

        Parameters:
        peak_only (bool): Whether to plot only regions with peaks.
        """
        if peak_only == True: 
            indice = self.with_peak
        else: 
            indice = range(len(self.locs))
        for i in indice: 
            plt.scatter(self.locs[i][1],self.locs[i][0], alpha=0.02)
            # plt.annotate(str(i+1), xy = self.centers[i], xytext = self.centers[i],color='white') 

    def center_map(self, peak_only = False, annot = False):
        """
        Plot the center map.

        Parameters:
        peak_only (bool): Whether to plot only centers of regions with peaks.
        annot (bool): Whether to annotate the centers.
        """
        if peak_only == True: 
            centers = [self.centers[i] for i in self.with_peak]
        else: 
            centers = self.centers
        for i, center in enumerate(centers): 
            plt.scatter(center[0], center[1], alpha=0.5, s = 100, edgecolors='black', c = 'b')
            if annot: 
                plt.annotate(str(i+1), xy = center, xytext = center,color='white') 

    def link_map(self, thresh = 0.8, peak_only=False):
        """
        Plot the link map.

        Parameters:
        thresh (float): The threshold for correlation to plot links.
        peak_only (bool): Whether to plot links only for regions with peaks.
        """
        link_fr = self._corr_map(peak_only = peak_only)
        for i in range(link_fr.shape[0]):
            link_fr.iloc[i,i] = 0
        link = np.where(link_fr >= 0.2) 
        x,y = link
        corr_v = np.array([np.array(link_fr)[x[i]][y[i]] for i in range(len(x))])
        corr_v = utils.normalization(corr_v)
        def get_color(corr):
            """
            Function to get color based on correlation.

            Parameters:
            corr (float): The correlation value.

            Returns:
            str: The color corresponding to the correlation value.
            """
            if corr >= 0.5:
                return 'purple'
            elif corr >= 0.4:
                return 'violet'
            elif corr >= 0.3:
                return 'pink'
            else:
                return 'grey'

        if peak_only == True: 
            centers = [self.centers[i] for i in self.with_peak]
        else: 
            centers = self.centers 
        for j in range(len(x)):
            plt.plot([centers[x[j]][0],centers[y[j]][0]],[centers[x[j]][1],centers[y[j]][1]],get_color(corr_v[j]), alpha = corr_v[j]/2)


class Suite2PCalciumSignal():
    """
    Class for handling Suite2P calcium signal data
    """
    def __init__(self, path, std=1.5):
        """
        Initialize the class with the path to the data and a standard deviation threshold.

        Parameters:
        path (str): The directory path containing the Suite2P data files.
        std (float): The standard deviation threshold for peak detection.
        """
        data = np.load(os.path.join(path, 'F.npy'))
        self.fr = pd.DataFrame(data)
        self.regu_fr = utils.regu(self.fr.copy())
        self.peak = utils.peak_std(self.regu_fr.copy(),p=std)
        self.with_peak = self.peak.loc[~(self.peak==0).all(axis=1)].index
        self.name = path.split('/')[-3]
        iscell_raw = np.load(os.path.join(path, 'iscell.npy'), allow_pickle=True)
        iscell_raw = iscell_raw[:,0]
        self.iscell = np.where(iscell_raw == 1)
        self.index = self.regu_fr.index

    def plot_curve(self, idx:int, show:str= False, save:bool= None):
        """
        Plot the firing curve for a specific index.

        Parameters:
        idx (int): The index of the region to plot.
        show (bool): Whether to display the plot.
        save (str): The directory path to save the plot.
        """
        # assert idx in self.regu_fr, 'The index is not in the peak region'
        plt.figure(figsize=(10,5))
        plt.plot(self.regu_fr.loc[idx], label='Original')
        plt.plot(self.peak.loc[idx], label='Peak')
        plt.legend()
        plt.ylim(-0.1, 1.5)
        plt.title(f'Firing curve of cell #{idx} in {self.name}')
        if show:
            plt.show()
        if save is not None:
            plt.savefig(os.path.join(save, self.name) + f'_curve_{idx}.svg')
        # plt.close()

    def plot_firing(self, peak_only = False, iscell =False, show = False, save = None):
        """
        Plot the firing pattern as a heatmap.

        Parameters:
        peak_only (bool): Whether to plot the heatmap only for regions with peaks.
        iscell (bool): Whether to plot only for identified cells.
        show (bool): Whether to display the plot.
        save (str): The directory path to save the plot.
        """
        regu_fr = self.regu_fr
        if peak_only: 
            regu_fr = self.regu_fr.loc[self.with_peak]

        if iscell:
            regu_fr = regu_fr.iloc[self.iscell]
            
        plt.figure(figsize=(10,5))
        sns.heatmap(regu_fr, vmax = 1.5, vmin=0, cmap='viridis', yticklabels=True)
        if show:
            plt.show()  
        if save is not None:
            plt.savefig(os.path.join(save, self.name) + '_firing.svg')


    

if __name__ == '__main__':
    """
    Main execution block
    """
    path = '/Volumes/zhongzh/Data/3D-Ca/MAT-10-26/CaSignal-Time-0701/CaSignal-20230701d1-4x-2zt-MaxIP_Alexa 488 antibody.mat' # Enter the path to the data file
    signal = CalciumSignal(path)
    save_path = './Examples/save_curve/' # Enter the path to save the curve
    os.makedirs(save_path, exist_ok=True)
    signal.plot_mask(link=True, peak_only=True, mask=True, save =save_path)

