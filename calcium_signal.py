import scipy.io as scio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from tqdm import tqdm
import seaborn as sns
from PIL import Image as im
import utils

from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase


class CalciumSignal():

    def __init__(self, path, std=1.5): 
        data_all = scio.loadmat(path) 
        data = data_all['analysis']
        mask = data['L'][0,0]
        self.name = path.split('/')[-1]
        self.image = data['image'][0,0]
        self.fr = pd.DataFrame(data['F_cell'][0,0])
        self.locs, self.centers = self._get_locs(mask)
        self.regu_fr = utils.regu(self.fr.copy())
        self.peak = utils.peak_std(self.regu_fr.copy(),p=std)
        self.with_peak = self.peak.loc[~(self.peak==0).all(axis=1)].index
        # print(self.with_peak)

    def _get_locs(self, mask):
        locs = []
        centers = [] 
        for i in range(0,mask.max()): 
            loc = np.where(mask==i+1)
            locs.append(loc)
            center = [np.mean(loc[1]),np.mean(loc[0])]
            centers.append(center)
        return locs, centers 
         
    def plot_mask(self, link = True, peak_only = False, mask = True, center = False, save=None, show=False): 
        pic = im.fromarray(self.image[:,:,1])
        plt.figure(figsize=(10,10))
        plt.imshow(pic, cmap='gray')
        if mask: 
            self.mask_map(peak_only=peak_only)
        if link: 
            self.link_map(peak_only=peak_only)
        if center:
            self.center_map(peak_only=peak_only, annot=True)

        if show:    
            plt.show()
        if save is not None:
            plt.savefig(os.path.join(save, self.name) + '_mask.svg')
    
    def plot_center(self, peak_only = False, show=False, save=None):
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
        if peak_only == True: 
            fr_t = self.fr.loc[self.with_peak].T
        else: 
            fr_t = self.fr.T
        corr_fr = fr_t.corr()
        return corr_fr
    
    def plot_heat(self, annot = False, peak_only = False, show = False, save = None):
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
            print('Error, Pass')
        if show:
            plt.show()
        if save is not None: 
            plt.savefig(os.path.join(save, self.name) + '.svg')
        n = len(corr_matrix)
        return (np.array(corr_matrix).mean() * n - 1)/(n-1)
    
    def plot_firing(self, peak_only = False, show = False, save = None):
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
        assert idx in self.regu_fr, 'The index is not in the peak region'
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
        if peak_only == True: 
            indice = self.with_peak
        else: 
            indice = range(len(self.locs))
        for i in indice: 
            plt.scatter(self.locs[i][1],self.locs[i][0], alpha=0.02)
            plt.annotate(str(i+1), xy = self.centers[i], xytext = self.centers[i],color='white') 

    def center_map(self, peak_only = False, annot = False):
        if peak_only == True: 
            centers = [self.centers[i] for i in self.with_peak]
        else: 
            centers = self.centers
        for i, center in enumerate(centers): 
            plt.scatter(center[0], center[1], alpha=0.5, s = 100, edgecolors='black', c = 'b')
            if annot: 
                plt.annotate(str(i+1), xy = center, xytext = center,color='white') 

    def link_map(self, thresh = 0.8, peak_only=False):
        link_fr = self._corr_map(peak_only = peak_only)
        for i in range(link_fr.shape[0]):
            link_fr.iloc[i,i] = 0
        # print(link_fr)
        link = np.where(link_fr >= 0.2)
        # print(link)
        x,y = link
        corr_v = np.array([np.array(link_fr)[x[i]][y[i]] for i in range(len(x))])
        corr_v = utils.normalization(corr_v)
        def get_color(corr):
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
	def __init__(self, path, std=1.5):
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
		regu_fr = self.regu_fr
		if peak_only: 
			regu_fr = self.regu_fr.loc[self.with_peak]

		if iscell:
			regu_fr = regu_fr.iloc[self.iscell]
			
		# plt.imshow(regu_fr)
		# sns.clustermap(regu_fr, vmax = 1.5, vmin=0, col_cluster=False, col_linkage=False, cmap='viridis')
		plt.figure(figsize=(10,5))
		sns.heatmap(regu_fr, vmax = 1.5, vmin=0, cmap='viridis', yticklabels=True)
		if show:
			plt.show()  
		if save is not None:
			plt.savefig(os.path.join(save, self.name) + '_firing.svg')


# TODO 完成未发生动作电位的筛选，并绘制相关性图 √
# TODO 钙信号形状示意图 √
    

if __name__ == '__main__':
    path = '/Volumes/zhongzh/Data/3D-Ca/MAT-10-26/CaSignal-Time-1009/CaSignal-20231009d10-4x-8zt-MaxIP_Alexa 488 antibody.mat'
    signal = CalciumSignal(path)
    save_path = '/Volumes/zhongzh/Data/3D-Ca/save_curve/'
    os.makedirs(save_path, exist_ok=True)
    id_list = signal.peak.index.tolist()
    # signal.plot_firing(peak_only=True, show=True)
    # signal.plot_center(peak_only=True, show=True)
    # signal.plot_curve(63, show=False, save=save_path)

    # signal.plot_mask(peak_only=True,center=True, link=True)
    # signal.plot_center(peak_only=True, show=False, save=save_path)
    signal.plot_mask(link=True, peak_only=True, mask=True, save =save_path)
    # signal.plot_curve(63, show=True, save=save_path)

