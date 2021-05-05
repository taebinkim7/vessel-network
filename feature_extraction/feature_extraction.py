#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: wangxuelin
"""

import os
import math
import numpy, scipy.io
from PIL import Image
import matplotlib.colors as mcolors
import cv2
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import operator
from scipy.spatial.distance import euclidean
import pandas as pd
from connectivity_matrix_test import *
from _utility import *
from skimage.morphology import thin
import pdb

# imgs_dir=os.getcwd() + "/img/"
imgs_dir=os.path.dirname(os.getcwd()) + "/data/test_data/predictions/"
print(imgs_dir)
save_dir=os.getcwd() + "/feature/"
for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
    for i in range(len(files)):
        print ("original image: " +files[i])
        image = Image.open(imgs_dir+files[i])

        skeleton=thin(image)
        graph=extract_graph(skeleton,image)
        print(graph.number_of_nodes())
        
        ############ merge nodes
        k=True
        
        while k:
            tmp=graph.number_of_nodes()
            attribute="line"
            attribute_threshold_value=5
            to_be_removed = [(u, v) for u, v, data in
                                         graph.edges(data=True)
                                         if operator.le(data[attribute],
                                                      attribute_threshold_value)]
            length=len(to_be_removed)
            for n in range(length):  
                nodes=to_be_removed[n]
                merge_nodes_2(graph,nodes)
            
            for n1,n2,data in graph.edges(data=True):
                line=euclidean(n1,n2)
                graph[n1][n2]['line']=line
            
            number_of_nodes=graph.number_of_nodes()
            k= tmp!=number_of_nodes
        
        print(graph.number_of_nodes())
        
        #Check connected
        compnt_size = 1
        operators ="smaller or equal"
        oper_str_value = operators
        operators = operator.le

        connected_components = sorted(list((graph.subgraph(c) for c in nx.connected_components(graph))),
            key=lambda graph: graph.number_of_nodes())
        
        to_be_removed = [subgraph for subgraph in connected_components
                                     if operators(subgraph.number_of_nodes(),
                                                            compnt_size)]
        for subgraph in to_be_removed:
            graph.remove_nodes_from(subgraph)
                     
        print ('discarding a total of', len(to_be_removed),
                           'connected components ...')
        
        ########
        
        nodes=[n for n in graph.nodes()]
        x=[x for (x,y) in nodes]
        y=[y for (x,y) in nodes]
        x1=int(np.min(x)+(np.max(x)-np.min(x))/2)
        y1=int(np.min(y)+(np.max(y)-np.min(y))/2)
        
        for n1,n2,data in graph.edges(data=True):
            centerdis1=euclidean((x1,y1),n2)
            centerdis2=euclidean((x1,y1),n1)
            #theta1=(math.atan2(-13,-14)/math.pi*180)%360
            #theta2=(math.atan2(-13,-14)/math.pi*180)%360
            
            if centerdis1>=centerdis2:
                centerdislow=centerdis2
                centerdishigh=centerdis1
            else:
                centerdislow=centerdis1
                centerdishigh=centerdis2
            graph[n1][n2]['centerdislow']=centerdislow
            graph[n1][n2]['centerdishigh']=centerdishigh
        
        
        
        ##############
        
        alldata=save_data(graph,center=False)
        data_name = files[i][0:6] + "_alldata.xlsx"
        print ("data name: " + data_name)

        #writer = pd.ExcelWriter('/Users/wangxuelin/Downloads/STARE-im/im0324_alldata.xlsx', engine='xlsxwriter')
        writer = pd.ExcelWriter(save_dir+data_name, engine='xlsxwriter')

        alldata.to_excel(writer,index=False)
        writer.save()
        
        degreedata=save_degree(graph,x1,y1)
        degree_name = files[i][0:6] + "_degreedata.xlsx"
        print ("degree data name: " + degree_name)

        #writer = pd.ExcelWriter('/Users/wangxuelin/Downloads/STARE-im/im0324_alldata.xlsx', engine='xlsxwriter')
        degreewriter = pd.ExcelWriter(save_dir+degree_name, engine='xlsxwriter')

        degreedata.to_excel(degreewriter,index=False)
        degreewriter.save()
    
        ########
        #Draw network
        NODESIZESCALING = 30
        EDGETRANSPARENCYDIVIDER = 5
        node_size=0.1
        
        pic=draw_graph2(np.asarray(image.convert("RGB")), graph,center=False)
        #plt.imshow(pic)
        pic_name = files[i][0:6] + "_network.png"
        print ("pic name: " + save_dir+pic_name)
        plt.imsave(save_dir+pic_name,pic)
        
        #plt.imsave('/Users/wangxuelin/Downloads/STARE-im/im0324_network.png',pic)
