# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 17:13:43 2022

@author: io
"""

import numpy as np
import matplotlib.pyplot as plt



class Grapher:
    
    """ provide graphical network functionalities  """
    
    
    def __init__(self, connections: list, nb_input: int, nb_output: int):
        
        """
        :param connections: list, list of tuples as [(pre, post)]
        :param nb_input: int, number of input nodes
        :param nb_output: int, number of output nodes
        return: None
        """
        
        self.nb_input = nb_input
        self.nb_output = nb_output
        
        self.connections = np.array([list(pair) for pair in connections])
        self.nodes = tuple(range(np.min(connections), np.max(connections) + 1))
        self.n = len(self.nodes)
        
        self.coordinates = np.zeros((self.n, 2))
        
        self.build()
        
        self.fig = ""
        self.ax = ""
        self.pause_time = 1
        
    def initialize(self):
        
        """
        initialize figure variables
        :return: None
        """
        
        self.fig = plt.figure(frameon=False)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.fig.canvas.manager.show()
        
    def live_graph(self, activations: np.ndarray):
        
        """
        given the activations of the nodes, print them on the plot
        :param activations, ndarray
        :return: None
        """
        
        # plt.figure(layout='constrained', frameon=0)
        #plt.axis([-0.3, .6, -0.3, .6])
        #plt.axis("equal")
        
        # self.ax.clear()
        self.ax.clear()
                
        # annotate activations
        for i, a in enumerate(activations):
            c = plt.Circle(self.coordinates[i], radius=0.1,
                           edgecolor='black', alpha=1)
            c.set_edgecolor('black')
            c.set_facecolor('pink')
            self.ax.text(x=self.coordinates[i, 0]-0.075, 
                         y=(self.coordinates[i, 1]-0.035),
                         s=f'{a:.2f}',
                         fontsize=14)
            self.fig.gca().add_artist(c)
            # self.ax.setp(txt)
            
        # connections
        for (j, i) in self.connections:
            xj, xi = self.coordinates[j, 0], self.coordinates[i, 0]
            yj, yi = self.coordinates[j, 1], self.coordinates[i, 1]
            self.ax.plot((xj, xi), (yj, yi), '--k', alpha=0.1)
            
        
        # nodes
        # plt.title('Live Connectivity graph')
        #plt.scatter(self.coordinates[:, 0], 
        #            self.coordinates[:, 1], color='black', s=100)
        self.ax.set_xlim((-0.3, 1.3))
        self.ax.set_ylim((-0.3, 1.3))
        self.ax.set_xticks(())
        self.ax.set_yticks(())
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(self.pause_time)
        #
        #self.fig.canvas.draw()
        #self.fig.show()


    def draw_graph(self):
        
        """
        draw a graph given a list of connections
        return: None
        """
        # plot #
        
        # connections
        for (j, i) in self.connections:
            xj, xi = self.coordinates[j, 0], self.coordinates[i, 0]
            yj, yi = self.coordinates[j, 1], self.coordinates[i, 1]
            plt.plot((xj, xi), (yj, yi), '-k', alpha=0.7)
            
        
        # nodes
        plt.title('Connectivity graph')
        plt.scatter(self.coordinates[:, 0], 
                    self.coordinates[:, 1], color='black', s=100)
        plt.xlim((-0.3, 1.3))
        plt.ylim((-0.3, 1.3))
        plt.xticks(())
        plt.yticks(())
        plt.show()
        
        
    def build(self):
        
        """
        define the coordinates for each node of a network
        :return: None
        """
        
        # x-axis
        X = [0] * self.nb_input
        X += [(i + 1) / (self.n - self.nb_input - self.nb_output + 1) 
              for i in range(self.n - self.nb_input - self.nb_output)]
        X += [1] * self.nb_output
        
        # y-axis
        Y = [(i) / (self.nb_input + 1) for i in range(self.nb_input, 0, -1)]
        
        y0 = np.random.uniform(0.1, 0.5)
        for _ in range(self.n - self.nb_input - self.nb_output):
            Y += [ 0.5 - (2*(y0 > 0.5)-1) * np.random.uniform(0.2, 0.6)]
            y0 = Y[-1]
        Y += [(i + 1) / (self.nb_output + 1) for i in range(self.nb_output)]
        
        self.coordinates[:, 0] = X
        self.coordinates[:, 1] = Y
        
    def set_pause(self, new_time: float):
        
        """
        set the pause time for the livestream graph
        :param new_time: float
        :return: None
        """
        
        self.pause_time = new_time
        
        
    
if __name__  == '__main__':
    
    C = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)]
    A = np.random.rand(5)
    grapher = Grapher(connections=C, nb_input=1, nb_output=1)
    #grapher.draw_graph()
    grapher.live_graph(activations=A)