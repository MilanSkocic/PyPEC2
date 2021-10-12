# -*- coding: utf-8 -*-
############## MODULES #######################
""" 

Graphical frontend for viewing dot files.

"""
__version__ = 'dev'
__author__ = 'M.Skocic'

from matplotlib import rcParams as rc
from matplotlib import rcdefaults
rc['text.usetex']='False'
rc['font.family']='serif'
rc['font.serif'] = 'Times New Roman'
rc['mathtext.default'] = 'rm'
rc['mathtext.fontset'] = 'stix'
rc['xtick.labelsize']=10
rc['ytick.labelsize']=10
rc['axes.titlesize']=12
rc['axes.labelsize']=12
rc['figure.subplot.left']=0.10  # the left side of the subplots of the figure
rc['figure.subplot.right']=0.98    # the right side of the subplots of the figure
rc['figure.subplot.bottom']=0.1 # the bottom of the subplots of the figure
rc['figure.subplot.top']=0.90
rc['figure.subplot.hspace']=0.5
rc['figure.subplot.wspace']=0.5
rc['backend']='TkAgg'
rc['legend.fontsize']=14
rc['legend.labelspacing']=0.17
rc['lines.markersize'] = 4
rc['lines.markeredgewidth'] = 1
rc['lines.linewidth'] = 1

import os
import sys

import datetime
import shutil
from itertools import izip

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.lines as lines
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

import Tkinter as tk
import ttk
import tkFileDialog
import tkMessageBox

import iph_functions


class Viewer(ttk.Frame):
    
    def __init__(self, master):

        ttk.Frame.__init__(self, master)
        self.pack(expand=tk.YES, fill=tk.BOTH)

        self.master = master

        ws = self.master.winfo_screenwidth()
        hs = self.master.winfo_screenheight()
        self.width = int(0.4*ws) 
        self.height = int(0.2*hs)
        self.x = (ws/2) - (self.width/20)
        self.y = hs/2 
        self.master.geometry('{}x{}+{}+{}'.format(self.width, self.height, self.x, self.y))
        self.master.resizable(height=True, width=True)


        self.row, self.col = 1, 1

        self.fig_window = FigureWindow(self, (int(ws*0.9), int(0.9*hs)), (ws/2-int(0.9*ws)/2, hs/2-int(0.9*hs)/2-25))
        self.datafile_frame = DataFileFrame(self)

        self.master.protocol("WM_DELETE_WINDOW", self._quit)


    def start(self):

        self.master.mainloop()

    def _quit(self):
        
        answer = tkMessageBox.askyesno('Quit?', 'Do you want the application?')
        if answer:
            self.master.destroy()


class FigureWindow(tk.Toplevel):

    def __init__(self, master, size, coordinates):
        
        tk.Toplevel.__init__(self, master)
        
        self.master = master
        #self.transient(self.master)
    
        self.x, self.y = coordinates
        self.width, self.height = size
        self.geometry('{}x{}+{}+{}'.format(self.width, self.height, self.x, self.y))
        self.resizable(height=True, width=True)

        self.fig_frame = FigureFrame(self)


        self.protocol("WM_DELETE_WINDOW", self._quit)


    def _quit(self):
        self.master._quit()

class FigureFrame(ttk.Frame):

    def __init__(self, master):
        ttk.Frame.__init__(self, master)
        self.pack(expand=tk.YES, fill=tk.BOTH)
    
        self.master = master

        self.row, self.col = 2, 1

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        
        for i in range(self.col):
            self.grid_columnconfigure(i, weight=1)

        self.figure = Figure(figsize=(8,6))
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nswe')

        self.toolbar=NavigationToolbar2TkAgg(self.canvas, self)
        self.toolbar.grid(row=1, column=0, sticky='w')


        self.axes = [self.figure.add_subplot(221),
                    self.figure.add_subplot(222),
                    self.figure.add_subplot(223)]
        
    def _grid(self):
        for axes in self.axes:
            axes.grid()
            axes.xaxis.set_minor_locator(AutoMinorLocator())
            axes.yaxis.set_minor_locator(AutoMinorLocator())
        
        
    def update(self, dotfiles):
        
        self._clear()
        self._grid()
        for dotfile in dotfiles:
            for axes, line in izip(self.axes, dotfile.get_lines()):
                axes.add_line(line)
        
        for axes in self.axes:
            axes.plot()

        self.redraw()    
        
    def redraw(self):
        
        for axes in self.axes:
            axes.legend(loc='best', fontsize = 12)
        
        self.autoscale()
        self.canvas.draw()

    def autoscale(self):
        for axes in self.axes:
            axes.set_autoscale_on(True)
            axes.relim()
            axes.autoscale_view(True,True,True)

    def _clear(self):
        for axes in self.axes:
            axes.clear()


class DataFileFrame(ttk.Frame):
    
    def __init__(self, master):

        ttk.Frame.__init__(self, master)
        self.pack(expand=tk.YES, fill=tk.BOTH)
        self.master = master
        self.last_folder = os.path.expanduser('~')
        
        self.col_names = ['Label',
                        'Eapp',
                        'Visible',
                        'Color',
                        'Marker',
                        'LineStyle',
                        'Closed',
                        'Delete']
                                                
        self.dotfiles = []                    
        
        self.col = len(self.col_names)
        self.row = len(self.dotfiles)
        self.header_row = 2
        self.start_row = self.header_row + 1
        self._grid_conf()
        
        self.col_wdg = [ttk.Entry]*1 + [ttk.Combobox, ttk.Checkbutton, ttk.Combobox,
                                        ttk.Combobox, ttk.Combobox, ttk.Checkbutton, ttk.Button]
        self.col_wdg_vbl = [tk.StringVar]*1 + [tk.DoubleVar, tk.IntVar, tk.StringVar,
                                                tk.StringVar, tk.StringVar, tk.IntVar, tk.StringVar]

        self.df_wdg = np.empty(shape=(self.row,), dtype={'names':self.col_names,
                                                    'formats':self.col*[np.object]})
        self.df_wdg_vbl = np.empty(shape=(self.row,), dtype={'names':self.col_names,
                                                    'formats':self.col*[np.object]})

        self.df_wdg_line = np.empty(shape=(1,), dtype={'names':self.col_names,
                                                    'formats':self.col*[np.object]})
        self.df_wdg_line_vbl = np.empty(shape=(1,), dtype={'names':self.col_names,
                                                    'formats':self.col*[np.object]})
        
        for j, name in enumerate(self.col_names):
            ttk.Label(self, text=name ).grid(row=self.header_row, column=j, sticky='nswe')
        
        self.master.master.bind('<Control-o>', self._import_dot_file)

        self.import_bt = ttk.Button(self, text='Import', command=lambda: self._import_dot_file(None))
        self.import_bt.grid(row=0, column=0, columnspan=self.col, sticky='ew')

        self.update_plot_bt = ttk.Button(self, text='Update Plot', command = self._update_plot)
        self.update_plot_bt.grid(row=1, column=0, columnspan=self.col, sticky='ew')


    def _grid_conf(self):   

        for i in range(self.start_row+self.row):
            self.grid_rowconfigure(i, weight=0)
        for j in range(self.col):
            self.grid_columnconfigure(j, weight=1)

    def _create_widget_line(self):

        for j in range(self.col):
            col_name = self.col_names[j]
            self.df_wdg_line_vbl[col_name][0] = self.col_wdg_vbl[j]()
            self.df_wdg_line[col_name][0] = self.col_wdg[j](self)

            if isinstance(self.df_wdg_line[col_name][0], ttk.Entry):
                
                if col_name == 'Label':
                    self.df_wdg_line[col_name][0].config(textvariable=self.df_wdg_line_vbl[col_name][0])

                if col_name == 'Color':
                   self.df_wdg_line[col_name][0].config(textvariable=self.df_wdg_line_vbl[col_name][0])
                   self.df_wdg_line[col_name][0]['values'] = ['k','r','b','g','c','y','m']  
                   self.df_wdg_line_vbl[col_name][0].set(self.df_wdg_line[col_name][0]['values'][0])

                if col_name == 'Marker':
                   self.df_wdg_line[col_name][0].config(textvariable=self.df_wdg_line_vbl[col_name][0])
                   self.df_wdg_line[col_name][0]['values'] = ['None','o','s','d','.','<','^','>','v']  
                   self.df_wdg_line_vbl[col_name][0].set(self.df_wdg_line[col_name][0]['values'][1])

                if col_name == 'LineStyle':
                   self.df_wdg_line[col_name][0].config(textvariable=self.df_wdg_line_vbl[col_name][0])
                   self.df_wdg_line[col_name][0]['values'] = ['None', '-', '--', '-.', ':']  
                   self.df_wdg_line_vbl[col_name][0].set(self.df_wdg_line[col_name][0]['values'][0])

                elif col_name == 'Eapp':
                   self.df_wdg_line[col_name][0].config(textvariable=self.df_wdg_line_vbl[col_name][0])
                   self.df_wdg_line[col_name][0]['values'] = [0.0, 10, 100]  
                   self.df_wdg_line_vbl[col_name][0].set(self.df_wdg_line[col_name][0]['values'][0])

                                           
            elif isinstance(self.df_wdg_line[col_name][0], ttk.Checkbutton):
                self.df_wdg_line[col_name][0].config(text='', variable=self.df_wdg_line_vbl[col_name][0])

            elif isinstance(self.df_wdg_line[col_name][0], ttk.Button):
                self.df_wdg_line[col_name][0].config(text='Remove',
                                                    command=lambda
                                                    widget=self.df_wdg_line[col_name][0]:self._on_remove(widget))
                


    def _create_widgets(self):
        
        for i in range(self.row):
            self._create_widget_line()
            for name in self.col_names:
                self.df_wdg_vbl[name][i] = self.df_wdg_line_vbl[name][0]
                self.df_wdg[name][i] = self.df_wdg_line[name][0]

    
    def _update_widget_values(self):
        for i in range(self.row):
            self.df_wdg_vbl['Label'][i].set(self.dotfiles[i].label)
            self.df_wdg_vbl['Eapp'][i].set(self.dotfiles[i].Eapp)
            self.df_wdg_vbl['Color'][i].set(self.dotfiles[i].color)
            self.df_wdg_vbl['Closed'][i].set(self.dotfiles[i].closed)
            self.df_wdg_vbl['Visible'][i].set(self.dotfiles[i].visible)
            self.df_wdg_vbl['Marker'][i].set(self.dotfiles[i].marker)
            self.df_wdg_vbl['LineStyle'][i].set(self.dotfiles[i].linestyle)


    def _update_dot_files(self):
        for i in range(self.row):
            self.dotfiles[i].set_label(self.df_wdg_vbl['Label'][i].get())
            self.dotfiles[i].set_Eapp(self.df_wdg_vbl['Eapp'][i].get())
            self.dotfiles[i].set_color(self.df_wdg_vbl['Color'][i].get(), self.df_wdg_vbl['Closed'][i].get())
            self.dotfiles[i].set_marker(self.df_wdg_vbl['Marker'][i].get())
            self.dotfiles[i].set_visible(self.df_wdg_vbl['Visible'][i].get())
            self.dotfiles[i].set_linestyle(self.df_wdg_vbl['LineStyle'][i].get())


    def _grid_widgets(self):
        self._update_widget_values()
        self._grid_conf()
        for i in range(self.row):
            for j, name in enumerate(self.col_names):
                self.df_wdg[name][i].grid(row=i+self.start_row, column=j, sticky='nswe')


    def _add_line(self):
        self.row += 1

        self._create_widget_line()
        self.df_wdg = np.append(self.df_wdg, self.df_wdg_line[0])
        self.df_wdg_vbl = np.append(self.df_wdg_vbl, self.df_wdg_line_vbl[0])
        
        self._grid_widgets()

    def _delete_line(self, line=-1):
        if self.df_wdg.shape[0] > 0:
            self.row -= 1
            for name in self.col_names:
                self.df_wdg[name][line].destroy()
                self.df_wdg[name][line] = np.empty

            self.df_wdg = np.delete(self.df_wdg, line)
            self.df_wdg_vbl = np.delete(self.df_wdg_vbl, line)

            self.dotfiles.remove(self.dotfiles[line])

            self._grid_widgets()


    def _import_dot_file(self, event):
        self.filepath = tkFileDialog.askopenfilename(title='Add exp File',
                                                         defaultextension='.dot' ,
                                                         filetypes=[('dot files','.dot')],
                                                         initialdir=self.last_folder,
                                                         parent=self)
            
        if os.path.isfile(self.filepath):
            self.filepath = os.path.abspath(self.filepath)
            self.last_folder = os.path.dirname(self.filepath)

            try:
                dotfile = DotFile(self.filepath)
                self.dotfiles.append(dotfile)
                self._add_line()
                    
            except ValueError:
                ext = os.path.basename(self.filepath).split('.')[1]
                tkMessageBox.showerror(title='Import Error', message='Corrupted {0:s} File'.format('*.'+ext))
      
    def _on_remove(self, button):
       self._update_dot_files()
       for i in range(self.row):
            if self.df_wdg['Delete'][i] is button:
                break
       self._delete_line(line=i)


    def _update_plot(self):
        self._update_dot_files()
        self.master.fig_window.fig_frame.update(self.dotfiles)

class DotFile(object):
    
    def __init__(self, filepath, label='', visible=True, color='k', marker='o', closed=False, linestyle='None'):

        self.filepath = filepath
        self.filename = os.path.basename(self.filepath)
        
        if label == '':
            self.label = self.filename
        else:
            self.label = label
        
        self.closed=closed
        self.color = color
        self.marker = marker
        self.linestyle = linestyle
  
        self.Eapp = 0.0
        self.visible = visible

        self.plot_lines =[lines.Line2D(xdata=self.get_hv(), ydata=self.get_Iph(), label=self.label),
                          lines.Line2D(xdata=self.get_hv(), ydata=self.get_IphB(), label=self.label),
                          lines.Line2D(xdata=self.get_hv(), ydata=self.get_phase(), label=self.label)]

    def __print__(self):
        return self.label

    def get_Iph(self):
        return self._import_data()[1]

    def get_hv(self):
        return self._import_data()[0]

    def get_IphN(self):
        return self._import_data()[2]

    def get_phase(self):
        return self._import_data()[3]

    def get_IphB(self):
        return self._import_data()[4]

    def get_lines(self):
        return self.plot_lines

    def _import_data(self):
        return iph_functions.get_exp_data(self.filepath)
                
    def set_label(self, label):
        if label == '':
            self.label = self.filename
        else:
            self.label = label

        for i in self.plot_lines:
            i.set_label(label)

    def set_color(self, color, closed=False):
        self.color = color
        self.closed = closed
        for i in self.plot_lines:
            i.set_color(self.color)
            i.set_markeredgecolor(self.color)
            if self.closed:
                i.set_markerfacecolor(self.color)
            else:
                i.set_markerfacecolor('w')
                        

    def set_marker(self, marker):
        self.marker = marker     
        for i in self.plot_lines:
            i.set_marker(self.marker)

    def set_linestyle(self, linestyle):
        self.linestyle = linestyle 
        for i in self.plot_lines:
            i.set_linestyle(self.linestyle)



    def set_visible(self, visible):
        self.visible = visible
        for i in self.plot_lines:
            i.set_visible(self.visible)


    def set_Eapp(self, Eapp):
        self.Eapp = Eapp




       

if __name__ == '__main__':

    root = tk.Tk()
    app = Viewer(root)
    app.start()
