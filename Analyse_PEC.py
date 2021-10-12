# -*- coding: utf-8 -*-
############## MODULES #######################
""" 

Graphical frontend for fitting photo-current spectra.

"""
__version__ = 'dev'
__author__ = 'M.Skocic'

from matplotlib import rcParams as rc
from matplotlib import rcdefaults
rc['text.usetex']='False'
rc['font.family']='sans-serif'
rc['font.serif'] = 'Times New Roman'
rc['font.sans-serif'] = 'Arial'
rc['mathtext.default'] = 'rm'
rc['mathtext.fontset'] = 'stixsans'
rc['xtick.labelsize']=10
rc['ytick.labelsize']=10
rc['axes.titlesize']=12
rc['axes.labelsize']=12
rc['figure.subplot.left']=0.10  # the left side of the subplots of the figure
rc['figure.subplot.right']=0.98    # the right side of the subplots of the figure
rc['figure.subplot.bottom']=0.1 # the bottom of the subplots of the figure
rc['figure.subplot.top']=0.90
rc['figure.subplot.hspace']=0.9
rc['figure.subplot.wspace']=0.9
rc['backend']='TkAgg'
rc['legend.fontsize']=14
rc['legend.labelspacing']=0.17
rc['lines.markersize'] = 4
rc['lines.markeredgewidth'] = 1
rc['lines.linewidth'] = 1

import Queue
import os
import sys
from itertools import izip

import datetime
import shutil

import numpy as np
import scipy as sp
from scipy.constants import k, h, e, c

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.lines as lines
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

from Tkinter import *
from ttk import *
import tkFileDialog
import tkMessageBox

import iph_functions

from Parallel_Process import MinimizationProcess,\
                             PlotSummaryProcess,\
                             initialize_processes,\
                             start_processes,\
                             get_cpu_number,\
                             get_queue
import multiprocessing

HV_COEF = h*c/(1e-9*e)

IPH_MARKER = 'o'
PHASE_MARKER = 'o'
RE_IPH_MARKER = 'o'
IM_IPH_MARKER = 'o'
DISTANCE_MARKER = 'o'
MARKER_SIZE = 4

_PROFILE_FAST = 0
_PROFILE_NORMAL = 1
_PROFILE_AGGRESSIVE = 2

_PROFILE_VALUES = {'NM iterations per prm': [50, 200, 500],
                        'NM iterations per prm': [50, 200, 500],
                        'NM fcalls per prm': [50, 200, 500],
                        'NM log10 xtol': [-8, -11, -11],
                        'NM log10 ftol': [-8, -23, -23],
                        'Nb cpu to use': [1, 1, int(get_cpu_number())],
                        'Nb run per process': [10, 3, 1],
                        'Nb fit per run': [200, 50, 20],
                        'Update every':[10, 5, 1]}



class Analyse_PEC(Frame):
    r"""

    Graphical front end for fitting Iph spectra.

    Parameters
    -----------
    master: tkinter widget
        Parent widget of the Tkinter Frame widget.

    Attributes
    -----------
    prm_init: 2d array
        Represents the values and states of the triplets :math:`(K_i, \theta_i, Eg_i)`.
        
    cpu_count: int
        Number of cpu available.

    cpu_to_use: int
        Number of cpu to use among the available cpus.
        
    stopped_process: int
        Counter of stopped process.

    run_per_process: int
        Number of runs per process.

    process_colors: list
        List of colors to be used for plotting the running processes.

    path_prm_file: string
        Filepath to the text file containing the values and states of the triplets :math:`(K_i, \theta_i, Eg_i)`.

    """

    
    def __init__(self,master=None):

        self.prm_init = np.asarray([[0.0,1,
                                     0.0,1,
                                     0.0,1,
                                     2,1,
                                     0,0,0]], dtype=iph_functions._FLOAT)
        self.cpu_count = get_cpu_number()
        self.cpu_to_use = 1 #int(self.cpu_count)
        self.stopped_process = 0
        self.running_flag = False
        self.run_per_process = 3
        self.process_colors = ['b', 'r', 'g', 'y', 'm', 'c', 'darkblue', 'darkred', 'darkgreen', 'lightblue', 'orange', 'lightgreen']
        self.path_prm_file = ''
        self.hv_min = 0.0
        self.hv_max = 6.2
        self.K_bound=(10**-12, 10**-1)
        self.theta_bound=(-180.0, 180.0)
        self.Eg_bound=(0.1, 6.2)

        self.iph_calc_complex_last_as_measured = [np.array([])]*self.cpu_to_use
        self.iph_calc_complex_last_true = [np.array([])]*self.cpu_to_use
        self.hv_last = [np.array([])]*self.cpu_to_use

        self.n_exponant = 0.0

        self.source_folder = os.path.dirname(sys.argv[0])
        self.last_data_folder = os.path.expanduser('~')
        self.last_prm_folder = os.path.expanduser('~')
        self.last_save_folder = os.path.expanduser('~')

        self.filepath = ''
        self.dirpath = 'C:\\'
        self.polling_time = 100
        self.flag_iphB = True
        self.flag_prm = False
        self.flag_data = False

        self.iph_labels = {'Iph': 'Iph', 'Iph*':r'$Iph^{\ast}$'}

        self.Nb_SC = IntVar()
        self.Nb_SC.set(self.prm_init.shape[0])
        
        
        # GUI
        Frame.__init__(self,master)
        self.pack(expand=YES, fill=BOTH)
        self.master.title('PEC Fitting - {0:s} - Running in Python {1:s}'.format(__version__, sys.version))
        #self.master.iconbitmap(os.path.abspath('./Icons/icon.ico'))
        self.master.protocol("WM_DELETE_WINDOW", self.ask_quit)
        
        # get screen width and height
        ws = self.master.winfo_screenwidth()
        hs = self.master.winfo_screenheight()
        #master.geometry(("%dx%d")%(ws,hs))
        width = int(0.9*ws) 
        height = int(0.9*hs)
        x = (ws/2) - (width/2)
        y = (hs/2) - (height/2)-25
        master.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        Grid.columnconfigure(self, 0,weight=1)
        Grid.columnconfigure(self, 1,weight=4)

        row = 0
        # Add Single File Button
        self.AddFiles_button=Button(self, text='Load exp File' , command=self.AddFiles_cb)
        self.AddFiles_button.grid(row=row, column=0, sticky='ew')
        
        # Name of exp files
        self.Files_List_Var = StringVar()
        self.Files_List_Entry = Entry(self, textvariable = self.Files_List_Var, state='readonly')
        self.Files_List_Entry.grid(row=row,column=1, sticky='ew')
        
        # PRM Button
        row += 1
        self.prm_button = Button(self, text='Set Parameters', command=self._set_parameters, state='normal')
        self.prm_button.grid(row=row, column=0, sticky='we')
        
        # Progil Scan: Fast, Normal, Aggressive
        row +=1 
        self.scan_profile_labelframe = LabelFrame(self, text='Choose Scan Profile')
        self.scan_profile_labelframe.grid(row=row, column=0, sticky='nswe')
        for i in range(3):
            Grid.columnconfigure(self.scan_profile_labelframe, i, weight=1)
        container = self.scan_profile_labelframe
        self.scan_profile_IntVar = IntVar() # 0:Fast, 1:Normal, 2: Aggressive
        self.scan_profile_IntVar.set(_PROFILE_NORMAL)
        self.scan_profile_fast_radbut = Radiobutton(container,
                                                    text='Fast',
                                                    variable=self.scan_profile_IntVar, 
                                                    value=0, 
                                                    command=self._on_profile, 
                                                    state='disabled')
        self.scan_profile_fast_radbut.grid(row=0, column=0, sticky='nswe')
        
        self.scan_profile_normal_radbut = Radiobutton(container,
                                                    text='Normal',
                                                    variable=self.scan_profile_IntVar, 
                                                    value=1, 
                                                    command=self._on_profile, 
                                                    state='disabled')
        self.scan_profile_normal_radbut.grid(row=0, column=1, sticky='nswe')

        self.scan_profile_aggressive_radbut = Radiobutton(container,
                                                    text='Aggressive',
                                                    variable=self.scan_profile_IntVar, 
                                                    value=2, 
                                                    command=self._on_profile, 
                                                    state='disabled')
        self.scan_profile_aggressive_radbut.grid(row=0, column=2, sticky='nswe')


        

        #linear preview
        row+=1
        self.linear_view_labelframe = LabelFrame(self, text='Linear View')
        self.linear_view_labelframe.grid(row=row, column=0, sticky='nswe')
        for i in range(3):
            Grid.columnconfigure(self.linear_view_labelframe, i, weight=1)
        container = self.linear_view_labelframe
        self.linear_view_DVar = DoubleVar()
        self.linear_view_DVar.set(0)
        self.linear_none_radbut = Radiobutton(container, text="None",variable=self.linear_view_DVar, value=0, command=self._on_linear_preview, state='disabled')
        self.linear_none_radbut.grid(row=0,column=0, sticky='nswe')
        self.linear_direct_radbut = Radiobutton(container, text="Direct Transitions",variable=self.linear_view_DVar, value=0.5, command=self._on_linear_preview, state='disabled')
        self.linear_direct_radbut.grid(row=0,column=1, sticky='nswe')
        self.linear_indirect_radbut = Radiobutton(container, text="Indirect Transitions",variable=self.linear_view_DVar, value=2.0, command=self._on_linear_preview, state='disabled')
        self.linear_indirect_radbut.grid(row=0,column=2, sticky='nswe')
        
        # Choice of values to fit
        row += 1
        self.fit_type_labelframe = LabelFrame(self, text='Values to fit')
        self.fit_type_labelframe.grid(row=row, column=0, sticky='nswe')
        for i in range(2):
            Grid.columnconfigure(self.fit_type_labelframe, i, weight=0)
        container = self.fit_type_labelframe
        self.fit_type_StrVar = StringVar()
        self.fit_type_StrVar.set('Iph*')
        self.fit_type_iph = Radiobutton(container, text='Iph*', value='Iph*', variable=self.fit_type_StrVar, state='disabled',
        command=self._on_fit_type)
        self.fit_type_iph.grid(row=0, column=0, sticky='nswe')
        self.fit_type_iphB = Radiobutton(container, text='Iph', value='Iph', variable=self.fit_type_StrVar, state='disabled',
        command=self._on_fit_type)
        self.fit_type_iphB.grid(row=0, column=1, sticky='nswe')

        # Choice of weights for calculating the errors
        row+=1
        self.weights_type_labelframe = LabelFrame(self, text='Weights for error estimation')
        self.weights_type_labelframe.grid(row=row, column=0, sticky='nswe')
        for i in range(3):
            Grid.columnconfigure(self.weights_type_labelframe, i, weight=1)
        container = self.weights_type_labelframe
        self.weights_type_StrVar = StringVar()
        self.weights_type_StrVar.set(u'Abs.Err.')
        self.weights_type_1 = Radiobutton(container, text='1', value='1', variable=self.weights_type_StrVar, state='disabled',
        command=self._on_weight_type)
        self.weights_type_1.grid(row=0, column=0, sticky='nswe')
        self.weights_type_invIph = Radiobutton(container, text=u'1/|Iph(*)|^2', value=u'1/|Iph(*)|^2',
        variable=self.weights_type_StrVar, state='disabled',
        command=self._on_weight_type)
        self.weights_type_invIph.grid(row=1, column=0, sticky='nswe')
        self.weights_type_sigma = Radiobutton(container, text='|IphN*/Noise(Iph)|^2', value='Abs.Err.', variable=self.weights_type_StrVar,
        state='disabled', command=self._on_weight_type)
        self.weights_type_sigma.grid(row=0, column=1, sticky='nswe')
        self.sigma_DblVar = DoubleVar()
        self.sigma_DblVar.set(1.0)
        self.sigma_entry = Entry(container, textvariable=self.sigma_DblVar, state='disabled', width=10)
        self.sigma_entry.grid(row=1, column=2,sticky='w')
        self.sigma_entry.bind('<Return>', self._on_sigma_entry)
        self.noise_label = Label(container, text='Average Noise (Iph)=', state='disabled')
        self.noise_label.grid(row=1, column=1, sticky='e')


        row+=1
        self.hvrange_labelframe = LabelFrame(self, text='Energy Range')
        self.hvrange_labelframe.grid(row=row, column=0, sticky='nswe')
        for i in range(4):
            if i%2==0:
                Grid.columnconfigure(self.hvrange_labelframe, i, weight=0)
            else:
                Grid.columnconfigure(self.hvrange_labelframe, i, weight=1)

        self.hv_start=DoubleVar()
        self.hv_end=DoubleVar()
        
        self.hv_start.set(self.hv_min)
        self.hv_end.set(self.hv_max)
        
        Label(self.hvrange_labelframe, text='hv start (eV): ').grid(row=0, column=0, sticky='nswe')
        self.hv_start_Entry=Entry(self.hvrange_labelframe, textvariable=self.hv_start,state='disabled', width=10)
        self.hv_start_Entry.grid(row=0,column=1,sticky='nswe')
        self.hv_start_Entry.bind('<Return>',self.on_hv_limits)
        Label(self.hvrange_labelframe, text='hv end (eV): ').grid(row=0,column=2,sticky='nswe')
        self.hv_end_Entry=Entry(self.hvrange_labelframe, textvariable=self.hv_end,state='disabled', width=10)
        self.hv_end_Entry.grid(row=0,column=3,sticky='nswe')
        self.hv_end_Entry.bind('<Return>',self.on_hv_limits)
        
        row+=1
        self.setup_range_labelframe = LabelFrame(self, text='Fit Setup')
        self.setup_range_labelframe.grid(row=row, column=0, sticky='nswe')
        for i in range(4):
            if i%2==0:
                Grid.columnconfigure(self.setup_range_labelframe, i, weight=0)
            else:
                Grid.columnconfigure(self.setup_range_labelframe, i, weight=1)
        container = self.setup_range_labelframe
        
        Label(container ,text='Processes: ').grid(row=0,column=0, sticky='w')
        
        self.nb_cpu_to_use_IntVar = IntVar()
        self.nb_cpu_to_use_IntVar.set(self.cpu_to_use)
        self.nb_cpu_to_use_spbox = Spinbox(container, from_=1,to=self.cpu_count, textvariable=self.nb_cpu_to_use_IntVar, increment=1,\
                                           command=self.update_nb_runs, state='disabled', width=10)
        self.nb_cpu_to_use_spbox.grid(row=0, column=1, sticky='nsew')
        
        Label(container ,text='Runs/Process: ').grid(row=1, column=0,sticky='nswe')
        self.nb_run_per_process_IntVar = IntVar()
        self.nb_run_per_process_IntVar.set(self.run_per_process)
        self.nb_run_per_process_spbox = Spinbox(container, textvariable=self.nb_run_per_process_IntVar, from_=1, to=1000000, increment=1,\
                                                command=self.update_nb_runs, state='disabled', width=10)
        self.nb_run_per_process_spbox.grid(row=1, column=1,sticky='nswe')

        Label(container, text='Runs: ').grid(row=2,column=0,sticky='nswe')
        self.NB_Run=IntVar()
        self.NB_Run.set(self.cpu_to_use*self.run_per_process)
        self.NB_Run_label=Label(container ,textvariable = self.NB_Run, width=10)
        self.NB_Run_label.grid(row=2,column=1,sticky='nswe')
        
        Label(container ,text='Minim./Run: ').grid(row=3,column=0,sticky='w')
        self.NB_Fit_in_Run=IntVar()
        self.NB_Fit_in_Run.set(50)
        self.NB_Fit_in_Run_spbox=Spinbox(container ,textvariable=self.NB_Fit_in_Run, from_=1, to=1000000, increment=1, state='disabled',\
                                         command=self.update_nb_fit_in_run, width=10)
        self.NB_Fit_in_Run_spbox.grid(row=3,column=1,sticky='nswe')

        Label(container ,text='Plot every: ').grid(row=3,column=2,sticky='w')
        self.update_every_IntVar = IntVar()
        self.update_every_IntVar.set(5)
        self.update_every_spbox=Spinbox(container ,textvariable=self.update_every_IntVar, from_ = 1, to = 1000000, increment=1, state='disabled',\
                                         command=self.update_nb_fit_in_run, width=10)
        self.update_every_spbox.grid(row=3, column=3, sticky='nswe')


        row+=1
        self.naming_labelframe = LabelFrame(self, text='Identification')
        self.naming_labelframe.grid(row=row, column=0, sticky='nswe')
        for i in range(6):
            if i%2==0:
                Grid.columnconfigure(self.naming_labelframe, i, weight=0)
            else:
                Grid.columnconfigure(self.naming_labelframe, i, weight=1)

        container = self.naming_labelframe
        
        Label(container ,text='Suffix: ').grid(row=0,column=0,sticky='nswe')
        self.suffix_entry_StrVar = StringVar()
        self.suffix_entry_StrVar.set('')
        self.suffix_entry = Entry(container, textvariable=self.suffix_entry_StrVar, state='disabled', width=10)
        self.suffix_entry.grid(row=0, column=1, sticky='nswe')

        Label(container ,text='Alloy: ').grid(row=0,column=2,sticky='w')
        self.alloy_entry_StrVar = StringVar()
        self.alloy_entry_StrVar.set('')
        self.alloy_entry = Entry(container, textvariable=self.alloy_entry_StrVar, state='disabled', width=10)
        self.alloy_entry.grid(row=0, column=3, sticky='nswe')

        Label(container ,text='ID: ').grid(row=0,column=4,sticky='w')
        self.alloy_id_entry_StrVar = StringVar()
        self.alloy_id_entry_StrVar.set('')
        self.alloy_id_entry = Entry(container, textvariable=self.alloy_id_entry_StrVar, state='disabled', width=10)
        self.alloy_id_entry.grid(row=0, column=5, sticky='nswe')

        #Label(self ,text='Initial Parameter Values').grid(row=14,column=0, columnspan = 2, sticky='we',padx=5,pady=5)
        row+=1
        self.run0_labelframe = LabelFrame(self,text='Initialization Run = 1')
        self.run0_labelframe.grid(row=row, column=0, sticky='nswe')
        Grid.columnconfigure(self.run0_labelframe, 0, weight=1)
        Grid.columnconfigure(self.run0_labelframe, 3, weight=1)
        Grid.columnconfigure(self.run0_labelframe, 1, weight=0)
        Grid.columnconfigure(self.run0_labelframe, 2, weight=0)

        container = self.run0_labelframe
        self.Init_type_0 = StringVar()
        self.Init_type_0.set('random')
        Radiobutton(container ,text="Random",variable=self.Init_type_0, value='random').grid(row=0,column=0, sticky='nswe')
        Radiobutton(container ,text="User", variable=self.Init_type_0, value='user').grid(row=0,column=3, sticky='nswe')
        Label(container ,text='->Random Loops: ').grid(row=0,column=1,sticky='nswe')
        self.random_loops_IntVar = IntVar()
        self.random_loops_IntVar.set(200)
        self.random_loops_sbox = Spinbox(container,textvariable = self.random_loops_IntVar, from_= 1, to = 1000000, increment=1, state='normal',width=5)
        self.random_loops_sbox.grid(row=0, column=2, sticky='nswe')

        row+=1
        self.runN_labelframe = LabelFrame(self,text='Initialize after non valid parameters')
        self.runN_labelframe.grid(row=row, column=0, sticky='nswe')
        for i in range(2):
            if i%2==0:
                Grid.columnconfigure(self.runN_labelframe, i, weight=0)
            else:
                Grid.columnconfigure(self.runN_labelframe, i, weight=1)
        container = self.runN_labelframe
        self.Init_type_validation = StringVar()
        self.Init_type_validation.set('random')
        Radiobutton(container ,text="Random",variable=self.Init_type_validation, value='random').grid(row=0,column=0,sticky='nswe')
        Radiobutton(container ,text="User", variable=self.Init_type_validation, value='user').grid(row=0,column=1, sticky='nswe')
        
        row+=1
        self.runN_labelframe = LabelFrame(self,text='Propagation Run > 1')
        self.runN_labelframe.grid(row=row, column=0, sticky='nswe')
        for i in range(4):
            Grid.columnconfigure(self.runN_labelframe, i, weight=1)
        container = self.runN_labelframe
        self.Init_type_N = StringVar()
        self.Init_type_N.set('random')
        Radiobutton(container ,text="Random",variable=self.Init_type_N, value='random').grid(row=0,column=0,sticky='nswe')
        Radiobutton(container ,text="User", variable=self.Init_type_N, value='user').grid(row=0,column=1, sticky='nswe')
        Radiobutton(container ,text="Min", variable=self.Init_type_N, value='min').grid(row=0,column=2, sticky='nswe')
        Radiobutton(container ,text="End", variable=self.Init_type_N, value='end').grid(row=0,column=3, sticky='nswe')

        #Nelder-Mead Options
        row+=1
        self.NelderMeadFrame = LabelFrame(self,text='Nelder-Mead Options (fminsearch)')
        self.NelderMeadFrame.grid(row=row, column=0, sticky='nswe')
        for i in range(4):
            if i%2==0:
                Grid.columnconfigure(self.NelderMeadFrame, i, weight=0)
            else:
                Grid.columnconfigure(self.NelderMeadFrame, i, weight=1)
        container = self.NelderMeadFrame

        self.NM_log10_xtol_IntVar = IntVar()
        self.NM_log10_xtol_IntVar.set(-11)
        Label(container ,text='log10(xtol): ').grid(row=0,column=0,sticky='nswe')
        self.NM_log10_xtol_sbox = Spinbox(container,textvariable = self.NM_log10_xtol_IntVar, from_= -50, to = 50, increment=1, state='normal',width=5)
        self.NM_log10_xtol_sbox.grid(row=0,column=1,sticky='nswe')
        
        self.NM_log10_ftol_IntVar = IntVar()
        self.NM_log10_ftol_IntVar.set(-23)
        Label(container ,text='log10(ftol): ').grid(row=0,column=2,sticky='nswe')
        self.NM_log10_ftol_sbox = Spinbox(container ,textvariable = self.NM_log10_ftol_IntVar, from_= -50, to = 50, increment=1, state='normal',width=5)
        self.NM_log10_ftol_sbox.grid(row=0, column=3, sticky='nswe')

        self.NM_iteration_IntVar = IntVar()
        self.NM_iteration_IntVar.set(200)
        Label(container ,text='Iter./prm.: ').grid(row=1,column=0,sticky='nswe')
        self.NM_iteration_sbox = Spinbox(container ,textvariable = self.NM_iteration_IntVar, from_= 10, to = 1000000, increment=10, state='normal', width=5)
        self.NM_iteration_sbox.grid(row=1,column=1,sticky='nswe')
        
        self.NM_fcalls_IntVar = IntVar()
        self.NM_fcalls_IntVar.set(200)
        Label(container ,text='fcalls/prm.: ').grid(row=1,column=2,sticky='nswe')
        self.NM_fcalls_sbox = Spinbox(container ,textvariable = self.NM_fcalls_IntVar, from_= 10, to = 1000000, increment=10, state='normal',width=5)
        self.NM_fcalls_sbox.grid(row=1, column=3, sticky='nswe')


        #Parameter Constraints Options
        row+=1
        self.ParameterFrame = LabelFrame(self,text='Parameter Constraints')
        self.ParameterFrame.grid(row=row, column=0, sticky='nswe')
        for i in range(4):
            if i%2==0:
                Grid.columnconfigure(self.ParameterFrame, i, weight=0)
            else:
                Grid.columnconfigure(self.ParameterFrame, i, weight=1)
        Grid.columnconfigure(self.ParameterFrame, 4, weight=1)

        container = self.ParameterFrame

        self.K_LBound_IntVar, self.K_UBound_IntVar = IntVar(), IntVar()
        self.K_LBound_IntVar.set(-12)
        self.K_UBound_IntVar.set(-1)

        Label(container ,text='log10(K min): ').grid(row=0,column=0,sticky='nswe')
        self.K_LBound_sbox = Spinbox(container ,textvariable = self.K_LBound_IntVar, from_= -30 , to = 30, increment=1, state='normal', width=5, command=self._on_K_limits)
        self.K_LBound_sbox.grid(row=0,column=1,sticky='nswe')

        Label(container ,text='log10(K Max): ').grid(row=0,column=2,sticky='nswe')
        self.K_UBound_sbox = Spinbox(container ,textvariable = self.K_UBound_IntVar, from_= -30 , to = 30, increment=1, state='normal',width=5,command=self._on_K_limits)
        self.K_UBound_sbox.grid(row=0,column=3,sticky='nswe')
        self.log_scan_IntVar = IntVar()

        self.log_scan_IntVar.set(True)
        Checkbutton(container ,text='Log Scan', variable=self.log_scan_IntVar).grid(row=0,column=4,sticky='nswe')


        self.theta_LBound_IntVar, self.theta_UBound_IntVar = IntVar(), IntVar()
        self.theta_LBound_IntVar.set(-180)
        self.theta_UBound_IntVar.set(180)
        
        Label(container ,text='theta min (°): ').grid(row=1,column=0,sticky='nswe')
        self.theta_LBound_sbox = Spinbox(container ,textvariable = self.theta_LBound_IntVar, from_= -360 , to = 360, increment=1, state='normal', width=5,command=self._on_theta_limits)
        self.theta_LBound_sbox.grid(row=1,column=1,sticky='nswe')

        Label(container ,text='theta max (°): ').grid(row=1,column=2,sticky='nswe')
        self.theta_UBound_sbox = Spinbox(container ,textvariable = self.theta_UBound_IntVar, from_= -360 , to = 360, increment=1, state='normal',width=5,command=self._on_theta_limits)
        self.theta_UBound_sbox.grid(row=1,column=3,sticky='nswe')


        self.Eg_LBound_DVar, self.Eg_UBound_DVar = DoubleVar(), DoubleVar()
        self.Eg_LBound_DVar.set(0.1)
        self.Eg_UBound_DVar.set(6.5)

        Label(container ,text='Eg min: ').grid(row=2,column=0,sticky='nswe')
        self.Eg_LBound_sbox = Spinbox(container ,textvariable = self.Eg_LBound_DVar, from_= 0 , to = 10, increment=0.1, state='normal', width=5,command=self._on_Eg_limits)
        self.Eg_LBound_sbox.grid(row=2,column=1,sticky='nswe')

        Label(container ,text='Eg max: ').grid(row=2,column=2,sticky='nswe')
        self.Eg_UBound_sbox = Spinbox(container ,textvariable = self.Eg_UBound_DVar, from_= 0.0 , to = 10, increment=0.1, state='normal',width=5,command=self._on_Eg_limits)
        self.Eg_UBound_sbox.grid(row=2,column=3,sticky='nswe')

        #Polling time
        row+=1
        self.PollingFrame = LabelFrame(self,text='Polling Parameters')
        self.PollingFrame.grid(row=row, column=0, sticky='nswe')
        for i in range(2):
            if i%2==0:
                Grid.columnconfigure(self.PollingFrame, i, weight=0)
            else:
                Grid.columnconfigure(self.PollingFrame, i, weight=1)
        container = self.PollingFrame

        Label(container ,text='Time (ms)').grid(row=0, column=0,sticky='nswe')
        self.polling_time_IntVar = IntVar()
        self.polling_time_IntVar.set(100)
        self.polling_time_sbox = Spinbox(container ,textvariable = self.polling_time_IntVar, from_= 10 , to = 1000000, increment=10,\
                                         command = self._update_polling_time,\
                                         state='normal',width=5)
        self.polling_time_sbox.grid(row=0, column=1,sticky='nswe')
        
        
        
        #File editor Button
        row+=1
        self.Fit_button=Button(self ,text='START' , command=self.Fit_cb, state='normal')
        self.Fit_button.grid(row=row,column=0, sticky='nswe')

        #Stop button - all processes at once
        row+=1
        self.stop_fit_button = Button(self, text='STOP', command=self.on_stop_button, state='disabled')
        self.stop_fit_button.grid(row=row, column=0, sticky='nswe')


        for i in range(row+1):
            Grid.rowconfigure(self, i, weight=1)
        
        # FIGURE
        self.figure=Figure(figsize=(12,8))
        # Instantiate canvas
        self.canvas = FigureCanvasTkAgg(self.figure ,self)
        self.canvas.get_tk_widget().grid(row=1, column=1, rowspan=row, sticky='nesw')
        # Instantiate and pack toolbar
        self.toolbar=NavigationToolbar2TkAgg(self.canvas, self)
        self.toolbar.grid(row=row+1, column=1, sticky='w')

               
        # CHARTS
        gs=gridspec.GridSpec(6,7)
        #Iph vs ,hv
        self.Iph_Graph=self.figure.add_subplot(gs[0:3,0:4])
        self.Iph_Graph.set_xlabel(r'$h\nu$ (eV)')
        self.Iph_Graph.set_ylabel(self.iph_labels['Iph'] + ' /A')
        self.Iph_Graph_line=lines.Line2D(xdata=[], ydata=[],\
                                         marker=IPH_MARKER, linestyle='none' ,\
                                         markersize = MARKER_SIZE ,color='k' ,mfc='w' ,mec='k',\
                                         label='exp')
        self.Iph_Graph.set_xlim(self.hv_start.get(), self.hv_end.get())
        self.Iph_Graph.add_line(self.Iph_Graph_line)

        self.Iph_true_Graph=self.figure.add_subplot(gs[3:6,0:4])
        self.Iph_true_Graph.set_xlabel(r'$h\nu$ (eV)')
        self.Iph_true_Graph.set_ylabel(self.iph_labels['Iph*'] + ' /A')
        self.Iph_true_Graph_line=lines.Line2D(xdata=[], ydata=[],\
                                         marker=IPH_MARKER, linestyle='none' ,\
                                         markersize = MARKER_SIZE ,color='k' ,mfc='w' ,mec='k',\
                                         label='exp')
        self.Iph_true_Graph.set_xlim(self.hv_start.get(), self.hv_end.get())
        self.Iph_true_Graph.add_line(self.Iph_true_Graph_line)

        #Phi vs ,hv
        self.Phi_Graph=self.figure.add_subplot(gs[0:2,4:7])
        self.Phi_Graph.set_ylabel(r'$\theta$ /$^{\circ}$')
        self.Phi_Graph_line=lines.Line2D(xdata=[], ydata=[], marker=PHASE_MARKER ,linestyle='-' ,markersize = MARKER_SIZE ,color='k' ,mfc='w' ,mec='k')
        self.Phi_Graph.set_xlim(self.hv_start.get(), self.hv_end.get())
        self.Phi_Graph.add_line(self.Phi_Graph_line)

        #Re+Im vs ,hv
        self.Re_Graph=self.figure.add_subplot(gs[2:4,4:7])
        self.Re_Graph.set_xlabel(r'$h\nu$ /eV')
        self.Re_Graph.set_ylabel('Re '+ self.iph_labels[self.fit_type_StrVar.get()] + ' /A, Im ' + self.iph_labels[self.fit_type_StrVar.get()] + ' /A')
        self.Re_Graph.set_xlim(self.hv_start.get(), self.hv_end.get())
        
        self.Re_Graph_line=lines.Line2D(xdata=[], ydata=[], marker=RE_IPH_MARKER ,linestyle='none' ,markersize = MARKER_SIZE ,color='k' ,mec='k' ,mfc='w')
        self.Re_Graph.add_line(self.Re_Graph_line)

        self.Im_Graph_line=lines.Line2D(xdata=[], ydata=[], marker=IM_IPH_MARKER, linestyle='none', markersize = MARKER_SIZE ,color='gray' ,mec='gray' ,mfc='w')
        self.Re_Graph.add_line(self.Im_Graph_line)

        #Evolution
        self.Evol_Graph=self.figure.add_subplot(gs[4:6,4:7])
        self.Evol_Graph.grid()
        #self.Evol_Graph.set_yscale('log')
        self.Evol_Graph.set_xlabel('Minimizations')
        self.Evol_Graph.set_ylabel('$\log10$(Distance)')
        

        self.create_fit_lines()
        self.update_legend(axes=self.Iph_Graph, run = 1, fit = 0)
        self.update_legend(axes=self.Iph_true_Graph, run = 1, fit = 0)
        self.Evol_Graph.set_xlim(0,self.NB_Fit_in_Run.get())
        

        #vertical lines for energy upper/lower limits
        self.Lim_Iph = self.Iph_Graph.axvspan(self.hv_start.get(),self.hv_end.get() ,facecolor='r' ,alpha=0.2)
        self.Lim_Iph_true = self.Iph_true_Graph.axvspan(self.hv_start.get(),self.hv_end.get() ,facecolor='r' ,alpha=0.2)
        self.Lim_Phi = self.Phi_Graph.axvspan(self.hv_start.get(),self.hv_end.get() ,facecolor='r' ,alpha=0.2)
        self.Lim_Re = self.Re_Graph.axvspan(self.hv_start.get(),self.hv_end.get() ,facecolor='r' ,alpha=0.2)


        self._on_profile()
        
  
    def update_nb_runs(self):
        r"""
        
        Update the number of cpu and run per process on the graphical interface.

        """
        self.cpu_to_use = self.nb_cpu_to_use_IntVar.get()
        self.run_per_process = self.nb_run_per_process_IntVar.get()
        
        nb_run = self.run_per_process * self.cpu_to_use
        self.NB_Run.set(nb_run)

        self.iph_calc_complex_last_as_measured = [np.array([])]*self.cpu_to_use
        self.iph_calc_complex_last_true = [np.array([])]*self.cpu_to_use
        self.hv_last = [np.array([])]*self.cpu_to_use


        self.remove_fit_lines()
        self.create_fit_lines()

    def _update_polling_time(self):

        self.polling_time = self.polling_time_IntVar.get()
        


    def update_nb_fit_in_run(self):
        
        self.Evol_Graph.set_xlim(0, self.NB_Fit_in_Run.get())
        self.update_legend(axes=self.Iph_Graph, run = 1, fit=0)
        self.update_legend(axes=self.Iph_true_Graph, run = 1, fit=0)


    def update_legend(self, axes, run=1, fit=0, location='upper left'):

        progress = self.get_progress(run, fit)
        lines = axes.get_lines()[1:]
        for ind, line in enumerate(lines):
            label = u'Process {0:d} - Run {1:03d}/{2:03d} - Fit {3:03d}/{4:03d} - Progress {5:03d}%'.format(ind+1, run, self.run_per_process, fit, self.NB_Fit_in_Run.get(), progress )
            line.set_label(label)
        axes.legend(loc=location, fontsize=12, ncol=1)

        self.update_figure()

    def get_progress(self, run=1, fit=0):

        total = self.NB_Fit_in_Run.get() * self.run_per_process
        current = fit + self.NB_Fit_in_Run.get()*(run-1)
        progress = int(current*100.0/total)

        return progress

    def on_stop_button(self):

        for worker in self.workers:
            if worker.is_alive():
                worker.shutdown()

            else:
                print('No more alive processe')
        

    def Fit_cb(self):

        if self.flag_prm and self.flag_data:
            self.dirpath = tkFileDialog.askdirectory(title='Choose folder for saving results',\
                                                         mustexist=False ,\
                                                         initialdir=self.last_save_folder,
                                                         parent=self)
            
            if os.path.isdir(self.dirpath):
                self.last_save_folder = self.dirpath 
                root = self.dirpath
                alloy = self.alloy_entry_StrVar.get().replace(' ','_')
                alloy_id = self.alloy_id_entry_StrVar.get().replace(' ','_')
                timestamp = datetime.datetime.now().strftime('%Y_%m_%d-%H%M%S')

                if len(alloy) == 0:
                    alloy = 'Un_Alloy'
                if len(alloy_id) == 0:
                    alloy_id = 'Un_ID'
                
                self.win_Fit1=Toplevel()
                self.win_Fit1.title("Fit Settings")
                
                Text=''
                #Text = Text+'FIT SETTINGS\n\n'
                Text = Text + 'Experimental Data File=' + self.filepath + '\n'
                Text = Text + 'Energy Range (eV)={0:s},{1:s}\n'.format(str(self.hv_start.get()), str(self.hv_end.get()))
                Text = Text + 'Fit type={0:s}\n'.format(self.fit_type_StrVar.get())
                Text = Text + 'Weights={0:s}\n'.format(self.weights_type_StrVar.get())
                Text = Text + 'Abs.Err.={0:+.16e}\n'.format(self.sigma_DblVar.get())
                   
                Text=Text + 'No of SC Contributions='+str(self.Nb_SC.get()) + '\n'
                Text=Text + 'Parameter File=' + self.path_prm_file + '\n'

                Text = Text + 'Alloy={0:s}'.format(alloy) + '\n'
                Text = Text + 'ID={0:s}'.format(alloy_id) + '\n'
                
                Text=Text + 'No of Processes=' + str(self.nb_cpu_to_use_IntVar.get()) + '\n'
                Text=Text + 'No of Runs per Process=' + str(self.nb_run_per_process_IntVar.get()) + '\n'
                Text=Text + 'No of Runs=' + str(self.NB_Run.get()) + '\n'
                Text=Text + 'No of Fits per Run=' + str(self.NB_Fit_in_Run.get()) + '\n'
                Text=Text + 'Suffix='+self.suffix_entry_StrVar.get() + '\n'

                Text = Text + 'Initialization Run 1={0:s} \n'.format(self.Init_type_0.get())
                Text = Text + 'Random Loops={0:d} \n'.format(self.random_loops_IntVar.get())
                Text = Text + 'Propagation Run > 1={0:s} \n'.format(self.Init_type_N.get())
                Text = Text + 'Initialize after non valid parameters={0:s} \n'.format(self.Init_type_validation.get())

                Text = Text + 'log10 xtol={0:d} \n'.format(self.NM_log10_xtol_IntVar.get())
                Text = Text + 'log10 ftol={0:d} \n'.format(self.NM_log10_ftol_IntVar.get())
                Text = Text + 'Iterations/Parameter={0:d} \n'.format(self.NM_iteration_IntVar.get())
                Text = Text + 'fcalls/Parameter={0:d} \n'.format(self.NM_fcalls_IntVar.get())

                Text = Text + 'K min, K max={0:.2e},{1:.2e} \n'.format(10**self.K_LBound_IntVar.get(), 10**self.K_UBound_IntVar.get())
                Text = Text + 'Log scan of K={0:b} \n'.format(bool(self.log_scan_IntVar.get()))
                Text = Text + 'theta min, theta max={0:.0f},{1:.0f} \n'.format(self.theta_LBound_IntVar.get(), self.theta_UBound_IntVar.get())
                Text = Text + 'Eg min, Eg max={0:.2f},{1:.2f} \n'.format(self.Eg_LBound_DVar.get(), self.Eg_UBound_DVar.get())
                    
                #folder pattern: root/alloy/alloy_id/%Y_%m_%d-%H%M%S-suffix-starteV-endeV-init_run1-init_after_notvalid-propagation
                self.alloy_folder = root + '/' + alloy + '-' + alloy_id
                self.fit_folder = self.alloy_folder + '/' + timestamp + '-' + alloy + '-' + alloy_id + '-' + self.suffix_entry_StrVar.get() + '-' +\
                                  self.Init_type_0.get()[0].capitalize() + self.Init_type_validation.get()[0].capitalize() + self.Init_type_N.get()[0].capitalize()+\
                                  '-' + '{0:.2f}eV_{1:.2f}eV'.format(self.hv_start.get(), self.hv_end.get())
                                  

                if not os.path.exists(self.alloy_folder):
                    os.mkdir(self.alloy_folder)       
                os.mkdir(self.fit_folder)
                
                Text=Text+'Result Folder=' + os.path.abspath(self.fit_folder)

                datafile_name, ext = os.path.basename(self.filepath).split('.')
                shortdatafile_name = alloy + '-' + alloy_id
                filepath = self.fit_folder + '/' + datafile_name + '.' + ext
                shutil.copy(self.filepath, filepath)

                filepath = self.fit_folder + '/' + shortdatafile_name + '-' + self.suffix_entry_StrVar.get() + '.' + iph_functions.FIT_SETTING_EXT
                fobj = open(filepath,'w')
                fobj.write(Text)
                fobj.close()
                
                filepath = self.fit_folder + '/' + shortdatafile_name + '-' + self.suffix_entry_StrVar.get() + '.' + iph_functions.PRM_INIT_EXT
                np.savetxt(filepath, self.prm_init, fmt=['%+.6e', '%d']*4+['%+.1e']*3, delimiter='\t', newline='\n')

                Label(self.win_Fit1 ,text=Text).grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky='w')
                Button(self.win_Fit1, text='OK' ,command=self.on_Run_Fit).grid(row=1 ,column=0, padx=5, pady=5, sticky='nsew')
                Button(self.win_Fit1, text='Cancel', command=self.remove_fit_folder).grid(row=1 ,column=1, padx=5, pady=5, sticky='nsew')

        else:
            if self.flag_prm is False:
                tkMessageBox.showinfo(title='Parameter Values', message='Parameter Values were not set.')
            elif self.flag_data is False:
                tkMessageBox.showinfo(title='Data Values', message='Data Values were not loaded.')

    def remove_fit_folder(self):

        shutil.rmtree(self.fit_folder)
        self.win_Fit1.destroy()

    def on_Run_Fit(self):

        self.remove_fit_lines()
        self.create_fit_lines()
        
        self.plot_ligne_V()
        
        self.win_Fit1.destroy()
        self.queue = get_queue()
        
        self.workers = []

        #mask_no_null = np.where(self.iph_exp_complex_all != (0.0+1j*0.0))
        #mask_no_null_CI = np.where(self.iph_exp_complex_CI_all != (0.0+1j*0.0))

        NelderMead_options = (10**self.NM_log10_xtol_IntVar.get(),\
                              10**self.NM_log10_ftol_IntVar.get(),\
                              self.NM_iteration_IntVar.get(),\
                              self.NM_fcalls_IntVar.get())

        constraints = [( 10**self.K_LBound_IntVar.get(), 10**self.K_UBound_IntVar.get() ),\
                       ( self.theta_LBound_IntVar.get(), self.theta_UBound_IntVar.get() ),\
                       ( self.Eg_LBound_DVar.get(), self.Eg_UBound_DVar.get() ),\
                       bool(self.log_scan_IntVar.get())]

        kwargs = {'output_queue':self.queue ,
                  'name':'',
                  'prm_init':self.prm_init,
                  'nb_run':self.run_per_process,
                  'nb_SC':self.Nb_SC.get(),
                  'init_type':(self.Init_type_0.get(),
                             self.Init_type_N.get(),
                             self.Init_type_validation.get()),
                  'random_loops': self.random_loops_IntVar.get(),
                  'hv':self.hv_all,
                  'iph_exp_complex':self.iph_exp_complex_all,
                  'iph_exp_complex_CI':self.iph_exp_complex_CI_all,
                  'phi_N': self.phi_N,
                  'phi_N_CI':self.phi_N_CI,
                  'weights': self.weights,
                  'hv_limits':(self.hv_start.get(), self.hv_end.get()),
                  'nb_fit_in_run':self.NB_Fit_in_Run.get(),
                  'fit_folder':self.fit_folder,
                  'filepath':self.filepath,
                  'suffix':self.suffix_entry_StrVar.get(),
                  'NelderMead_options' : NelderMead_options,
                  'ParameterConstraints' : constraints,
                  'update_every' : self.update_every_IntVar.get()}
        
        self.workers = initialize_processes(self.cpu_to_use, True, **kwargs)
        self.process_queue()
        self.on_start_workers()


    def remove_fit_lines(self):

        for line in self.Iph_Graph_fit_lines:
            line.remove()
        for line in self.Iph_true_Graph_fit_lines:
            line.remove()
        for line in self.Phi_Graph_fit_lines:
            line.remove()
        for line in self.Evol_Graph_lines:
            line.remove()
        for line in self.Re_Graph_fit_lines:
            line.remove()
        for line in  self.Im_Graph_fit_lines:
            line.remove()

    def create_fit_lines(self):

        self.Iph_Graph_fit_lines = []
        self.Iph_true_Graph_fit_lines = []
        self.Evol_Graph_lines = []
        self.Phi_Graph_fit_lines = []
        self.Re_Graph_fit_lines = []
        self.Im_Graph_fit_lines = []
        
        for i in range(self.cpu_to_use):
            
                line = lines.Line2D(xdata=[], ydata=[], linestyle='-', color=self.process_colors[i])
                self.Iph_Graph_fit_lines.append(line)
                self.Iph_Graph.add_line(line)

                line = lines.Line2D(xdata=[], ydata=[], linestyle='-', color=self.process_colors[i])
                self.Iph_true_Graph_fit_lines.append(line)
                self.Iph_true_Graph.add_line(line)

                
                line = lines.Line2D(xdata=[], ydata=[], linestyle='-' , color=self.process_colors[i])
                self.Phi_Graph_fit_lines.append(line)
                self.Phi_Graph.add_line(line)

                line = lines.Line2D(xdata=[], ydata=[] ,linestyle='-', color = self.process_colors[i])
                self.Re_Graph_fit_lines.append(line)
                self.Re_Graph.add_line(line)

                line = lines.Line2D(xdata=[], ydata=[] ,linestyle='-', color = self.process_colors[i])
                self.Im_Graph_fit_lines.append(line)
                self.Re_Graph.add_line(line)
                
                line = lines.Line2D(xdata=[], ydata=[] ,linestyle='-',\
                                    marker=DISTANCE_MARKER, markersize = MARKER_SIZE ,\
                                    mfc='w', mec=self.process_colors[i] ,\
                                    color = self.process_colors[i])
                self.Evol_Graph_lines.append(line)
                self.Evol_Graph.add_line(line)

        self.update_legend(axes = self.Iph_Graph, run = 1, fit=0)
        self.update_legend(axes = self.Iph_true_Graph, run = 1, fit=0)
            
    def on_start_workers(self):

        start_processes(self.workers)

        self._lock_widgets()

        self.running_flag = True

            
        

    def process_queue(self):
        
        #poll the queue while the stopped_process counter is lower than the number of cpu_to_use
        if self.stopped_process < self.cpu_to_use:
            try:
                name ,run ,fit_nb , data_type ,data = self.queue.get(0)
                if data_type == 'data':
                    self.hv_calc = data[0]
                    self.iph_exp_complex = data[1]
                    if self.fit_type_StrVar.get() == 'Iph':
                        self.iph_calc_complex = data[2]
                        self.iph_calc_complex_true = data[2]/self.phi_N_[self.mask]
                    else:
                        self.iph_calc_complex_true = data[2]
                        self.iph_calc_complex = data[2]*self.phi_N_[self.mask]
                    self.Suivi_fit = data[3]
                    self.plot_fit_lines(name, run, fit_nb)
                elif data_type == 'done':
                    self.stopped_process += 1
                self.master.after(self.polling_time, self.process_queue)
                
            except Queue.Empty:
                #print('Still Polling...')
                self.master.after(self.polling_time, self.process_queue)
        else:
            #print('exiting polling')
            self.stopped_process = 0
            self._unlock_widgets()
            self.running_flag = False

            
            answer = tkMessageBox.askyesno(title='Save Results', message = 'Do you want to keep the results saved in ' + self.fit_folder)
            if answer == False:
               shutil.rmtree(self.fit_folder)
            else:
                plot_process = PlotSummaryProcess(self.fit_folder)
                plot_process.daemon = False
                plot_process.start()
            



    def on_hv_limits(self , *args):

        #check if the lower and upper limits are identical
        #if identical, shift both limits by 0.1 eV.
        if self.hv_start.get() == self.hv_end.get():
            self.hv_start.set(self.hv_start.get() - 0.1)
            self.hv_end.set(self.hv_end.get() + 0.1)

        #swap the limits if the lower limit is greater than the upper limit
        if self.hv_start.get() > self.hv_end.get():
            temp = self.hv_start.get()
            self.hv_start.set(self.hv_end.get())
            self.hv_end.set(temp)

        #check if the lower the limit is lower than the minimal hv value
        if self.hv_start.get() < self.hv_min:
            self.hv_start.set(self.hv_min)

        #check if the upper limit is greater than the maximal hv value
        if self.hv_end.get() > self.hv_max:
            self.hv_end.set(self.hv_max)

        self.Eg_UBound_DVar.set(round(self.hv_end.get(),1))


        self.mask, = np.where((self.hv_all >= self.hv_start.get()) & (self.hv_all <= self.hv_end.get()))
        self.hv = self.hv_all[self.mask]
        self.hv_calc = np.zeros(shape=self.hv_all[self.mask].shape, dtype=np.complex128)
        self.iph_calc_complex_true = np.zeros(shape=self.hv_all[self.mask].shape, dtype=np.complex128)
        self.iph_calc_complex = np.zeros(shape=self.hv_all[self.mask].shape, dtype=np.complex128)
            
        #self._on_weight_type()    
        self.plot_ligne_V()
        self.update_figure()

    def _on_K_limits(self):

        if self.K_LBound_IntVar.get() == self.K_UBound_IntVar.get():
            self.K_LBound_IntVar.set( self.K_LBound_IntVar.get() - 1 )
            self.K_UBound_IntVar.set( self.K_UBound_IntVar.get() + 1  )

        if self.K_LBound_IntVar.get() > self.K_UBound_IntVar.get():
            temp = self.K_LBound_IntVar.get()
            self.K_LBound_IntVar.set( self.K_UBound_IntVar.get())
            self.K_UBound_IntVar.set( temp )

    def _on_theta_limits(self):

        if self.theta_LBound_IntVar.get() == self.theta_UBound_IntVar.get():
            self.theta_LBound_IntVar.set( self.theta_LBound_IntVar.get() - 1 )
            self.theta_UBound_IntVar.set( self.theta_UBound_IntVar.get() + 1  )

        if self.theta_LBound_IntVar.get() > self.theta_UBound_IntVar.get():
            temp = self.theta_LBound.get()
            self.theta_LBound_IntVar.set( self.theta_UBound_IntVar.get())
            self.theta_UBound_IntVar.set( temp )

    def _on_Eg_limits(self):

        if self.Eg_LBound_DVar.get() == self.Eg_UBound_DVar.get():
            self.Eg_LBound_DVar.set( self.Eg_LBound_DVar.get() - 0.1 )
            self.Eg_UBound_DVar.set( self.Eg_UBound_DVar.get() + 0.1  )

        if self.Eg_LBound_DVar.get() > self.Eg_UBound_DVar.get():
            temp = self.Eg_LBound_DVar.get()
            self.Eg_LBound_DVar.set( self.Eg_UBound_DVar.get())
            self.Eg_UBound_DVar.set( temp )


    def _unlock_widgets(self):

        self.prm_button.config(state='normal')
        self.AddFiles_button.config(state='normal')

        self.scan_profile_fast_radbut.config(state='normal')
        self.scan_profile_normal_radbut.config(state='normal')
        self.scan_profile_aggressive_radbut.config(state='normal')

        self.weights_type_1.config(state='normal')
        self.weights_type_invIph.config(state='normal')
        self.weights_type_sigma.config(state='normal')
        self.sigma_entry.config(state='normal')
        self.noise_label.config(state='normal')

        self.fit_type_iph.config(state='normal')
        self.fit_type_iphB.config(state='normal')

        self.linear_none_radbut.config(state='normal')
        self.linear_direct_radbut.config(state='normal')
        self.linear_indirect_radbut.config(state='normal')

        self.Fit_button.config(state='normal')
        self.stop_fit_button.config(state='disabled')

        self.hv_start_Entry.config(state='normal')
        self.hv_end_Entry.config(state='normal')
        self.nb_cpu_to_use_spbox.config(state='normal')
        self.nb_run_per_process_spbox.config(state='normal')
        self.NB_Fit_in_Run_spbox.config(state='normal')
        self.update_every_spbox.config(state='normal')
        
        self.suffix_entry.config(state='normal')
        self.alloy_entry.config(state='normal')
        self.alloy_id_entry.config(state='normal')


    def _lock_widgets(self):

        self.prm_button.config(state='disabled')
        self.AddFiles_button.config(state='disabled')

        self.scan_profile_fast_radbut.config(state='disabled')
        self.scan_profile_normal_radbut.config(state='disabled')
        self.scan_profile_aggressive_radbut.config(state='disabled')

        self.weights_type_1.config(state='disabled')
        self.weights_type_invIph.config(state='disabled')
        self.weights_type_sigma.config(state='disabled')
        self.sigma_entry.config(state='disabled')
        self.noise_label.config(state='disabled')

        self.fit_type_iph.config(state='disabled')
        self.fit_type_iphB.config(state='disabled')

        self.linear_none_radbut.config(state='disabled')
        self.linear_direct_radbut.config(state='disabled')
        self.linear_indirect_radbut.config(state='disabled')
        

        self.Fit_button.config(state='disabled')
        self.stop_fit_button.config(state='active')

        self.hv_start_Entry.config(state='disabled')
        self.hv_end_Entry.config(state='disabled')
        self.nb_cpu_to_use_spbox.config(state='disabled')
        self.nb_run_per_process_spbox.config(state='disabled')
        self.NB_Fit_in_Run_spbox.config(state='disabled')
        self.update_every_spbox.config(state='disabled')

        self.suffix_entry.config(state='disabled')
        self.alloy_entry.config(state='disabled')
        self.alloy_id_entry.config(state='disabled')
        
        
    def prm_binary(self):

        suffix = ''
        for i in range(self.Nb_SC.get()):
            bin_str = ''.join(self.prm_init[i, 1:7:2].astype(np.int32).astype(np.str).tolist())
            suffix = suffix + str(int(bin_str,2))
            
        self.suffix_entry_StrVar.set(suffix)
        self.flag_prm = True 
                
                        
    def AddFiles_cb (self):  
        self.filepath = tkFileDialog.askopenfilename(title='Add exp File',
                                                     defaultextension='.dot' ,
                                                     filetypes=[('dot files','.dot'),
                                                                ('data files', '.data'),
                                                                ('txt files', '.txt'),
                                                                ('all files', '.*')],
                                                     initialdir=self.last_data_folder,
                                                     parent=self)
        
        if os.path.isfile(self.filepath):
            self.filepath = os.path.abspath(self.filepath)
            self.last_data_folder = os.path.dirname(self.filepath)

            try:
                # import in local arrays
                hv, Iph_exp, Iph_exp_N, phase_exp, IphB_exp, flag_iphB = iph_functions.get_exp_data(self.filepath)
                # Explicit memory allocation for global array containing all data
                nb_points = hv.size
               
                self.hv_all = np.zeros(shape=(nb_points,), dtype=np.float64)

                
                self.iph_true_exp_complex_all = np.zeros(shape=(nb_points,), dtype=np.complex128)
                self.iph_trueN_exp_complex_all = np.zeros(shape=(nb_points,), dtype=np.complex128)
                self.iphB_exp_complex_all = np.zeros(shape=(nb_points,), dtype=np.complex128)
                self.iph_exp_complex_all = np.zeros(shape=(nb_points,), dtype=np.complex128)
                self.iph_exp_complex_CI_all = np.zeros(shape=(nb_points,), dtype=np.complex128)
                
                self.phi_N_ = np.zeros(shape=(nb_points), dtype=np.float64)
                self.phi_N = np.zeros(shape=(nb_points), dtype=np.float64)
                self.phi_N_CI = np.zeros(shape=(nb_points), dtype=np.float64)
                self.phi_1 = np.ones(shape=(nb_points), dtype=np.float64)

                self.weights_1 = np.ones(shape=(nb_points), dtype=np.complex128)
                self.weights = np.zeros(shape=(nb_points), dtype=np.complex128)
                self.weights[:] = self.weights_1[:]

                self.hv_all[:] = hv[:]
                self.hv_min = np.min(self.hv_all) 
                self.hv_max = np.max(self.hv_all)
                self.hv_start.set(self.hv_min)
                self.hv_end.set(self.hv_max)
                
                self.mask, = np.where((self.hv_all >= self.hv_start.get()) & (self.hv_all <= self.hv_end.get()))
                self.hv = self.hv_all[self.mask]

                self.hv_start_Entry.config(state='normal')
                self.hv_end_Entry.config(state='normal')
                
                self.iph_true_exp_complex_all[:] = Iph_exp[:]*np.exp(1j*phase_exp[:]*np.pi/180.0)
                self.iph_trueN_exp_complex_all[:] = Iph_exp_N[:]*np.exp(1j*phase_exp[:]*np.pi/180.0)
                self.iphB_exp_complex_all[:] = IphB_exp[:]*np.exp(1j*phase_exp[:]*np.pi/180.0)
                self.iph_exp_complex_all[:] = self.iph_true_exp_complex_all[:]
                self.iph_exp_complex_CI_all[:] = self.iph_true_exp_complex_all[:]

                self.phi_N_[:] = np.absolute(self.iphB_exp_complex_all[:])/np.absolute(self.iph_true_exp_complex_all[:])
                self.phi_N[:] = self.phi_N_[:]
                self.phi_N_CI[:] = self.phi_N_[:]

                self.flag_iphB = flag_iphB
                self.fit_type_StrVar.set('Iph*')
                self.weights_type_StrVar.set(u'1')
                self._on_fit_type()
                self._on_weight_type()

                self.plot_ligne_V()
                self.plot_Graph()
            
                self._unlock_widgets()
                self.flag_data = True

                self.Files_List_Var.set(os.path.basename(self.filepath))
                
            except ValueError:

                self.Files_List_Var.set('Corrupted Data File')
                ext = os.path.basename(self.filepath).split('.')[1]
                tkMessageBox.showerror(title='Import Error', message='Corrupted {0:s} File'.format('*.'+ext))
            
            


    def plot_ligne_V(self):
        self.Lim_Iph.remove()
        self.Lim_Iph_true.remove()
        self.Lim_Phi.remove()
        self.Lim_Re.remove()

        self.Lim_Iph=self.Iph_Graph.axvspan(self.hv_start.get(),self.hv_end.get() ,facecolor='r' ,alpha=0.2)
        self.Lim_Iph_true=self.Iph_true_Graph.axvspan(self.hv_start.get(),self.hv_end.get() ,facecolor='r' ,alpha=0.2)
        self.Lim_Phi=self.Phi_Graph.axvspan(self.hv_start.get(),self.hv_end.get() ,facecolor='r' ,alpha=0.2)
        self.Lim_Re=self.Re_Graph.axvspan(self.hv_start.get(),self.hv_end.get() ,facecolor='r' ,alpha=0.2)

        self.update_figure()


    def _on_profile(self):
        
        if self.scan_profile_IntVar.get() not in [_PROFILE_FAST, _PROFILE_NORMAL, _PROFILE_AGGRESSIVE]:
            self.scan_profile_IntVar.set(_PROFILE_NORMAL)
           
        self.nb_cpu_to_use_IntVar.set(_PROFILE_VALUES['Nb cpu to use'][self.scan_profile_IntVar.get()])
        self.nb_run_per_process_IntVar.set(_PROFILE_VALUES['Nb run per process'][self.scan_profile_IntVar.get()])
        self.NB_Fit_in_Run.set(_PROFILE_VALUES['Nb fit per run'][self.scan_profile_IntVar.get()])
        self.update_every_IntVar.set(_PROFILE_VALUES['Update every'][self.scan_profile_IntVar.get()])

        self.NM_iteration_IntVar.set(_PROFILE_VALUES['NM iterations per prm'][self.scan_profile_IntVar.get()])
        self.NM_fcalls_IntVar.set(_PROFILE_VALUES['NM fcalls per prm'][self.scan_profile_IntVar.get()])
        self.NM_log10_xtol_IntVar.set(_PROFILE_VALUES['NM log10 xtol'][self.scan_profile_IntVar.get()])
        self.NM_log10_ftol_IntVar.set(_PROFILE_VALUES['NM log10 ftol'][self.scan_profile_IntVar.get()])


           

    def _on_linear_preview(self):

        self.n_exponant = self.linear_view_DVar.get()
        
        self.plot_Graph()

    def _on_fit_type(self):

        if self.flag_iphB is False:
            tkMessageBox.showinfo(title='Warning', message='As-measured values are not available.')
            self.fit_type_StrVar.set('Iph*')
            self.weights_type_StrVar.set('1')

        if self.fit_type_StrVar.get() == 'Iph':
            self.iph_exp_complex_all[:] = self.iphB_exp_complex_all[:]
            self.phi_N[:] = self.phi_N_[:]
            self.phi_N_CI[:] = self.phi_N_[:]
            self.Iph_Graph.patch.set_facecolor('w')
            self.Iph_true_Graph.patch.set_facecolor([0.9]*3)
        elif self.fit_type_StrVar.get() == 'Iph*':
            self.iph_exp_complex_all[:] = self.iph_true_exp_complex_all[:]
            self.phi_N[:] = self.phi_1[:]
            self.phi_N_CI[:] = self.phi_1[:]
            self.Iph_Graph.patch.set_facecolor([0.9]*3)
            self.Iph_true_Graph.patch.set_facecolor('w')



        #self.plot_Graph()
        self.plot_Re_Im()
        self.update_figure()

    def _on_weight_type(self, silent=False):

        self.iph_exp_complex_CI_all[:] = self.iph_exp_complex_all[:]

        if self.weights_type_StrVar.get() == '1':
            self.weights[:] = self.weights_1[:]
        elif self.weights_type_StrVar.get() == u'1/|Iph(*)|^2':
            self.weights[:] = 1.0/self.iph_exp_complex_all[:]
        elif self.weights_type_StrVar.get() == 'Abs.Err.':
            self.iph_exp_complex_CI_all[:] = self.iphB_exp_complex_all[:]
            self.phi_N_CI[:] = self.phi_N_[:]
            # search the maximum og the true iph in the domain defined by hv_start and hv_end
            # define a new iph normalized to this new max to be used as weights
            mask_true, = np.where( (self.hv_all >= self.hv_start.get()) & (self.hv_all<=self.hv_end.get()))
            iph_true_max = np.max(np.absolute(self.iph_true_exp_complex_all[mask_true]))
            iph_true = np.absolute(self.iph_true_exp_complex_all)/iph_true_max * np.exp(1j*np.angle(self.iph_true_exp_complex_all, deg=False))
            self.weights[:] = iph_true[:]*1.0/self.sigma_DblVar.get()
        print('phi_N')
        print(self.phi_N[0:5])
        print('phi_N_CI')
        print(self.phi_N_CI[0:5])
        print('Iph')
        print(self.iph_exp_complex_all[0:5])
        print('Iph CI')
        print(self.iph_exp_complex_CI_all[:5])
        print('Weights')
        print(self.weights[0:5])
        print('\n\n')
        if self.fit_type_StrVar.get() == 'Iph*' and self.weights_type_StrVar.get() == 'Abs.Err.':
            if silent is False:
                tkMessageBox.showwarning(title='Warning',
                                  message='The true photocurrent Iph* is fitted through the distance function and the confidence intervals are computed using the chi2 function and the as-measured photocurrent Iph.')

                                  
        #self.plot_Graph()  
        self.plot_Re_Im()

    def _on_sigma_entry(self, *args):
        
        self.weights_type_StrVar.set('Abs.Err.')
        if self.sigma_DblVar.get() == 0.0:
            self.sigma_DblVar.set(1.0)
            tkMessageBox.showerror(title='Value Error', message='The absolute error cannot be equal to 0.0')
        self._on_weight_type()

    def plot_Re_Im(self):

        X = self.hv_all

        self.Re_Graph_line.set_xdata(X)
        self.Re_Graph_line.set_ydata(np.real(self.iph_exp_complex_all))
        
        self.Im_Graph_line.set_xdata(X)
        self.Im_Graph_line.set_ydata(np.imag(self.iph_exp_complex_all))

        for ind, (Re_line, Im_line) in enumerate(izip(self.Re_Graph_fit_lines, self.Im_Graph_fit_lines)):
            if len(Re_line.get_xdata()) > 1:
                    X = self.hv_last[ind]
                    if self.fit_type_StrVar.get() == 'Iph':
                        Im = np.real(self.iph_calc_complex_last_as_measured[ind])
                        Re = np.imag(self.iph_calc_complex_last_as_measured[ind])
                    else:
                        Im = np.real(self.iph_calc_complex_last_true[ind])
                        Re = np.imag(self.iph_calc_complex_last_true[ind])

                    Re_line.set_xdata(X)
                    Re_line.set_ydata(Re)
                    Im_line.set_xdata(X)
                    Im_line.set_ydata(Im)


        self.autoscale(self.Phi_Graph)
        self.autoscale(self.Re_Graph)


            
    def plot_Graph(self):
        
        X = self.hv_all
        Y = np.absolute(self.iphB_exp_complex_all)
        Ylin = np.absolute(self.iph_true_exp_complex_all)
        self.Re_Graph.set_ylabel('Re '+ self.iph_labels[self.fit_type_StrVar.get()] + ' /A, Im ' + self.iph_labels[self.fit_type_StrVar.get()] + ' /A')
        self.Iph_true_Graph.set_ylabel(self.iph_labels[self.fit_type_StrVar.get()] + ' /A')
        self.Iph_Graph.set_ylabel(self.iph_labels[self.fit_type_StrVar.get()] + ' /A')
        if self.n_exponant > 0:
            Ylin = (Ylin*X)**(1.0/self.n_exponant)
            if self.n_exponant == 0.5:
                self.Iph_true_Graph.set_ylabel(r'(' + self.iph_labels[self.fit_type_StrVar.get()] + r' $\cdot h\nu$)$^{2}$ /$A^{2} \cdot eV^{2}$')
            elif self.n_exponant == 2:
                self.Iph_true_Graph.set_ylabel(r'('+ self.iph_labels[self.fit_type_StrVar.get()] + r' $\cdot h\nu$)$^{1/2}$ /$A^{1/2} \cdot eV^{1/2}$')
        
        self.Iph_Graph_line.set_xdata(X)
        self.Iph_Graph_line.set_ydata(Y)

        self.Iph_true_Graph_line.set_xdata(X)
        self.Iph_true_Graph_line.set_ydata(Ylin)
        
        self.Phi_Graph_line.set_xdata(X)
        self.Phi_Graph_line.set_ydata(np.angle(self.iph_exp_complex_all, deg=True))
        
        self.plot_Re_Im()
     
        #self.autoscale(self.Iph_Graph)
        self.Iph_Graph.set_xlim(np.min(X), np.max(X))
        self.Iph_Graph.set_ylim(np.min(Y), np.max(Y))

        self.Iph_true_Graph.set_xlim(np.min(X), np.max(X))
        self.Iph_true_Graph.set_ylim(np.min(Ylin), np.max(Ylin))
        
        self.autoscale(self.Phi_Graph)
        self.autoscale(self.Re_Graph)

                    
          
        for ind, (line, line_true) in enumerate(izip(self.Iph_Graph_fit_lines, self.Iph_true_Graph_fit_lines)):
            if len(line.get_xdata()) > 1:
                    X = self.hv_last[ind]
                    Ylin = np.absolute(self.iph_calc_complex_last_true[ind])
                    Y = np.absolute(self.iph_calc_complex_last_as_measured[ind])
                    if self.n_exponant > 0:
                        Ylin = (Ylin*X)**(1.0/self.n_exponant)
                    line.set_xdata(X)
                    line.set_ydata(Y)
                    line_true.set_xdata(X)
                    line_true.set_ydata(Ylin)

        self.update_figure()

    def plot_fit_lines(self, *args):

    
        name, run, fit_nb = args
        
        process_index = int(name.split(': ')[1]) - 1
        self.hv_last[process_index] = self.hv_calc
        self.iph_calc_complex_last_as_measured[process_index] = self.iph_calc_complex
        self.iph_calc_complex_last_true[process_index] = self.iph_calc_complex_true
        X = self.hv_calc
        Ylin = np.absolute(self.iph_calc_complex_true)
        Y = np.absolute(self.iph_calc_complex)

        if self.n_exponant > 0:
            Ylin = (Ylin*X)**(1.0/self.n_exponant)

        self.Iph_Graph_fit_lines[process_index].set_xdata(X)
        self.Iph_Graph_fit_lines[process_index].set_ydata(Y)

        self.Iph_true_Graph_fit_lines[process_index].set_xdata(X)
        self.Iph_true_Graph_fit_lines[process_index].set_ydata(Ylin)

        
        self.Phi_Graph_fit_lines[process_index].set_xdata(self.hv)
        self.Phi_Graph_fit_lines[process_index].set_ydata(np.angle(self.iph_calc_complex, deg=True))
        
        self.Re_Graph_fit_lines[process_index].set_xdata(self.hv)
        #self.Re_Graph_fit_lines[process_index].set_ydata(np.real(self.iph_calc_complex))
        
        #self.Im_Graph_fit_lines[process_index].set_xdata(self.hv)
        #self.Im_Graph_fit_lines[process_index].set_ydata(np.imag(self.iph_calc_complex))

        self.plot_Re_Im()

        mask,= np.where(self.Suivi_fit[:,0] != 0)
        self.Evol_Graph_lines[process_index].set_xdata(self.Suivi_fit[mask, 0])
        self.Evol_Graph_lines[process_index].set_ydata(self.Suivi_fit[mask, 2])

        self.Evol_Graph.set_xlim(0,self.NB_Fit_in_Run.get())
        self.Evol_Graph.relim()
        self.Evol_Graph.autoscale_view(True,False,True)

        progress = self.get_progress(run, fit_nb)
        lines = self.Iph_Graph.get_lines()[1:]
        label = u'Process {0:d} - Run {1:03d}/{2:03d} - Fit {3:03d}/{4:03d} - Progress {5:03d}%'.format(process_index+1, run, self.run_per_process, fit_nb, self.NB_Fit_in_Run.get(), progress )
        lines[process_index].set_label(label)
        self.Iph_Graph.legend(loc='upper left', fontsize=12, ncol=1)
        
        lines = self.Iph_true_Graph.get_lines()[1:]
        label = u'Process {0:d} - Run {1:03d}/{2:03d} - Fit {3:03d}/{4:03d} - Progress {5:03d}%'.format(process_index+1, run, self.run_per_process, fit_nb, self.NB_Fit_in_Run.get(), progress )
        lines[process_index].set_label(label)
        self.Iph_true_Graph.legend(loc='upper left', fontsize=12, ncol=1)
        self.update_figure()
        
              

    def autoscale(self,chart):
        chart.set_autoscale_on(True)
        chart.relim()
        chart.autoscale_view(True,True,True)
        
        
    def update_figure(self):
        self.canvas.draw()

    def _set_parameters(self):
        
        prm_win = ParameterWindow(self, self.prm_init, self.last_prm_folder)
        self.prm_init = prm_win.get_prm()
        self.prm_filepath, self.last_prm_folder = prm_win.get_paths()
        self.Nb_SC.set(self.prm_init.shape[0])
        self.prm_binary()


    def run(self):
        self.mainloop()

    def ask_quit(self):

        if self.running_flag:
            tkMessageBox.showwarning('Warning', 'You must stop the running fit before closing down the application.')
        else:
            if tkMessageBox.askyesno("Exit", "Do you want to quit the application?"):
                self.master.destroy()



class ParameterWindow(Toplevel):

    def __init__(self, master, prm_init, last_prm_folder):

        Toplevel.__init__(self, master)
        self.transient(master)

        self.master = master
        self.title('Set Parameters')

        self.prm_init = prm_init

        self.grab_set()

        self.initial_focus = self

        self.protocol("WM_DELETE_WINDOW", self._quit)

        ws = self.master.winfo_screenwidth()
        hs = self.master.winfo_screenheight()
        self.width = int(0.6*ws) 
        self.height = int(0.45*hs)
        self.x = (ws/2) - (self.width/2)
        self.y = (hs/2) - (self.height/2)
        self.geometry('{}x{}+{}+{}'.format(self.width, self.height, self.x, self.y))
        self.resizable(height=True, width=True)
        
        self.parameter_frame = ParameterTable(self, self.prm_init, last_prm_folder, scrolled='both')

        self.initial_focus.focus_set()
        self.wait_window(self)


    def get_prm(self):

        return self.parameter_frame.prm

    def get_paths(self):
        return self.parameter_frame.path_prm_file, self.parameter_frame.last_prm_folder

    def _quit(self):

        self.parameter_frame._update_prm_values()
        self.master.focus_set()
        self.destroy()


class ScrolledFrame(Frame):

    def __init__(self, master, **kwargs):

        Frame.__init__(self, master)

        self._default_options = {'scrolled':'y'}

        for i in kwargs.keys():
            if i not in self._default_options.keys():
                raise TclError('Unknow option --' + i)

        self._default_options.update(kwargs) 

        self.pack(expand=True, fill=BOTH)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)

        self.yscrollbar = Scrollbar(self, orient=VERTICAL)
        self.xscrollbar = Scrollbar(self, orient=HORIZONTAL)
        
        if self._default_options['scrolled'] == 'y':
            self.yscrollbar.grid(row=0, column=1, sticky='ns')
        elif self._default_options['scrolled'] == 'x':
            self.xscrollbar.grid(row=1, column=0, sticky='ew')
        elif self._default_options['scrolled'] == 'both':
            self.yscrollbar.grid(row=0, column=1, sticky='ns')
            self.xscrollbar.grid(row=1, column=0, sticky='ew')
        else:
            raise TclError('Bad scroll style \"' + self._default_options['scrolled'] + '\" must be x, y or both')

        self.canvas = Canvas(self, bd=0, relief=FLAT, yscrollcommand=self.yscrollbar.set,
                            xscrollcommand=self.xscrollbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')

        self.yscrollbar.config(command=self.canvas.yview)
        self.xscrollbar.config(command=self.canvas.xview)
        
        self.canvas.config(scrollregion=self.canvas.bbox(all))

        self.frame = Frame(self.canvas)
        self.pack(expand=True, fill=BOTH)

        self.canvas_window_id = self.canvas.create_window(0,0, window=self.frame, anchor='nw')
        self.canvas.itemconfig(self.canvas_window_id, width=self.frame.winfo_reqwidth())
        self.canvas.bind("<Configure>", self._update_canvas_window_size)

    def _update_canvas_window_size(self, event):
        if event.width <= self.frame.winfo_reqwidth():
            self.canvas.itemconfig(self.canvas_window_id, width=self.frame.winfo_reqwidth())
        else:
            self.canvas.itemconfig(self.canvas_window_id, width=event.width)

        if event.height <= self.frame.winfo_reqheight():
            self.canvas.itemconfig(self.canvas_window_id, height=self.frame.winfo_reqheight())
        else:
            self.canvas.itemconfig(self.canvas_window_id, height=event.height)
        
        self._update_canvas_bbox()

    def _update_canvas_bbox(self):    
        self.canvas.config(scrollregion=self.canvas.bbox(ALL))


        

class ParameterTable(ScrolledFrame):

    def __init__(self, master, prm, last_prm_folder, **kwargs):

        ScrolledFrame.__init__(self, master, **kwargs)
        self.pack(expand=True, fill=BOTH)

        self.path_prm_file = ''
        self.last_prm_folder = last_prm_folder

        self.prm = prm
        self.default_prm_values = np.asarray([[0.0,1,
                                              0.0,1,
                                              0.0,1,
                                              2,1,
                                              0,0,0]], dtype=iph_functions._FLOAT)


        self.imported_prm = prm
        
        self.row, self.col = self.prm.shape[0], self.prm.shape[1]-3
        self.header_row = 1
        self.all_button_row = self.header_row + 1
        self.start_row = self.all_button_row + 1

        self.prm_wdg = np.empty(shape=(self.row, self.col), dtype=np.object)
        self.prm_wdg_line = np.empty(shape=(1, self.col), dtype=np.object)
        
        self.prm_wdg_vbl = np.empty(shape=(self.row, self.col), dtype=np.object)
        self.prm_wdg_line_vbl = np.empty(shape=(1, self.col), dtype=np.object)

        # All rows have no weight --> minimize space
        for i in range(self.start_row+self.row):
            Grid.rowconfigure(self.frame, i, weight=0)

        # K, theta, Eg have a weight of 1 for automatic resizing
        for j in range(0,6,2):
            Grid.columnconfigure(self.frame, j, weight=3)
        # K?, theta?, Eg? have a weight of 0 for automatic resizing
        for j in range(1,7,2):
            Grid.columnconfigure(self.frame, j, weight=1)
        # n, powhv have no weight
        for j in [6,7]:
            Grid.columnconfigure(self.frame, j, weight=1)

        self.header = ['K', 'K?',
                        'theta', 'theta?',
                        'Eg', 'Eg?',
                        'n','powhv']
        for j in range(self.col):
            Label(self.frame, text=self.header[j] ).grid(row=self.header_row, column=j, sticky='nswe')

        self._create_widgets()
        self._grid_widgets()

        Button(self.frame, text='+1 SC', command=self._add_line).grid(row=0,
                                                              column=0,
                                                              sticky='nswe',
                                                              pady=5)
        Button(self.frame, text='-1 SC', command=self._delete_line).grid(row=0,
                                                                 column=2,
                                                                 sticky='nswe',
                                                                 pady=5)

        Button(self.frame, text='Import Parameter File', command=self._import_prm_file).grid(row=0,
                                                                                       column=4,
                                                                                       sticky='nswe',
                                                                                       pady=5)
        self.check_all_K_vbl = IntVar()
        self.check_all_K_vbl.set(1)
        self.check_all_theta_vbl = IntVar()
        self.check_all_theta_vbl.set(1)
        self.check_all_Eg_vbl = IntVar()
        self.check_all_Eg_vbl.set(1)

        self.check_all_K_button = Checkbutton(self.frame, text='All', variable=self.check_all_K_vbl,
                                              command=self._on_check_all_K)
        self.check_all_K_button.grid(row=self.all_button_row, column=1, sticky='nswe')
        self.check_all_theta_button = Checkbutton(self.frame, text='All', variable=self.check_all_theta_vbl,
                                                    command=self._on_check_all_theta)
        self.check_all_theta_button.grid(row=self.all_button_row, column=3, sticky='nswe')
        self.check_all_Eg_button = Checkbutton(self.frame, text='All', variable=self.check_all_Eg_vbl,
                                                command=self._on_check_all_Eg)
        self.check_all_Eg_button.grid(row=self.all_button_row, column=5, sticky='nswe')

                                              
    def _create_widget_line(self):

        for j in range(0, self.col-2, 2):
            self.prm_wdg_line_vbl[0,j] = DoubleVar()
            self.prm_wdg_line[0,j] = Entry(self.frame, textvariable=self.prm_wdg_line_vbl[0,j])
         
        for j in range(1, self.col-1, 2):
            self.prm_wdg_line_vbl[0,j] = IntVar()
            self.prm_wdg_line[0,j] = Checkbutton(self.frame, text=' ', variable=self.prm_wdg_line_vbl[0,j])

        self.prm_wdg_line_vbl[0,6] = DoubleVar()
        frame = Frame(self.frame)
        Radiobutton(frame, text='Dir. Trans.', variable=self.prm_wdg_line_vbl[0,6], value=0.5).pack(side=LEFT, expand=True, fill=BOTH) 
        Radiobutton(frame, text='Ind. Trans.', variable=self.prm_wdg_line_vbl[0,6], value=2.0).pack(side=LEFT, expand=True, fill=BOTH) 
        self.prm_wdg_line[0,6] = frame

        self.prm_wdg_line_vbl[0,7] = IntVar()
        frame = Frame(self.frame)
        Radiobutton(frame, text='0',variable=self.prm_wdg_line_vbl[0,7], value=0).pack(side=LEFT, expand=True, fill=BOTH) 
        Radiobutton(frame, text='1', variable=self.prm_wdg_line_vbl[0,7], value=1).pack(side=LEFT, expand=True, fill=BOTH) 
        self.prm_wdg_line[0,7] = frame 


    def _create_widgets(self):
        
        for i in range(0, self.row):
            self._create_widget_line()
            self.prm_wdg[i,:] = self.prm_wdg_line[0,:]
            self.prm_wdg_vbl[i,:] = self.prm_wdg_line_vbl[0,:]


    def _update_widget_values(self):

        for i,j in np.ndindex(self.prm_wdg.shape):
            if j in [1, 3, 5, 7]:
                self.prm_wdg_vbl[i,j].set(int(self.prm[i,j]))
            else:
                self.prm_wdg_vbl[i,j].set(self.prm[i,j])



    def _update_prm_values(self):

        for i,j in np.ndindex(self.prm_wdg.shape):
            self.prm[i,j] = self.prm_wdg_vbl[i,j].get()


    def _grid_widgets(self):
        self._update_widget_values()
        for i,j in np.ndindex(self.prm_wdg.shape):
            self.prm_wdg[i,j].grid(row=self.start_row+i, column=j, sticky='nswe')


    def _add_line(self):
        
        self.row += 1

        self._create_widget_line()
        self.prm_wdg = np.append(self.prm_wdg, self.prm_wdg_line, axis=0)
        self.prm_wdg_vbl = np.append(self.prm_wdg_vbl, self.prm_wdg_line_vbl, axis=0)

        self.prm = np.append(self.prm, self.default_prm_values, axis=0)
        self._grid_widgets()

    
    def _delete_line(self):
        
        if self.prm_wdg.shape[0] > 1:
            self.row -= 1

            for j in range(self.col):
                self.prm_wdg[-1, j].destroy()
                self.prm_wdg[-1, j] = np.empty
            self.prm_wdg = np.delete(self.prm_wdg, -1, axis=0)
            self.prm_wdg_vbl = np.delete(self.prm_wdg_vbl, -1, axis=0)
            self._grid_widgets()

            self.prm = np.delete(self.prm, -1, axis=0)

        self._update_widget_values()

    def _copy_imported_prm(self):
        
        row_old, col_old = self.prm.shape
        row_new, col_new = self.imported_prm.shape

        d_row = row_new - row_old
        
        if d_row != 0:
            sign = d_row/abs(d_row)

            for i in range(abs(d_row)):
                if sign < 0:
                    self._delete_line()
                else:
                    self._add_line()

        self.prm[:,0:8] = self.imported_prm[:,0:8]

        self._update_widget_values()


    def _import_prm_file(self):
        File_PRM =  tkFileDialog.askopenfilename(title="Choose your parameters file",\
                                                 defaultextension='.'+iph_functions.PRM_INIT_EXT,\
                                                 filetypes=[('prm init', '.'+iph_functions.PRM_INIT_EXT),\
                                                            ('prm min', '.'+iph_functions.PRM_MIN_EXT),\
                                                            ('prm end', '.'+iph_functions.PRM_END_EXT),\
                                                            ('All files', '.txt')],\
                                                 initialdir=self.last_prm_folder,\
                                                 parent=self)
        if os.path.isfile(File_PRM):
            self.path_prm_file = os.path.abspath(File_PRM)
            self.last_prm_folder = os.path.dirname(self.path_prm_file)

            try:
                self.imported_prm = iph_functions.import_prm_file(self.path_prm_file)
                self._copy_imported_prm()

            except ValueError:
                print('Incorrect PRM file')

            except IOError:
                print('Importing canceled')


    def _on_check_all_K(self):

        for i in range(self.row):
            self.prm_wdg_vbl[i, 1].set(self.check_all_K_vbl.get())

    def _on_check_all_theta(self):

        for i in range(self.row):
            self.prm_wdg_vbl[i, 3].set(self.check_all_theta_vbl.get())

    def _on_check_all_Eg(self):

        for i in range(self.row):
            self.prm_wdg_vbl[i, 5].set(self.check_all_Eg_vbl.get())

if __name__=='__main__':
    if sys.platform.startswith('win'):
        multiprocessing.freeze_support()
    root = Tk()
    app = Analyse_PEC(master=root)
    app.run()






