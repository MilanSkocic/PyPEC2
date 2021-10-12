r"""DocString"""

import iph_functions
import numpy as np
from multiprocessing import Process, cpu_count, Queue, Event, forking
import os
import sys

def get_queue():

    r"""
    DocString
    """

    return Queue()


def get_cpu_number():

    r"""
    DocString
    """

    return cpu_count()


def initialize_processes(n, daemon=True, **kwargs):

    r"""
    Create N processes as daemons and return them in a list.

    Parameters
    -----------
    name: string
        Name of the process.
    n: int
        Number of processes to be created.
    daemon: bool
        Flag for creating daemon process.
    kwargs: dict
        See :class:`MinimizationProcess`
    """
    
    workers = []
    for i in range(n):
        kwargs['name'] = 'Process: ' + str(i + 1)
        worker = MinimizationProcess(**kwargs)
        worker.daemon = daemon
        workers.append(worker)

    return workers


def start_processes(workers):
    for worker in workers:
        worker.start()

    return 0

class _Popen(forking.Popen):
    def __init__(self, *args, **kw):
        if hasattr(sys, 'frozen'):
            # We have to set original _MEIPASS2 value from sys._MEIPASS
            # to get --onefile mode working.
            # Last character is stripped in C-loader. We have to add
            # '/' or '\\' at the end.
            os.putenv('_MEIPASS2', sys._MEIPASS + os.sep)
        try:
            super(_Popen, self).__init__(*args, **kw)
        finally:
            if hasattr(sys, 'frozen'):
                # On some platforms (e.g. AIX) 'os.unsetenv()' is not
                # available. In those cases we cannot delete the variable
                # but only set it to the empty string. The bootloader
                # can handle this case.
                if hasattr(os, 'unsetenv'):
                    os.unsetenv('_MEIPASS2')
                else:
                    os.putenv('_MEIPASS2', '')
    

class ProcessCompile(Process):

    _Popen = _Popen


class PlotSummaryProcess(ProcessCompile):

    """
    Parameters
    ----------
    fit_folder: str
        Path to the folder where the summary files will be saved.

    Attributes
    -----------
    fit_folder: str
        Path to the folder where the summary files will be saved.
    """
    def __init__(self, fit_folder):
        Process.__init__(self)
        self.fit_folder = fit_folder

    def run(self):
        iph_functions.get_summary(self.fit_folder)
        iph_functions.plot_summary(self.fit_folder)


class MinimizationProcess(ProcessCompile):
    def __init__(self,
                 output_queue,
                 name,
                 prm_init,
                 nb_run,
                 nb_SC,
                 init_type,
                 random_loops,
                 hv,
                 iph_exp_complex,
                 iph_exp_complex_CI,
                 phi_N,
                 phi_N_CI,
                 weights,
                 hv_limits,
                 nb_fit_in_run,
                 fit_folder,
                 filepath,
                 suffix,
                 NelderMead_options=(1e-11, 1e-23, 200, 200),
                 ParameterConstraints=[(1e-12, 1e-1),
                                       (-180, 180),
                                       (0.1, 6.2),
                                       True],
                 update_every=5):

        Process.__init__(self)
        self.exit_event = Event()
        self.name = name
        self.process_id = int(name.split(':')[1])
        self.output_queue = output_queue

        self.prm_array = np.zeros(shape=prm_init.shape, dtype=np.float64)
        self.prm_array_init = np.zeros(shape=prm_init.shape, dtype=np.float64)
        self.PRM_min_fit = np.zeros(shape=prm_init.shape, dtype=np.float64)
        self.PRM_end_fit = np.zeros(shape=prm_init.shape, dtype=np.float64)

        self.prm_array[:] = prm_init[:]
        self.prm_array_init[:] = prm_init[:]
        self.PRM_min_fit[:] = prm_init[:]
        self.PRM_end_fit[:] = prm_init[:]

        self.prm_values_flatten = self.prm_array[:, 0:6:2].flatten()
        self.prm_states_flatten = self.prm_array[:, 1:7:2].flatten()
        self.prm_to_fit_mask, = np.where(self.prm_states_flatten == 1.0)
        self.nb_prm_to_fit, = self.prm_to_fit_mask.shape
        self.xtol, self.ftol, self.iteration_per_prm, self.func_calls_per_prm = NelderMead_options

        self.max_fmin_iteration = self.nb_prm_to_fit * self.iteration_per_prm
        self.max_fmin_func_calls = self.nb_prm_to_fit * self.func_calls_per_prm

        self.nb_run = nb_run
        self.nb_SC = nb_SC
        self.init_type_0, self.init_type_N, self.init_type_validation = init_type
        self.random_loops = random_loops

        self.hv = hv
        self.phi_N = phi_N
        self.phi_N_CI = phi_N_CI
        self.weights = weights
        self.iph_exp_complex = iph_exp_complex
        self.iph_exp_complex_CI = iph_exp_complex_CI
        self.iph_calc_complex = np.zeros(shape=self.iph_exp_complex.shape, dtype=np.complex128)

        hv_start, hv_end = hv_limits
        self.mask, = np.where((self.hv >= hv_start) & (self.hv <= hv_end))
        self.nb_fit_in_run = nb_fit_in_run
        self.fit_folder = fit_folder
        self.filepath = filepath
        self.Suffix = suffix

        self.K_limits, self.theta_limits, self.Eg_limits, self.Ki_log_flag = ParameterConstraints

        self.Dist_min_fit = 0.0
        self.Dist_end_fit = 0.0

        self.update_every = update_every

        self.header2 = 'Nb of minimization \t Valid \t np.log10(D) \t LCC Module \t LCC Phase \t LCC Re \t LCC Im \t ' \
                       + 'slope Modulus \t slope Phase \t slope Re \t slope Im \t ' \
                       + 'intercept Module \t intercept Phase \t intercept Re \t intercept Im \t'
        for i in range(self.nb_SC):
            self.header2 = self.header2 + 'K_' + str(i + 1) + '\t'
            self.header2 = self.header2 + 'Phi_' + str(i + 1) + '\t'
            self.header2 = self.header2 + 'Eg_' + str(i + 1) + '\t'


    def run(self):

        for run in xrange(self.nb_run):

            if run == 0:
                if self.init_type_0 == 'random':
                    self.prm_array = iph_functions._random_scan(hv=self.hv[self.mask], \
                                                                prm_array=self.prm_array, \
                                                                iph_exp_complex=self.iph_exp_complex[self.mask], \
                                                                phi_N = self.phi_N[self.mask],
                                                                weights=self.weights[self.mask],
                                                                loops=self.random_loops, \
                                                                K_bound=self.K_limits, \
                                                                theta_bound=self.theta_limits, \
                                                                Eg_bound=self.Eg_limits, \
                                                                phase_flag=True)
                elif self.init_type_0 == 'user':
                    self.prm_array[:] = self.prm_array_init[:]
            else:
                if self.init_type_N == 'random':
                    self.prm_array = iph_functions.get_random_prm_values(self.prm_array, \
                                                                         K_bound=self.K_limits, \
                                                                         theta_bound=self.theta_limits, \
                                                                         Eg_bound=self.Eg_limits, \
                                                                         phase_flag=True)
                elif self.init_type_N == 'user':
                    self.prm_array[:] = self.prm_array_init[:]
                elif self.init_type_N == 'min':
                    self.prm_array[:] = self.PRM_min_fit[:]
                elif self.init_type_N == 'end':
                    self.prm_array[:] = self.PRM_end_fit[:]

            self.prm_array = iph_functions.sort_prm_Eg(self.prm_array)
            self.iph_calc_complex = iph_functions.get_Iph_calc(self.hv, self.prm_array, self.phi_N)

            self.PRM_min_fit[:] = self.prm_array[:]
            self.PRM_end_fit[:] = self.prm_array[:]

            self.Dist_min_fit = iph_functions.get_distance(self.iph_exp_complex[self.mask],
                                                           self.iph_calc_complex[self.mask])
            #self.Dist_min_fit = iph_functions._get_chi2(self.prm_array[:,0:6:2].flatten(),
            #                                            self.hv,
            #                                            self.prm_array,
            #                                            self.iph_exp_complex,
            #                                            self.phi_N,
            #                                            self.weights)


            row = self.nb_fit_in_run
            col = 15 + len(self.prm_array[:, 0:6:2].flatten())
            self.Suivi_fit = np.zeros(shape=(row, col))

            for fit_in_run in xrange(self.nb_fit_in_run):

                # check if the shutdown signal was sent
                #if shutdown signal is true, break the fit loop
                #otherwise execute the else statement
                if self.exit_event.is_set():
                    break
                else:
                    self.prm_array, distance = iph_functions.minimize(hv=self.hv[self.mask], \
                                                                      iph_exp_complex=self.iph_exp_complex[self.mask], \
                                                                      phi_N=self.phi_N[self.mask],
                                                                      weights=self.weights[self.mask],
                                                                      prm_array=self.prm_array, \
                                                                      Ki_log_flag=self.Ki_log_flag, \
                                                                      maxiter=self.max_fmin_iteration,
                                                                      maxfun=self.max_fmin_func_calls, \
                                                                      xtol=self.xtol, ftol=self.ftol, \
                                                                      full_output=True, retall=False, disp=False,
                                                                      callback=None)

                    self.iph_calc_complex = iph_functions.get_Iph_calc(self.hv, self.prm_array, self.phi_N)

                    LCC_results = iph_functions.get_LCC(self.iph_exp_complex[self.mask],
                                                        self.iph_calc_complex[self.mask])

                    valid = iph_functions.validate_prm(self.prm_array, K_bound=self.K_limits, Eg_bound=self.Eg_limits)
                    self.prm_array = iph_functions.shift_phase(self.prm_array, theta_bound=self.theta_limits)

                    self.Suivi_fit[fit_in_run] = np.hstack((
                        fit_in_run + 1, int(valid), np.log10(distance), LCC_results, self.prm_array[:, 0:6:2].flatten()))

                    self.PRM_end_fit[:] = self.prm_array[:]
                    self.Dist_end_fit = distance

                    #Output for the main process - Queue data
                    if (fit_in_run + 1) % self.update_every == 0:
                        output = [self.name, run + 1, fit_in_run + 1, 'data',
                                  [self.hv[self.mask], self.iph_exp_complex[self.mask],
                                   self.iph_calc_complex[self.mask], self.Suivi_fit]]
                        self.output_queue.put(output)

                    if valid is False:
                        if self.init_type_validation == 'random':
                            self.prm_array = iph_functions.get_random_prm_values(self.prm_array, \
                                                                                 K_bound=self.K_limits, \
                                                                                 theta_bound=self.theta_limits, \
                                                                                 Eg_bound=self.Eg_limits, \
                                                                                 phase_flag=False)

                        elif self.init_type_validation == 'user':
                            self.prm_array[:] = self.prm_array_init[:]

                    elif valid:
                        if (self.Dist_min_fit > distance):
                            self.Dist_min_fit = distance
                            self.PRM_min_fit[:] = self.prm_array[:]



            else:
                # Saving results
                # Executed when no break signal was triggered
                args = (self.hv[self.mask], self.PRM_end_fit, self.iph_exp_complex_CI[self.mask], self.phi_N_CI[self.mask], self.weights[self.mask])
                self.PRM_end_fit[:,8:11] = iph_functions._get_prm_error(iph_functions._get_residuals, iph_functions._EPSILON, *args)
                args = (self.hv[self.mask], self.PRM_min_fit, self.iph_exp_complex_CI[self.mask], self.phi_N_CI[self.mask], self.weights[self.mask])
                self.PRM_min_fit[:,8:11] = iph_functions._get_prm_error(iph_functions._get_residuals, iph_functions._EPSILON, *args)

                iph_functions.save_results(run, self.process_id,\
                             self.fit_folder, self.filepath, self.Suffix, self.hv, self.mask, self.iph_exp_complex, self.phi_N,\
                             self.PRM_min_fit, self.PRM_end_fit, self.Dist_min_fit, self.Dist_end_fit,\
                             self.Suivi_fit, self.header2)
            # When a break signal is triggered the else statement is skipped
            #Save fits results before breaking the run loop
            if self.exit_event.is_set():
                args = (self.hv[self.mask], self.PRM_end_fit, self.iph_exp_complex_CI[self.mask], self.phi_N_CI[self.mask], self.weights[self.mask])
                self.PRM_end_fit[:,8:11] = iph_functions._get_prm_error(iph_functions._get_residuals, iph_functions._EPSILON, *args)
                args = (self.hv[self.mask], self.PRM_min_fit, self.iph_exp_complex_CI[self.mask], self.phi_N_CI[self.mask], self.weights[self.mask])
                self.PRM_min_fit[:,8:11] = iph_functions._get_prm_error(iph_functions._get_residuals, iph_functions._EPSILON, *args)

                iph_functions.save_results(run, self.process_id,\
                             self.fit_folder, self.filepath, self.Suffix, self.hv, self.mask, self.iph_exp_complex, self.phi_N,\
                             self.PRM_min_fit, self.PRM_end_fit, self.Dist_min_fit, self.Dist_end_fit,\
                             self.Suivi_fit, self.header2)
                break

            output = [self.name, run + 1, fit_in_run + 1, 'data',
                      [self.hv[self.mask], self.iph_exp_complex[self.mask], self.iph_calc_complex[self.mask],
                       self.Suivi_fit]]
            self.output_queue.put(output)

        # print 'Process '+ str(self.process_id) + ' done.'
        output = [self.name, -1, -1, 'done', []]
        self.output_queue.put(output)

    def shutdown(self):
        # print(self.name + ' was shutdown.')
        self.exit_event.set()
