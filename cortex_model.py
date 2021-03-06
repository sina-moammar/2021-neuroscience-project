import numpy as np
import networkit as nk
from networkit.graph import Graph
from numpy import float32 as float32, float64 as float64
from typing import List
from typing_extensions import Literal


class cortex_model:
    MODE = Literal['mask', 'full']
    
    def __init__(self, graph: Graph, inh_per: float, v_th: float32, v_reset: float32,
                v_rev: float32, t_m: float32, t_ref: float32, t_delay: float32,
                t_stdp: float32, theta_stdp: float32, g_c: float32, g_levels: int = 1,
                mode: MODE = 'full') -> None:
        """Constructs a model of braincortex with STDP

        Args:
            graph (Graph): network of neurons connections
            inh_per (float): percentage of inhibitory neurons
            v_th (float32): threshold potential for firing
            v_reset (float32): reset potential after firing
            v_rev (float32): reversal potential of ion channel in dendrites
            t_m (float32): timescale of potentials' dynamics
            t_ref (float32): refractory period. Should be multiple of t_delay
            t_delay (float32): travel-time delay from pre- to post-synaptic neurons
            t_stdp (float32): time constant of spikes tracing
            theta_stdp (float32): threshold for synaptic plasticity
            g_c (float32): maximum synaptic conductances
            mode (MODE, optional): how to count arrived spikes matrix. Defaults to 'full'.
            g_levels (int, optional): lambda(learning rate) = 1 / g_levels. Defaults to 1.
        """
        
        if mode == 'mask':
            assert g_levels == 1, "Only `g_levels = 1` is supported in `mask` mode"
        
        self.graph = graph
        self.v_th = v_th
        self.v_reset = v_reset
        self.v_rev = v_rev
        self.t_m = t_m
        self.t_ref = t_ref
        self.t_delay = t_delay
        self.t_stdp = t_stdp
        self.theta_stdp = theta_stdp
        self.g_c = np.float32(g_c / g_levels)
        self.g_levels = g_levels
        self.is_full_model = (mode == 'full')
        
        # number of neurons
        self.size = graph.numberOfNodes()
        # relative conductance matrix (g_ij / g_c)
        if self.is_full_model:
            self.g_s = nk.algebraic.adjacencyMatrix(self.graph, matrixType='dense').astype(np.int16) * g_levels
        # potentials of neurons
        self.v_s = np.zeros(self.size, np.float32)
        # which neuron have been fired last step
        self.is_fired = np.zeros(self.size, bool)
        self.fired_neurons = np.array([])
        # which neurons are in refractory period
        self.is_in_ref = np.zeros(self.size, bool)
        # steps since last spike for checking refractory period
        self.steps_after_spike = np.zeros(self.size, np.uint16)
        # list of post synaptics neurons of every neuron
        self.post_syn_neurons = {}
        # which neurons are inhibitory
        self.is_inh = np.random.choice((True, False), self.size, p=(inh_per, 1 - inh_per))
        # total spikes arrived from fired pre-synaptic neurons
        self.pre_syn_spikes = np.zeros(self.size, dtype=np.int16)
        
        # v_n = v_{n - 1} * exp(-t_delay / t_m)
        self.v_exp_step = np.exp(-self.t_delay / self.t_m)
        self.x_exp_step = np.exp(-self.t_delay / self.t_stdp)
        # refractory time in t_delay timescale
        self.ref_steps = np.uint16(self.t_ref / self.t_delay)
        # neurons' number
        self.neurons = np.arange(self.size, dtype=np.uint16)
        
        
    def restart(self, abv_th_per: float = 0.02) -> None:
        """restarts the system for new neurons' dynamics

        Args:
            abv_th_per (float, optional): percentage of neurons which should set above v_th.
                                          Defaults to 0.02.
        """
        
        # assigning random potentials to neurons
        self.v_s = np.random.rand(self.size).astype(np.float32) * self.v_th
        
        # set `abv_th_per` percentage of neurons fired
        self.is_fired = np.random.choice((True, False), self.size, p=(abv_th_per, 1 - abv_th_per))
        self.fired_neurons = self.neurons[self.is_fired]
        
        self.is_in_ref[:] = False
        self.is_in_ref[self.is_fired] = True
        
        self.steps_after_spike[:] = 0
    
    
    def get_post_syn_neurons(self, neuron: int) -> List[np.uint16]:
        """find post-synaptic neurons of given neuron.

        Args:
            neuron (int): neuron number

        Returns:
            List[np.uint16]: list of post-synaptic neurons of given neuron
        """
        
        if self.post_syn_neurons.get(neuron) is None:
            self.post_syn_neurons[neuron] = np.fromiter(self.graph.iterNeighbors(neuron), np.uint16)
            
        return self.post_syn_neurons[neuron]
        
        
    def neurons_dynamics(self) -> None:
        """dynamics of neurons' potential
        """
        
        # restet pre-synaptic spikes count
        self.pre_syn_spikes[:] = 0
        
        self.steps_after_spike[self.is_in_ref] += 1
        # find which neurons refractory period is ended
        is_ref_ended = (self.steps_after_spike == self.ref_steps)
        self.is_in_ref[is_ref_ended] = False
        self.steps_after_spike[is_ref_ended] = 0
        
        # evolve neurons' potential and set neurons which their refractory period is ended
        # to v_reset. Do not reset all neurons in refractory period because after spikes
        # their potentials will change and need to reset them again.
        self.v_s *= self.v_exp_step
        self.v_s[is_ref_ended] = self.v_reset
        
        # count relative spikes for every post-synaptic neurons and update their potentials
        for neuron in self.fired_neurons:
            if self.is_full_model:
                if self.is_inh[neuron]:
                    self.pre_syn_spikes -= self.g_s[neuron]
                else:
                    self.pre_syn_spikes += self.g_s[neuron]
            else:
                post_syn_neurons = self.get_post_syn_neurons(neuron)
                self.pre_syn_spikes[post_syn_neurons] += -1 if self.is_inh[neuron] else +1
        self.v_s += (self.v_rev - self.v_s) * (self.g_c * self.pre_syn_spikes)
        
        # set neurons in refractory period to v_reset
        self.v_s[self.is_in_ref] = self.v_reset
        
        # find new fired neurons
        self.is_fired = self.v_s >= self.v_th
        self.fired_neurons = self.neurons[self.is_fired]
        self.is_in_ref[self.fired_neurons] = True
        
    
    def STDP_dynamics(self, t_sim: int) -> None:
        """STDP update rule

        Args:
            t_sim (int): max number of steps for simulation of neurons
        """

        # choose random neurons indices = [i(post-synaptic), j(pre-synaptic)] to pick c_ij
        indices = np.random.randint(low=0, high=self.size, size=2, dtype=np.uint16)
        # initialize tracing variables
        c_ij = 0
        x_s = np.zeros(2, dtype=np.float32)
        n_s = np.zeros(2, dtype=np.uint16)

        step = 0
        # check if the activity is stopped or reaches to t_sim
        while step < t_sim and len(self.fired_neurons):
            # evolve neurons' potentials
            self.neurons_dynamics()
            
            # check if pre- or post- synaptic neurons are fired and update their tracing variables
            are_fired = self.is_fired[indices]
            x_s[are_fired] += 1
            n_s[are_fired] += 1
            
            # update c_ij
            if are_fired[0]:
                c_ij += x_s[1]
            if are_fired[1]:
                c_ij -= x_s[0]
                
            x_s *= self.x_exp_step
            step += 1
                
        # updates topology
        n_sum = np.sum(n_s)
        if n_sum:
            ratio = c_ij / n_sum
            if ratio >= self.theta_stdp:
                if self.is_full_model:
                    self.g_s[indices[1], indices[0]] = self.g_levels
                elif not self.graph.hasEdge(indices[1], indices[0]):
                    self.graph.addEdge(indices[1], indices[0])
                    self.post_syn_neurons[indices[1]] = None
            else:
                if self.is_full_model:
                    self.g_s[indices[1], indices[0]] = 0
                elif self.graph.hasEdge(indices[1], indices[0]):
                    self.graph.removeEdge(indices[1], indices[0])
                    self.post_syn_neurons[indices[1]] = None
    
    
    def c_syn(self) -> float64:
        """compute order parameter C_syn

        Returns:
            float64: order parameter C_syn
        """
        
        return np.square(np.mean(self.is_fired))
