import numpy as np
import networkit as nk
from networkit.graph import Graph
from numpy import float32 as float32


class cortex_model:
    def __init__(self, graph: Graph, v_0: float32, v_th: float32, v_reset: float32,
                 v_rev: float32, t_m: float32, t_ref: float32, t_delay: float32,
                 t_stdp: float32, theta_stdp: float32, g_c: float32, Lambda: float32 = 1.) -> None:
        """Constructs a model of braincortex with STDP

        Args:
            graph (Graph): network of neurons connections
            v_0 (float32): rest potential
            v_th (float32): threshold potential for firing
            v_reset (float32): reset potential after firing
            v_rev (float32): reversal potential of ion channel in dendrites
            t_m (float32): timescale of potentials' dynamics
            t_ref (float32): refractory period
            t_delay (float32): travel-time delay from pre- to post-synaptic neurons
            t_stdp (float32): time constant of spikes tracing
            theta_stdp (float32): threshold for synaptic plasticity
            g_c (float32): maximum synaptic conductances
            Lambda (float32, optional): leaning rate. Defaults to 1..
        """
        
        self.graph = graph
        self.v_0 = v_0
        self.v_th = v_th
        self.v_reset = v_reset
        self.v_rev = v_rev
        self.t_m = t_m
        self.t_ref = t_ref
        self.t_delay = t_delay
        self.t_stdp = t_stdp
        self.theta_stdp = theta_stdp
        self.g_c = g_c
        self.Lambda = Lambda
        
        # number of neurons
        size = graph.numberOfNodes()
        # potentials of neurons
        self.v_s = np.zeros(size, np.float32)
        # dummy variable for tracing neurons' spikes
        self.x_s = np.zeros(size, np.float32)
        # number of neurons' spikes
        self.n_s = np.zeros(size, np.int32)
        # which neuron have been fired last step
        self.is_fired = np.zeros(size, bool)
        # which neurons are in refractory period
        self.is_in_ref = np.zeros(size, bool)
        # steps since last spike for checking refractory period
        self.steps_after_spike = np.zeros(size, np.uint16)
        
        
    def restart(self, abv_th_per: float = 0.02):
        """restarts the system for new neurons' dynamics

        Args:
            abv_th_per (float, optional): percentage of neurons which should set above v_th.
                                          Defaults to 0.02.
        """
        
        pass
        
        
    def neurons_dynamics(self) -> None:
        """dynamics of neurons' potential
        """
        
        pass
    
    
    def STDP_dynamics(self, size: int = -1) -> None:
        """STDP update rule

        Args:
            size (int, optional): number of the synapses which should be updated.
                                  `-1` means all the synapses. Defaults to -1.
        """
        
        pass
