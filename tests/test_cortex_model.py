import numpy as np
import networkit as nk
from collections import defaultdict
from typing import Dict
import pytest

from cortex_model import cortex_model


def default_params() -> Dict[str, any]:
    size = 4
    graph = nk.graph.Graph(size, directed=True)
    graph.addEdge(0, 1)
    graph.addEdge(0, 2)
    graph.addEdge(1, 3)
    graph.addEdge(2, 3)
    return {
        'graph': graph,
        'inh_per': 0.2,
        'v_th': np.float32(15),
        'v_reset': np.float32(13.5),
        'v_rev': np.float32(33.5),
        't_m': np.float32(30),
        't_ref': np.float32(3),
        't_delay': np.float32(1),
        't_stdp': np.float32(5),
        'theta_stdp': np.float32(0.4),
        'g_c': np.float32(.15)
    }
    

def test_constructor():
    params = default_params()
    graph = params['graph']
    g_levels = np.random.randint(2, 100, 1)

    model = cortex_model(**params, g_levels=g_levels)

    assert model.v_th == params['v_th']
    assert model.v_reset == params['v_reset']
    assert model.v_rev == params['v_rev']
    assert model.t_m == params['t_m']
    assert model.t_ref == params['t_ref']
    assert model.t_delay == params['t_delay']
    assert model.t_stdp == params['t_stdp']
    assert model.theta_stdp == params['theta_stdp']
    assert model.g_c == np.float32(params['g_c'] / g_levels)

    assert model.size == graph.numberOfNodes()
    assert np.array_equal(model.g_s, np.array([[0, 1, 1, 0],
                                               [0, 0, 0, 1],
                                               [0, 0, 0, 1],
                                               [0, 0, 0, 0]]) * g_levels)
    assert np.all(model.v_s == np.zeros(model.size, np.float32))
    assert np.all(model.is_fired == np.zeros(model.size, bool))
    assert np.all(model.fired_neurons == np.array([]))
    assert np.all(model.is_in_ref == np.zeros(model.size, bool))
    assert np.all(model.steps_after_spike == np.zeros(model.size, np.uint16))
    assert model.post_syn_neurons == defaultdict(list)
    assert len(model.is_inh) == model.size

    assert model.v_exp_step == np.exp(-model.t_delay / model.t_m)
    assert model.ref_steps == np.uint16(model.t_ref / model.t_delay)
    assert np.all(model.pre_syn_spikes == np.zeros(model.size))
    assert np.all(model.neurons == np.arange(model.size))
    
    
def test_post_syn_neurons():
    params = default_params()
    
    model = cortex_model(**params)
    
    assert np.array_equal(model.get_post_syn_neurons(0), [1, 2])
    assert np.array_equal(model.get_post_syn_neurons(1), [3])
    assert np.array_equal(model.get_post_syn_neurons(2), [3])
    assert np.array_equal(model.get_post_syn_neurons(3), [])
    
    
def test_neurons_dynamics():
    params = default_params()
    v_exp_step = np.exp(-params['t_delay'] / params['t_m'])

    # full mode
    g_levels = np.random.randint(3, 100, 1)
    model = cortex_model(**params, g_levels=g_levels, mode='full')
    
    model.is_inh[:] = [0, 0, 1, 0]
    v_s_0 = np.array([10, 0.98, 0.95, 0.3], np.float32) * params['v_th']
    model.v_s[:] = v_s_0
    model.g_s[0, [1, 2]] -= 1
    
    # step 1
    model.neurons_dynamics()
    v_s_1 = v_s_0 * v_exp_step
    assert np.array_equal(model.v_s, v_s_1)
    assert np.array_equal(model.is_fired, [1, 0, 0, 0])
    assert np.array_equal(model.fired_neurons, [0])
    assert np.array_equal(model.is_in_ref, [1, 0, 0, 0])
    assert np.array_equal(model.steps_after_spike, [0, 0, 0, 0])
    
    # step 2
    model.neurons_dynamics()
    v_s_2 = v_s_1 * v_exp_step
    v_s_2[0] = params['v_reset']
    v_s_2[1:3] += (params['v_rev'] - v_s_2[1:3]) * np.float32(params['g_c'] / g_levels) * model.g_s[0, [1, 2]]
    assert np.array_equal(model.v_s, v_s_2)
    assert np.array_equal(model.is_fired, [0, 1, 1, 0])
    assert np.array_equal(model.fired_neurons, [1, 2])
    assert np.array_equal(model.is_in_ref, [1, 1, 1, 0])
    assert np.array_equal(model.steps_after_spike, [1, 0, 0, 0])
    
    # step 3
    model.neurons_dynamics()
    v_s_3 = v_s_2 * v_exp_step
    v_s_3[0:3] = params['v_reset']
    assert np.array_equal(model.v_s, v_s_3)
    assert np.array_equal(model.is_fired, [0, 0, 0, 0])
    assert np.array_equal(model.fired_neurons, [])
    assert np.array_equal(model.is_in_ref, [1, 1, 1, 0])
    assert np.array_equal(model.steps_after_spike, [2, 1, 1, 0])
    
    # step 4
    model.neurons_dynamics()
    assert np.array_equal(model.is_in_ref, [0, 1, 1, 0])
    assert np.array_equal(model.steps_after_spike, [0, 2, 2, 0])
    
    
    # mask mode
    g_levels = 2
    with pytest.raises(AssertionError):
        model = cortex_model(**params, g_levels=g_levels, mode='mask')
        
    model = cortex_model(**params, mode='mask')
    
    model.is_inh[:] = [0, 0, 1, 0]
    v_s_0 = np.array([10, 0.98, 0.95, 0.3], np.float32) * params['v_th']
    model.v_s[:] = v_s_0
    
    # step 1
    model.neurons_dynamics()
    v_s_1 = v_s_0 * v_exp_step
    assert np.array_equal(model.v_s, v_s_1)
    assert np.array_equal(model.is_fired, [1, 0, 0, 0])
    assert np.array_equal(model.fired_neurons, [0])
    assert np.array_equal(model.is_in_ref, [1, 0, 0, 0])
    assert np.array_equal(model.steps_after_spike, [0, 0, 0, 0])
    
    # step 2
    model.neurons_dynamics()
    v_s_2 = v_s_1 * v_exp_step
    v_s_2[0] = params['v_reset']
    v_s_2[1:3] += (params['v_rev'] - v_s_2[1:3]) * params['g_c']
    assert np.array_equal(model.v_s, v_s_2)
    assert np.array_equal(model.is_fired, [0, 1, 1, 0])
    assert np.array_equal(model.fired_neurons, [1, 2])
    assert np.array_equal(model.is_in_ref, [1, 1, 1, 0])
    assert np.array_equal(model.steps_after_spike, [1, 0, 0, 0])
    
    # step 3
    model.neurons_dynamics()
    v_s_3 = v_s_2 * v_exp_step
    v_s_3[0:3] = params['v_reset']
    assert np.array_equal(model.v_s, v_s_3)
    assert np.array_equal(model.is_fired, [0, 0, 0, 0])
    assert np.array_equal(model.fired_neurons, [])
    assert np.array_equal(model.is_in_ref, [1, 1, 1, 0])
    assert np.array_equal(model.steps_after_spike, [2, 1, 1, 0])
    
    # step 4
    model.neurons_dynamics()
    assert np.array_equal(model.is_in_ref, [0, 1, 1, 0])
    assert np.array_equal(model.steps_after_spike, [0, 2, 2, 0])


def test_c_syn():
    params = default_params()

    model = cortex_model(**params)
    
    model.is_inh[:] = [0, 0, 1, 0]
    v_s_0 = np.array([10, 0.98, 0.95, 0.3], np.float32) * params['v_th']
    model.v_s[:] = v_s_0
    
    # step 0
    assert model.c_syn() == 0
    
    # step 1
    model.neurons_dynamics()
    assert model.c_syn() == 1 / 16
    
    # step 2
    model.neurons_dynamics()
    assert model.c_syn() == 1 / 4
    
    # step 3
    model.neurons_dynamics()
    assert model.c_syn() == 0
    
    
def test_restart():
    params = default_params()

    model = cortex_model(**params)
    
    model.is_inh[:] = [0, 0, 1, 0]
    v_s_0 = np.array([10, 0.98, 0.95, 0.3], np.float32) * params['v_th']
    model.v_s[:] = v_s_0
    
    model.neurons_dynamics()
    model.restart()
    
    assert np.all((0 <= model.v_s) & (model.v_s < params['v_th']))
    assert np.array_equal(model.neurons[model.is_fired], model.fired_neurons)    
    assert np.array_equal(model.is_in_ref, model.is_fired)
    assert np.array_equiv(model.steps_after_spike, 0)
    
    fired_count = 0
    sample_size = 1000
    abv_th_per = np.random.rand()
    for _ in range(sample_size):
        model.restart(abv_th_per)
        fired_count += len(model.fired_neurons)
    p_fired = fired_count / sample_size / model.size
    # 6 sigma error margin
    error_margin = 6 * np.sqrt(abv_th_per * (1 - abv_th_per) / model.size / sample_size)
    assert (p_fired - error_margin) < abv_th_per < (p_fired + error_margin)
    