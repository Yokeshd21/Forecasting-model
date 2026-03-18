import numpy as np
import do_mpc
from casadi import *

def setup_hvac_model():
    model_type = 'continuous' # or 'discrete'
    model = do_mpc.model.Model(model_type)
    
    # States
    T_room = model.set_variable(var_type='_x', var_name='T_room', shape=(1,1))
    
    # Inputs (Control Variables)
    T_supply = model.set_variable(var_type='_u', var_name='T_supply', shape=(1,1))
    Flowrate = model.set_variable(var_type='_u', var_name='Flowrate', shape=(1,1))
    
    # Disturbances
    T_ambient = model.set_variable(var_type='_tvp', var_name='T_ambient')
    
    # System Dynamics (Simplified Heat Balance)
    # dT/dt = (1/C) * [ UA*(T_amb - T_room) + m_dot*Cp*(T_supply - T_room) ]
    # Normalized parameters for demo
    UA = 0.5 
    C = 10.0
    Cp = 1.0
    
    dT_dt = (1/C) * (UA * (T_ambient - T_room) + Flowrate * Cp * (T_supply - T_room))
    model.set_rhs('T_room', dT_dt)
    
    # Model Setup
    model.setup()
    return model

def setup_mpc(model):
    mpc = do_mpc.controller.MPC(model)
    
    setup_settings = {
        'n_horizon': 20,
        't_step': 0.1,
        'store_full_solution': True,
    }
    mpc.set_settings(**setup_settings)
    
    # Objective Function: Minimize energy (Flowrate^2) and stabilize temperature
    mterm = (model.x['T_room'] - 22)**2 # Setpoint 22C
    lterm = (model.x['T_room'] - 22)**2 + 0.1 * model.u['Flowrate']**2
    
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(T_supply=0.1, Flowrate=0.1) # Penalize input changes
    
    # Constraints
    mpc.bounds['lower','_x', 'T_room'] = 20
    mpc.bounds['upper','_x', 'T_room'] = 26
    
    mpc.bounds['lower','_u', 'T_supply'] = 15
    mpc.bounds['upper','_u', 'T_supply'] = 25
    
    mpc.bounds['lower','_u', 'Flowrate'] = 0
    mpc.bounds['upper','_u', 'Flowrate'] = 10
    
    # Time-varying parameters (T_ambient)
    tvp_template = mpc.get_tvp_template()
    def tvp_fun(t_now):
        # Sample ambient temp variation
        tvp_template['_tvp', :, 'T_ambient'] = 30 + 5*np.sin(t_now)
        return tvp_template
    mpc.set_tvp_fun(tvp_fun)
    
    mpc.setup()
    return mpc

if __name__ == "__main__":
    model = setup_hvac_model()
    mpc = setup_mpc(model)
    
    # Initial State
    x0 = np.array([25.0])
    mpc.x0 = x0
    mpc.set_initial_guess()
    
    # Run a simple step
    u0 = mpc.make_step(x0)
    print(f"Optimal Control Inputs: {u0}")
    print("MPC Controller initialized successfully.")
