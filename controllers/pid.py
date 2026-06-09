import numpy as np
import matplotlib.pyplot as plt

class PID:
    # define a PID controller class with methods for P, I, D terms and a method to compute the control output given the current error.
    # torque(e,theta)= K_p*e + K_i*integrator + K_d*de/dt + torque_ff(theta)
    # where:
    # e = theta_d - theta (error)
    # integrator += e * dt (integral/accumulated error)
    # de/dt = rate of change of error, (D often also can be -K_d*dtheta/dt)
    # saturation array tracks saturation, needs an array of torque limits for motors to call calc torque

    # how to use
    # 1. initialize with gains and initial theta
    # 2. at each timestep, update theta and compute torque output using calc_torque
    # 3. calc error based on desired theta_d and current theta, update integrator, compute P, I, D terms, and sum for total torque output
    # 4. apply torque to the arm, get new theta from the arm dynamics, and repeat

    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integrator = np.zeros(12)  # assuming 12 joints for the quadruped
        self.error = np.zeros(12)  # assuming 12 joints for the quadruped
        self.theta = np.zeros(12)  # current joint angles
        self.last_theta = np.zeros(12)  # to compute derivative term, assuming 12 joints
        self.saturation = np.zeros((12) , dtype = bool)  # array to track saturation of motor
        self.torques = np.zeros(12)  # array to store computed torques for each joint

    def update_thetas(self, current_angles):
        self.last_theta = self.theta
        self.theta = current_angles.copy()
        return self.theta
    
    def update_errors(self, current_angles, desired_angles):
        self.last_error = self.error
        self.error = desired_angles - current_angles
        return self.error
    
    def calc_P(self):
        return self.kp * self.error
    
    def calc_I(self, saturation):
        self.integrator[~saturation] += self.error[~saturation] * self.dt
        return self.ki * self.integrator
       
    def calc_D(self):
        # using dtheta/dt, will add filtering later
        if self.last_theta is None:
            return 0.0
        self.theta_dot = (self.theta - self.last_theta) / self.dt
        return -self.kd * self.theta_dot
        
    def calc_torque(self, ff, torque_limits):
    # full provisional command, integrator NOT yet updated
        u = self.calc_P() + self.ki * self.integrator + self.calc_D() + ff
        u_applied = np.clip(u, -torque_limits, torque_limits)   # what will actually go to ctrl
        saturated = u != u_applied                               # detect from the real clip
        winding_up = np.sign(self.error) == np.sign(u)           # error pushing further into the rail
        freeze = saturated & winding_up
        # recompute with the updated integrator and return the clipped command
        u = self.calc_P() + self.calc_I(freeze) + self.calc_D() + ff
        self.torques = np.clip(u, -torque_limits, torque_limits)
        return self.torques


    


        


 