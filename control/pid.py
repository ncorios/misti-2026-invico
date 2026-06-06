import numpy as np
import matplotlib.pyplot as plt

class PID:
    # define a PID controller class with methods for P, I, D terms and a method to compute the control output given the current error.
    # torque(e,theta)= K_p*e + K_i*integrator + K_d*de/dt + torque_ff(theta)
    # where:
    # e = theta_d - theta (error)
    # integrator += e * dt (integral/accumulated error)
    # de/dt = rate of change of error, (D often also can be -K_d*dtheta/dt)
    # torque_ff(theta) = torque computed from a model, not inherently standard but complements. pid can usually eliminate error without modeling

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
        self.integrator = 0
        self.theta = None
        self.last_theta = None
        self.error = 0.0

    def update_theta(self, theta):
        self.last_theta = self.theta
        self.theta = theta
        return self.theta
    
    def update_error(self, theta_d):
        self.last_error = self.error # dead but useful if we want to use de/dt instead of -dtheta/dt
        self.error = theta_d - self.theta
        return self.error
    
    def calc_P(self):
        return self.kp * self.error
        
    
    def calc_I(self):
        self.integrator += self.error * self.dt
        self.integrator = np.clip(self.integrator, -2.0, 2.0)
        return self.ki * self.integrator
       
    def calc_D(self):
        # using dtheta/dt, will add filtering later
        if self.last_theta is None:
            return 0.0
        theta_dot = (self.theta - self.last_theta) / self.dt
        return -self.kd * theta_dot
        
    def calc_torque(self, theta_d):
        self.update_error(theta_d)
        return self.calc_P() + self.calc_I() + self.calc_D()

# feed forward control to compliment the PID controller, can be used to compute a torque based on the arm model to help eliminate steady state error and improve response. This is not strictly necessary for a PID controller but can enhance performance if the model is accurate.


    


        


 