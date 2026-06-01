# PID theory notes

# TODO before website writeup: pass through for typos, comment code better, do math for saturation and real world asks of gains for pidf.

Working notes from learning PID 

- [big_picture_formula]

torque(e,theta)= K_p*e + K_i*integrator + K_d*de/dt + torque_ff(theta)

where:
e = theta_d - theta (error)
integrator += e * dt (integral/accumulated error)
de/dt = rate of change of error, (D often also can be -K_d*dtheta/dt)
torque_ff(theta) = torque computed from a model, not inherently standard but complements. pid can usually eliminate error without modeling

- [error]:
    say our system has theta, our current angle, and theta_d, our desired angle. error (e) = desired angle - current angle. e = theta_d - theta. positive error means we under shot, needs to move forward. negative error means we over shot, we need to move back. every controller takes in error as an input to a function and outputs some change in the angle.
- [proportional]
    P term: physical intuition, steady-state error under load

    this is the simplest controller. multiply your error by a constant, that output is your command.

    example: torque = K_p * error (where K_p is a gain, a constant setting how aggresively the controller responds)
    P acts as a spring, pushing harder the larger the error is.

    trouble shooting:
    small K_p takes forever to reach the setpoint. might never reach with other forces
    big K_p causes fast response, but the system never reaches a small enough error to stabilize, it ends up oscillating.

    why is P alone not enough?
    imagine an arm at angle theta, of mass m, distance l with gravity g. torque on the arm is tau = -m g l sintheta, and the desired angle is theta_d = pi/2. for the controller to hold that angle, it must cancel out the gravitational torque at theta_d, so mgl. thus mgl = e * k_p, e = mgl/K_p. error can never be zero, because then torque from controller is zero, and gravity causes the arm to fall. P must hold a permanent error to generate the torque it needs. error must be non zero for torque from the controller to match gravitational torque. this is called STEADY-STATE ERROR. P only controllers can not eliminate steady-state error against constant load. steady state error in this case = mgl/k_p. basically force_on_system/gain, formally disturbance/gain.

    whats the solution?
    cranking k_p shrinks e_ss (steady state error), but can never eliminate it (asympotically), and a large k_p will eventually cause overshooting and oscillations. you cant tune P to eliminate e_ss, you need another term (I)

    thoughts i had:
    (note that i derived gravity compensation independently. thinking about how to improve things is useful for developing intuition here.)

    well then why not make theta_d past pi/2? example theta_d = pi/2 + phi (offset)
    then the error when the arm is at pi/2 = phi. thus phi = mgl/k_p. this works! its actually called feedforward compensation, or gravity compensation in this case. 

    standard way to go about this:
    torque = k_p * e + t_ff (torque feed forward)
    t_ff is a function of theta, modeling gravity
    t_ff = mglsin(theta)

    this works for perfect systems, but mass can change, friction exists, there can be gusts of wind, or any disturbance. then the model breaks. this is standard on arm controllers. feeforward handles what you can model, and PID handle the residuals. because P can never fully eliminate the residuals, this is where I comes in.

- [integral] 
    I term: why it kills steady-state error, windup

    I (integral)
    torque_integral = k_i * integrator
    integrator += e * dt (error * timestep)

    the integral term has memory. it uses accumulated error to increae torque until error is zero. at this point e = 0, the integrator stops increasing and the torque is held constant at theta_d, and in the case of the arm we've been talking about would be equal to graviational torque. so at theta_d in the arm at pi/2 case, torque_p = 0, torque_i = mgl.
    k_i tuning behaves similarly to k_p. small k_i takes forever. large k_i causes oscillations.

    if we have a pretty good model of our system, then p + ff should take care of the majority of output, and then the i term will handle our disturbances. this is because i does not require a model, it uses pure error to calculate output and does not fall into steady state error. the integrator grows until e = 0, at which point the controller "learns" the torque required without any modeling. i alone is unstable. p handles fast transient errors, ff handles modeled dynamics, i handles smaller disturbances, residuals, steady state error, etc.

    the integrator 'discovers' the steady-state torque by accumulation alone. no model required. this is why I works for any constant disturbance without modeling. 

    integral windup:

    I has a problem that formulas dont know: actuator saturation
    motors have limits on torque, when the integrator i is unlimited, the torque accumulated by the I term can grow past this limit. once theta_d is reached, and the p term contributes no torque, the I term produces much larger torque than needed, causing massive overshoot, and eventually oscillations.

    how do we fix this?

    flag on the integral term
    "if not saturated" as a conditon for integration, but you want to allow helpful integration so if theres saturation in the direction opposing error thats fine. 

    pseudo(ish)code:
    if not saturated:
    integrator += e * dt
    else:  # saturated
        if sign(e) != sign(integrator):
            integrator += e * dt   # this would shrink integrator, allow
        # else: skip integration, would make windup worse
    
    back-calculation
    take the lost torque (commanded-applied) and feed it back to the integrator with a negative gain

    integrator claming
    integ = clip(integ,-max,+max) 
    works but u have to tune limits

    the larger lesson is that pid doesnt always carry cleanly into the real world. pid is simple. correct pid isn't.

- [derivative] 
    D term: damping, noise amplification, derivative kick

    2 forms

    error:
    torque_d = k_d * de/dt


    measurement:
    torque_d = -k_d * dtheta/dt
    negative bc error = theta_d - theta, dtheta_d/dt = 0

    de/dt = (e(n) - e(n -1))/dt
    behavior:
    setpoint = theta_d
    setpoint doesnt change: identical
    setpoint changes smoothly: both are fine, but measurement ignores the changes in setpoint
    setpoint change by step: error spikes, measurement is calm

    use -dtheta/dt for hardware applications, avoiding derivative kick (spikes with de/dt)

    what does d do?
    smooths the transient response, reduces overshoot, damps
    doesnt kill steady state error (de/dt = 0)
    d slows approach to setpoint. over tuned k_d makes system genuinely sluggish.
    d doesnt fix noise, it creates noise problems without filtering

    tuning k_d:

    small k_d: system doesnt damp much, still oscillates
    large k_d: system is overdamped and sluggish
    
    D is damping. it pushes against motion.

    D suffers heavily from noise
    check signal-processing.md for noise and more


- [discretization]: 
    continuous → discrete-time form
    formula is continous. integrals, derivatives break with discrete time. 
    
    what do we need to change to make this codeable?

    P: 
    fine, linear.

    I: 
    integrator += error * dt
    tau_I = k_i * integrator
    this is forward euler. fine for now. can do fancier if needed.

    D:
    e_dot = (e - e_prev) / dt
    OR
    tau_D = k_d * e_dot
    theta_dot = (theta - theta_prev)/dt
    tau_D = -k_d * theta_dot

    dt needs to be atleast 10x faster than the fastest dynamic in your system (rule of thumb), but not too large that FE breaks.


- [tuning_intuition_+_Ziegler-Nichols]
    tuning is kinda vibes. theres formulas, but in practice its kinda just guess and adjust.

    classical method:
    Ziegler-Nichols
    1. k_i = 0, k_d = 0, k_p = low
    2. increase k_p until there are constant amplitude oscillations
    3. record this as k_u, ultimate gain, and the period as t_u, ultimate period.
    4. set with these tables:
    | controller | Kp        | Ki              | Kd          |
    |------------|-----------|-----------------|-------------|
    | P only     | 0.5 * Ku  | —               | —           |
    | PI         | 0.45 * Ku | 1.2 * Kp / Tu   | —           |
    | PID        | 0.6 * Ku  | 2 * Kp / Tu     | Kp * Tu / 8 |

    this gives a starting point. assumes first order dynamics + delay. 
    finding K_u can damage hardware lol. 
    works kinda but u can get better results with a simpler method.

    practical method:
     1. k_i = 0, k_d = 0, k_p = low
     2. increase until fast response with maybe 20% overshoot. dont go to instability.
     3. add k_d to kill overshoot. k_p/10, then increase until smooth response.
     4. add k_i to kill steady state error. ki = kp/5, adjust based on how fast you want steady state error to die.

    what to watch:
    1. slow rise time, no overshoot means low k_p
    2. fast rise, big overshoot, decaying oscillations, add k_d
    3. steady state error, need k_i
    4. slow oscillations that take forevver to die, k_i too high (windup/integrator instability)
    5. buzzing/jittering, system runs hot. k_d high, d needs filtering

    also can fix with feedforward, windup mitigation, filtering, etc so dont spend forever tuning if you havent done anything else.


- [my_approach]
    i would prob want to write something like this for the pid-arm
    t = pid + t_ff

    t_ff = torque_feedforward(theta) = mglsin(theta) for arm
    this models basic gravity, makes our controller faster and compliments so I term can eliminate other constant distrubances we cant model without worrying abt gravity
    not much noise so filter kinda just adding lag for no reason in a sim, but ill make it to where it can be turned on if needed when noise injected

    so rought structure ( too lazy to write pseudocode):

    lowpass:
    alpha from cutoff frequency
    current signal, update last signal
    return alpha * signal + (1-alpha) * last signal

    torque feed forward = model

    pid:
    def p (one line)
    def i (integrator, t_i)
    def d (derivative(filtered), t_d)

    useful structure:
    class PID:
    def __init__(self, kp, ki, kd, dt, 
                 tau_max=None,           # actuator limit
                 use_anti_windup=False,
                 d_filter_alpha=1.0):

    whats good here is that i can use those flags off initially to toggle more advanced features, so removing filter from no noise sim for reduced lag, or not worrying abt windup for unlimited torque sim for intitial testing

    main:
    output = pid + tff
    also graph error over time
    