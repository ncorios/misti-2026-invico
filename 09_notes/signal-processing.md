 # Basic signal processing notes. why not.

scope: intuition-level cooverage of signals, noise, frequency-content, and basic filters. enough to make informed decisions with control code.

- [signal]
    what is a signal:

    value that changes over time. any number that depends on time.

    continous vs discrete:

    continous signal:
    defined at every t. real, physical signals. voltage, position, etc.

    discrete signal:
    defined only at specific moments, typically even time steps. ex: sensor reading a value every 1 ms.

    you bridge them through sampling

- [noise]
    measurement is a signal, we will use it as an example but this can be extrapolated.
    
    measurement has some angle theta_true, or theta_t
    theta_measured = theta_m = theta_t + n(t)
    where n(t) is noise, random unwanted signals
    shows up as a high frequency jitter on graphs

    contextualizing with D in PID:

    differentiation amplifies this noise heavily:
    dtheta = (theta(t) - theta(t-1))/dt
    say theta_t = pi/2, but there is noise. a jitter of +- 0.001 rad on a super small timestep, say a millisecond.
    then dtheta = +- 1 rad/s, a huge number compared to a tiny jitter
    multiplied by k_d, this would be a huge thrashing oscillating torque output

    on real hardware, this needs to be filtered.


- [frequency-content]
    every signal is a mix of oscillations at different speeds.

    every signal can be decomposed as a sum of sine waves. fourier theorem. (obv theres nuance tho)

    any signal has a frequency content, a "recipe" for which sin waves you need to reconstruct it. that is called a signal's spectrum/

    a signal in the time domain can be converted to the frequency domain, with how much of each frequency it contains. 

    time domain: time, signal

    frequency domain: frequency, amplitude. 

    complex signals can be decomposed into spikes at diff frequencies.

    filtering becomes obvious in the frequency domain.

    simple intuition:

    fast chnage = high frequency

    slow change = low frequency

    use that to think about what we need to filter out.

    connecting to our pid-arm:

    arms angle has low frequency real movement over fractions of a second. maybe 1-20 Hz

    it also has ultra high frequency jitters from noise. 100s-1000s of Hz

    how do we fix this from messing with our controller?

    low pass filter.

- [filters]
    noisy signal in, cleaner signal out.

    some frequencies get amplified, killed, etc

    which frequencies define the filter

    four basic types:

    1. low-pass filter (LPF)
        passes low, blocks high
        use case: removing noise from slowly changing signal
    2. high-pass filter (HPF)
        passes high, blocks low
        use case: removing slow drift, DC offset, wander
    3. band-pass filter (BPF)
        passes a band of frequencies, passes above and below
        use case: isolating a specific range, known frequencies

    4. band-stop filter (BSF)
        blocks a band, passes everything else
        use case: killing a specific intefering frequence (ex: 60 Hz power-line hum in audio)

    visualization (thanks claude):

    low-pass:     ▔▔▔▔▔▔▏______
              0 →  fc  → ∞ Hz

    high-pass:    _____▕▔▔▔▔▔▔
                0 → fc → ∞ Hz

    band-pass:    _____▕▔▔▔▔▏_____
                0  flow  fhigh  ∞ Hz

    band-stop:    ▔▔▔▔▏_____▕▔▔▔▔
                0  flow  fhigh  ∞ Hz

    real filters dont have hard cutoff like this tho

    you can kinda build everything from a LPF

    HPF = signal - LPF output
    BPF = HPF output with LPF applied
    BSF = signal - BPF output

    two key parameters:

    1. cutoff frequencies
        - boundaries. two for BPF,BSF, one for LPF,HPF
    2. roll-off/sharpness
        - how abruptly the filter transitions from passing to blocking. increases with order.

    tradeoff:

    gentle filters (first-order, gradual roll-off)
    low lag, blocks gradually 

    sharp filter (high-order, fast roll-off)
    blocks cleanly, significant lag, may ring/oscillate

    all filters have lag. contextualizing, excessive lag can destabilize PID loops.

    vocab:

    linear vs nonlinear

    linear filters: output is a linear combination of inputs. low-pass, high-pass, etc. all linear.

    nonlinear filters: median filter, max filter, etc. used in image processing, not control.  

    IIR vs FIR

    IIR (infinite impulse response):
    depends on previous outputs. efficient, low computation, harder to design carefully.
    FIR (finite impulse response):
    filter has only feedforward, depends only on input history.

    for control, linear IIR filters are pretty standard.

    limitations:

    cant recover destroyed information

    cant tell signal from noise without assumptions

    filters always introduce lag

    aggressive filters cause more lag, can destabilize loops

this is all interesting, not incredibly useful right now. lets deep dive on low pass, then leave more signal processing to other projects.

- [LPF]
    assuming our true signal is smooth and slowly varying, there is rapid oscilating noise.

    solution to removing noise is a low pass filter. let low frequencies pass through, remove high frequencies

    assumption: real signal slow, noise fast. perfect filter would leave the true signal, we approximate with some lag

    with random noise, mean is roughly 0 so averaging out the noise takes it out. take the last N samples, average them. thats your filtered value. most basic filter.

    low pass is this but doesnt remember all samples, just the last N amount and blends with each new sample. old samples decay exponentially.

    new_output = (a little bit of new sample) + (a lot of old output)

    y[k] = alpha * x[k] + (1 - alpha) * y[k-1]

    y[k] = filtered signal
    alpha = smoothing coefficient, 0-1. requires tuning
    x[k] = raw signal
    y[k-1] = filtered signal at previous timestep

    output becomes a weighted sum of all past inputs, with weight alpha(1-alpha)^n. 

    alpha:
    alpha = 1, no filtering
    alpha = 0.5, half new, half old. light filtering
    alpha = 0.1, heavy weight on memory, strong filtering. smooth output
    alpha = 0.01, extreme filtering. output barely moves
    alpha = 0, dead filter. output never changes

    small alpha = smpooth output, lots of lag, lots of memory
    large alpha = responsive output, less memorry, less noise rejection

    how to actually find alpha:

    real formula: alpha = (2*pi*f_c*dt)/(1+2*pi*f_c*dt)
    f_c = cutoff frequency

    u can approximate for small f_c*d
    alpha = 2*pi*f_c*dt

    code:


   
import numpy as np

class LowPassFilter:
    def __init__(self, cutoff_hz, sample_rate_hz, initial_value=0.0):
        """
        cutoff_hz: -3dB cutoff frequency
        sample_rate_hz: how often update() is called
        """
        dt = 1.0 / sample_rate_hz
        omega_c = 2 * np.pi * cutoff_hz
        self.alpha = omega_c * dt / (1 + omega_c * dt)
        self.y = initial_value
        self.initialized = False
    
    def update(self, x):
        if not self.initialized:
            self.y = x
            self.initialized = True
        else:
            self.y = self.alpha * x + (1 - self.alpha) * self.y
        return self.y
    
    def set_cutoff(self, cutoff_hz, sample_rate_hz):
    """change cutoff on the fly. doesn't reset state."""
        dt = 1.0 / sample_rate_hz
        omega_c = 2 * np.pi * cutoff_hz
        self.alpha = omega_c * dt / (1 + omega_c * dt)
    
    def reset(self, initial_value=0.0):
        self.y = initial_value
        self.initialized = False

     # this is a lot, but if we dont intialize it will take many samples to startup. startup transient. also this allows for cutoff frequency to be changed

    # allows for easy setup like: 
    # filter velocity, cut anything above 30 Hz, sampling at 1 kHz
    velocity_filter = LowPassFilter(cutoff_hz=30, sample_rate_hz=1000)

    #ENDCODE

    also can write one with alpha easily with the approximation

    make f_c atleast 5-10x lower than sample rate.