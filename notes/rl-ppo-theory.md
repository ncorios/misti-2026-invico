# RL and PPO Theory

## Reinforcement Learning



# Reading PPO Training Output + Tuning — DogEnv  

Run TensorBoard:
```bash
cd ~/dogzilla-control/controllers/ppo
tensorboard --logdir tb_logs
```
-
## PART 1 — The four lines that matter, in order

1. `ep_len_mean` → is it surviving?
2. `reward_forward` → is it actually moving forward?
3. `explained_varoance` → is it learning at all, or is the machinery broken?
4. `video` → what is it ACTUALLY doing? (ground truth)

The console number hints. The video is truth.

---

## PART 2 — rollout/ (the behavior — look here first)

### `ep_rew_mean` — average total reward per episode
Never read raw because: its a sum of terms, and terms that are meant to compliment forward movement cant dominate.
A high value can mean walked well OR survived well by staying still/ going in circles. Decompose it + watch video.

### `ep_len_mean` — steps before episode ends
how long before episodes end, should climb to 100

---

## PART 3 — train/ (the learning machinery)

### `explained_variance` — critic prediction quality (0=useless, 1=perfect)
It measures how well the critic predicts actual results. should climb to 1

### `std` — the policy's action-distribution spread (exploration)
spead of the action distribution, should be high early on and decrease, if it gets flat it means the policy isnt commiting. sharpens with lower ent_coef or decay learning rate, not more steps

### `approx_kl` / `clip_fraction` / `clip_range` — the "don't change too fast" safety
~0.01–0.03 = controlled, healthy updates. Spiking >0.1 = policy lurching, unstable (lower learning rate).
~0.1–0.3 = normal. Consistently >0.5 = updates constantly hitting the brake (lower the learning_rate).
These answer "is training stable," not "is it walking."

### `value_loss` / `loss` / `policy_gradient_loss`
"Is the math healthy," not "is it walking." Largish value_loss early is expected.

### `learning_rate` / `n_updates` — ____.
value_loss etc. — largish value_loss early is normal (critic still learning).
learning_rate / n_updates — config echo + a counter; ignore for reading progress.

## PART 4 — time/ (speed, not quality)
`fps` high = parallel envs working. 

---

## PART 5 — The reward breakdown (TensorBoard, per-term curves)

rewards ( that you should keep always ):
forward: learning to move if high, farming survival if low
survive: high means upright, low means dying fast
smoothness: if very negative, it means joints are thrashing
y_drift: small = good, staying near center.


tweak with weights, understand failure modes

failure modes:
forward: too high causes lunging, too low causes standing
survive: opposite of forward
smoothness: too high causes the robot to barely move
y_drift: a bit weird. too high can cause either standstill or it will turn 90 degrees and then cease moving as any movement is a penalty. too low and you have drift if gait is assymetrical/strange



---

## PART 6 — Hyperparameters (what they are + how to tune)

> A hyperparameter is a setting you choose BEFORE training that the learning
> process does NOT adjust itself. (Weights are learned; hyperparameters are set.)

| Hyperparameter | What it controls | Too high | Too low | Sane range |
|total_timesteps| how long it trains | wasted compute/ brittle policy| undertrained, erratic, uncommited| 5-50M steps.
|n_envs. parallel experience + actual wall clock speed. range should be physical cores. i use 8 bc m4 chip has 12 and 4 should be used for physics and computer processes. diff cores have different purposes
learning_rate: weight update, step size. too high is super unstable and erratic. noisy. too low means your model learns very slow. 0.001-0.004 is a good range
n_steps: data collected per env per update. too high makes slower updates, too low is noisy and gives unstable gradients. 2048-2096 is a good range
batch size: mini batch per graident step. loo low gives noiser gradients. 64-512 is a good range.
ent_coef is exploration bonus. keeps std up, if its too high your model explores forever and never commits. too low, it collapses early and wont find the optimal gait.
gamma: future vs immediate reward. too high and it overvalues distant reward. too low = myopic
gae_lambda: advantage smoothing. honestly dont touch.
clip_range: ppo update clip. too high too aggresuve updates, too low its myopic. 0.1-0.2 is good

### The tuning decision tree (fill the symptom → lever)
- Symptom: no learning. explained variance not approaching 0 -> lower learnng rate, check that reward/env is working fine
- Symptom: erratic/noisy learning, lower learning rate
- Symptom: plateaued, std flat, won't sharpen: lower ent_coef
- Symptom: good deterministic but brittle stochastic: domain randomization + training noise. the whole point of rl is to give a good robust policy that works in a variety of situations. you need a robust policy otherwise you shouldbve just used pid/mpc. also rl dosnt need any data/ a model ig.
- Symptom: too slow wall-clock -> parallel envs, more n_envs
- Symptom: falls fast / lunges bring down forward reward.


