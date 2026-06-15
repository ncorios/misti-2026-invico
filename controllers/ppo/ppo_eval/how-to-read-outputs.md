# Reading PPO Training Output — DogEnv


run in terminal for plots: 
cd ~/misti-2026-invico/controllers/ppo
tensorboard --logdir tb_logs

A reference for interpreting the Stable Baselines3 console tables and TensorBoard
curves while training the DOGZILLA quadruped. Tuned to this env's reward
(forward + survival − smoothness) and the v2 change (orientation termination +
forward weight raised to 20).

---

## TL;DR — the four lines that matter, in order

1. **`ep_len_mean`** → is it surviving? (and is the fall-termination firing?)
2. **`reward_forward`** (TensorBoard) → is it actually moving forward?
3. **`explained_variance`** → is it learning at all, or is the machinery broken?
4. **the rollout video** (`ppo_log.py`) → what is it *actually* doing?

The console number hints. The video is ground truth.

---

## Example console output

```
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 146         |
|    ep_rew_mean          | 155         |
| time/                   |             |
|    fps                  | 12202       |
|    iterations           | 32          |
|    time_elapsed         | 53          |
|    total_timesteps      | 655360      |
| train/                  |             |
|    approx_kl            | 0.011241903 |
|    clip_fraction        | 0.115       |
|    clip_range           | 0.2         |
|    entropy_loss         | -15.4       |
|    explained_variance   | 0.554       |
|    learning_rate        | 0.0003      |
|    loss                 | 79.6        |
|    n_updates            | 310         |
|    policy_gradient_loss | -0.0119     |
|    std                  | 0.874       |
|    value_loss           | 156         |
-----------------------------------------
```

What this particular table says: episodes last ~146 steps and total reward is ~155.
Since survival pays ~1.0/step, ~146 of that 155 is just the survival bonus — only
~9 is forward progress. **Translation: it's mostly staying "alive" and barely moving
forward.** (In v1 this was the dog lying on its back, counted as alive by the
height-only check. v2's orientation termination is meant to close that.)

---

## rollout/ — the behavior (look here first)

### `ep_rew_mean` — average total reward per episode
The headline, but **never read it raw.** Reward = forward + survival − smoothness.
Survival pays ~1.0/step, so a long episode accumulates reward even with no walking.

- High value can mean "walked well" **or** "survived a long time doing little."
- You cannot tell which from this line alone — decompose it (see TensorBoard) and
  watch the video.

### `ep_len_mean` — average steps before the episode ends  *(KEY for v2)*
TimeLimit caps episodes at 1000 steps. Falls now end the episode (orientation
termination).

| What you see | What it means |
|---|---|
| ep_len **drops** | termination firing — falls are being caught. *Good* (exploit closed). |
| ep_len **climbs toward 1000** over training | learning to stay upright longer — real progress. |
| ep_len **stuck very low** (<50) | falling almost immediately; forward weight (20) may be too aggressive → lunging into faceplants. |

Counterintuitive but important: right after adding the fall-termination, a *shorter*
ep_len is good news — it means falling is finally being detected instead of farmed.

---

## train/ — the learning machinery (confirms it's *learning*, even if behavior is bad)

### `explained_variance` — critic prediction quality (0 = useless, 1 = perfect)
Your "is the machinery working" gauge.
- Climbing toward 1 over training = healthy.
- Stuck near 0 = learning is broken.
- Example value 0.554 = mid; critic is learning but not great yet (fine early on).

### `entropy_loss` / `std` — how much the policy is still exploring
`std` is the spread of the action distribution.
- High early = exploring varied actions (expected).
- Should slowly **decrease** as the policy commits to a behavior.
- Collapses to ~0 too fast = stopped exploring early (can get stuck).
- Never declines = never commits.

### `approx_kl` / `clip_fraction` / `clip_range` — PPO's "don't change too fast" safety
- `approx_kl` ~0.01 = healthy, controlled updates. Spiking >0.05 = unstable.
- `clip_fraction` ~0.1 = normal.
- Glance at these to confirm nothing's exploding; not where you read "is it walking."

### `value_loss` / `loss` / `policy_gradient_loss` — internal optimization losses
- Largish `value_loss` early is normal (critic still learning).
- "Is the math healthy," not "is it walking." Don't over-read them.

### `learning_rate` / `n_updates` — config + bookkeeping. Ignore.

---

## time/ — speed, not quality

| Field | Meaning |
|---|---|
| `fps` | steps/sec — high means parallel envs are working |
| `total_timesteps` | how far into the run |
| `iterations` | rollout→update cycles completed |
| `time_elapsed` | wall-clock seconds |

These tell you how *fast*, never how *good*.

---

## The reward breakdown — most important, and it's in TensorBoard (not the console)

The env logs `reward_forward`, `reward_survive`, `reward_smoothness` in `reward_info`.
These appear as separate curves in TensorBoard:

```bash
tensorboard --logdir tb_logs
# then open http://localhost:6006
```

This split is the real diagnostic — it breaks the headline reward into its parts:

| Curve | Climbing / high | Flat / low | Very negative |
|---|---|---|---|
| `reward_forward` | learning to move forward — **the one you want up** | standing still / farming survival | — |
| `reward_survive` | staying upright | dying fast | — |
| `reward_smoothness` | — | — | thrashing the joints (consider raising smoothness weight) |

**The classic failure pattern:** `reward_survive` high + `reward_forward` flat near zero
= standing still. Fix = raise `forward_reward_weight` or lower `healthy_reward`.

---

## The reading workflow

1. **`ep_len_mean`** — did it drop (termination firing) then climb (staying up longer)?
2. **`reward_forward`** (TensorBoard) — climbing off zero?
3. **`explained_variance`** — climbing? (confirms it's learning, not broken)
4. **Watch the video** — `python ppo_log.py <version>`. Numbers hint; video confirms.

---

## v2 hypothesis being tested

Closing the on-back exploit (orientation termination) + stronger forward pull
(weight 5 → 20) should make it try to walk instead of flopping.

**Confirmed if:** ep_len behaves sanely (drops on falls, trends up as it learns to
stay up), `reward_forward` climbs off zero, and the video shows forward attempts
rather than lying down.

**If reward stays pure-survival and the video shows standing:** lower `healthy_reward`
or raise `forward_reward_weight` further.

**If it lunges and faceplants constantly (ep_len stuck tiny):** forward weight at 20
may be too aggressive — back it toward 10.

---

## One-line summary

`ep_len_mean` tells you if it's **surviving**, `reward_forward` tells you if it's
**moving**, `explained_variance` tells you if it's **learning**, and the **video**
tells you what's actually happening. Those four, in that order.