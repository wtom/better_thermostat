# Room control and low-temperature heating: development brief

Status: independent experimental production mode integrated and tested; no controller deployed to the house. See iteration 8 for current status.

## Goal

Every controlled room should hold its own scheduled target without unnecessary
overshoot, using as little gas as possible. Supply temperature should be as low
as comfort permits. Valve opening is an actuator, not an efficiency score.
Rooms with independent underfloor heating retain their own control authority.
Window contact handling remains unchanged.

## Existing building blocks inspected

- `utils/calibration/pid.py`: per-loop state, measurement derivative, conditional
  anti-windup and conservative tuning already exist. Reuse these mechanisms
  rather than introducing a parallel controller framework.
- `utils/calibration/mpc_v2/compute.py`: consumes supply temperature and heating
  availability. The diagnostic baseline is commit bbe37a72; it changes logging,
  not actuator decisions.
- `tests/benchmark`: existing plant, actuator, sensor and multi-TRV runners provide
  a comparison harness; simulations are not proof of real gas savings.

## Candidate to evaluate, not yet selected

Compare current MPC with the existing PID path first. Only add supply-aware
feed-forward and slower room correction if that comparison demonstrates a gap.
Account for warming trend and radiator afterheat. Integral learning must not
accumulate fictitious demand during unavailable heating. A burner-off interval
alone does not prove that circulating water cannot still heat a radiator.
Preserve OFF, window, sensor-failure, setpoint and multi-TRV adapter behavior.
Keep tuning per room; do not hard-code the five main rooms as the control scope.

## Evaluation and rollout order

1. Inventory all currently controlled rooms and actual actuators, including
   existing per-room calibration mode and temperature sources.
2. Compare the baseline and candidate under steady targets, target increases,
   target decreases, variable supply, heating interruptions, sensor lag and
   multi-TRV imbalance. Report overshoot/deficit magnitude and duration,
   settling time, valve travel and saturation. An initial reporting band of
   +/-0.2 K is provisional, not an agreed comfort guarantee.
3. Implement only the smallest justified change in existing paths, with focused
   regression checks for the scenarios it affects. Commit and push the tested
   candidate with an exact installable revision and rollback revision.
4. Validate actual actuator writes and room trajectories in a bounded live
   trial before expanding. Do not simultaneously remove the pause pilot or
   alter the heating curve: those are separate experimental variables.
5. Once room control is stable across all rooms, tune the heating curve slowly
   against the least adequately heated room. Compare gas per heating degree
   hour over completed comparable windows, plus comfort and schedule changes.

Deployment support from Wolfgang is available if agent access is insufficient.
No deployment is necessary for this design document.

## Evaluation checkpoint 2026-11-07

- Read-only live inventory captured all 10 BT room entities and 11 radiator
  actuators (two in the living room), including bathroom, WC, kitchen,
  hallway and heating cellar. All expose MPC v2 diagnostics. Bathroom
  independent underfloor heating remains outside the actuator scope.
- Repaired a benchmark input omission: simulated supply temperature now reaches
  MPC v2 through both single- and multi-TRV drive paths. Before this patch the
  controller kept its model supply default even on low-temperature plants.
- Focused adapter/runner/multi-TRV tests: 117 passed. Entire benchmark test
  directory: 434 passed. Ruff check and diff whitespace check passed.
- Existing 37-scenario single-TRV comparison: MPC v2 overall 0.879 versus PID
  0.783; supply-aware run 0.880 versus 0.783. These are synthetic composite
  scores, not measured gas efficiency or deployment acceptance.
- In the 30-scenario multi-TRV groups, PID wins in all three simulated plant
  profiles. Baseline overall MPC/PID: symmetric 0.574/0.655, asymmetric
  0.539/0.664, heterogeneous 0.576/0.684. Supply propagation does not change
  these 65 C plants. Single-TRV ranking must not be generalized to groups.
- Additional 30 low-supply runs compare PID, default MPC and MPC with a
  2000-minute room prior at 35/45 C. Large deficits occur for all controllers
  on some of these synthetic plants; they are not fitted home models and
  do not establish the appropriate home heating curve. Changing only the
  room time constant does not resolve these deficits.
- Autoreview found no actionable issues in this benchmark patch; its filesystem
  sandbox prevented repository inspection, so the review is bundle-only.

Evidence is retained in workspace artifacts/heizung/room-control-20261107:
inventory.json, baseline.txt, supply-aware.txt, live-prior-metrics.json,
focused-tests.txt, benchmark-tests.txt and review-run.txt. The next regulator
work must distinguish group model/distribution behavior from single-radiator
tracking, then test changing supply and interruptions explicitly. No production
controller, schedule, heating curve, window behavior or pause pilot was changed.

## Group-model fix checkpoint

The group command is applied to every radiator. The fallback prior now sums
room coupling across the configured radiators while retaining the response of
one representative radiator (gain and radiator time constant unchanged).
Single-radiator priors are unchanged. This is a count-based prior, not a fitted
hydraulic model for unequal radiator sizes. The benchmark group factory mirrors
this prior without reading actual simulated gains or coupling strengths.

Across 30 scenarios per profile, mean tracking RMSE changes from 0.716 to
0.636 K (symmetric), 0.738 to 0.632 K (asymmetric), and 0.720 to 0.640 K
(heterogeneous). Mean peak overshoot changes from 1.067/1.130/1.074 K to
0.571/0.641/0.565 K; mean valve travel falls from 2084/2001/2203 to
270/298/312 percentage points. Mean peak deficits worsen by 0.16–0.19 K,
although integrated deficits improve across these scenario sets.

A separate 12-hour synthetic test steps supply 45→35→45 C and disconnects
heating for one hour. With two radiators, overshoot falls 0.123→0.034 K and
travel 608→336 points, but integrated deficit rises 2.880→3.004 K h and
RMSE 0.481→0.498 K. Three radiators show a similar deficit tradeoff. One
radiator is exactly unchanged. No gas-saving or all-room comfort claim follows.

Validation: 833 passed, 2 expected xfails in MPC-v2, sensor-fallback and benchmark
suites. Autoreview local returned clean, bundle-only (reviewer shell unavailable).
Evidence: workspace artifacts/heizung/room-control-20261107/group-comparison.json,
dynamic-group.json, group-fix-tests.txt, group-fix-review.txt. Baseline/rollback:
27d1e01f. The wider all-room control and heating-curve goal remains open.

## Agreed direction: slow valve calibration and critical-room heating curve

The preceding MPC work fixes a separate group-model defect. It is not the
chosen architecture for this next experiment. The user explicitly prefers
hours-long observation at fixed valve positions, small subsequent corrections,
and earlier intervention only for significant cold/warm deviations. PID is
not assumed adequate either.

### 1. Specify observations and provisional limits

Include all 10 BT rooms and their 11 existing radiator actuators. Respect the
bathroom's independent floor-heating controller; do not add its actuators.
Record room target, measured temperature and trend, actual valve writes/echoes,
supply/outdoor temperatures, contact states, heating availability and schedule
changes. Use room-level targets even for multiple radiators. Inspect individual
radiator balancing separately: an existing temperature-boost distributor must
not silently defeat a fixed-valve observation window.

Initial simulation parameters, not agreed deployment settings: 2–3 h normal
observation windows, 2–3 percentage-point valve corrections, +/-0.2 K room band,
fast intervention at approximately +/-0.5 K sustained error or a clear trend
that predicts crossing it. Calibrate these against real sensor resolution,
noise and room response; fast close for genuine excessive warmth must remain
possible. Not every room needs the same timescale.

### 2. Build a room-level sample-and-hold calibration prototype

Maintain an explicit chosen baseline opening and observation window per room.
Start from a plausible previously applied opening, not an arbitrary full-open
reset. Hold the value over the observation window. Use sustained error and
trend together: no extra opening if the room is already warming adequately,
and reduce ahead of afterheat overshoot. Once settled, make only one small
correction, then observe again. Keep an accepted opening when it works.

Separate exceptional recovery from normal calibration: a substantial deficit
can request more opening promptly; excessive warmth can close promptly. Do
not learn a normal baseline from that emergency interval. Target changes,
windows, explicit circuit unavailability, substantial supply changes and
solar disturbances invalidate or segment the observation. Burner off alone
does not establish unavailable radiator heat. Persist accepted baselines and
explain every intervention; neither endless hold nor rapid correction chatter
is an acceptable fallback.

### 3. Validate with the heating curve fixed

Use existing single/multi-radiator simulations and recorded home conditions.
Cover steady targets, cold start, small target steps, target drops, solar gain,
window events, changing supply, heating interruptions and sensor noise/delay.
Compare against the installed MPC and PID as references, not assumed winners.
Report room error, time above/below the proposed band, peak deviations,
recovery time, actual command changes/travel, and time at 0/100 percent.
Reject a quiet controller if it merely leaves rooms persistently cold.
Valve opening/integral is not measured gas consumption.

### 4. Bounded live trial, then all rooms

Add an explicitly selectable calibration mode in the BT fork; do not disguise
it as MPC. Reuse existing write, OFF/contact, cap, persistence and diagnostic
paths. Start with one single-radiator room, then the two-radiator living room;
retain other rooms' working modes until the new mode passes the trial. Check
actual successful writes/echoes as well as recommendations. Record exact
install and rollback revisions. Observe for completed day/night cycles before
expanding to all rooms. Do not change heating curve or pause pilot in this
stage. Bathroom floor heating retains independent authority.

### 5. Outer loop: lower the heating curve slowly

After room calibration is stable, evaluate all rooms together. Lower supply in
small bounded heating-curve steps only when all scheduled comfort targets are
being met and heat reserve is present. During the following observation phase
freeze normal valve-baseline learning (fast comfort protection remains active).
Then allow room recalibration on the new supply level. Do not adjust both
levels simultaneously. Require a long stable assessment after each change,
initially at least a completed day; exclude warm-up, open windows, DHW and
heating outages from evidence of inadequate curve temperature.

The user's endpoint is explicit: the thermally limiting room holds its target
with 100% valve opening, while other rooms hold theirs with smaller openings.
This is sustained holding, not a transient target crossing. Re-evaluate which
room is limiting as weather/targets change. If a normally used room remains
below target with 100% opening and available heat, stop lowering and roll back
the last curve reduction. A failed sensor, open window or defective radiator
must be diagnosed rather than drive an unlimited whole-house curve increase.
Slope versus offset changes must be based on outdoor-temperature evidence,
not blindly changed together from one mild day's result.

### 6. Acceptance and efficiency evidence

Acceptance is stable room comfort at every room's own scheduled target, rare
normal valve corrections, reliable exceptional recovery, and the lowest
comfort-safe heating curve. Formal tolerance/duration limits remain to be
chosen from measured room/sensor response. Compare gas per heating degree-hour
(m3/degree-hour) across completed comparable windows; report both normalized
values and their percentage difference, with room comfort and schedule changes
separately. Do not claim savings from simulation or lower valve values alone.

Next deliverable: a tested simulation prototype of the slow room calibration,
with explicit provisional parameters and comparison report. No automatic
heating-curve changes are authorized by this plan alone.

## Implementation checkpoint: first simulation prototype

`tests/benchmark/adapters/slow_adapter.py` implements the first experimental
sample-and-hold path. Select it with `uv run python -m tests.benchmark.runner
--controller slow --controller pid --controller mpc_v2`. It is benchmark-only:
no new production calibration choice, actuator writes, or heating-curve changes.

Provisional behaviour: hold 3 hours; normal corrections 3 percentage points;
use mean error and an hour-ahead trend projection. Sustained 0.5 K deviations
trigger separate recovery after 10 minutes, with opening increments at most
once per 15 minutes or closure for excessive warmth. Recovery does not train
the normal baseline. Window, solar and supply disturbances restart observation;
only baseline opening persists across adapter restarts.

Eight behavioural tests and the full benchmark suite (443 tests) pass. The
37-scenario single-radiator composite score is 0.600 for the prototype versus
0.783 PID and 0.880 MPC v2. These synthetic scores are not gas measurements.
The prototype is NOT suitable for live rollout. In particular, recovery can
repeat around a poor retained baseline: fixed small steps alone do not yet
identify an adequate operating point. Missing forecast-based early protection,
room-specific response learning and multi-radiator validation remain open.
The benchmark starts from its actual last command (initially zero), not an
invented optimally fitted opening. This cold-start limitation is part of the
first result, not grounds to tune against hidden plant truth.

### Adaptive balancing research direction

Homematic IP's published Fraunhofer IEE summary describes adaptive flow
reduction in over-supplied rooms; this supports the principle, not a claim of
access to their implementation. Reference:
https://homematic-ip.com/sites/default/files/downloads/IEE-Bericht153-242-2020-0530-dynamisch-adaptiver-abgleich-_Zusammenfassung.pdf

Learn each room's approximate holding opening conditional on available supply
and weather, delay before a command has measurable effect, incremental response
and radiator afterheat. Learn from ordinary stable operation, excluding known
disturbances. Do not equate valve percentage with flow or heat output. Keep
multiple radiators under one room target; individual distribution and changing
hydraulic interactions require separate validation. This is a proposed next
iteration, not a tested identification algorithm or certified hydraulic balance.

Closeout: autoreview `--mode local` returned clean after fixing discarded
observation intervals leaking into a new window (regression test included).
The reviewer could inspect only the supplied bundle because its shell sandbox
was unavailable. Ruff check/format and diff whitespace checks pass.

### Iteration 2: bounded recovery feedback and observation record

Simulation only. On completing an undisturbed comfort intervention, move the
normal baseline by one calibration step (3 percentage points), not to the
transient recovery opening. Target/window/supply/solar changes discard the
intervention rather than training from it. Cold recovery holds its command
while approaching target instead of increasing it every 15 minutes regardless
of the remaining deficit.

Each completed normal hold window records applied opening, mean error,
temperature trend, duration, target, outdoor and supply temperature. This record
survives simulator restart with the baseline. Recovery duration is recorded
separately: it is NOT a measured room time constant, dead time or post-heating
response. No weather feed-forward or response-time inference is implemented yet.
The record describes the opening that was actually held, not the next command.

### Tuesday house-test target: 10 November 2026

The user requests a bounded house test on Tuesday, not an immediate rollout.
The date is a delivery target, not evidence that the controller is ready.

- Saturday: harden observation validity; gradual supply/weather drift must not
  masquerade as valve response. Compare with the beginning of each hold, not
  only consecutive samples. A 3 K cumulative change restarts the experiment;
  loss/return of supply telemetry also restarts it. These are provisional
  simulation thresholds, not learned thermal parameters.
- Sunday: implement and test room-response observations and conservative reuse
  of context-matched holding points. Run multi-day learning, changing-weather,
  sensor-delay/noise and recovery scenarios; report comfort and valve travel
  separately rather than optimizing the composite score alone.
- Monday: deliver a separately selectable BT mode using existing control/write,
  persistence and diagnostics paths. Validate OFF, contacts, sensor loss,
  restart, caps, recovery and multi-TRV distribution. Prepare exact install and
  rollback revisions and an initial room choice from current measurements.
- Tuesday: start with one single-radiator room only if simulation and integration
  gates pass. Verify actual writes and echoes, then observe a full day/night
  cycle before expanding. Other rooms retain their current controller.

Trial gate: no known sustained unexplained comfort deficit, no normal rapid
valve corrections, all existing safety/override paths preserved, inspectable
held/recovery state and baseline, and a verified rollback. Before enabling,
record numerical room-specific comfort/abort thresholds from sensor resolution
and measured response; they are not yet validated. If these gates are not met,
Tuesday may be an observation-only shadow test, not an unsafe actuator trial.
Heating curve, pause pilot and independent bathroom floor control stay fixed.

Iteration 3 adds three regressions for cumulative supply/weather drift and
supply-telemetry loss. It does not yet implement response-time identification
or context-dependent baseline reuse. A passing test is not live readiness.

### Iteration 4: conservative context-matched holding points

The simulation adapter retains up to 16 settled observations across restart.
Only complete undisturbed holds with mean and final error within 0.2 K,
absolute trend at most 0.05 K/h and available supply telemetry qualify.
At restart or a detected context transition, the latest matching point may
move the baseline by at most one normal step. Matching requires target within
0.1 K, outdoor temperature within 1 K and supply within 2 K. Ordinary samples
do not repeatedly apply the point. No interpolation or extrapolation occurs;
recovery and disturbed windows cannot teach holding points. All thresholds are
provisional simulation choices. Points have no age expiry yet; reuse is bounded
but cannot distinguish changed hydraulics from old conditions. Response-time
and radiator-afterheat identification remain open, as do multi-day validation
and production integration. This is not a live candidate.

### Iteration 5: closed-loop thermal week, not input replay

S39 and S40 add RC3 wall mass (1100 min), slow radiator (30 min), 120 s
transport delay, 45 °C supply, 90 s sensor lag and 0.05 K sensor noise.
Both start with room AND radiator at 21 °C, a zero-opening cold controller,
no oracle warm-up, and a controller-only restart after 72 hours. S39 holds
21 °C/5 °C outdoor; S40 cycles outdoor 0–8 °C daily and drops target from
21 to 20 °C after 96 hours. A plant regression verifies that closing a hot
radiator does not erase its stored heat: the room continues warming.

Measured 7-day experiment (30 s steps, exact physical 604800 s):

| Scenario / controller | Day 1 RMSE (K) | Day 7 RMSE (K) | Day 7 max under / over (K) | Day 7 valve travel (percentage points) |
| --- | ---: | ---: | ---: | ---: |
| S39 slow | 0.363 | 0.185 | 0.207 / 0.304 | 6 |
| S39 PID | 0.208 | 0.264 | 0.561 / 0.396 | 4143 |
| S40 slow | 0.365 | 0.338 | 0.580 / 0.616 | 150 |
| S40 PID | 0.229 | 0.266 | 0.597 / 0.472 | 4025 |

These are model results, not house measurements or gas savings. The valve
travel proxy is summed absolute requested opening changes, not physical motor
travel. It excludes each day's boundary transition. The PID benchmark uses its
existing simplified filtering/slope path, not a full production HA replay.
The deterministic sensor seed depends on scenario name; each controller within
a scenario sees the same noise generation, but not the same room trajectory.

S39 shows useful settling with quiet actuation and no evident restart collapse;
S40 still gives worse tracking RMSE than PID and repeated comfort interventions.
Cold-start maximum undershoot is 0.776 K (S39) / 0.815 K (S40). This is not a
live-ready result. Weather-dependent equilibrium learning and separating delayed
response/afterheat from true heat deficit remain the next implementation work.
No response time constant has been identified by this experiment.

Reproduction: the workspace artifact `artifacts/heizung/slow-stage5-20261107/`
contains `run.py` and raw daily `results.json`. From this checkout run
`PYTHONPATH=. uv run python <artifact-directory>/run.py`. It deliberately uses
`_drive_adapter` with exact seven-day duration; the general benchmark CLI's
`run_scenario` automatically extends nominal durations for slow plants. Running
S39/S40 through that CLI therefore does NOT reproduce these exact seven-day
numbers (the scenario duration is nominal there). Neither route warms the
plant with oracle control for these two scenarios.

### Iteration 6: measured trend guard during cold recovery

The simulation adapter now fits a least-squares room-temperature trend over
at most 30 minutes, requiring at least three samples spanning 15 minutes.
After the first cold recovery command, further opening increments wait if
that measured warming trend projects entry into the 0.2 K target band within
one hour. Stalled warming permits escalation again. Context disturbances and
restart discard the trend; it is neither persisted nor claimed to identify a
room time constant. First intervention and warm closure remain unchanged.
The projection and windows are provisional, not fitted house parameters.

Focused regressions cover still-cold but warming recovery, renewed escalation
when warming stalls, and trend invalidation on target change/restart. Existing
window, missing heat, learning and persistence tests remain in force.

Repeating the exact iteration-5 seven-day experiment gives effectively the
same day-7 S39 result (RMSE 0.185 K; 6 percentage points travel), and identical
S40 results (RMSE 0.338 K; under/over 0.580/0.616 K; 150 points travel).
This guard addresses a local recovery escalation case but DOES NOT solve the
weather-tracking gap; no improved weekly comfort or gas efficiency is claimed.
The reproducible runner and outputs are in workspace artifact
`artifacts/heizung/slow-stage6-20261107/`. Production integration and bounded
live readiness remain pending. Next inspect the actual S40 recovery/context
transitions and holding-baseline evolution before adding further learning.

The S40 transition trace narrows the next investigation: on day 7 the normal
baseline moves 21 -> 24 -> 27 -> 24 -> 21 percent. Warm recoveries request zero
at t=561540 and 577020 s; between them the controller returns to a 24-percent
baseline while the reported room remains 0.455 K above target (t=573600 s).
Context transitions can end an intervention before entry into the comfort band;
separating evidence invalidation from comfort-command cancellation should be
examined next. These are simulator observations, not inferred house behaviour.

### Iteration 7: separate evidence resets from comfort cancellation

Weather/supply changes now discard the observation window and recent trend
without cancelling an active comfort intervention. Target changes, windows,
solar disturbances and unavailable heat still cancel it. A recovery spanning
an environmental change cannot teach a baseline correction when it completes.
Regression checks cover warm closure, cold command continuity, invalidated
learning, and explicit cancellation conditions. Restart semantics are unchanged.

All 460 benchmark tests pass. Repeating the exact seven-day experiment leaves
both S39 and S40 day-7 metrics unchanged from iteration 6. Thus this is a local
contract fix, NOT evidence that the observed S40 reopening was caused by an
environment reset. That earlier causal hypothesis is not confirmed. The next
trace must distinguish actual comfort-band completion and subsequent afterheat
from context cancellation before changing the controller further. Reproduction
artifacts: workspace `artifacts/heizung/slow-stage7-20261107/`.

The full transition trace corrects the earlier abbreviated interpretation:
at t=562800 s S40 is back inside the +0.2 K band (20.189 C at target 20 C),
so warm recovery completes normally and baseline drops from 27 to 24 percent.
At t=573600 s, 10800 s later, the room is warm again (20.455 C); this is not
an immediate reset-induced reopening. A second completion at t=578280 s is
likewise inside the band (20.168 C). The remaining issue is rebound under the
resumed holding opening, not a demonstrated weather-triggered cancellation.

## Iteration 8 — independent production integration (7 November)

The selectable **Slow valve calibration (experimental)** mode now uses
`utils/calibration/slow.py`; the benchmark is a thin adapter over that same
controller. It does not run MPC, Kalman filtering, a QP or PID. The `v2` branch
is the integration base, not the chosen controller. Earlier production MPC
changes were removed from this PR's net diff; the preceding sections are a
historical research log, not a description of changes still shipped here.

The mode requires direct valve control on **every** radiator of a room.
Mixed calibration modes and non-valve outputs are rejected in the configuration
flow. One shared baseline drives all room radiators, bounded by the lowest
configured cap; no temperature-dependent MPC distributor changes the held
opening. A room's existing OFF/contact/kernel safety and actuator write paths
remain in charge. Bathroom floor heating is outside this integration.

Accepted learning is persisted under its own `slow` Store section. Restart,
OFF/contact, sensor fallback and observation gaps longer than 15 minutes discard
active windows. Unconfirmed or different actuator feedback cannot count as a
held baseline experiment. Feedback still requires live verification: successful
adapter writes and device-reported position are not proof of hydraulic flow.
Configured supply/DHW inputs reuse the existing sensor settings (their internal
names still contain `mpc_v2`); they do not invoke MPC. Solar classification
invalidates learning but no longer disables warm-room protection.

A measured 30-minute trend can begin warm recovery before a predicted +0.5 K
excursion: current excess must exceed 0.2 K, warming at least 0.1 K/h predicts
+0.5 K within 30 minutes, and the condition must persist for 10 minutes.
Normal corrections remain 3 percentage points after 3 hours. These constants
remain experimental, not tuned per-house deployment settings.

Exact seven-day S39/S40 reruns, using the same 30-second simulation step as the
previous runs, give last-day results:

| Scenario/controller | RMSE K | max under/over K | valve travel pp |
| --- | ---: | ---: | ---: |
| S39 slow | 0.185 | 0.207 / 0.304 | 6 |
| S39 PID | 0.264 | 0.561 / 0.396 | 4143 |
| S40 slow | 0.300 | 0.583 / 0.516 | 170 |
| S40 PID | 0.266 | 0.597 / 0.472 | 4025 |

The warmer-weather case improves from the prior slow RMSE 0.338 K but still
misses a strict +/-0.5 K envelope and is not superior to PID on RMSE. A tested
alternative (waiting until target before exiting warm recovery) worsened peaks
and was rejected. No valve-travel number is a gas-consumption estimate.
Artifacts: workspace `artifacts/heizung/slow-stage8-20261107/`.

### Tuesday 10 November: bounded test and rollback

1. Before any deployment, record the exact candidate commit from PR #11 and the
   **actually installed** BT version/commit, save the installed component and
   relevant HA configuration/Store backup, and verify the backup is readable.
   Do not assume that the fork's `v2` head equals the installed revision.
2. Keep heating curve, schedules, demand-pause automation and bathroom unchanged.
   Start with one single-radiator room. Keep all other rooms on their current
   modes. Do not change several control variables together.
3. Confirm configuration shows `slow_calibration` and direct valve output. Record
   target/room temperature, the `slow_calibration` entity attribute (baseline, reason, trend),
   requested percentage, successful writes and device echoes. Check OFF and
   window closure before unattended observation. Never infer delivery from a
   diagnostic recommendation alone.
4. Observe a full day/night cycle. Stop the pilot if an occupied room remains
   more than 0.5 K cold for 30 minutes despite available heat, if excess exceeds
   0.5 K and the valve does not close, if writes/echoes diverge, or if controller
   state is non-finite/unavailable. Investigate recovery and supply separately.
   The simulated weather envelope is not yet a blanket live-comfort acceptance.
5. Roll back the selected room to its recorded prior calibration mode first.
   If the integration itself is faulty, restore the exact backed-up component
   and restart HA; restore the configuration/Store backup if needed. Verify the
   prior mode and actual writes after rollback. Never remove the whole HA Store.
6. Only after a successful single-room cycle test the two-radiator living room,
   with both heads selected together. No all-room rollout or heating-curve
   reduction is implied by this PR. If deployment prerequisites or actuator
   checks are missing, Tuesday remains an observation/replay test, not actuation.

No production device or HA configuration was changed while preparing this code.


Closeout checks also cover recovery feedback: after one minute of write/echo
allowance, a divergent or missing recovery echo invalidates its learning without
cancelling comfort control. Natural cooling after a failed close cannot teach a
lower baseline. The `slow_calibration` live attribute is excluded from Recorder,
like existing controller diagnostics; explicitly collect it during a pilot.


## House pilot update — 2026-11-09

Default normal observation windows are now 1 hour. At window completion, both
mean and current absolute error within 0.35 K select a 1-percentage-point
correction; larger errors retain the 3-point coarse correction. The existing
0.2 K acceptance band and trend projection still decide whether any correction
is needed. Recovery remains independent and unchanged. A 1-point actuator
mismatch now invalidates learning (feedback tolerance below 0.5 points).
Learning persistence and DHW exclusion are unchanged. Historical benchmark
results above describe the previous 3-hour configuration, not this new default;
legacy scenario tests retain an explicit 3-hour window.


## House-heating review and sunshine correction — 2026-11-09

The weather service's sunshine estimate is no longer a disturbance gate or a
context-reset trigger in SlowCalibration. It does not establish heat gain in
any individual room (orientation, shading and shutters are unknown). Both
comfort protection and normal observation continue under sunny/changing reports.
The field remains an input for compatibility; it has no effect on decisions.
Actual indoor temperature and its measured trend drive the response. Window,
heat-unavailable, DHW, supply-context, feedback and setpoint gates are unchanged.
Outdoor temperature still qualifies the reuse of old holding points and marks
large load-context changes; it does not determine whether a cold room may heat.
Earlier references to solar invalidation above describe superseded behaviour.

### Critical assessment against the user's goal

The sample-and-hold architecture is a useful **experimental room-demand
balancer**, not yet a proven dynamic hydraulic balance for this house:

- Normal observation is 1 h with 1/3-point steps. A one-hour linear projection
  is used at window completion. This is a heuristic, not an identified thermal
  model: radiator lag, valve nonlinearity and afterheat are not learned.
- Cold protection still starts at a sustained 0.5 K deficit (10 min), with
  fixed 10-point steps at least 15 min apart. The warming-trend guard stops
  subsequent increases; cold protection has no symmetric predictive entry
  equivalent to warm protection. Step size does not yet scale with deficit
  or measured room response. These remain next-iteration work, not claims
  that this sunshine patch implements adaptive gain learning.
- Finishing cold recovery returns to the old baseline plus 3 points. For an
  initially inadequate baseline this may repeat recovery/cooling cycles.
  Blindly persisting the temporary boost would also be wrong; it is not the
  equilibrium opening. A future change needs closed-loop recovery tests.
- A settled window is only a conditional holding observation. Unknown indoor
  gains can make an apparently good opening temporary. No temperature-only
  observer can uniquely attribute heat to radiator versus sun/people/cooking.
  Old points must remain bounded hints; repeated confirmation and recency
  deserve attention before relying on them for automatic curve reduction.
- Shared pump pressure, valve flow curves, minimum circuit flow and differing
  radiator sizes are not modeled. Valve percent is not flow percent. The
  current pilot consists of three single-TRV rooms; whole-house and multi-TRV
  hydraulic interaction remains unproven. Bathroom floor-loop authority is
  unchanged.
- There is no automatic outer heating-curve optimizer. First establish stable
  room comfort, then lower supply separately and evaluate all rooms over long
  windows. A persistently cold room at 100% with available heat is a stop/
  rollback condition, not a reason to keep lowering supply. Exclude faults,
  windows, warm-up and DHW. Boiler minimum flow/cycling also matter.

### Evidence and boundaries

506 benchmark/dispatch/HA-integration tests passed after the sunshine fix,
including sunny-versus-shaded identical room histories, weather changes during
warm recovery, cold recovery, normal calibration and holding-point learning.
Two seven-day closed-loop RC3 simulations were rerun with current 1 h defaults,
sensor noise/lag and a restart. On the final day, the constant-condition test
settled at 25% with <0.001 K true-plant error; the varying-weather/target test
had 0.234 K RMSE, 0.495 K peak overshoot and 0.385 K peak deficit, with 61 points
of valve travel. These synthetic results demonstrate both capability and limits;
not house calibration, proof of all-room comfort, or gas savings. The comparable
PID fixture had 0.266 K RMSE and 4025 points travel in the latter test; this is
not a general PID-versus-slow ranking.

Evidence: workspace artifacts/heizung/slow-weather-audit-20261109/.
No curve setting, room target, actuator scope or other calibration mode changes
are part of this patch. Continue the bounded pilot, not an all-room rollout.

## Bounded response-aware correction — 2026-11-09 afternoon

The next Flur/Küche iteration changes only normal cold calibration, not recovery
or the actuator path. When heat is available, both mean and current deficit are
above 0.2 K, and both the full held-window trend and the recent 15–30 minute
trend are at most +0.05 K/h, a correction can occur after 30 minutes (or the
configured shorter maximum window). That stalled-cold case uses the existing
3-point coarse step even at a 0.25–0.35 K deficit. A room already warming faster
keeps its configured observation window and the existing fine-step/projection
rules. Settled holding points still require the full window and existing tests.

This removes the one-point-per-hour crawl through ineffective openings without
hardcoding a 7/8/9% minimum. It does **not** identify a valve dead zone, measure
flow, or learn a room time constant. The 30-minute lower observation bound and
0.05 K/h trend threshold are explicit pilot heuristics. Heat, DHW, context,
feedback and restart interruption still discard incomplete observations.
Recovery-to-baseline handover and use of internal TRV temperatures remain
separate next iterations; neither is silently claimed fixed by this change.
The WC controller and heating curve are outside this rollout. Existing Sonoff
bump/pending/range logic is unchanged.
