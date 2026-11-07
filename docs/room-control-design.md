# Room control and low-temperature heating: development brief

Status: experimental simulation prototype implemented; no new production controller deployed.

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
