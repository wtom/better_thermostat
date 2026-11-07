# Room control and low-temperature heating: development brief

Status: design only; no new controller implemented or deployed.

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
