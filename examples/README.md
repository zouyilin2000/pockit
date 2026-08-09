# pockit examples

These standalone programs use the same structure: `build_problem`,
`initial_guess`, `solve_problem`, `plot_solution`, and `main`. Each solver
routine checks its result before plotting it.

Install the example dependencies and an Ipopt backend, then run a script from
the repository root:

```console
pip install -e ".[examples,ipopt]"
python examples/brachistochrone.py
```

All examples accept `--no-show` and `--save [PATH]`. Longer examples also
offer `--quick` for a coarse smoke test.

The directory contains 33 maintained examples spanning 23 conservatively
grouped application areas. `_plotting.py` is a shared helper and is not counted
as an example.

| Area | Examples |
| --- | --- |
| Mathematics and variational problems | `brachistochrone.py` |
| Control theory and numerical benchmarks | `double_integrator.py`, `hyper_sensitive.py`, `linear_quadratic_regulator.py` |
| Robotics | `drone_stabilization.py`, `free_flying_robot.py`, `humanoid_motion_retargeting.py`, `humanoid_whole_body_control.py`, `planar_quadrotor.py`, `robot_arm.py` |
| Aerospace engineering and astrodynamics | `orbit_transfer.py`, `multiphase_two_stage_rocket.py`, `rocket_powered_descent.py` |
| Planetary science and small-body operations | `asteroid_soft_landing.py` |
| Astronomy and precision pointing | `flexible_telescope_slew.py` |
| Physical oceanography | `ocean_inertial_current.py` |
| Structural engineering | `structural_vibration_control.py` |
| Chemistry and reaction engineering | `multiphase_batch_reactor.py`, `parallel_reaction_selectivity.py` |
| Transportation engineering | `multiphase_electric_vehicle.py` |
| Epidemiology and public health | `sir_epidemic_control.py` |
| Biomedical engineering and neuroscience | `neural_stimulation.py` |
| Building science | `building_hvac_control.py` |
| Energy systems and electric power | `battery_energy_arbitrage.py` |
| Pharmacology | `pharmacokinetic_dosing.py` |
| Quantum physics | `quantum_state_transfer.py` |
| Macroeconomics | `ramsey_growth.py` |
| Quantitative finance | `optimal_trade_execution.py` |
| Water-resources engineering | `reservoir_flood_control.py` |
| Agricultural water management | `irrigation_scheduling.py` |
| Ecology and resource economics | `bioeconomic_fishery.py` |
| Communications engineering | `wireless_data_transmission.py` |
| Scientific machine learning | `neural_ode_xor.py` |

Generated figures and symbolic compilation caches are runtime artifacts and
do not belong in this directory.
