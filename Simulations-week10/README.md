# Week-10 conference dataset pipeline

This directory defines the reproducible InTAS dataset matrix used for the
conference-paper evaluation. It does not modify `mosaic-week9`. The week-10
application is a private copy under `CamApp/` with deterministic MOSAIC-seeded
randomness and append-only JSON output.

## Experiment matrix

The four settings use the original full-day routes from midnight, enable V2X
communication for exactly 300 seconds, and stop at the end of that window.
Vehicle applications preserve SUMO traffic history but defer radio and
perception activation until a vehicle is inside the configured area during the
communication window. This avoids accumulating unused OMNeT++ nodes all day.

| Setting | Communication | Simulation stop |
|---|---:|---:|
| `InTAS_urban_2AM_300sec` | 7200–7500 s | 7500 s |
| `InTAS_urban_7AM_300sec` | 25200–25500 s | 25500 s |
| `InTAS_highway_2AM_300sec` | 7200–7500 s | 7500 s |
| `InTAS_highway_7AM_300sec` | 25200–25500 s | 25500 s |

Simulation seed `1` is enabled initially. Seeds `2` and `3` are already
defined in `experiment.json`; add them to `active_simulation_seeds` when the
paper requires three independent clean simulations. Attacks use seeds 1–3 and
the existing 20% attacker ratio.

The runtime is the pinned image:

```text
ghcr.io/vs-uulm/veremi-nextgen:recreation@sha256:cdddb6e0ddcb350f9fb3602128b8fa046ecc0b5699082566cdb4ff01db68d4b8
```

MOSAIC runs with its federate watchdog disabled (`-w 0`). Daytime InTAS traffic
produces synchronization steps longer than both the image's 30-second default
and a tested five-minute threshold. Exact stop-time and completion-code checks
still prevent an interrupted or incomplete run from being promoted.

## Fresh-machine setup

The runner targets Linux or WSL2 and requires Docker, Python 3.12 or newer,
JDK 17 or newer, and Maven. Docker must be available to the current user
without an interactive `sudo` prompt. From the repository root, create the
ignored local Python environment and pull the pinned simulation image:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r Simulations-week10/requirements.txt

docker pull \
  ghcr.io/vs-uulm/veremi-nextgen:recreation@sha256:cdddb6e0ddcb350f9fb3602128b8fa046ecc0b5699082566cdb4ff01db68d4b8
```

Do not commit `.venv`; `requirements.txt` is the reproducible description of
its pipeline dependencies. The pipeline uses GNU `cp --reflink=auto`, Linux
user/group IDs, and Docker bind mounts, so native Windows and macOS hosts are
not currently supported directly.

## Commands

Run commands from the repository root:

```bash
# Build the week-10 app, prepare all active scenarios, and validate configuration.
.venv/bin/python Simulations-week10/run_week10.py validate

# Short end-to-end Docker run. This does not create a production dataset.
.venv/bin/python Simulations-week10/run_week10.py smoke

# Run production simulations sequentially (the safe default for dense 7 AM traffic).
.venv/bin/python Simulations-week10/run_week10.py simulate

# Generate the 24 attacked datasets after clean simulations finish.
.venv/bin/python Simulations-week10/run_week10.py attack

# Run missing clean simulations and then all attacks.
.venv/bin/python Simulations-week10/run_week10.py all
```

Use `--settings`, `--simulation-seeds`, `--attacks`, and `--attack-seeds` to
run a subset. `--jobs N` controls parallelism, but production simulations
default to one job because a single dense 7 AM container can use roughly
17 GiB RAM. Existing completed destinations are validated and skipped; the
runner never silently overwrites them.

Examples:

```bash
.venv/bin/python Simulations-week10/run_week10.py simulate \
  --settings InTAS_highway_2AM_300sec --simulation-seeds 1 --jobs 1

.venv/bin/python Simulations-week10/run_week10.py attack \
  --settings InTAS_highway_2AM_300sec \
  --attacks constantPositionOffset --attack-seeds 1 --jobs 1
```

## Output and recovery

Clean datasets are written to:

```text
seeded-simulations-300sec/<simulation-seed>/<setting>/{cam,cpm,ego}
```

Attacked datasets are written to:

```text
attacks/<attack>/<attack-seed>/simulation-seed-<simulation-seed>/<setting>/{cam,cpm,ego}
```

Each completed dataset contains `metadata.json`. Container and attack logs are
under `run-logs/`. In-progress data remains under `.work/` and is never
promoted if the container, finalization, or validation step fails. Remove only
the relevant incomplete `.work` directory if manual cleanup is needed. On a
retry, the runner first recovers a fully completed staging run and otherwise
discards only that incomplete staging directory. Completed datasets should be
retained as immutable paper artifacts.

Generated scenarios, datasets, logs, and build/runtime work are ignored by Git.
Commit the manifest, runner, dependency requirements, week-10 application
source, tests, and this README.
