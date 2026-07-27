# mir_core/beats/experiments

Beat tracking experiment identity utilities and canonical paper preset registry.

## Hashing

`experiment_hash(config_dict)` — returns `btk-{SHA-256[:16]}`.  
Format: always 20 characters (`btk-` + 16 hex chars).  
Configs must NOT have env vars expanded before hashing.

## Presets

Each file in `presets/` is a self-contained canonical experiment definition.
The **filename stem is the experiment hash** (e.g. `btk-a14aef639058a4c7.json`).

These presets reproduce historical paper-specific dataset universes and carry
`data.allow_legacy_split: true` explicitly. New multi-lab thesis experiments
must use the universal `data.split_plan` runtime in `mir-train-hpc`; the opt-in
prevents a paper preset from silently being mistaken for that shared protocol.

### Available presets

| Hash | Key | Paper |
|------|-----|-------|
| `btk-724a5e4d0d1e8edf` | `rapini2024_salsaset_beatnet` | Rapini & Jordanous 2024 (LAMIR) |
| `btk-68a1f54a14e999c9` | `rapini2024_salsaset_bocktcn` | Rapini & Jordanous 2024 (LAMIR) |
| `btk-65bd103f0edc1432` | `heydari2021_beatnet` | Heydari et al. 2021 (ISMIR) |
| `btk-6e688b9bba34efba` | `davies2019_bocktcn` | Davies & Böck 2019 (EUSIPCO) |

Run `python -c "from mir_core.beats.experiments import PRESETS_BY_KEY; print(list(PRESETS_BY_KEY))"` to list current keys.

### Python API

```python
from mir_core.beats.experiments import (
    PRESETS, PRESETS_BY_KEY, get_by_hash, get_by_key, experiment_hash
)

preset = get_by_key("rapini2024_salsaset_beatnet")
preset.hash        # "btk-724a5e4d0d1e8edf"
preset.citation    # full citation string
preset.config      # complete config dict (unexpanded)
preset.notes       # list of discrepancy / methodology notes

# Compute hash for a new config
h = experiment_hash(config_dict)  # "btk-{16 hex chars}"
```

### Adding a preset

1. Build the config dict with raw (unexpanded) env var strings.
2. Compute hash:
   ```bash
   python -c "from mir_core.beats.experiments import experiment_hash; import json; print(experiment_hash(json.load(open('my_preset.json'))['config']))"
   ```
3. Write `presets/{hash}.json` with `key`, `hash`, `citation`, `notes`, `config` fields.
4. Commit. No Python changes required.

## BeatNet preprocessing discrepancy

**Paper text** (Heydari et al. 2021): 93 ms window, 46 ms hop → ~22 fps.  
**Official implementation** (all published results): win=1408 samples (64 ms), hop=441 samples (20 ms) → **50 fps**.  
All BeatNet presets use 50 fps to match published numbers.
