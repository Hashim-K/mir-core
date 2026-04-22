# mir_core/experiments

Experiment identity utilities and canonical paper preset registry.

## Hashing

`stable_hash(config_dict)` — SHA-256 of canonical JSON, first 16 hex chars.
Configs must NOT have env vars expanded before hashing (hash `${MIR_DATA_ROOT}`,
not `/home/user/data`). Use `load_config()` from `beatlab.config` for
runtime expansion.

## Presets

Each file in `presets/` is a self-contained canonical experiment definition.
The **filename stem is the experiment hash**. Open any file to see the full
config and paper notes.

### Available presets

| Hash | Key | Paper |
|------|-----|-------|
| `a14aef639058a4c7` | `rapini2024_salsaset_beatnet` | Rapini & Jordanous 2024 (LAMIR) |
| `875884b5ff6ad78a` | `rapini2024_salsaset_bocktcn` | Rapini & Jordanous 2024 (LAMIR) |
| `8fec2f384db1fde8` | `heydari2021_beatnet` | Heydari et al. 2021 (ISMIR) |
| `b75eba7a86c16914` | `davies2019_bocktcn` | Davies & Böck 2019 (EUSIPCO) |

Run `python -c "from mir_core.experiments import PRESETS_BY_KEY; print(list(PRESETS_BY_KEY))"` to list current keys with their hashes.

### Python API

```python
from mir_core.experiments import PRESETS, PRESETS_BY_KEY, get_by_hash, get_by_key

preset = get_by_key("rapini2024_salsaset_beatnet")
preset.hash        # experiment hash
preset.citation    # full citation string
preset.config      # complete config dict (unexpanded)
preset.notes       # list of discrepancy / methodology notes
```

### Adding a preset

1. Build the config dict with raw (unexpanded) env var strings.
2. Compute hash: `python -c "from mir_core.experiments import stable_hash; print(stable_hash(<config>))"`
3. Write `presets/{hash}.json` with `key`, `hash`, `citation`, `notes`, `config` fields.
4. Commit. No Python changes required.

## BeatNet preprocessing discrepancy

**Paper text** (Heydari et al. 2021): 93 ms Hann window, 46 ms hop → ~22 fps.  
**Official implementation** (BeatNet GitHub, used in all published results): win=1408 samples (64 ms), hop=441 samples (20 ms) → **50 fps**.

We follow the official implementation so results are comparable to published
numbers. The paper text likely describes an earlier prototype. All BeatNet
presets use 50 fps accordingly.
