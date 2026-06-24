# Bock TCN Baseline Checkpoints

These resources package the canonical madmom TCN beat model ensemble from:

`madmom/madmom/models/beats/2019/beats_tcn_[1-8].pkl`

The selector names map to the eight ensemble members:

| Selector | Source file |
| --- | --- |
| `baseline` | `beats_tcn_1.pkl` |
| `baseline_alt0` | `beats_tcn_2.pkl` |
| `baseline_alt1` | `beats_tcn_3.pkl` |
| `baseline_alt2` | `beats_tcn_4.pkl` |
| `baseline_alt3` | `beats_tcn_5.pkl` |
| `baseline_alt4` | `beats_tcn_6.pkl` |
| `baseline_alt5` | `beats_tcn_7.pkl` |
| `baseline_alt6` | `beats_tcn_8.pkl` |

These are madmom pickle resources for the published inference ensemble, not
native PyTorch `BockTCN` state dictionaries.
