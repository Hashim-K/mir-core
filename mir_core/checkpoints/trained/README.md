# Packaged trained models

This directory contains the completed five-fold model candidates used by
BeatLab, ClassifierLab, SystemLab, and deployment applications. Checkpoints are
stored as:

```text
<model-family>/<target>/<condition>/checkpoints/seed_42_fold_<fold>.pt
```

Beat-tracking bundles attach both the original stock online postprocessors and
every completed tuned causal candidate under
`postprocessors/<candidate>/params.json`. Stock choices use `stock-*` names;
tuned choices use matching `tuned-*` names. The public id pattern is
`<kind>-<method>[-<variant>]`; historical experiment ids remain in each tuned
record as `source_id` provenance. Classifier bundles have no beat postprocessor
because their causal routing policy is embedded in each checkpoint.

The stable method ids are `stock-1d`/`tuned-1d`,
`stock-dbn`/`tuned-dbn`, and
`stock-particle-filter`/`tuned-particle-filter`. `tuned-1d` is present only for
bundles with the completed immediate causal-activation candidate; the older
12-frame past-snap candidate is not packaged as a runnable tuned option.

All bundles are deliberately marked `candidate`. Model and postprocessor
selection remains a separate scientific decision; the `default_postprocessor`
is only a convenient runnable default for validation smoke tests.

Use the stable Python API instead of constructing paths:

```python
from mir_core.checkpoints import (
    trained_checkpoint_path,
    trained_postprocessor_path,
)

checkpoint = trained_checkpoint_path(
    "beatnet", "candombe", "scratch", fold_index=2
)
tuned_postprocessor = trained_postprocessor_path(
    "beatnet", "candombe", "scratch", "tuned-dbn"
)
stock_postprocessor = trained_postprocessor_path(
    "beatnet", "candombe", "scratch", "stock-dbn"
)
classifier = trained_checkpoint_path(
    "classifier", "latin_router", "efficientat", fold_index=2
)
```

`TrainedModelBundle.postprocessors` contains both groups, while
`stock_postprocessors` and `tuned_postprocessors` expose them separately. The
default remains `tuned-dbn`; choose the corresponding `stock-*` name explicitly
to reproduce the original behavior. The legacy combined stock catalog remains
available through `beatnet_stock_postprocessor_selection_path()`.

Every file is byte-bound by its bundle manifest. All current bundles use split
contract `e2e-537f350dbaf7e925`; consumers must select the same fold before
combining classifier and beat-tracking components.
