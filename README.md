# mir-core
 
Shared Python package for the MSc thesis MIR project.
Successor to `mir-beat-env` from the prototype repo.
 
## Structure
 
```
mir_core/
  models/        — model definitions (beat detection, classifier)
  preprocessing/ — audio preprocessing pipeline
  evaluation/    — evaluation metrics and reporting
  datasets/      — dataset loader adapters and metadata interfaces
  export/        — checkpoint loading and export helpers
mir_env/
  verify_installation.py — environment sanity check
```
 
## Install
 
```bash
pip install -e .
```
 
## Usage
 
```python
import mir_core
```
 
## Rules
 
All other code repos (`mir-train-hpc`, `mir-desktop-app`, `mir-webapp`,
`mir-embedded-ai`) import from this package only.
No sibling repo imports another sibling repo directly.
