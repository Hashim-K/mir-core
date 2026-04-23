"""Official BEAST model code vendored from WildHoneyPie/BEAST.

The files in this package preserve the upstream model path and parameter names
for checkpoint compatibility. Only package-relative imports are adjusted.

Upstream: https://github.com/WildHoneyPie/BEAST
Commit: f680f17f9d175bd76e152549fe2281853e228ae6
License: MIT, see LICENSE in this directory.
"""

from .StreamingTransformer import TransformerModel

__all__ = ["TransformerModel"]
