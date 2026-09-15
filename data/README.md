# Dataset

This project fine-tunes on **GLUE MRPC** (Microsoft Research Paraphrase
Corpus), loaded automatically at runtime via Hugging Face's `datasets`
library:

```python
from datasets import load_dataset
load_dataset("nyu-mll/glue", "mrpc")
```

No files are stored in this folder — `datasets` downloads and caches
MRPC (~1MB, 3.7k train / 408 validation examples) directly to Hugging
Face's local cache the first time `src/train.py` is run. This folder
exists only to document that, since Git doesn't track empty directories.
