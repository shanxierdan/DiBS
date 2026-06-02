# External Sources Manifest (Table4 Extension)

This manifest tracks candidate external datasets for the unified Table4 extension experiment.

Policy: strict admission only.

- License must be explicit and compatible with research use.
- Download path must be reproducible.
- Format must be parseable into `puzzle` + `solution`.
- If any requirement fails, source is rejected and synthetic generation fills the gap.

## Fields

- `source_id`: stable source identifier
- `task_family`: `generalized_sudoku` or `nqueens`
- `size`: `4x4` / `16x16` / `25x25` / `8` / `9` / `10`
- `url`: canonical source URL
- `license`: SPDX-like value or explicit license text
- `download`: direct download endpoint or deterministic retrieval instructions
- `format`: raw format and parser notes
- `status`: `candidate` / `accepted` / `rejected`
- `reason`: rejection reason if applicable

## Candidate Sources

These are seeded examples and must still pass `dataset/discover_external_data.py`.

```json
[
  {
    "source_id": "local_generalized_sudoku_4x4",
    "task_family": "generalized_sudoku",
    "size": "4x4",
    "url": "file://dataset/generalized_sudoku/sudoku_4x4_puzzles.txt",
    "license": "local-project-data",
    "download": "already-in-repo",
    "format": "plain-text puzzle lines (length 16)",
    "status": "candidate"
  },
  {
    "source_id": "local_nqueens_8x8",
    "task_family": "nqueens",
    "size": "8",
    "url": "file://dataset/n_queens/nqueens_8x8_puzzles.txt",
    "license": "local-project-data",
    "download": "already-in-repo",
    "format": "plain-text puzzle lines (length 64, '.'/'Q')",
    "status": "candidate"
  },
  {
    "source_id": "local_nqueens_9x9",
    "task_family": "nqueens",
    "size": "9",
    "url": "file://dataset/n_queens/nqueens_9x9_puzzles.txt",
    "license": "local-project-data",
    "download": "already-in-repo",
    "format": "plain-text puzzle lines (length 81, '.'/'Q')",
    "status": "candidate"
  },
  {
    "source_id": "local_nqueens_10x10",
    "task_family": "nqueens",
    "size": "10",
    "url": "file://dataset/n_queens/nqueens_10x10_puzzles.txt",
    "license": "local-project-data",
    "download": "already-in-repo",
    "format": "plain-text puzzle lines (length 100, '.'/'Q')",
    "status": "candidate"
  }
]
```

## Output Artifacts

`dataset/discover_external_data.py` writes:

- `dataset/accepted_sources.json`
- `dataset/rejected_sources.json`

These files are consumed by `dataset/build_table4_extension_data.py`.
