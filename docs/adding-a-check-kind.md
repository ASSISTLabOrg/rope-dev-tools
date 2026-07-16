# Adding a new check kind

A check kind is a plain function, registered under a name.

## One step

Write the function in `src/rope_dev_tools/validation/checks/<kind_name>.py`:

```python
from rope_dev_tools.validation.checks import register_kind

@register_kind("my_new_check")
def my_new_check(model, *, id=None, out_dir=None, suite_dir=None, **your_own_fields) -> dict:
    ...
    return {...}  # JSON-serializable; include "passed": bool if it has a pass/fail concept
```

`model` is a `validation.model_interfaces.ModelInterface` (`.forecast(start, end)`, `.query(...)`, `.query_grid(...)`, `.backend_name`). `id`/`out_dir`/`suite_dir` come from the runner; other keywords are your kind's own fields. There is no separate schema to write — required fields with no default raise `TypeError` if omitted; anything a signature can't express (grid bounds, row alignment, horizon limits) is a `raise ValueError(...)` in the function body.

A check using this kind can then appear in any validation suite JSON: `{"id": "...", "kind": "my_new_check", ...your_own_fields}`.

## Backend gate

If a kind should only run against a real exported model (e.g. because of memory/perf), don't special-case the kind — accept a field like `requires_exported_model` and check it inline:

```python
if requires_exported_model and model.backend_name != "exported_dir":
    raise ValueError(f"check {id!r} requires an exported model, got backend {model.backend_name!r}")
```

## Saved data + standalone regeneration

If your kind produces plots, save the comparison data it plotted from via `validation/data_artifacts.py` (`save_csv` for point/scalar series, `save_npz` for grids), list it in the output as `"data": [...]` next to `"plots": [...]`, and add a matching `replot_<kind_name>(loaded, *, id, out_dir, **fields)` function in the same module. `scripts/generate_validation_plots.py` calls `replot_<kind_name>` to re-render plots from saved data alone, with no model involved.
