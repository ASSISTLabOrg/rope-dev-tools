# Adding a new check kind

A check kind is a plain function plus a matching schema.

## Two steps

1. **Write the function**, in `src/rope_dev_tools/validation/checks/<kind_name>.py`:

   ```python
   from rope_dev_tools.validation.checks import register_kind

   @register_kind("my_new_check")
   def my_new_check(model, *, id=None, out_dir=None, suite_dir=None, **your_own_fields) -> dict:
       ...
       return {...}  # JSON-serializable; include "passed": bool if it has a pass/fail concept
   ```

   `model` is a `validation.model_interfaces.ModelInterface` (`.forecast(start, end)`, `.query(...)`, `.query_grid(...)`). `id`/`out_dir`/`suite_dir` come from the runner; other keywords are your kind's own fields. `model` is optional — see `lonlat_density_plot`'s raw-array calling convention.

2. **Write the schema**, in `rope-registry`: `schemas/checks/<kind_name>.schema.json` for your function's fields, and an entry in `check_kinds.json` with `"status": "draft"`. Flip to `"stable"` once in use.

A check using this kind can then appear in any validation suite JSON: `{"id": "...", "kind": "my_new_check", ...your_own_fields}`.
