# Adding a new check kind

A check kind is a single plain function plus a matching schema — that's the whole extension surface. Kinds don't need to agree with each other on field names, time representation, or output shape.

## Two steps

1. **Write the function**, in `src/rope_dev_tools/validation/checks/<kind_name>.py`:

   ```python
   from rope_dev_tools.validation.checks import register_kind

   @register_kind("my_new_check")
   def my_new_check(model, *, id=None, out_dir=None, suite_dir=None, **your_own_fields) -> dict:
       ...  # do the one discrete thing this kind does
       return {...}  # any JSON-serializable value; include "passed": bool if it has a pass/fail concept
   ```

   `model` is a `validation.model_interfaces.ModelInterface` (`.forecast(start, end)`, `.query(...)`, `.query_grid(...)`) — call it yourself, on whatever window this kind actually needs; there's no suite-level forecast orchestration to hook into. `id`/`out_dir`/`suite_dir` are always supplied by the runner; every other keyword is whatever fields you declared for this kind. A function doesn't have to require `model` at all — see `lonlat_density_plot`'s second calling convention (raw arrays, no model) for a kind that's also useful as a standalone utility outside the validation-suite system entirely.

2. **Write the schema**, in `rope-registry`: add `schemas/checks/<kind_name>.schema.json` describing exactly the fields your function needs (besides `id`/`kind`, which are already covered by the envelope schema), and add an entry to `check_kinds.json` with `"status": "draft"`. Flip to `"stable"` once the function is in real use.

That's it — nothing else needs to change. A check using this kind can now appear in any validation suite JSON: `{"id": "...", "kind": "my_new_check", ...your_own_fields}`.
