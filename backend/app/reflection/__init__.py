"""Reflection + auto-fix layer.

Diagnoses capability failures and applies deterministic, artifact-level
repairs before escalating to the planner. See ``docs/reflection_layer_design.md``.

Importing this package registers every built-in repair in ``REPAIR_REGISTRY``
(registration is an import side effect of ``repairs``). Selection and the reflect
node depend on the registry being populated, so this import must not be pruned as
"unused" — it is the guarantee that ``select_repair`` sees the built-in repairs
regardless of module import order.
"""

from app.reflection import repairs as _repairs  # noqa: F401  (registers repairs)
