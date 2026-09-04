"""Centralized registry for Helion codegen backends.

All backend lookup and instantiation should go through this module.
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
import types
from typing import TYPE_CHECKING

from .backend import CuteBackend
from .backend import MetalBackend
from .backend import PallasBackend
from .backend import TileIRBackend
from .backend import TritonBackend

if TYPE_CHECKING:
    from .backend import Backend

_BUILTIN_BACKENDS: list[type[Backend]] = [
    TritonBackend,
    PallasBackend,
    CuteBackend,
    TileIRBackend,
    MetalBackend,
]

_REGISTRY: dict[str, type[Backend]] = {}
_CODEGEN_REPAIR_LOCK = threading.RLock()
_REPAIRED_CODEGEN_NAMES: frozenset[str] = frozenset()

# Backend override env var and the compiled-in default backend name. Kept here
# so both settings defaults and device routing share one definition.
BACKEND_ENV_VAR = "HELION_BACKEND"
DEFAULT_BACKEND_NAME = "triton"


def register_compiler_backend(backend_class: type[Backend]) -> None:
    """Register a compiler backend.

    The backend's ``name`` property is used as the registry key.
    Built-in backends are registered at module load time below.

    Args:
        backend_class: A :class:`Backend` subclass.
    """
    global _REPAIRED_CODEGEN_NAMES

    backend = backend_class()
    with _CODEGEN_REPAIR_LOCK:
        _REGISTRY[backend.name] = backend_class
        codegen_names, _ = _codegen_repair_scope(backend.codegen_name)
        _REPAIRED_CODEGEN_NAMES = _REPAIRED_CODEGEN_NAMES.difference(codegen_names)


def get_backend_class(name: str) -> type[Backend]:
    """Look up a registered backend class by name."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown backend: {name!r}. Available backends: {list_backends()}"
        )
    return _REGISTRY[name]


def list_backends() -> list[str]:
    """Return the names of all registered backends."""
    return list(_REGISTRY.keys())


def find_backend_for_device(device_type: str) -> str | None:
    """Return the name of a registered backend that targets ``device_type``.

    A backend participates in device routing only when its ``device_types``
    class attribute is non-empty. When several backends declare the same
    device type, the first registered wins (registration order is stable).
    Returns ``None`` when no backend targets the device.
    """
    for name, backend_cls in _REGISTRY.items():
        if device_type in backend_cls.device_types:
            return name
    return None


def is_default_backend_selection(backend_name: str) -> bool:
    """Whether ``backend_name`` is eligible for automatic device routing: the
    compiled-in default backend name with ``HELION_BACKEND`` unset. An explicit
    ``HELION_BACKEND`` or a ``Settings(backend=<other name>)`` is never routed;
    a programmatic ``Settings(backend=<default name>)`` is indistinguishable
    from the implicit default and is eligible."""
    return not os.environ.get(BACKEND_ENV_VAR) and backend_name == DEFAULT_BACKEND_NAME


def resolve_backend_name(device_type: str, backend_name: str) -> str:
    """Resolve the backend to use for ``device_type``.

    Returns ``backend_name`` unchanged unless routing applies: the selection
    is the implicit default (:func:`is_default_backend_selection`), that
    backend opts into routing via ``device_types`` but does not target
    ``device_type``, and another registered backend does -- in which case
    that backend's name is returned (first registered wins). Explicit choices
    and devices no backend declares are returned unchanged.

    Called at bind time in :class:`CompileEnvironment`; code that reads the
    backend before then (e.g. ``settings.backend`` at Kernel construction)
    sees the unresolved name. One such construction-time path is the
    zero-warmup TPU compile-capture registration, which therefore still
    requires an explicit ``HELION_BACKEND=pallas``.
    """
    if not is_default_backend_selection(backend_name):
        return backend_name
    selected_cls = get_backend_class(backend_name)
    if selected_cls.device_types and device_type not in selected_cls.device_types:
        routed = find_backend_for_device(device_type)
        if routed is not None:
            return routed
    return backend_name


def all_reserved_launch_param_names() -> frozenset[str]:
    """Union of reserved launch param names across all registered backends.

    Reserving all names ensures kernel portability. A variable name
    that collides with any backend's launch params is avoided regardless
    of which backend is currently active.
    """
    result: set[str] = set()
    for backend_cls in _REGISTRY.values():
        result.update(backend_cls.reserved_launch_param_names())
    return frozenset(result)


def import_backend_codegen() -> None:
    """Import every registered backend's per-op codegen modules.

    Each backend lists its own codegen modules in
    ``helion/_compiler/<backend>/_codegen_modules.py``.  Importing that module
    runs the backend's ``@_decorators.codegen(op, "<backend>")`` /
    ``register_codegen("<backend>")`` handlers, wiring them onto the op and
    aten-lowering objects they extend.

    This is called once from ``helion.language`` after all language ops are
    defined (so the eager registration timing matches the old per-file bottom
    imports).  Because it is driven by the registry, adding a backend requires
    no edits to the core ``helion/language`` files -- only registering the
    backend class (below) and adding its ``_codegen_modules`` module.
    """
    seen: set[str] = set()
    for backend_cls in _REGISTRY.values():
        # e.g. "helion._compiler.cute.backend" -> "helion._compiler.cute".
        # Subclasses that share a folder (e.g. TileIRBackend in the triton
        # package) collapse to one import.
        package = backend_cls.__module__.rsplit(".", 1)[0]
        if package in seen:
            continue
        seen.add(package)
        module = f"{package}._codegen_modules"
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as e:
            # A backend without any per-op codegen modules is allowed; only
            # swallow the missing _codegen_modules module itself, never a
            # broken import inside it.
            if e.name != module:
                raise


def _codegen_repair_scope(
    codegen_name: str,
) -> tuple[frozenset[str], frozenset[str]]:
    entries = [
        (
            backend_class().codegen_name,
            backend_class.__module__.rsplit(".", 1)[0],
        )
        for backend_class in _REGISTRY.values()
    ]
    codegen_names = {codegen_name}
    packages: set[str] = set()
    while True:
        next_packages = packages.union(
            package for name, package in entries if name in codegen_names
        )
        next_codegen_names = codegen_names.union(
            name for name, package in entries if package in next_packages
        )
        if next_packages == packages and next_codegen_names == codegen_names:
            return frozenset(codegen_names), frozenset(packages)
        packages = next_packages
        codegen_names = next_codegen_names


def _reload_backend_codegen(codegen_name: str) -> None:
    if not _REGISTRY:
        for _cls in _BUILTIN_BACKENDS:
            register_compiler_backend(_cls)

    _, packages = _codegen_repair_scope(codegen_name)
    preexisting_modules = frozenset(sys.modules)
    seen_packages: set[str] = set()
    for backend_cls in _REGISTRY.values():
        package = backend_cls.__module__.rsplit(".", 1)[0]
        if package not in packages or package in seen_packages:
            continue
        seen_packages.add(package)
        module_name = f"{package}._codegen_modules"
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            if e.name != module_name:
                raise
            continue

        # Reload the registry module first so all of its leaf modules are bound,
        # then rerun each leaf module's registration decorators.
        if module_name in preexisting_modules:
            importlib.reload(module)
        reloaded_modules: set[str] = set()
        for value in list(vars(module).values()):
            if (
                isinstance(value, types.ModuleType)
                and value.__name__.startswith(package + ".")
                and value.__name__ in preexisting_modules
                and value.__name__ not in reloaded_modules
            ):
                reloaded_modules.add(value.__name__)
                importlib.reload(value)


def repair_backend_codegen(codegen_name: str) -> None:
    """Force-complete backend codegen registrations skipped due to partial import.

    :func:`import_backend_codegen` relies on ``importlib.import_module`` to run
    each backend's ``@_decorators.codegen`` / ``register_codegen`` handlers. When
    a codegen module is already present in ``sys.modules`` in a partially
    initialized state -- a circular import during package init cached it before
    its module-scope registrations ran -- ``import_module`` returns the
    incomplete module and the registrations are silently skipped, leaving
    ``APIFunc._codegen`` empty for that backend. That surfaces later as
    :class:`~helion.exc.BackendImplementationMissing` at codegen time (notably
    under ``torch.compile`` in a packaged runtime, where the init import order
    differs).

    This reloads the requested backend's codegen leaf modules so their
    registrations run. Repairs are serialized across compilation threads and a
    backend is marked repaired only after a successful reload, allowing a failed
    attempt to be retried. Every backend dispatch crosses this barrier before
    reading a handler, so no compiler thread can use a handler while its module
    is being reloaded.
    """
    global _REPAIRED_CODEGEN_NAMES

    if codegen_name in _REPAIRED_CODEGEN_NAMES:
        return
    with _CODEGEN_REPAIR_LOCK:
        from ..language._decorators import _begin_codegen_repair
        from ..language._decorators import _codegen_repair_in_progress
        from ..language._decorators import _end_codegen_repair

        if codegen_name in _REPAIRED_CODEGEN_NAMES:
            return
        if _codegen_repair_in_progress():
            return
        if not _REGISTRY:
            for _cls in _BUILTIN_BACKENDS:
                register_compiler_backend(_cls)
        codegen_names, _ = _codegen_repair_scope(codegen_name)
        _begin_codegen_repair(codegen_names)
        try:
            _reload_backend_codegen(codegen_name)
            _REPAIRED_CODEGEN_NAMES = _REPAIRED_CODEGEN_NAMES.union(codegen_names)
        finally:
            _end_codegen_repair()


# register built-in backends
for _cls in _BUILTIN_BACKENDS:
    register_compiler_backend(_cls)
