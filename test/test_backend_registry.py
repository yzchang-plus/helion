from __future__ import annotations

import os
from types import SimpleNamespace
from typing import cast
import unittest
from unittest import mock

import torch

import helion
from helion import exc
from helion._compiler import backend_registry
from helion._compiler.aten_lowering import AtenLowering
from helion._compiler.backend import Backend
from helion._compiler.backend_registry import _REGISTRY
from helion._compiler.backend_registry import BACKEND_ENV_VAR
from helion._compiler.backend_registry import all_reserved_launch_param_names
from helion._compiler.backend_registry import get_backend_class
from helion._compiler.backend_registry import list_backends
from helion._compiler.backend_registry import register_compiler_backend
from helion._compiler.backend_registry import repair_backend_codegen
from helion._compiler.compile_environment import CompileEnvironment
import helion.language as hl
from helion.language._decorators import APIFunc
from helion.language._decorators import api
from helion.language._decorators import codegen


class TestBackendRegistry(unittest.TestCase):
    def test_list_backends_contains_all_builtins(self) -> None:
        names = list_backends()
        for expected in ("triton", "pallas", "cute", "tileir", "metal"):
            self.assertIn(expected, names)

    def test_get_backend_class_and_instantiate_all(self) -> None:
        for name in list_backends():
            cls = get_backend_class(name)
            instance = cls()
            self.assertIsInstance(instance, Backend)
            self.assertEqual(instance.name, name)

    def test_get_backend_class_raises_for_unknown(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown backend: 'nonexistent'"):
            get_backend_class("nonexistent")

    def test_default_fake_subscript_shape(self) -> None:
        backend = get_backend_class("triton")()
        tensor = torch.empty(4, 8)

        self.assertEqual(
            backend.fake_subscript_shape(tensor, [slice(None), None, slice(None)]),
            [4, 1, 8],
        )
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "narrowing kernel-tensor subscripts"
        ):
            backend.fake_subscript_shape(tensor, [0, slice(None)])
        with self.assertRaises(exc.InvalidIndexingType):
            backend.fake_subscript_shape(tensor, [object(), slice(None)])
        with self.assertRaises(exc.InvalidIndexingType):
            backend.fake_subscript_shape(tensor, [slice(0, 4, 0), slice(None)])

    def test_register_custom_backend(self) -> None:
        class _TestBackend(Backend):
            @property
            def name(self) -> str:
                return "_test_custom"

            def dtype_str(self, dtype: object) -> str:
                return ""

            def acc_type(self, dtype: object) -> str:
                return ""

            @property
            def default_launcher_name(self) -> str:
                return ""

            @property
            def constexpr_type(self) -> str:
                return ""

            @property
            def function_decorator(self) -> str:
                return ""

            @property
            def library_imports(self) -> dict[str, str]:
                return {}

        register_compiler_backend(_TestBackend)
        try:
            self.assertIn("_test_custom", list_backends())
            self.assertIs(get_backend_class("_test_custom"), _TestBackend)
            # custom backends default to experimental=True
            self.assertTrue(_TestBackend().experimental)
        finally:
            _REGISTRY.pop("_test_custom", None)

    def test_all_reserved_launch_param_names_is_union(self) -> None:
        result = all_reserved_launch_param_names()
        self.assertIsInstance(result, frozenset)
        # must contain at least the triton names
        for expected in ("grid", "warmup", "num_warps", "num_stages"):
            self.assertIn(expected, result)
        # must be the union of all backends
        for cls in _REGISTRY.values():
            self.assertTrue(cls.reserved_launch_param_names().issubset(result))

    def test_builtin_device_types(self) -> None:
        from helion._compiler.backend_registry import find_backend_for_device

        # Triton lowers for NVIDIA (cuda; ROCm reports the same type), Intel
        # GPUs (xpu) and MTIA; pallas targets TPU; metal targets MPS.
        self.assertEqual(find_backend_for_device("cuda"), "triton")
        self.assertEqual(find_backend_for_device("xpu"), "triton")
        self.assertEqual(find_backend_for_device("mtia"), "triton")
        self.assertEqual(find_backend_for_device("tpu"), "pallas")
        self.assertEqual(find_backend_for_device("mps"), "metal")
        # No builtin backend opts into routing for cpu or unknown devices.
        self.assertIsNone(find_backend_for_device("cpu"))
        self.assertIsNone(find_backend_for_device("does_not_exist"))
        # Experimental backends opt out of device routing entirely (in
        # particular TileIRBackend must not inherit Triton's device_types).
        self.assertEqual(get_backend_class("cute").device_types, frozenset())
        self.assertEqual(get_backend_class("tileir").device_types, frozenset())

    def _register_routing_backend(
        self, name: str, device_types: frozenset[str]
    ) -> type[Backend]:
        from helion._compiler.triton.backend import TritonBackend

        class _RoutingTestBackend(TritonBackend):
            @property
            def name(self) -> str:
                return name

        _RoutingTestBackend.device_types = device_types
        # register_compiler_backend also mutates the _REPAIRED_CODEGEN_NAMES
        # global; isolate it, and unregister on cleanup so test registration
        # leaves no global state behind.
        patcher = mock.patch.object(
            backend_registry, "_REPAIRED_CODEGEN_NAMES", frozenset()
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(_REGISTRY.pop, name, None)
        register_compiler_backend(_RoutingTestBackend)
        return _RoutingTestBackend

    def test_resolve_routes_to_backend_declaring_device(self) -> None:
        from helion._compiler.backend_registry import resolve_backend_name

        self._register_routing_backend("_route_test", frozenset({"cpu"}))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(BACKEND_ENV_VAR, None)
            self.assertEqual(resolve_backend_name("cpu", "triton"), "_route_test")

    def test_resolve_does_not_route_when_backend_explicit(self) -> None:
        from helion._compiler.backend_registry import resolve_backend_name

        # Use a device a builtin backend claims (mps -> metal): a broken
        # env-var guard would reroute to "metal", making this discriminating.
        with mock.patch.dict(os.environ, {BACKEND_ENV_VAR: "triton"}, clear=False):
            self.assertEqual(resolve_backend_name("mps", "triton"), "triton")

    def test_resolve_does_not_route_programmatic_backend_choice(self) -> None:
        # A backend chosen programmatically via Settings(backend=...) (with no
        # HELION_BACKEND set) is an explicit choice and must never be rerouted,
        # even when another registered backend targets the active device.
        from helion._compiler.backend_registry import resolve_backend_name

        self._register_routing_backend("_route_explicit", frozenset({"tpu"}))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(BACKEND_ENV_VAR, None)
            # Device "tpu" makes this discriminating: a broken explicit-name
            # guard would reroute to the builtin "pallas" backend.
            self.assertEqual(
                resolve_backend_name("tpu", "_route_explicit"), "_route_explicit"
            )

    def test_resolve_keeps_default_when_no_backend_declares_device(self) -> None:
        from helion._compiler.backend_registry import resolve_backend_name

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(BACKEND_ENV_VAR, None)
            self.assertEqual(resolve_backend_name("cpu", "triton"), "triton")

    def test_resolve_skips_routing_when_default_backend_opts_out(self) -> None:
        # A default backend that opts out of device routing (empty
        # device_types) is never rerouted, even when another registered
        # backend declares the active device.
        from helion._compiler.backend_registry import resolve_backend_name
        from helion._compiler.triton.backend import TritonBackend

        self._register_routing_backend("_route_test", frozenset({"cpu"}))
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch.object(TritonBackend, "device_types", frozenset()),
        ):
            os.environ.pop(BACKEND_ENV_VAR, None)
            self.assertEqual(resolve_backend_name("cpu", "triton"), "triton")

    def test_min_dot_size_is_route_aware_for_tpu(self) -> None:
        # TPU MXU tile floors must apply for a tpu device even when the
        # implicit default backend is still "triton" (a kernel routed to
        # pallas at bind time), not just with HELION_BACKEND=pallas.
        from helion._compat import min_dot_size

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(BACKEND_ENV_VAR, None)
            # A tpu-typed device (torch_tpu registers the type on TPU
            # machines); a Mock stands in since this torch build lacks it.
            tpu_device = mock.Mock()
            tpu_device.type = "tpu"
            self.assertEqual(
                min_dot_size(tpu_device, torch.float16, torch.float16),
                (8, 128, 128),
            )
            self.assertEqual(
                min_dot_size(torch.device("cpu"), torch.float16, torch.float16),
                (16, 16, 16),
            )

    def test_environment_applies_device_routing_end_to_end(self) -> None:
        # One integration test: CompileEnvironment routes the default backend
        # to the backend declaring the active device at bind time.
        self._register_routing_backend("_route_test", frozenset({"cpu"}))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(BACKEND_ENV_VAR, None)
            env = CompileEnvironment(
                torch.device("cpu"),
                helion.Settings(backend="triton"),
            )
        self.assertEqual(env.backend_name, "_route_test")


class TestBackendCodegenRepair(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.object(
            backend_registry, "_REPAIRED_CODEGEN_NAMES", frozenset()
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_repair_runs_once_per_codegen_backend(self) -> None:
        with mock.patch.object(
            backend_registry, "_reload_backend_codegen"
        ) as reload_mock:
            repair_backend_codegen("triton")
            repair_backend_codegen("triton")

        reload_mock.assert_called_once_with("triton")

    def test_failed_repair_can_be_retried(self) -> None:
        with mock.patch.object(
            backend_registry,
            "_reload_backend_codegen",
            side_effect=[RuntimeError("broken import"), None],
        ) as reload_mock:
            with self.assertRaisesRegex(RuntimeError, "broken import"):
                repair_backend_codegen("triton")
            repair_backend_codegen("triton")
            repair_backend_codegen("triton")

        self.assertEqual(reload_mock.call_count, 2)

    def test_controlled_repair_allows_reregistration(self) -> None:
        @api()
        def op() -> None:
            return None

        def original(state: object) -> str:
            return "original"

        def replacement(state: object) -> str:
            return "replacement"

        codegen(op, "triton")(original)

        def reload_backend(codegen_name: str) -> None:
            self.assertEqual(codegen_name, "triton")
            codegen(op, "triton")(replacement)

        with mock.patch.object(
            backend_registry,
            "_reload_backend_codegen",
            side_effect=reload_backend,
        ):
            repair_backend_codegen("triton")

        self.assertIs(
            dict.__getitem__(cast("APIFunc", op)._codegen, "triton"), replacement
        )

    def test_repair_rejects_second_registration_for_same_slot(self) -> None:
        @api()
        def op() -> None:
            return None

        def first(state: object) -> str:
            return "first"

        def second(state: object) -> str:
            return "second"

        def reload_backend(codegen_name: str) -> None:
            codegen(op, codegen_name)(first)
            codegen(op, codegen_name)(second)

        with (
            mock.patch.object(
                backend_registry,
                "_reload_backend_codegen",
                side_effect=reload_backend,
            ),
            self.assertRaisesRegex(
                AssertionError, "codegen already registered for backend 'triton'"
            ),
        ):
            repair_backend_codegen("triton")

    def test_real_reload_restores_missing_handler(self) -> None:
        codegen_impls = cast("APIFunc", hl.load)._codegen
        original = dict.__getitem__(codegen_impls, "triton")
        del codegen_impls["triton"]
        self.addCleanup(codegen_impls.__setitem__, "triton", original)

        restored = codegen_impls["triton"]

        self.assertEqual(restored.__module__, "helion._compiler.triton.memory_ops")
        self.assertIsNot(restored, original)

    def test_resident_plan_metadata_survives_codegen_reload(self) -> None:
        from helion._compiler.pallas import view_ops

        plan = view_ops._ResidentPlan(False, (), None)
        node = mock.Mock(meta={view_ops.RESIDENT_PLAN_META: plan})

        class ReloadedResidentPlan:
            pass

        with mock.patch.object(view_ops, "_ResidentPlan", ReloadedResidentPlan):
            self.assertIs(view_ops._resident_plan(node), plan)


class TestAtenCodegenRepair(unittest.TestCase):
    def test_exact_backend_passes_repair_barrier(self) -> None:
        def triton(ctx: object, node: object) -> str:
            return "triton"

        lowering = AtenLowering(codegen_impls={"triton": triton})
        env = SimpleNamespace(codegen_name="triton", backend_name="triton")

        with (
            mock.patch.object(CompileEnvironment, "current", return_value=env),
            mock.patch(
                "helion._compiler.backend_registry.repair_backend_codegen"
            ) as repair_mock,
        ):
            result = lowering.codegen(mock.Mock(), mock.Mock())

        self.assertEqual(result, "triton")
        repair_mock.assert_called_once_with("triton")

    def test_missing_backend_repairs_before_common_fallback(self) -> None:
        def common(ctx: object, node: object) -> str:
            return "common"

        def triton(ctx: object, node: object) -> str:
            return "triton"

        lowering = AtenLowering(codegen_impls={"common": common})
        env = SimpleNamespace(codegen_name="triton", backend_name="triton")

        def repair(codegen_name: str) -> None:
            lowering.codegen_impls[codegen_name] = triton

        with (
            mock.patch.object(CompileEnvironment, "current", return_value=env),
            mock.patch(
                "helion._compiler.backend_registry.repair_backend_codegen",
                side_effect=repair,
            ) as repair_mock,
        ):
            result = lowering.codegen(mock.Mock(), mock.Mock())

        self.assertEqual(result, "triton")
        repair_mock.assert_called_once_with("triton")

    def test_rejects_duplicate_registration_outside_repair(self) -> None:
        lowering = AtenLowering()

        def first(ctx: object, node: object) -> None:
            return None

        def second(ctx: object, node: object) -> None:
            return None

        lowering.register_codegen("triton")(first)
        with self.assertRaisesRegex(
            AssertionError, "codegen already registered for backend 'triton'"
        ):
            lowering.register_codegen("triton")(second)


if __name__ == "__main__":
    unittest.main()
