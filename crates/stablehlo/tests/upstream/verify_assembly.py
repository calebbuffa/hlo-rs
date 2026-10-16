"""Optional independent parser/verifier, not a Rust runtime dependency."""

import os
from pathlib import Path
import sys

import jaxlib

# Windows does not search sibling extension directories automatically.
if os.name == "nt":
    root = Path(jaxlib.__file__).parent
    dll_directories = [
        os.add_dll_directory(str(root)),
        os.add_dll_directory(str(root / "mlir" / "_mlir_libs")),
    ]

from jaxlib.mlir import ir
from jaxlib.mlir._mlir_libs import _jax_mlir_ext
from jaxlib.mlir.dialects import stablehlo

version = stablehlo.get_current_version()
if (jaxlib.__version__, version) != ("0.11.2", "1.20.0"):
    raise RuntimeError(
        f"Upstream audit requires jaxlib 0.11.2 / StableHLO 1.20.0, "
        f"got {jaxlib.__version__} / {version}"
    )
print(f"VERSION\t{jaxlib.__version__}\t{version}", flush=True)

registry = ir.DialectRegistry()
_jax_mlir_ext.register_dialects(registry)
execute = sys.argv[1:] == ["--execute"]
if execute:
    import numpy as np
    from jaxlib import xla_client, _jax

    client = xla_client.make_cpu_client()
    devices = client.devices()
with ir.Context() as context:
    context.append_dialect_registry(registry)
    stablehlo.register_dialect(context)
    context.allow_unregistered_dialects = False
    for index, line in enumerate(sys.stdin):
        source = bytes.fromhex(line.strip()).decode("utf-8")
        try:
            arguments = []
            if execute:
                count, payload = source.split("\n", 1)
                rows = payload.split("\n", int(count))
                source = rows[-1]
                for row in rows[:-1]:
                    dtype, dimensions, data = row.split("\t")
                    dimensions = tuple(map(int, dimensions.split(","))) if dimensions else ()
                    array = np.frombuffer(bytes.fromhex(data), dtype=np.dtype(dtype)).copy()
                    array = array.reshape(dimensions)
                    arguments.append(
                        _jax.dlpack_managed_tensor_to_buffer(
                            array.__dlpack__(), devices[0], None, True
                        )
                    )
            target = None
            if sys.argv[1:] == ["--serialize"]:
                target, source = source.split("\t", 1)
            module = ir.Module.parse(source, context=context)
            if not module.operation.verify():
                raise RuntimeError(f"Verifier returned false for module {index}")
            if execute:
                executable = client.compile_and_load(str(module), devices)
                arrays = [np.asarray(value) for value in executable.execute(arguments)]
                result = "\n".join(
                    "\t".join(
                        (
                            array.dtype.str,
                            ",".join(map(str, array.shape)),
                            array.tobytes(order="C").hex(),
                        )
                    )
                    for array in arrays
                )
            elif target is not None:
                result = stablehlo.serialize_portable_artifact(module, target).hex()
            else:
                result = str(module)
            print(f"{index}\tOK\t{result.encode('utf-8').hex()}", flush=True)
        except (ir.MLIRError, RuntimeError) as error:
            print(f"{index}\tERROR\t{str(error).encode('utf-8').hex()}", flush=True)
