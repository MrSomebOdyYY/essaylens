"""Generate v6 stage metadata from the published v5 graphs using ONNX.

Example: python build_position_stages.py path/to/stages [output-directory]
It only copies external weight references. No model weights are required.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import onnx


def build(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    layout = json.loads((source / "layout.json").read_text())
    stages = layout["stages"]
    zero = onnx.load(source / stages[0]["file"], load_external_data=False).graph
    norm = next(node for node in zero.node if node.name == "/model/encoder/LayerNorm/LayerNormalization")
    table = norm.output[0]
    init = {tensor.name: tensor for tensor in zero.initializer}
    source_ranges = {(offset, size): file_offset for file_offset, offset, size in stages[0]["ranges"]}

    for i, stage in enumerate(stages[1:], 1):
        model = onnx.load(source / stage["file"], load_external_data=False)
        graph = model.graph
        slot = next(value for value in graph.input if value.name == table)
        graph.input.remove(slot)
        stage["inputs"].remove(table)
        # The normalized position table belongs to the fixed model, no matter
        # which essay, sentence, or batch enters the layer. Recreate it here so
        # ONNX Runtime can fold the dependent projection nodes at session setup.
        graph.node.insert(0, norm)
        for name in norm.input:
            tensor = deepcopy(init[name])
            fields = {entry.key: entry for entry in tensor.external_data}
            size = int(fields["length"].value)
            original = source_ranges[(int(fields["offset"].value), size)]
            fields["offset"].value = str(stage["size"])
            stage["ranges"].append([original, stage["size"], size])
            stage["size"] += size
            graph.initializer.append(tensor)
        stage["file"] = f"layer-group-{i}-fixed-position-v6.onnx"
        (destination / stage["file"]).write_bytes(model.SerializeToString())
    (destination / "layout-v6.json").write_text(json.dumps(layout, separators=(",", ":")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path, nargs="?")
    args = parser.parse_args()
    build(args.source, args.destination or args.source)
