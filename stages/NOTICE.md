# Model graph credits

The ONNX graph fragments and offset table in this directory were derived from the fp32 conversion of Desklib's AI Text Detector v1.01 by GabeuxDev. EssayLens groups the original layers for browser execution. The numerical weights are fetched separately from the pinned model repository.

- Original model and classifier by Desklib. [Model card](https://huggingface.co/desklib/ai-text-detector-v1.01)
- ONNX conversion by GabeuxDev. [Pinned conversion](https://huggingface.co/GabeuxDev/ai-text-detector-v1.01-onnx/tree/1017cdcc0fa315991a32c5a0eb7e1f8c5e74bf34)

Both model cards identify the model license as MIT.

## Sentence inference metadata

`layout-v6.json` uses the original first group and three alternate graphs for sentence batches. Each alternate graph includes the original position-table normalization and references the same three tensors in the cached model weights. This lets ONNX Runtime precompute fixed position projections during session setup. The full-essay score and independent checks near the 98% cutoff continue to use the original graphs.

To regenerate the alternate graphs from this directory's original metadata, install the Python package `onnx` and run `python tools/build_position_stages.py stages` from the repository root. The generator reads no essays and requires no weight download.

## MIT License

Copyright holders: Desklib and the respective contributors to the model and its ONNX conversion.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
