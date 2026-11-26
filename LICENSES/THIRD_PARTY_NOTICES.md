# Third-party notices and provenance

## ComfyUI-Sharp integration

Selected loader and inference logic is adapted from `PozzettiAndrea/ComfyUI-Sharp`, commit `92c0015af61b71e6fd88301cfb336e7034f19796`, under GPL-3.0. The complete upstream license text is retained in `COMFYUI-SHARP-GPL-3.0.txt`. Modified integration files are identified in their module docstrings.

## Apple SHARP architecture

The files under `gaussian_stereo/sharp/` originate from the vendored SHARP implementation in ComfyUI-Sharp and retain Apple copyright headers. Apple Software License: `APPLE-SOFTWARE-LICENSE.txt`. Original Apple repository: https://github.com/apple-aiml-research/ml-sharp

The model checkpoint is not included. If downloaded, it is subject to `APPLE-MODEL-LICENSE.txt` and the attribution statement in that agreement.

## Gaussian viewer

`web/gaussian_preview.js` and `web/viewer_gaussian.html` are based on the Gaussian preview implementation from `PozzettiAndrea/comfy-3d-viewers` (GPL-3.0-or-later), with substantial modifications for stereo rendering, camera controls, and the current UI. Upstream: https://github.com/PozzettiAndrea/comfy-3d-viewers

## gsplat.js

`web/js/gsplat-bundle.js` bundles `gsplat.js` 1.2.9, originally published by Dylan Ebert and distributed under the MIT License. Upstream source: https://github.com/dylanebert/gsplat.js. The bundled file is retained from the project's previously approved viewer build. Its bundle does not contain a complete standalone copy of the package's license text; the upstream MIT license should be consulted at the source repository.

## Other dependencies

The runtime model code uses ComfyUI-provided PyTorch operations, attention and model-management facilities. It does not vendor ComfyUI or PyTorch. The browser bundle and upstream projects may include additional third-party components; consult their upstream package manifests and license notices for the complete transitive dependency inventory.
