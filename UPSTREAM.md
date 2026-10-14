# nanoVLM v0.1 provenance

- Repository: https://github.com/huggingface/nanoVLM
- Tag: `v0.1`
- Commit: `6ba9082e16f1fc8c21a1f8d0c54b26c9233c8771`
- Release: https://github.com/huggingface/nanoVLM/releases/tag/v0.1
- Original reference files: `vendor/nanovlm_v01/`
- Retrieved tree and file hashes: `vendor/nanovlm_v01/tree.json` / `manifest.json`

Eight original files are retained byte-for-byte, including the three backbone
components, configuration, original VLM, training loop and README. The stored
Git blob SHA1 matches the upstream pinned tree; SHA256 is included for local
verification. Run `python -m scripts.verify_upstream` to check them offline.
The vendor directory is reference material and is never imported at runtime.

Runtime `models/vision_transformer.py`, `models/language_model.py`, and
`models/config.py` are derived from that pinned release. Changes:

- Add immutable HF revisions, spatial vocabulary and laptop configuration.
- Make missing/mismatched backbone weights fatal instead of silently retaining random weights.
- Combine causality and padding in an explicit SDPA mask for backend portability.
- Compute RMSNorm statistics in FP32.
- Add thin native backbone wrappers for non-reentrant activation checkpointing.
- Keep pixel shuffle + one linear projector, causal image prefix and tied LM embeddings.
- Move token shifting/answer supervision into one shared model contract.
- Replace the original monolithic trainer with one stage-driven Trainer.

HF backbone snapshots used for integration verification:

| Component | Repository | Revision |
|---|---|---|
| Vision | `google/siglip-base-patch16-224` | `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` |
| Decoder | `HuggingFaceTB/SmolLM2-135M` | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` |
| Tokenizer | `HuggingFaceTB/cosmo2-tokenizer` | `4ce2318a3628e77279c939ed6a9f3f03034402de` |

The v0.1 tree contains no standalone LICENSE file. Its original VLM model-card
template states `license: mit`; this is preserved in the vendor copy. The local
LICENSE explicitly covers original additions; upstream-derived files retain
their original provenance and terms. Backbone weights and third-party packages
retain the licenses provided by their respective repositories.

PaliGemma-style coordinate tokens follow the documented 1024-bin y0/x0/y1/x1
representation: https://huggingface.co/blog/paligemma. This project does not copy
PaliGemma weights, implement its prefix-LM attention, or include its segmentation decoder.
