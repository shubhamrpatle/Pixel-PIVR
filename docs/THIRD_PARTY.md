# Third-Party Boundary

Pixel-PIVR is an integration around NVIDIA LocateAnything. It does not include model
weights, NVIDIA remote-code files, or the Eagle training framework.

Users must obtain those components under their respective licenses. The configured
LocateAnything directory must include its `config.json`, tokenizer/processor files,
trusted model Python files, and weight shards. `EAGLE_ROOT` must point to the
`Eagle/Embodied` directory.

The tested 16K data uses materialized lossless crop images and needs no virtual-crop
extension. If a private large-scale curriculum stores crops as dictionaries, its
LocateAnything processor and Eagle media loader must support that private format;
this repository intentionally does not patch third-party source at runtime.

