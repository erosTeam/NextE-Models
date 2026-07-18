# NextE-Models Agent Rules

This repository builds and publishes optional local image models for NextE.

## Hard stops

- Never commit or upload calibration pages, evaluation pages, reader cache files, cookies, or
  account data. Real pages are local-only inputs.
- Never publish a model before its source hash, artifact hash, runtime contract, license, and
  device-validation record are complete.
- Do not replace an accepted artifact in an existing release. Publish a new version instead.
- Do not treat a successful conversion as runtime validation. A candidate must load and run on the
  intended HarmonyOS device before it becomes `published` in `manifests/models-v1.json`.
- Keep large checkpoints and generated models out of Git. Release assets are the distribution
  boundary.

## Required checks

Run before committing:

```bash
python3 -m compileall -q scripts tests
python3 -m unittest discover -s tests -v
python3 scripts/validate_repository.py
git diff --check
```
