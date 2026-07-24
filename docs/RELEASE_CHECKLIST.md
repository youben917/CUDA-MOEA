# Pre-release Checklist

This checklist records verifiable release work for CUDA-MOEA. It does not
authorize publishing and does not replace a maintainer-approved release
procedure.

## Current blockers

- [ ] Add a maintainer-approved license file and align package metadata with it.
- [ ] Define and verify the package publishing procedure and target registry.
- [ ] Decide whether public contribution, support, and security-reporting
      channels will be offered; document only the channels that are established.
- [ ] Record a tested platform matrix, or clearly state the narrower environment
      used to validate the release candidate.

## Metadata and source

- [ ] Confirm that `pyproject.toml`, `CMakeLists.txt`, and
      `python/cuda_moea/__init__.py` use the same release version.
- [ ] Confirm the project name, description, Python requirement, and dependency
      declarations in `pyproject.toml`.
- [ ] Review public exports in `python/cuda_moea/__init__.py` against
      `docs/API.md`.
- [ ] Search tracked source, docs, examples, notebooks, fixtures, and generated
      reports for credentials, private endpoints, email addresses, personal
      paths, and internal infrastructure names.
- [ ] Remove temporary investigation scripts and notebooks that are not intended
      for the release artifact.

## Build and test

- [ ] Record Python, PyTorch, CUDA Toolkit, GPU, driver, compiler, CMake, and
      operating-system versions.
- [ ] Run `python -m compileall -q python examples tests`.
- [ ] Build and install the extension for each supported CUDA architecture.
- [ ] Run `python -m unittest discover -s tests/python -v`.
- [ ] Run the examples that are intended as supported user paths.
- [ ] Build the wheel and inspect its file list.
- [ ] Build and inspect the source archive after a source-distribution procedure
      has been established.
- [ ] Install each artifact in a clean environment and repeat the quick start.

## Documentation and benchmarks

- [ ] Confirm that English files are authoritative and every optional Chinese
      translation links back to its English counterpart.
- [ ] Check all local Markdown links and remove unresolved template markers.
- [ ] Recheck every documented version, command, path, default, and tensor shape
      against source and automation.
- [ ] Validate benchmark records before regenerating reports.
- [ ] Ensure benchmark claims remain limited to the recorded environments and
      do not imply general superiority.
- [ ] Confirm that committed figures and derived tables can be regenerated from
      retained raw results or a documented data archive.

## Release candidate sign-off

- [ ] Review known limitations and decide which must be release notes.
- [ ] Confirm that no publishing command will overwrite an existing version.
- [ ] Assign an immutable source revision to the candidate.
- [ ] Obtain maintainer approval before any registry upload or public
      announcement.

The tag-triggered release workflow lives in `.github/workflows/publish.yml`.
It verifies version consistency, runs the test suite, and builds the sdist
and a Linux CUDA wheel. The sdist is published to PyPI through OIDC trusted
publishing (PyPI rejects non-manylinux Linux wheels), and both archives are
attached to the GitHub Release. Before the first upload, a maintainer must
register the pending publisher (or the `pypi` environment publisher) on
pypi.org as described in the workflow header.
