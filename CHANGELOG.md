# Changelog

## 0.1.3

Documentation and packaging metadata only. Statistical calculations, numerical outputs, the result schema (`1.0`) and the public API are unchanged from `0.1.2`; the only output difference is `provenance.package_version`.

- Clarified that persistence is a recurrence measure: the share of exposure-eligible entities with at least two false positives, relative to a fixed-count, context-conditioned reference. It is not a monotonic measure of overall false-positive concentration, intensity or severity. Added a worked counterexample and guidance to report the per-entity false-positive count distribution separately when concentration matters.
- Replaced "concentration" wording in the package description, quick-start text and example docstring.
- Replaced the deprecated `project.license` TOML table with the SPDX expression `Apache-2.0` and `license-files`; the build now requires `setuptools>=77`.
- Updated version references and the citation metadata (concept DOI).

## 0.1.2

Public-package hardening release (see the v0.1.2 GitHub release notes and https://doi.org/10.5281/zenodo.23304468).
