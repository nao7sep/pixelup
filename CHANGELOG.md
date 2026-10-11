# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Choose a models folder before downloading and change it later in Managed models, preserving existing files and already-queued jobs.
- Check model readiness in the background, including after folder changes and at queue submission.

- Keep rejected saved settings and unknown settings entries across unrelated saves, with a startup notice naming settings running on defaults.
- Keep settings backup history per session without delaying saves; write diagnostics off the window thread.
- Check offered images and load previews in the background.
- Label the combined Records filter “Warnings and errors” and allow keyboard resizing of its list pane.
- Simplify file coordination for the supported single-instance workflow.
- Clarify the Settings discard confirmation in every language.

### Fixed

- Apply saved settings even when their dialog closes before the save finishes.
- Preserve multiple queue requests while their models download, including each request's parameters; removing an image also removes its waiting requests.
- Attempt cleanup of partial sidecars during forced quit and use exclusive output creation when hard links are unsupported, including Windows exFAT.

## [0.1.0] - 2026-07-08

### Added

- First public release.
