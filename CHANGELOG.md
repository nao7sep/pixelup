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

- Keep combo-box hover separate from keyboard navigation and the committed choice, preserving selection and focus marks.
- Show hover on selected collection rows and checked options without hiding their keyboard focus marks.
- Prevent another PixelUp copy from writing to the same data folder while the first is running or finishing shutdown.
- Keep source pixels, dimensions, metadata and sidecar provenance consistent if the source changes during processing.
- Preserve photo orientation when stripping metadata, grayscale tonal range and indexed-image transparency; show oriented previews and dimensions.
- Honour cancellation during image and sidecar writing, remove only the job's owned files and preserve committed results.
- Accept literal braces in image paths and keep large-batch output planning proportional to the number of jobs.
- Prepare queues in the background, preserve unrelated requests after a planning failure, and discard removed or cancelled requests before adoption.
- Keep retry cancellation and quit ownership until each worker has stopped.
- Keep landed settings saves successful when optional backups cannot start, and retain full save and backup diagnostics.
- Complete save outcomes and failed-save choices when settings storage becomes unavailable, even if optional backup diagnostics fail.
- Retain failed submitted settings for retry after their dialog closes, and apply landed settings when quitting is cancelled.
- Preserve multiple queue requests while their models download, including each request's parameters; removing an image also removes its waiting requests.
- Choose unused output names across image formats, including when an older image has no sidecar.
- Keep background reads owned through bounded quit and save newer parameter edits during OS shutdown when time permits.
- Attempt cleanup of partial sidecars during forced quit and use exclusive output creation when hard links are unsupported, including Windows exFAT.

## [0.1.0] - 2026-07-08

### Added

- First public release.
