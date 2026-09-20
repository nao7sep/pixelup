# PixelUp's areas, and the tests that stand for them

`uv run pytest` runs this whole suite apart from the `heavy` tests: at a few seconds it is already a
fixed, balanced run, so nothing selects a subset of it. `uv run pytest -m "heavy or not heavy"` is
the full gate and adds the two `heavy` tests, which install every managed Real-ESRGAN artifact
through PixelUp's own model manager and check each file against its pinned SHA-256, then upscale a
photo from the shared test-fixture corpus with every upscale model and measure that each output
still resembles the source it was enlarged from.

This file is the balance judgement the `tests-folder-conventions` require — which areas PixelUp has,
and which tests stand for each — so a reader can tell what a green run covered, and an area with no
test standing for it is visible rather than merely absent. `test_area_map.py` holds every path below
to what is on disk.

The `heavy` tests stand for no area, because the default run never executes them. Two files below
hold one apiece beside their ordinary tests; it is those ordinary tests that stand for the area.
`conftest.py` is shared infrastructure — the offscreen `QApplication`, the singleton teardowns, the
corpus and managed-model fixtures — and not itself a representative of anything.

Paths are relative to this folder.

| Area | What it covers | Tests standing for it |
|---|---|---|
| The main window | The single window the app is: the dropped-image list, the Parameters panel, the queue actions, the selected-image preview, the per-job summaries, and the remembered position and size | `test_gui_main_window.py`, `test_gui_helpers.py`, `test_gui_layout.py` |
| The queue and the job runner | Turning images and models into jobs, scheduling them under the concurrency limit, reporting progress, and cancelling or retrying them | `test_jobs.py`, `test_runner.py` |
| Upscale options and the plan | Validating a job's scale, denoise, alpha mode, format, quality and tile size, and planning the run those options describe | `test_upscale_validate_options.py`, `test_upscale_plan.py` |
| Inference and the device | The Real-ESRGAN network subset PixelUp carries, its tiled upsampler, the refusal to unpickle weights on an unsafe torch, and resolving auto/MPS/CUDA/CPU | `test_inference.py`, `test_realesrgan_runtime.py`, `test_devices.py` |
| Writing the output image | Where an output lands beside its source, the atomic no-clobber write, format and quality handling, alpha flattening, and the colour profile carried or converted | `test_imaging.py`, `test_paths.py`, `test_icc_profiles.py` |
| The output bundle and its sidecar | Reserving the output and its `.json` companion so two runs or two processes cannot claim the same pair, and the sidecar record that makes a result reproducible | `test_output_reservation.py`, `test_sidecar.py` |
| Managed models | The pinned model registry, verified downloads, the install manager behind Managed models, and the dialog that discloses and repairs what is missing | `test_models.py`, `test_model_management.py`, `test_model_manager.py`, `test_managed_models_dialog.py` |
| Storage paths | Resolving the home, models, temp and state directories from overrides and the environment, quarantining a corrupt file, and the nanoid the derived temp and quarantine names are built from | `test_config.py`, `test_nanoid.py` |
| Settings | The `config.json` PixelUp loads, repairs and saves, and the Settings dialog that edits the part the main window does not show | `test_app_config.py`, `test_settings_dialog.py` |
| The write-through backup store | The SQLite store that records every managed write, and that the app's own saves reach it | `test_backup_store.py`, `test_managed_write_records.py` |
| The session log | The JSONL session log, its redaction and excepthook, and the UTC timestamps the log, the sidecar and the derived filenames share | `test_session_log.py`, `test_timestamps.py` |
| Appearance and shared controls | The one owned style sheet and its uniform control sizes, the UI font, the window minimum derived from its panes, and the shared widgets the panels are built from | `test_theme.py`, `test_fonts.py`, `test_gui_chrome.py`, `test_widgets.py` |
| The interface text | What the app says to the user: the About box, the quit confirmation, the shortcuts sheet, the parameters help, and the authored message dialogs | `test_about_dialog.py`, `test_quit_dialog.py`, `test_shortcuts_dialog.py`, `test_parameters_help_dialog.py`, `test_message_dialogs.py` |
| Packaging and the repository's own contracts | The pinned runtime dependency stack, the macOS and Windows packaging scripts and installer, the licence and third-party notices they ship, the release-tag check, and this map | `test_packaging.py`, `test_release_tag.py`, `test_area_map.py` |
