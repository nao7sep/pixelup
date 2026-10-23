from pathlib import Path

from PySide6.QtCore import QModelIndex, Qt

from pixelup.i18n import localizer
from pixelup.i18n.message import Message
from pixelup.jobs import Job, JobSettings
from pixelup.queue_table import QUEUE_COLUMNS, QueueTableModel, QueueTableView


def _model() -> QueueTableModel:
    return QueueTableModel(
        status_message=lambda status: Message(f"queue.status{status.title()}"),
        scale_label=lambda scale: f"scale:{scale}",
    )


def _job(job_id: int) -> Job:
    return Job(
        id=job_id, input_path=Path(f"/input/image-{job_id}.png"), model="test model",
        output_path=Path(f"/output/result-{job_id}.png"), settings=JobSettings(scale=4),
    )


def test_cells_read_the_existing_job_and_keep_full_tooltips(qapp) -> None:
    model = _model()
    job = _job(7)
    model.append([job])
    def display(column):
        return model.data(model.index(0, column))

    def tooltip(column):
        return model.data(model.index(0, column), Qt.ItemDataRole.ToolTipRole)

    assert [display(column) for column in range(5)] == [
        job.input_path.name, job.model, "scale:4", job.output_path.name,
        localizer.of(Message("queue.statusPending")),
    ]
    assert tooltip(0) == str(job.input_path)
    assert tooltip(1) == job.model
    assert tooltip(2) is None
    assert tooltip(3) == str(job.output_path)
    job.output_path = Path("/output/renamed.png")
    job.message = Message("progress.loadingModel")
    job.warnings = [Message("queue.statusFailed")]
    assert display(3) == "renamed.png"
    assert display(4) == localizer.of(job.message)
    assert tooltip(4) == "\n".join(localizer.of(value) for value in [job.message, *job.warnings])
    job.status = "cancelling"
    assert display(4) == localizer.of(Message("queue.statusCancelling"))
    assert tooltip(4).startswith(localizer.of(Message("queue.statusCancelling")))


def test_batch_append_uses_one_insert_and_refreshes_the_mapped_row(qapp) -> None:
    model = _model()
    events = []
    model.rowsAboutToBeInserted.connect(
        lambda parent, first, last: events.append(("before", first, last, model.rowCount()))
    )
    model.rowsInserted.connect(
        lambda parent, first, last: events.append(("after", first, last, model.rowCount()))
    )
    model.append([])
    assert events == []
    jobs = [_job(50), _job(3)]
    model.append(jobs)
    model.append([_job(90)])
    assert events == [
        ("before", 0, 1, 0), ("after", 0, 1, 2),
        ("before", 2, 2, 2), ("after", 2, 2, 3),
    ]
    changes = []
    model.dataChanged.connect(lambda first, last, roles: changes.append((first, last, roles)))
    model.refresh(jobs[1])
    first, last, roles = changes[0]
    assert (first.row(), first.column(), last.row(), last.column()) == (1, 3, 1, 4)
    assert set(roles) == {Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole}
    assert model.rowCount() == 3
    assert model.columnCount() == 5
    assert model.rowCount(model.index(0, 0)) == 0
    assert model.columnCount(model.index(0, 0)) == 0
    assert model.data(QModelIndex()) is None
    assert model.data(model.index(0, 0), Qt.ItemDataRole.DecorationRole) is None


def test_language_change_notifies_headers_and_all_existing_rows(qapp) -> None:
    model = _model()
    model.append([_job(1), _job(2)])
    headers = []
    changes = []
    model.headerDataChanged.connect(lambda *args: headers.append(args))
    model.dataChanged.connect(lambda first, last, _roles: changes.append((first, last)))
    localizer.use("ja", ("en",))
    assert headers == [(Qt.Orientation.Horizontal, 0, 4)]
    first, last = changes[0]
    assert (first.row(), first.column(), last.row(), last.column()) == (0, 0, 1, 4)
    assert [model.headerData(column, Qt.Orientation.Horizontal) for column in range(5)] == [
        localizer.t(key) for key in QUEUE_COLUMNS
    ]
    assert model.data(model.index(0, 4)) == localizer.of(Message("queue.statusPending"))
    assert model.headerData(0, Qt.Orientation.Vertical) is None


def test_empty_view_message_follows_rows_and_language(qapp) -> None:
    model = _model()
    view = QueueTableView(model)
    assert view.empty_state_visible
    assert view.accessibleDescription() == localizer.of(Message("queue.empty"))
    localizer.use("ja", ("en",))
    assert view.empty_text == localizer.of(Message("queue.empty"))
    assert view.accessibleDescription() == view.empty_text
    model.append([_job(1)])
    assert not view.empty_state_visible
    assert view.accessibleDescription() == ""
    view.close()
