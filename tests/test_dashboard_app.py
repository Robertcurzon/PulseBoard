from streamlit.testing.v1 import AppTest


def test_upload_mode_waits_for_csv_before_rendering_dashboard() -> None:
    app = AppTest.from_file("dashboard/app.py", default_timeout=180)
    app.run(timeout=180)
    app.radio[0].set_value("Upload CSV")
    app.run(timeout=180)

    assert len(app.exception) == 0
    assert len(app.metric) == 0
    assert any("Minimum required columns" in markdown.value for markdown in app.markdown)
