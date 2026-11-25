"""
Check that the notebook dependencies execute cells and export their output.
"""

import sys

import nbformat
import pytest
from jupyter_client import KernelManager
from nbclient import NotebookClient


def test_notebook_execution_and_html_export(tmp_path):
    exporter = pytest.importorskip("nbconvert").HTMLExporter()
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "import sys\nimport phsi\n"
                "model = phsi.model()\n"
                "print(sys.executable)\n"
                "print('PHSI notebook check')"
            )
        ]
    )
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    NotebookClient(
        notebook,
        km=manager,
        timeout=60,
        resources={"metadata": {"path": str(tmp_path)}},
    ).execute(cleanup_kc=True)
    output = "".join(item.get("text", "") for item in notebook.cells[0].outputs)
    assert sys.executable in output
    assert "PHSI notebook check" in output
    html, _ = exporter.from_notebook_node(notebook)
    assert "PHSI notebook check" in html
