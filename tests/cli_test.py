from __future__ import annotations

from typer.testing import CliRunner

from observer.cli import app

runner = CliRunner()


# def test_help():
#     result = runner.invoke(app)
#     assert result.exit_code == 0
# assert "Hello Camila" in result.stdout
# assert "Let's have a coffee in Berlin" in result.stdout
