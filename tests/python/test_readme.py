"""The Python examples of the README run as written, top to bottom."""

import pathlib
import re

README = pathlib.Path(__file__).resolve().parents[2] / "README.md"


def test_the_code_blocks_of_the_readme_run(capsys):
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), flags=re.S)
    assert len(blocks) >= 5
    namespace = {}
    exec(compile("\n".join(blocks), str(README), "exec"), namespace)
    printed = capsys.readouterr().out.splitlines()
    # the quick start prints the three products bought with a laptop, closest first
    assert [line.split(":")[0] for line in printed[:3]] == ["mouse", "monitor", "keyboard"]
