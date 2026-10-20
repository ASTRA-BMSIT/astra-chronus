"""Guard the link between README section 4 and the code it specifies.

README section 4 allows a contract change only by a pull request that updates
the README in the same change. A reviewer can miss the half of that rule where
the file changes but the documented block does not, so this test holds the two
together by comparing field names and annotations as written. Comments are
ignored, because they are prose rather than contract.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_CLASSES = ("Detection", "TrackState", "Prediction", "GimbalCommand")


def readme_section_4_source() -> str:
    """The Python block under the 'Interfaces (week-1 contract)' heading."""
    text = (REPO_ROOT / "README.md").read_text()
    heading = text.index("## 4. Interfaces")
    start = text.index("```python", heading) + len("```python")
    end = text.index("```", start)
    return text[start:end]


def declared_fields(source: str) -> dict[str, list[str]]:
    """Field name and annotation of each contract class, as written in `source`."""
    found: dict[str, list[str]] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name in CONTRACT_CLASSES:
            found[node.name] = [
                f"{item.target.id}: {ast.unparse(item.annotation)}"
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            ]
    return found


def test_readme_contract_block_is_complete() -> None:
    """Success path: the documented block declares all four classes."""
    assert set(declared_fields(readme_section_4_source())) == set(CONTRACT_CLASSES)


def test_code_matches_the_documented_contract() -> None:
    """Boundary: the file implements the README, it does not amend it."""
    documented = declared_fields(readme_section_4_source())
    implemented = declared_fields(
        (REPO_ROOT / "src" / "astra_chronus" / "interfaces.py").read_text()
    )
    assert implemented == documented, (
        "src/astra_chronus/interfaces.py no longer matches README section 4; "
        "update both in the same pull request"
    )
