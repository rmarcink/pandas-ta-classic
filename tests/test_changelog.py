"""Each CHANGELOG.md entry appears once in its version section.

.gitattributes merges CHANGELOG.md with the ``union`` driver, which keeps both
sides instead of stopping on a conflict. A branch that edits an existing entry
therefore lands as the old and the new version side by side, and an entry that
reaches a merge from two positions lands twice. Both copies keep the entry's
bold title, so a title repeated within one version is the trace of such a merge.
"""

import re
from collections import Counter
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"


def test_no_entry_title_repeats_within_a_version():
    for section in re.split(r"^(?=## \[)", CHANGELOG.read_text(encoding="utf-8"), flags=re.MULTILINE):
        titles = Counter(re.findall(r"^\* (\*\*.+?\*\*)", section, flags=re.MULTILINE))
        repeated = sorted(title for title, count in titles.items() if count > 1)
        assert not repeated, f"{section.splitlines()[0]} repeats {len(repeated)} entry title(s): {repeated}"
