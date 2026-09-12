from html.parser import HTMLParser
from pathlib import Path
import json
import re


ROOT = Path(__file__).resolve().parent


class PageLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if "id" in attributes:
            assert attributes["id"] not in self.ids, attributes["id"]
            self.ids.add(attributes["id"])
        if tag == "a" and "href" in attributes:
            self.links.append(attributes["href"])


def validate_slices(text, expected_size):
    cursor = 0
    count = 0
    for start, end, width in re.findall(
        r"^\| (\d+):(\d+) \| (\d+) \|", text, re.MULTILINE
    ):
        start, end, width = int(start), int(end), int(width)
        assert start == cursor, (start, cursor)
        assert end - start == width, (start, end, width)
        cursor = end
        count += 1
    assert cursor == expected_size, (cursor, expected_size)
    return count


def main():
    experiments = (ROOT / "EXPERIMENTS.md").read_text()
    single_start = experiments.index("单臂 `md-arm-table-v1`")
    dual_start = experiments.index("双臂 `md-dualarm-table-v1`")
    contract_end = experiments.index("## 3.", dual_start)
    single_rows = validate_slices(experiments[single_start:dual_start], 66)
    dual_rows = validate_slices(experiments[dual_start:contract_end], 116)
    page = (ROOT / "index.html").read_text()
    parser = PageLinks()
    parser.feed(page)
    for link in parser.links:
        if link.startswith("#"):
            assert link[1:] in parser.ids, link
        elif not re.match(r"[a-z]+:", link):
            assert (ROOT / link.split("#")[0]).exists(), link
    for path in ROOT.glob("*.md"):
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if not re.match(r"[a-z]+:|#", link):
                assert (path.parent / link.split("#")[0]).exists(), (path, link)
    cases = [
        "arm-reach-v1", "arm-pick-place-v1", "arm-relocate-v1",
        "arm-carry-v1", "arms-handover-v1", "arms-co-carry-v1",
    ]
    for case in cases:
        assert case in experiments and case in page, case
    assert "66 维" in page and "116 维" in page
    assert "56 维" not in page and "109 维" not in page
    assert "未训练" in page and "未连接硬件" in page
    assert not re.search(r"<video\b|<iframe\b|<script[^>]+src=", page)
    static_torque = 9.81 * (
        0.018 * (0.065 + 0.120 + 0.135 + 0.145)
        + 0.020 * 0.0325 + 0.015 * 0.0925
        + 0.015 * 0.135 + 0.020 * 0.150
    )
    assert abs(static_torque - 0.1514) < 0.00001
    assert abs(6 * 1.47 - 8.82) < 1e-12
    print(json.dumps({
        "status": "passed",
        "scope": "design consistency only, not simulation or hardware",
        "single_observation_rows": single_rows,
        "single_observations": 66,
        "dual_observation_rows": dual_rows,
        "dual_observations": 116,
        "case_count": len(cases),
        "shoulder_static_assumption_nm": round(static_torque, 8),
        "local_links": "valid",
        "invented_video_embeds": "none",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
