"""Fast tests of vesselnet.energy_check's bookkeeping (the fits themselves run in the energy check)."""
import json

import numpy as np

from vesselnet import energy_check as EC


def test_summarize_counts_per_variant(tmp_path, capsys):
    rows = [dict(kind="truth"),
            dict(kind="split", dnll=1.0, dE={"vesselmap": 5.0, "strict": 4.0}),
            dict(kind="split", dnll=2.0, dE={"vesselmap": -1.0, "strict": 3.0}),
            dict(kind="delete_hidden", dnll=0.5, dE={"vesselmap": -2.0, "strict": 1.0})]
    p = tmp_path / "e.jsonl"
    p.write_text(json.dumps(dict(seed=1, rows=rows)) + "\n")
    EC.summarize(str(p))
    out = capsys.readouterr().out
    assert "| split | 2 | +1.5 | 1/2 (+2) | 2/2 (+3.5) |" in out
    assert "| delete_hidden (should fall) | 1 | +0.5 | 1/1 (-2) | 0/1 (+1) |" in out


def test_windows_pick_the_most_crossings_without_overlap():
    js = [dict(type_visible="crossing", members=[dict(type="crossing", xy=[x, y])])
          for x, y in [(300, 300)] * 5 + [(900, 300)] * 3 + [(310, 290)]]
    w = EC.windows(js, 1200, 1920, 512, 2)
    y0, x0 = w[0]
    assert y0 + 48 < 290 and 310 < y0 + 512 - 48 and x0 + 48 < 300 and 310 < x0 + 512 - 48
    assert all(abs(a[0] - b[0]) >= 512 or abs(a[1] - b[1]) >= 512 for a, b in [(w[0], w[1])])
