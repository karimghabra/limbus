"""Fast tests of vesselnet.data: splits, presets, folders, the compact round trip."""
import json
import os

import numpy as np
import pytest

from vesselnet import data as D


def test_splits_do_not_overlap_and_cover_the_plan():
    spans = sorted(D.SPLITS.values())
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    assert D.SPLITS["val"][1] - D.SPLITS["val"][0] == 50
    assert D.SPLITS["test"][1] - D.SPLITS["test"][0] == 100
    assert D.split_of(0) == "train" and D.split_of(1_000_049) == "val" and D.split_of(2_000_000) == "test"
    with pytest.raises(ValueError):
        D.split_of(1_000_050)


def test_preset_depends_on_the_seed_alone_and_follows_the_shares():
    assert [D.preset_of(s) for s in range(20)] == [D.preset_of(s) for s in range(20)]
    n = 4000
    got = [D.preset_of(s) for s in range(n)]
    for name, share in D.PRESET_SHARES:
        assert abs(got.count(name) / n - share) < 0.03, name


def test_scene_dir_shards_by_hundred(tmp_path):
    p = D.scene_dir(str(tmp_path), 1_000_123 - 100)
    assert p.endswith(os.path.join("val", "1000000", "s1000023"))
    assert os.path.join("train", "0000400", "s0000417") in D.scene_dir("x", 417)


def test_load_and_list_compact(tmp_path):
    f = D.scene_dir(str(tmp_path), 7)
    os.makedirs(f)
    np.savez_compressed(os.path.join(f, "scene.npz"), img=np.ones((4, 5), np.float32), valid=np.ones((4, 5), bool))
    assert D.list_scenes(str(tmp_path), "train") == []          # no scene.json yet: unfinished
    with open(os.path.join(f, "scene.json"), "w") as fh:
        json.dump(dict(seed=7), fh)
    assert D.list_scenes(str(tmp_path), "train") == [f]
    S = D.load_compact(f)
    assert S["img"].shape == (4, 5) and S["scene"]["seed"] == 7


def test_jsonable_handles_numpy():
    o = D._jsonable({1: np.float32(2.5), "a": (np.int64(3), np.arange(3)), "b": np.zeros(100)})
    assert json.loads(json.dumps(o)) == {"1": 2.5, "a": [3, [0, 1, 2]], "b": "<array float64 (100,)>"}
