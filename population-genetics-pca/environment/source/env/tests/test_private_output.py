import json
import stat

import pytest

from grader.grade import _write_private_json


def test_reward_output_is_owner_private_and_exclusive(tmp_path):
    reward = tmp_path / "reward.json"

    _write_private_json(reward, {"reward": 0.75})

    assert stat.S_IMODE(reward.stat().st_mode) == 0o600
    assert json.loads(reward.read_text()) == {"reward": 0.75}
    with pytest.raises(FileExistsError):
        _write_private_json(reward, {"reward": 1.0})


def test_reward_output_refuses_symlink(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("unchanged")
    reward = tmp_path / "reward.json"
    reward.symlink_to(target)

    with pytest.raises(OSError):
        _write_private_json(reward, {"reward": 1.0})

    assert target.read_text() == "unchanged"
