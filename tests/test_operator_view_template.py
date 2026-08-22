import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "teleoperation-episode-recording.json"


def test_episode_recording_operator_view_references_live_graph_contracts():
    graph = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    nodes = graph["node_meta"]
    view = graph["metadata"]["operator_view"]

    assert view["schema_version"] == 1
    assert view["title"] == "Collect episodes"
    assert view["run_target"]["mode"] == "live"
    assert "follower motion remains disarmed" in view["run_target"]["confirm"].lower()

    targets = [view["run_target"]]
    for section in view["sections"]:
        for widget in section["widgets"]:
            if "source" in widget:
                source = widget["source"]
                assert source["node_id"] in nodes
                assert source["port"] in nodes[source["node_id"]]["outputs"]
            for item in widget.get("items", []):
                if "port" in item:
                    assert item["node_id"] in nodes
                    assert item["port"] in nodes[item["node_id"]]["outputs"]
                if "param" in item:
                    assert item["node_id"] in nodes
                    assert item["param"] in nodes[item["node_id"]]["params"]
                for update in item.get("updates", []):
                    assert update["node_id"] in nodes
                    assert update["param"] in nodes[update["node_id"]]["params"]
                if "control" in item:
                    assert nodes[item["control"]["node_id"]]["type"] == "EpisodeRecorder"
                if "cook_target" in item:
                    targets.append(item["cook_target"])

    for target in targets:
        assert target["node_id"] in nodes
        assert target["port"] in nodes[target["node_id"]]["outputs"]


def test_episode_recording_operator_view_preserves_motion_and_data_safety():
    graph = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    view = graph["metadata"]["operator_view"]
    actions = {
        action["id"]: action
        for section in view["sections"]
        for widget in section["widgets"]
        if widget["type"] == "actions"
        for action in widget["items"]
    }

    assert graph["node_meta"]["armed"]["params"]["value"] is False
    assert graph["node_meta"]["follow"]["params"]["armed"] is False
    assert actions["arm-follower"]["confirm"]
    assert actions["arm-follower"]["updates"][0]["value"] is True
    assert actions["disarm-follower"]["updates"][0]["value"] is False
    assert actions["discard-recording"]["confirm"]
    assert actions["discard-recording"]["control"]["action"] == "discard"

