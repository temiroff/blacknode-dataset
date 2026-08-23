import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "teleoperation-episode-recording.json"


def test_episode_recording_operator_view_references_live_graph_contracts():
    graph = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    nodes = graph["node_meta"]
    view = graph["metadata"]["operator_view"]

    assert view["schema_version"] == 1
    assert view["id"] == "collect-episodes"
    assert view["title"] == "Collect episodes"
    assert view["icon"] == "record"
    assert view["settings"]["groups"]
    assert all("title" not in section and "description" not in section for section in view["sections"])
    assert next(section for section in view["sections"] if section["id"] == "controls")["region"] == "parameters"
    assert view["run_target"]["mode"] == "live"
    assert view["run_target"]["live_source"] == {"node_id": "follow", "port": "running"}
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
                    assert item["control"]["node_id"] in nodes
                if "deactivate_control" in item:
                    assert item["deactivate_control"]["node_id"] in nodes
                if "cook_target" in item:
                    targets.append(item["cook_target"])

    for target in targets:
        assert target["node_id"] in nodes
        assert target["port"] in nodes[target["node_id"]]["outputs"]

    for group in view["settings"]["groups"]:
        for item in group["items"]:
            setting_targets = [item, *item.get("apply_to", [])]
            for target in setting_targets:
                assert target["node_id"] in nodes
                assert target["param"] in nodes[target["node_id"]]["params"]
            for pair in item.get("swap_pairs", []):
                for target in (pair["left"], pair["right"]):
                    assert target["node_id"] in nodes
                    assert target["param"] in nodes[target["node_id"]]["params"]

    connection = next(group for group in view["settings"]["groups"] if group["id"] == "connection")
    ros_host = next(item for item in connection["items"] if item["param"] == "host")
    assert {target["node_id"] for target in ros_host["apply_to"]} == {
        "leader_robot", "follower_robot", "leader_release", "follow",
    }
    devices = next(group for group in view["settings"]["groups"] if group["id"] == "devices")
    calibration_pickers = [
        item for item in devices["items"] if item.get("input") == "calibration_file"
    ]
    assert {(item["node_id"], item["param"]) for item in calibration_pickers} == {
        ("leader_robot", "calibration"),
        ("follower_robot", "calibration"),
    }
    for node_id in ("leader_robot", "follower_robot"):
        assert "calibration" in nodes[node_id]["inputs"]
        assert nodes[node_id]["input_types"]["calibration"] == "Dict"
    role_swap = next(item for item in devices["items"] if item.get("input") == "swap")
    assert role_swap["disabled_when"] == {"node_id": "follow", "port": "running"}
    assert {
        (pair["left"]["param"], pair["right"]["param"])
        for pair in role_swap["swap_pairs"]
    } == {
        ("selection", "selection"),
        ("serial_port", "serial_port"),
        ("profile_id", "profile_id"),
        ("calibration_hardware_id", "calibration_hardware_id"),
        ("calibration", "calibration"),
    }


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
    toggle = actions["toggle-follower"]
    assert toggle["confirm"]
    assert toggle["state"] == {"node_id": "follow", "port": "armed"}
    assert toggle["control"]["action"] == "arm"
    assert toggle["deactivate_control"]["action"] == "disarm"
    assert "cook_target" not in toggle
    assert actions["discard-recording"]["confirm"]
    assert actions["discard-recording"]["control"]["action"] == "discard"


def test_episode_recording_declares_hardware_and_transport_dependency_closure():
    graph = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    metadata = graph["metadata"]

    assert "blacknode-drivers" in metadata["required_packages"]
    assert {
        "blacknode-drivers/feetech",
        "blacknode-robot/devices",
        "blacknode-ros2/rosbridge",
    }.issubset(metadata["required_components"])
    assert "blacknode-drivers/feetech@ros2" in metadata["required_adapters"]
