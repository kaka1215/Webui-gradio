import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import yaml

from webui.config import EVALUATIONS, Settings, load_settings
from webui.evaluation import run_evaluation
from webui.payloads import build_payload, port_number, preview_payload
from webui.processes import ProcessManager, processes
from webui.results import MAX_JSON_BYTES, find_result, read_result
from webui.services import launch_service, stop_service


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name).resolve()
        self.settings = Settings(
            base_dir=self.root,
            tasks={
                "objects": {
                    "RSICDObject": [
                        {"label": "default", "infer": "object_eval", "eval": "object_eval"}
                    ]
                }
            },
        )
        self.values = {
            "task": "objects",
            "dataset": "RSICDObject",
            "scheme": "default",
            "infer_type": "object_eval",
            "eval_type": "object_eval",
            "allow_custom": False,
            "model_name": "fixture-model",
            "port": "3000",
            "output_dir": "results",
            "key": "private-token",
            "gpu": "2",
            "interference": "jpeg",
            "attack": "APGD",
            "dataset2": "FactFailure",
            "dataset2_type": "null",
            "infer_type2": "fact",
            "eval_type2": "fact",
            "http_proxy": "",
            "https_proxy": "",
            "attack_kwargs": "steps: 4",
            "defense": "keg",
            "attack_eval_type": "acc",
            "knowledge_base": "llava",
            "dino_port": "3001",
            "llava_port": "3002",
            "use_label": True,
        }

    def tearDown(self):
        processes.close()
        self.directory.cleanup()

    def backend(self, code):
        script = self.root / "fixture backend.py"
        script.write_text(code, encoding="utf-8")
        command = ["{python}", str(script), "{config}"]
        for kind in EVALUATIONS:
            self.settings.evaluations[kind] = {
                "command": command,
                "results": ["{output_dir}/metrics.json"],
            }

    def test_six_payload_contracts(self):
        for kind in EVALUATIONS:
            with self.subTest(kind=kind):
                payload = build_payload(kind, self.values, self.settings)
                self.assertEqual(payload["model"]["port"], 3000)
                self.assertIsNone(payload["data"][0]["type"])
                self.assertEqual(payload["data"][0]["dataset_name"], "RSICDObject")
                self.assertEqual(yaml.safe_load(yaml.safe_dump(payload)), payload)
        accuracy = build_payload("accuracy", self.values, self.settings)
        self.assertEqual(
            accuracy["evaluate"], [{"infer_type": "object_eval", "eval_type": "object_eval"}]
        )
        interference = build_payload("interference", self.values, self.settings)
        self.assertEqual(interference["evaluate"][0]["interference"], "jpeg")
        self.assertIsNone(interference["evaluate"][0]["interference_kwargs"])
        generalization = build_payload("generalization", self.values, self.settings)
        self.assertEqual(len(generalization["data"]), 2)
        self.assertEqual(generalization["evaluate"][1], {"infer_type": "fact", "eval_type": "fact"})
        defense = build_payload("defense", self.values, self.settings)
        self.assertEqual(defense["evaluate"][0]["attack_kwargs"], {"steps": 4})
        self.assertEqual(defense["evaluate"][0]["defense_kwargs"]["dino_port"], 3001)

    def test_attack_specific_key_and_metric_contract(self):
        self.settings.attack_presets = {"APGD": {"RSICDObject": {"steps": 8}}}
        payload = build_payload("attack", self.values, self.settings)
        self.assertEqual(payload["model"]["keys"], [])
        self.assertEqual(payload["evaluate"][0]["keys"], ["private-token"])
        self.assertEqual(payload["evaluate"][0]["attack_eval_type"], "acc")
        self.assertEqual(payload["evaluate"][0]["attack_kwargs"], {"steps": 8})
        self.settings.tasks["objects"]["Other"] = self.settings.tasks["objects"]["RSICDObject"]
        payload = build_payload("attack", self.values | {"dataset": "Other"}, self.settings)
        self.assertIsNone(payload["evaluate"][0]["attack_eval_type"])
        self.assertEqual(payload["evaluate"][0]["keys"], [])

    def test_locked_scheme_enforced_server_side(self):
        self.settings.tasks["objects"]["RSICDObject"][0]["locked"] = True
        with self.assertRaises(ValueError):
            build_payload(
                "accuracy",
                self.values | {"allow_custom": True, "infer_type": "changed"},
                self.settings,
            )
        self.settings.tasks["objects"]["RSICDObject"][0]["locked"] = False
        build_payload(
            "accuracy", self.values | {"allow_custom": True, "infer_type": "changed"}, self.settings
        )

    def test_invalid_ports_and_missing_values(self):
        for value in ("", "abc", 0, 65536, "3.5", "3000;echo bad"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                port_number(value)
        with self.assertRaises(ValueError):
            build_payload("accuracy", self.values | {"output_dir": ""}, self.settings)

    def test_invalid_attack_parameters_are_not_silently_discarded(self):
        for value in ("[1, 2]", "steps: [", "text"):
            with self.subTest(value=value), self.assertRaises((ValueError, yaml.YAMLError)):
                build_payload("defense", self.values | {"attack_kwargs": value}, self.settings)

    def test_preview_redacts_credentials_without_mutating_payload(self):
        values = self.values | {"http_proxy": "http://name:password@proxy"}
        preview = preview_payload("defense", values, self.settings)
        self.assertNotIn("private-token", preview)
        self.assertNotIn("password", preview)
        self.assertEqual(
            build_payload("defense", values, self.settings)["model"]["keys"], ["private-token"]
        )

    def test_local_config_validation(self):
        path = self.root / "config.yaml"
        for invalid in (
            {"typo": {}},
            {"models": []},
            {"evaluations": {"accuracy": {"command": "python run.py"}}},
            {"evaluations": {"accuracy": {"command": ["python", "run.py"]}}},
            {"tasks": {"task": {"dataset": []}}},
        ):
            path.write_text(yaml.safe_dump(invalid))
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                load_settings(path)
        path.write_text("models: {}\n")
        self.assertEqual(load_settings(path).base_dir, self.root)

    def test_all_six_execute_and_remove_temporary_configs(self):
        self.backend("""import json, os, pathlib, sys, yaml
p = pathlib.Path(sys.argv[1])
config = yaml.safe_load(p.read_text())
pathlib.Path('received.yaml').write_text(yaml.safe_dump(config))
pathlib.Path('temporary-path.txt').write_text(str(p))
out = pathlib.Path(config['output']['output_dir'])
out.mkdir(exist_ok=True)
(out / 'metrics.json').write_text(json.dumps({'fixture': True, 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}))
print('fixture output private-token', flush=True)
""")
        for kind in EVALUATIONS:
            with self.subTest(kind=kind):
                updates = list(run_evaluation(kind, self.values, self.settings, kind))
                self.assertEqual(updates[-1].status, "已完成")
                self.assertEqual(updates[-1].result.data, {"fixture": True, "gpu": "2"})
                self.assertNotIn("private-token", updates[-1].log)
                self.assertFalse(Path((self.root / "temporary-path.txt").read_text()).exists())
                self.assertEqual(
                    yaml.safe_load((self.root / "received.yaml").read_text()),
                    build_payload(kind, self.values, self.settings),
                )

    def test_failed_backend_never_displays_a_stale_result(self):
        (self.root / "results").mkdir()
        stale = self.root / "results/metrics.json"
        stale.write_text('{"old":true}')
        os.utime(stale, (1, 1))
        self.backend("import sys; print('backend failed'); sys.exit(3)")
        update = list(run_evaluation("accuracy", self.values, self.settings, "failed"))[-1]
        self.assertIn("退出码 3", update.status)
        self.assertIsNone(update.result)
        self.backend("print('no new result')")
        update = list(run_evaluation("accuracy", self.values, self.settings, "stale"))[-1]
        self.assertIsNone(update.result)
        self.assertIn("未找到本次运行", update.status)

    def test_validation_and_unconfigured_backend_errors_are_yielded(self):
        update = list(run_evaluation("accuracy", self.values, self.settings, "missing"))[-1]
        self.assertIn("尚未配置", update.status)
        update = list(
            run_evaluation("accuracy", self.values | {"port": "bad"}, self.settings, "bad")
        )[-1]
        self.assertIn("端口", update.status)

    def test_process_stop_and_duplicate_protection(self):
        manager = ProcessManager()
        stream = manager.stream(
            "job", [sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(30)"]
        )
        next(stream)
        self.assertIn("ready", next(stream).log)
        with self.assertRaises(ValueError):
            next(manager.stream("job", [sys.executable, "-c", "pass"]))
        self.assertTrue(manager.stop("job"))
        events = list(stream)
        self.assertTrue(events[-1].cancelled)
        self.assertIsNotNone(events[-1].returncode)
        self.assertFalse(manager.stop("unmanaged"))

    def test_closing_stream_stops_process(self):
        manager = ProcessManager()
        stream = manager.stream("job", [sys.executable, "-c", "import time; time.sleep(30)"])
        next(stream)
        stream.close()
        self.assertFalse(manager.stop("job"))

    def test_result_preview_and_path_template(self):
        path = self.root / "metrics.json"
        path.write_text(json.dumps({"long": "a" * MAX_JSON_BYTES}))
        result = read_result(path)
        self.assertIsNone(result.data)
        self.assertIn("truncated", result.preview)
        self.assertEqual(
            find_result(
                ["{output_dir}/metrics.json"], {"output_dir": "."}, time.time() - 10, self.root
            ),
            path,
        )
        path.write_text("not json")
        self.assertEqual(read_result(path).preview, "not json")

    @unittest.skipUnless(os.name == "posix", "Bash service contract")
    def test_service_arguments_and_managed_stop(self):
        script = self.root / "service with spaces.sh"
        script.write_text('printf "%s|%s|%s\\n" "$1" "$2" "$CUDA_VISIBLE_DEVICES"\n')
        self.settings.models["local"] = {"script": script.name}
        output = list(launch_service(self.settings, "models", "local", "0,1", "3000"))[-1]
        self.assertIn("0,1|3000|0,1", output)
        self.assertIn("退出码 0", output)
        self.assertIn("没有正在管理", stop_service("models", "3000"))

    def test_ui_builds_all_six_modes_without_running_backends(self):
        from webui.ui import build_ui

        ui = build_ui(self.settings)
        labels = {item.label for item in ui.blocks.values() if type(item).__name__ == "Tab"}
        self.assertEqual(labels, set(EVALUATIONS.values()))


if __name__ == "__main__":
    unittest.main()
