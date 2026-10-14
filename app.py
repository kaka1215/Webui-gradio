"""Command-line entry point for the evaluation workspace."""

import argparse
import atexit

import yaml

from webui.config import load_settings
from webui.processes import processes


def main():
    parser = argparse.ArgumentParser(description="EvalDeck — Multimodal Evaluation Workspace")
    parser.add_argument("--config", help="Path to a deployment YAML file")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: local only)")
    parser.add_argument("--port", type=int, default=7980)
    parser.add_argument(
        "--check-config", action="store_true", help="Validate configuration without starting the UI"
    )
    args = parser.parse_args()
    try:
        settings = load_settings(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.check_config:
        print(
            f"Configuration valid: {len(settings.models)} models, {len(settings.tasks)} tasks, "
            f"{sum(bool(v.get('command')) for v in settings.evaluations.values())} configured evaluation backends."
        )
        return
    from webui.ui import build_ui

    atexit.register(processes.close)
    build_ui(settings).launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
