"""Launch configured foreground services without shell interpolation."""

import shutil
from contextlib import closing

from .config import Settings
from .payloads import port_number
from .processes import processes


def service_id(group, port):
    return f"service:{group}:{port_number(port)}"


def launch_service(settings: Settings, group, name, gpu, port):
    try:
        mapping = settings.models if group == "models" else settings.attacks
        if name not in mapping:
            raise ValueError("请先在配置文件中添加并选择服务。")
        script = settings.resolve(mapping[name]["script"])
        if not script.is_file():
            raise ValueError(f"脚本不存在：{script}")
        bash = shutil.which("bash")
        if not bash:
            raise ValueError("启动服务需要 Bash。")
        command = [bash, str(script), str(gpu)]
        if group == "models":
            command.append(str(port_number(port)))
        with closing(
            processes.stream(service_id(group, port), command, cwd=script.parent, gpu=gpu)
        ) as stream:
            for event in stream:
                if event.returncode is None:
                    yield event.log
                else:
                    status = (
                        "已停止" if event.cancelled else f"脚本结束 · 退出码 {event.returncode}"
                    )
                    yield f"{event.log}\n{status}"
    except (OSError, ValueError) as exc:
        yield f"无法启动：{exc}"


def stop_service(group, port):
    try:
        stopped = processes.stop(service_id(group, port))
        return "已发送停止请求。" if stopped else "本应用没有正在管理的对应服务。"
    except ValueError as exc:
        return str(exc)
