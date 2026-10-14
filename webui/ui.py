"""Gradio components and event wiring."""

from functools import partial
from uuid import uuid4

import gradio as gr
import yaml

from .config import ATTACKS, EVALUATIONS, INTERFERENCES, Settings
from .evaluation import preview_payload, run_evaluation
from .processes import processes
from .services import launch_service, stop_service


def _scheme_fields(settings, task, dataset, scheme, allow=False):
    options = settings.tasks.get(task, {}).get(dataset, [])
    selected = next((s for s in options if s["label"] == scheme), None)
    editable = bool(allow and selected and not selected.get("locked"))
    return tuple(
        gr.update(value=selected[key] if selected else "", interactive=editable)
        for key in ("infer", "eval")
    )


def _selectors(settings):
    tasks = list(settings.tasks)
    initial_task = tasks[0] if tasks else None
    datasets = list(settings.tasks.get(initial_task, {}))
    initial_dataset = datasets[0] if datasets else None
    options = settings.tasks.get(initial_task, {}).get(initial_dataset, [])
    labels = [s["label"] for s in options]
    with gr.Row():
        task = gr.Dropdown(tasks, value=initial_task, label="任务", interactive=True)
        dataset = gr.Dropdown(datasets, value=initial_dataset, label="数据集", interactive=True)
    scheme = gr.Dropdown(
        labels, value=labels[0] if labels else None, label="推荐搭配", interactive=True
    )
    with gr.Row():
        infer = gr.Textbox(
            value=options[0]["infer"] if options else "", label="推理类型", interactive=False
        )
        evaluate = gr.Textbox(
            value=options[0]["eval"] if options else "", label="评测类型", interactive=False
        )
    custom = gr.Checkbox(label="自定义 infer / eval（锁定搭配除外）", value=False)

    def task_changed(value):
        names = list(settings.tasks.get(value, {}))
        return gr.update(choices=names, value=names[0] if names else None), gr.update(value=False)

    def dataset_changed(task_value, dataset_value):
        schemes = settings.tasks.get(task_value, {}).get(dataset_value, [])
        names = [s["label"] for s in schemes]
        return gr.update(choices=names, value=names[0] if names else None)

    def edit_changed(allow, task_value, dataset_value, label):
        schemes = settings.tasks.get(task_value, {}).get(dataset_value, [])
        selected = next((s for s in schemes if s["label"] == label), {})
        editable = bool(allow and selected and not selected.get("locked"))
        return gr.update(interactive=editable), gr.update(interactive=editable)

    task.change(task_changed, task, [dataset, custom], queue=False)
    dataset.change(dataset_changed, [task, dataset], scheme, queue=False)
    scheme.change(
        partial(_scheme_fields, settings),
        [task, dataset, scheme, custom],
        [infer, evaluate],
        queue=False,
    )
    custom.change(edit_changed, [custom, task, dataset, scheme], [infer, evaluate], queue=False)
    return {
        "task": task,
        "dataset": dataset,
        "scheme": scheme,
        "infer_type": infer,
        "eval_type": evaluate,
        "allow_custom": custom,
    }


def _free_dataset():
    with gr.Row():
        dataset = gr.Textbox(label="数据集", placeholder="Dataset registered in your backend")
        dataset_type = gr.Textbox(label="数据集类型（可选）", placeholder="null")
    with gr.Row():
        infer = gr.Textbox(label="推理类型")
        evaluate = gr.Textbox(label="评测类型")
    return {
        "dataset": dataset,
        "dataset_type": dataset_type,
        "infer_type": infer,
        "eval_type": evaluate,
    }


def _extra_fields(kind, settings, fields):
    if kind == "interference":
        fields["interference"] = gr.Dropdown(
            INTERFERENCES, value=INTERFERENCES[0], label="干扰方式"
        )
    if kind == "attack":
        choices = list(dict.fromkeys([*settings.attacks, *settings.attack_presets, *ATTACKS]))
        fields["attack"] = gr.Dropdown(
            choices, value=choices[0], label="攻击方式", allow_custom_value=True
        )
        gr.Markdown("RSICDObject 使用 `acc` 指标和下方代理 / key；其他数据集保留后端默认指标。")
    if kind == "generalization":
        gr.Markdown("#### 第二数据集")
        with gr.Row():
            fields["dataset2"] = gr.Textbox(label="第二数据集")
            fields["dataset2_type"] = gr.Textbox(label="第二数据集类型（可选）", placeholder="null")
        with gr.Row():
            fields["infer_type2"] = gr.Textbox(label="第二数据集推理类型")
            fields["eval_type2"] = gr.Textbox(label="第二数据集评测类型")
    if kind == "defense":
        with gr.Row():
            fields["attack"] = gr.Textbox(label="攻击方式", value="APGD")
            fields["attack_eval_type"] = gr.Textbox(label="攻击评测指标", value="acc")
        fields["attack_kwargs"] = gr.Textbox(label="攻击参数（YAML / JSON）", value="null", lines=3)
        with gr.Row():
            fields["defense"] = gr.Textbox(label="防御策略", value="keg")
            fields["knowledge_base"] = gr.Textbox(label="知识注入模型", value="llava")
        with gr.Row():
            fields["dino_port"] = gr.Textbox(label="DINO 端口", value="3000")
            fields["llava_port"] = gr.Textbox(label="LLaVA 端口", value="4000")
        fields["use_label"] = gr.Checkbox(label="使用标签", value=False)


def _evaluation_panel(kind, settings):
    with gr.Row():
        with gr.Column(scale=5):
            fields = (
                _selectors(settings)
                if kind in {"accuracy", "interference", "attack"}
                else _free_dataset()
            )
            _extra_fields(kind, settings, fields)
            with gr.Row():
                fields["model_name"] = gr.Textbox(
                    label="模型名称", placeholder="Name registered in your backend"
                )
                fields["port"] = gr.Textbox(label="模型端口", value="3000")
            with gr.Row():
                fields["output_dir"] = gr.Textbox(
                    label="输出目录", value=f"outputs/{kind}", scale=3
                )
                fields["gpu"] = gr.Textbox(label="GPU 编号", value="0", scale=1)
            with gr.Accordion("API 与代理（可选）", open=False):
                fields["key"] = gr.Textbox(label="API key", type="password")
                if kind in {"attack", "generalization", "defense"}:
                    fields["http_proxy"] = gr.Textbox(label="HTTP 代理", type="password")
                    fields["https_proxy"] = gr.Textbox(label="HTTPS 代理", type="password")
        with gr.Column(scale=5):
            configured = bool(settings.evaluations.get(kind, {}).get("command"))
            gr.Markdown("#### 运行控制")
            status = gr.Textbox(
                label="状态",
                value="就绪" if configured else "后端未配置 · 可先预览 YAML",
                interactive=False,
            )
            with gr.Row():
                preview_button = gr.Button("预览配置", min_width=80)
                run_button = gr.Button(
                    "开始评测", variant="primary", interactive=configured, min_width=80
                )
                stop_button = gr.Button("停止", min_width=80)
            preview = gr.Code(
                label="配置预览（凭据已隐藏）", language="yaml", lines=12, interactive=False
            )
            log = gr.Textbox(
                label="运行日志（最近输出）",
                lines=8,
                max_lines=8,
                interactive=False,
                show_copy_button=True,
            )
    with gr.Row():
        result_json = gr.JSON(label="评测结果", visible=False)
        result_text = gr.Code(
            label="文件预览（大文件或非标准 JSON）",
            language="json",
            visible=False,
            interactive=False,
        )
    result_file = gr.File(label="下载完整结果", visible=False, interactive=False)
    job_id = gr.State(value=lambda: str(uuid4()))
    names, inputs = list(fields), list(fields.values())

    def preview_config(*args):
        try:
            return preview_payload(
                kind, dict(zip(names, args)), settings
            ), "配置有效 · 可查看右侧 YAML"
        except (ValueError, yaml.YAMLError) as exc:
            return "", f"配置无效：{exc}"

    def execute(identifier, *args):
        values = dict(zip(names, args))

        def hidden():
            return gr.update(value=None, visible=False)

        yield "准备运行", "", hidden(), hidden(), hidden()
        for update in run_evaluation(kind, values, settings, identifier):
            result = update.result
            download = None
            if result:
                try:
                    download = result_file.move_resource_to_block_cache(result.path)
                except OSError:
                    update.status += " · 下载文件准备失败，请检查文件读取权限"
            yield (
                update.status,
                update.log,
                gr.update(value=result.data, visible=True)
                if result and result.preview is None
                else hidden(),
                gr.update(value=result.preview, visible=True)
                if result and result.preview is not None
                else hidden(),
                gr.update(value=download, visible=True) if download else hidden(),
            )

    def stop(identifier):
        gr.Info("已发送停止请求。" if processes.stop(identifier) else "当前没有正在执行的任务。")

    preview_button.click(preview_config, inputs, [preview, status], queue=False)
    run_button.click(
        execute,
        [job_id, *inputs],
        [status, log, result_json, result_text, result_file],
        concurrency_id="evaluations",
        concurrency_limit=1,
        api_name=False,
    )
    stop_button.click(stop, job_id, queue=False, api_name=False)


def _service_panel(settings, group, index=0):
    model = group == "models"
    mapping = settings.models if model else settings.attacks
    name = gr.Dropdown(list(mapping), label="模型服务" if model else "攻击服务", interactive=True)
    with gr.Row():
        gpu = gr.Textbox(label="GPU 编号", value="0")
        port = gr.Textbox(label="端口 / 管理标识", value=str(3000 + index) if model else "1337")
    with gr.Row():
        start = gr.Button("启动", interactive=bool(mapping))
        stop = gr.Button("停止", min_width=80)
    output = gr.Textbox(label="服务日志", lines=5, max_lines=5, interactive=False)
    start.click(
        partial(launch_service, settings, group),
        [name, gpu, port],
        output,
        concurrency_limit=None,
        api_name=False,
    )
    stop.click(partial(stop_service, group), port, output, queue=False, api_name=False)


def build_ui(settings: Settings | None = None) -> gr.Blocks:
    settings = settings or Settings()
    with gr.Blocks(
        title="Multimodal Evaluation WebUI",
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
        delete_cache=(3600, 86400),
    ) as demo:
        gr.Markdown(
            "# Multimodal Evaluation WebUI\n模型服务管理 · 六维评测 · 配置预览 · 实时日志与结果"
        )
        if not settings.evaluations:
            gr.Markdown(
                "**开始使用：** 界面已就绪。复制 `configs/example.yaml` 为本地配置，填写后端信息后使用 `python app.py --config configs/local.yaml` 启动。"
            )
        with gr.Accordion("模型与攻击服务", open=False):
            with gr.Row():
                for index in range(3):
                    with gr.Column():
                        gr.Markdown(f"#### 模型 {index + 1}")
                        _service_panel(settings, "models", index)
            with gr.Accordion("攻击服务", open=False):
                _service_panel(settings, "attacks")
                gr.Markdown(
                    "攻击脚本接收 GPU 参数；端口字段仅用于标识本应用管理的进程。脚本需以前台方式运行。"
                )
        with gr.Tabs():
            for kind, label in EVALUATIONS.items():
                with gr.Tab(label):
                    _evaluation_panel(kind, settings)
        with gr.Accordion("已接入的后端与任务", open=False):
            rows = ["| 评测 | 后端命令 | 结果路径规则 |", "| --- | --- | --- |"]
            for kind, label in EVALUATIONS.items():
                backend = settings.evaluations.get(kind, {})
                rows.append(
                    f"| {label} | {'已配置' if backend.get('command') else '待配置'} | {len(backend.get('results', []))} |"
                )
            gr.Markdown("\n".join(rows))
            rows = ["| 任务 | 数据集 | 推理 → 评测 |", "| --- | --- | --- |"]
            for task, datasets in settings.tasks.items():
                for dataset, schemes in datasets.items():
                    for scheme in schemes:
                        rows.append(
                            f"| {task} | {dataset} | {scheme['infer']} → {scheme['eval']} |"
                        )
            gr.Markdown("\n".join(rows))
        gr.Markdown("后端负责模型推理与指标计算。本界面负责配置、进程管理与结果展示。")
    return demo.queue(max_size=16)
