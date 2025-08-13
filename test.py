import gradio as gr
import subprocess
import os
import yaml
from tempfile import NamedTemporaryFile

# 模型和攻击算法脚本目录
MODEL_SCRIPT_DIR = ""
ATTACK_SCRIPT_DIR = ""
ATTACK_SCRIPT_PATH = ""

import os, time, json, glob

MAX_JSON_BYTES = 0.5 * 1024 * 1024          # ≤512KB 才在 gr.JSON 中完整展示
PREVIEW_HEAD_BYTES = 12_000              # 大文件只读前 12KB 作为预览
def _fmt_size(n): return f"{n/1024/1024:.2f} MB"

def build_json_preview_from_file(path, head_bytes=PREVIEW_HEAD_BYTES):
    try:
        size = os.path.getsize(path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            head = f.read(head_bytes)
        if size > head_bytes:
            head += "\n...\n/* truncated preview: file is " + _fmt_size(size) + " */"
        return head, size
    except Exception as e:
        return f"无法读取结果文件用于预览：{e}", -1


# 各评测的结果路径模板（可按需继续扩展 attack、robustness 等）
# 支持精确文件 & 通配符。变量：{output_dir},{model_name},{dataset}
RESULT_PATTERNS = {

}

PREFERRED_JSON_BASENAMES = {
    "results.json", "result.json", "metrics.json", "eval.json", "evaluation.json", "summary.json"
}


def _expand_patterns(eval_kind, **vars_dict):
    patterns = RESULT_PATTERNS.get(eval_kind, [])
    expanded = []
    for pat in patterns:
        pat_fmt = pat.format(**{k: str(v).strip().rstrip("/") for k, v in vars_dict.items()})
        if any(ch in pat_fmt for ch in "*?[]"):
            expanded.extend(glob.glob(pat_fmt))
        else:
            expanded.append(pat_fmt)
    # 去重保序
    seen, uniq = set(), []
    for p in expanded:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq

def _pick_newest_json(candidates, start_ts, relax_time=False):
    scored = []
    for path in candidates:
        try:
            st = os.stat(path)
            if relax_time or st.st_mtime >= start_ts - 1:
                name = os.path.basename(path).lower()
                score = 3 if name in PREFERRED_JSON_BASENAMES else 1
                scored.append((score, st.st_mtime, path))
        except FileNotFoundError:
            continue
    if not scored:
        return None
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]

def locate_result_json(eval_kind, output_dir, model_name, dataset, start_ts, **extra):
    """
    仅按 RESULT_PATTERNS 展开的候选进行匹配：
      1) 先使用严格时间过滤（mtime >= start_ts-1）
      2) 若没有命中，再放宽时间过滤
    不做任何目录兜底扫描。
    兼容 {interference} / {interference_type} 两种写法。
    """
    dataset_str = str(dataset).strip()
    vars_dict = dict(
        output_dir=str(output_dir).rstrip("/"),
        model_name=str(model_name).strip(),
        dataset=dataset_str,
        dataset_lower=dataset_str.lower(),
        dataset_slug=dataset_str.replace(" ", "_").lower(),
        **extra
    )

    # 1) 模板展开 → 严格时间筛选
    expanded = _expand_patterns(eval_kind, **vars_dict)  # 只用模板！
    path = _pick_newest_json(expanded, start_ts, relax_time=False)
    if path:
        return path

    # 2) 放宽时间（仍只在模板展开的候选里挑）
    path = _pick_newest_json(expanded, start_ts, relax_time=True)
    return path  # 可能为 None；由上层决定如何提示



# 工具：列出脚本文件
def list_sh_scripts(directory):
    return [f for f in os.listdir(directory) if f.endswith(".sh")]


# 通用脚本加载逻辑（可带 port）
def launch_script(script_dir, script_name, gpu, port=None):
    if not script_name:
        yield "⚠️ 未选择脚本"
        return

    script_path = os.path.join(script_dir, script_name)
    if not os.path.exists(script_path):
        yield f"❌ 找不到脚本文件: {script_path}"
        return

    command = f"CUDA_VISIBLE_DEVICES={gpu} bash {script_path}"
    if port:
        command += f" {gpu} {port}"
    else:
        command += f" {gpu}"

    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            output_lines.append(line.strip())
            yield "\n".join(output_lines)

        process.wait()
        if process.returncode == 0:
            yield "✅ 加载完成"
        else:
            yield "❌ 启动失败，请检查脚本"

    except Exception as e:
        yield f"❌ 启动失败: {str(e)}"


# 模型关闭（按端口）
def stop_model(port):
    try:
        command = f"fuser -k {port}/tcp"
        subprocess.run(command, shell=True, check=True)
        return f"🛑 已关闭端口 {port} 上的服务"
    except subprocess.CalledProcessError:
        return f"⚠️ 未找到服务或关闭失败"


# 攻击算法关闭（按 端口）
def stop_attack(port=1337):
    try:
        command = f"fuser -k {port}/tcp"
        subprocess.run(command, shell=True, check=True)
        return f"🛑 已关闭端口 {port} 上的攻击算法进程"
    except subprocess.CalledProcessError:
        return f"⚠️ 未找到相关进程或关闭失败"

# ===== 固定映射：模型名称 → 脚本文件名 =====
MODEL_SCRIPTS_MAP = {

}

# ===== 固定映射：攻击算法名称 → 脚本文件名 =====
ATTACK_SCRIPTS_MAP = {
}


# ===== 全局：任务/数据集 → 合法 infer/eval 搭配（通用于所有评测）=====
CONFIG_MATRIX = {
    }

# —— 联动：引用 CONFIG_MATRIX ——
def _on_task_change(task):
    if not task:
        return (gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                gr.update(value=""), gr.update(value=""),
                gr.update(value=False))
    dsets = list(CONFIG_MATRIX[task].keys())
    return (gr.update(choices=dsets, value=None),
            gr.update(choices=[], value=None),
            gr.update(value=""), gr.update(value=""),
            gr.update(value=False))


def _on_dataset_change(task, dataset):
    if not (task and dataset):
        return gr.update(choices=[], value=None), gr.update(value=""), gr.update(value="")
    opts = CONFIG_MATRIX[task][dataset]
    scheme_choices = [o.get("label") or f"{o['infer']} / {o['eval']}" for o in opts]
    infer, eval_ = opts[0]["infer"], opts[0]["eval"]
    return gr.update(choices=scheme_choices, value=scheme_choices[0]), gr.update(value=infer), gr.update(
        value=eval_)


def _on_scheme_change(task, dataset, scheme_label):
    if not (task and dataset and scheme_label):
        return gr.update(), gr.update()
    for o in CONFIG_MATRIX[task][dataset]:
        lbl = o.get("label") or f"{o['infer']} / {o['eval']}"
        if lbl == scheme_label:
            return gr.update(value=o["infer"]), gr.update(value=o["eval"])
    return gr.update(), gr.update()


def _toggle_custom(allow, task, dataset, scheme_label):
    locked = False
    if task and dataset and scheme_label:
        for o in CONFIG_MATRIX[task][dataset]:
            lbl = o.get("label") or f"{o['infer']} / {o['eval']}"
            if lbl == scheme_label:
                locked = o.get("locked", False)
                break
    can_edit = bool(allow and not locked)
    return gr.update(interactive=can_edit), gr.update(interactive=can_edit)


# --- shared validators (全局共享) ---
def validate_combo(task, dataset, scheme_label, infer, eval_, allow_custom, config=CONFIG_MATRIX):
    if not (task and dataset):
        gr.Warning("请选择任务与数据集")
        return False
    if allow_custom:
        return True
    for o in config.get(task, {}).get(dataset, []):
        lbl = o.get("label") or f"{o['infer']} / {o['eval']}"
        if lbl == scheme_label and o["infer"] == infer and o["eval"] == eval_:
            return True
    gr.Error("所选组合不在支持矩阵中；请更换“推荐搭配”或勾选“允许自定义”。")
    return False

def launch_script_by_map(script_dir, mapping, name, gpu, port=None):
    if not name:
        yield "⚠️ 未选择名称"
        return
    if name not in mapping:
        yield f"❌ 未找到对应脚本：{name}"
        return

    script_file = mapping[name]
    script_path = os.path.join(script_dir, script_file)
    if not os.path.exists(script_path):
        yield f"❌ 找不到脚本文件: {script_path}"
        return

    command = f"CUDA_VISIBLE_DEVICES={gpu} bash {script_path} {gpu}"
    if port:
        command += f" {port}"

    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output_lines = []
        for line in iter(process.stdout.readline, ""):
            output_lines.append(line.strip())
            yield "\n".join(output_lines)

        process.wait()
        if process.returncode == 0:
            yield "✅ 加载完成"
        else:
            yield "❌ 启动失败，请检查脚本"
    except Exception as e:
        yield f"❌ 启动失败: {str(e)}"


# ========== Gradio UI ==========
with gr.Blocks() as demo:
    gr.Markdown("# 模型服务 + 攻击算法加载 + 多维度评测")

    # === 模型加载 ===
    for i in range(1, 4):
        with gr.Accordion(label=f"模型 {i} 配置", open=(i == 1)):
            model_name = gr.Dropdown(label="选择模型", choices=list(MODEL_SCRIPTS_MAP.keys()), interactive=True)
            gpu = gr.Textbox(label="GPU 编号", value="0")
            port = gr.Textbox(label="端口号", value=str(3000 + i - 1))
            output = gr.Textbox(label="输出信息", lines=10)

            launch = gr.Button(f"启动模型 {i}")
            stop = gr.Button(f"关闭模型 {i}")


            def _launch_model(name, gpu, port):
                yield from launch_script_by_map(MODEL_SCRIPT_DIR, MODEL_SCRIPTS_MAP, name, gpu, port)

            launch.click(fn=_launch_model, inputs=[model_name, gpu, port], outputs=output)

            stop.click(fn=stop_model, inputs=port, outputs=output)

    # === 攻击算法加载 ===
    gr.Markdown("## 攻击算法加载")
    with gr.Accordion("攻击算法部署", open=False):
        attack_name = gr.Dropdown(label="选择攻击算法", choices=list(ATTACK_SCRIPTS_MAP.keys()), interactive=True)
        # 旁边放一条提示
        attack_gpu = gr.Textbox(label="GPU 编号", value="0")
        attack_output = gr.Textbox(label="输出信息", lines=10)

        attack_launch = gr.Button("加载攻击算法")
        attack_stop = gr.Button("关闭攻击算法")


        def _launch_attack(name, gpu):
            yield from launch_script_by_map(ATTACK_SCRIPT_DIR, ATTACK_SCRIPTS_MAP, name, gpu)

        attack_launch.click(fn=_launch_attack, inputs=[attack_name, attack_gpu], outputs=attack_output)

        attack_stop.click(fn=stop_attack, outputs=attack_output)


    def build_support_matrix_md(config):
        lines = ["| 任务 | 数据集 | 推荐搭配（infer → eval） |", "|---|---|---|"]
        for task, dsets in config.items():
            for ds, opts in dsets.items():
                pairs = " / ".join([f"{o['infer']} → {o['eval']}" for o in opts])
                lines.append(f"| {task} | {ds} | {pairs} |")
        return "\n".join(lines)


    with gr.Accordion("📚 支持矩阵（全局，适用于所有评测）", open=False):
        gr.Markdown(build_support_matrix_md(CONFIG_MATRIX))

    # === 准确性评测 ===
    gr.Markdown("## 准确性评测任务")

    with gr.Accordion("运行准确性评测", open=False):
        # 选择器（引用全局 CONFIG_MATRIX）
        with gr.Row():
            acc_task = gr.Dropdown(label="任务", choices=list(CONFIG_MATRIX.keys()), interactive=True)
            acc_dataset = gr.Dropdown(label="数据集", choices=[], interactive=True)
            acc_scheme = gr.Dropdown(label="推荐搭配", choices=[], interactive=True)

        with gr.Row():
            acc_infer_type = gr.Textbox(label="推理类型", interactive=False)
            acc_eval_type = gr.Textbox(label="评测类型", interactive=False)
        acc_allow_custom = gr.Checkbox(label="允许自定义 infer/eval（非特殊任务）", value=False)

        with gr.Row():
            acc_model = gr.Textbox(label="模型名称", value="qwen-vl")
            acc_port = gr.Textbox(label="模型端口", value="3000")

        with gr.Row():
            acc_output_dir = gr.Textbox(label="输出目录",
                                        value="")
            acc_gpu = gr.Textbox(label="GPU 编号", value="0")

        # 补回 key（你的 _acc_run 里需要）
        acc_key = gr.Textbox(label="密钥 key（可选）", value="")

        acc_button = gr.Button("运行准确性评测", variant="primary")
        acc_output = gr.Textbox(label="输出日志", lines=20, show_copy_button=True)

        # 新增三件套：小文件 JSON、大文件预览、下载
        acc_result_json = gr.JSON(label="评测结果（≤512KB 自动展示）", visible=False)
        acc_result_preview = gr.Code(label="结果预览（大文件截断）", language="json", visible=False)
        acc_result_file = gr.File(label="下载完整结果", visible=False)




        acc_task.change(_on_task_change, inputs=acc_task,
                        outputs=[acc_dataset, acc_scheme, acc_infer_type, acc_eval_type, acc_allow_custom], queue=False)
        acc_dataset.change(_on_dataset_change, inputs=[acc_task, acc_dataset],
                           outputs=[acc_scheme, acc_infer_type, acc_eval_type], queue=False)
        acc_scheme.change(_on_scheme_change, inputs=[acc_task, acc_dataset, acc_scheme],
                          outputs=[acc_infer_type, acc_eval_type], queue=False)
        acc_allow_custom.change(_toggle_custom, inputs=[acc_allow_custom, acc_task, acc_dataset, acc_scheme],
                                outputs=[acc_infer_type, acc_eval_type], queue=False)


        # —— 运行 & 大文件处理（与干扰性评测同逻辑）——
        def _acc_run(task, dataset, scheme_label, infer, eval_, allow_custom,
                     model_name, port, output_dir, key, gpu):
            import time, os, json  # 确保可用
            # 参数校验
            if not validate_combo(task, dataset, scheme_label, infer, eval_, allow_custom):
                return "❌ 评测参数无效，请检查选择。", gr.update(visible=False), gr.update(visible=False), gr.update(
                    visible=False)

            try:
                start_ts = time.time()

                config = {
                    "data": [{"dataset_name": dataset, "type": None}],
                    "model": {"model_name": model_name, "port": int(port), "keys": [key] if key else []},
                    "output": {"output_dir": output_dir},
                    "evaluate": [{"infer_type": infer, "eval_type": eval_}],
                }

                with NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as temp_config:
                    yaml.dump(config, temp_config, allow_unicode=True)
                    temp_config_path = temp_config.name

                command = ()

                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True)

                output_lines = [f"启动准确性评测：task={task}, dataset={dataset}, infer/eval={infer}/{eval_}"]
                # 初始占位：只更新日志
                yield "\n".join(output_lines), gr.update(), gr.update(), gr.update()

                for line in iter(process.stdout.readline, ''):
                    output_lines.append(line.rstrip("\n"))
                    yield "\n".join(output_lines), gr.update(), gr.update(), gr.update()

                process.wait()

                if process.returncode == 0:
                    json_path = locate_result_json("accuracy", output_dir, model_name, dataset, start_ts)
                    if json_path and os.path.exists(json_path):
                        size = os.path.getsize(json_path)
                        if size <= MAX_JSON_BYTES:
                            with open(json_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            output_lines.append(f"✅ 评测完成，结果文件：{json_path}（{_fmt_size(size)}）")
                            yield "\n".join(output_lines), gr.update(value=data, visible=True), gr.update(
                                visible=False), gr.update(value=json_path, visible=True)
                        else:
                            preview, _ = build_json_preview_from_file(json_path)
                            output_lines.append(f"✅ 评测完成，结果较大（{_fmt_size(size)}），显示预览并提供下载")
                            yield "\n".join(output_lines), gr.update(visible=False), gr.update(value=preview,
                                                                                               visible=True), gr.update(
                                value=json_path, visible=True)
                    else:
                        output_lines.append("⚠️ 评测完成，但未找到结果 JSON（请检查 RESULT_PATTERNS 命名与大小写）。")
                        yield "\n".join(output_lines), gr.update(visible=False), gr.update(visible=False), gr.update(
                            visible=False)
                else:
                    output_lines.append("❌ 评测失败，请检查日志")
                    yield "\n".join(output_lines), gr.update(visible=False), gr.update(visible=False), gr.update(
                        visible=False)

            except Exception as e:
                return f"❌ 运行失败: {str(e)}", gr.update(visible=False), gr.update(visible=False), gr.update(
                    visible=False)


        acc_button.click(
            _acc_run,
            inputs=[acc_task, acc_dataset, acc_scheme, acc_infer_type, acc_eval_type, acc_allow_custom,
                    acc_model, acc_port, acc_output_dir, acc_key, acc_gpu],
            outputs=[acc_output, acc_result_json, acc_result_preview, acc_result_file]
        )

    # === 抗干扰性评测 ===
    gr.Markdown("## 抗干扰性评测任务")

    with gr.Accordion("运行抗干扰性评测", open=False):
        # 任务/数据集/推荐搭配（复用全局 CONFIG_MATRIX）
        with gr.Row():
            inter_task = gr.Dropdown(label="任务", choices=list(CONFIG_MATRIX.keys()), interactive=True)
            inter_dataset = gr.Dropdown(label="数据集", choices=[], interactive=True)
            inter_scheme = gr.Dropdown(label="推荐搭配", choices=[], interactive=True)

        with gr.Row():
            inter_infer_type = gr.Textbox(label="推理类型", interactive=False)
            inter_eval_type = gr.Textbox(label="评测类型", interactive=False)
        inter_allow_custom = gr.Checkbox(label="允许自定义 infer/eval（非特殊任务）", value=False)

        with gr.Row():
            inter_model = gr.Textbox(label="模型名称", value="qwen-vl")
            inter_port = gr.Textbox(label="模型端口", value="3000")

        inter_output_dir = gr.Textbox(
            label="输出目录",
            value=""
        )

        with gr.Row():
            # 仅保留下拉：干扰类型
            INTERFERENCE_CHOICES = ["salt", "gaussian_noise", "motion_blur", "jpeg", "resize"]
            interference_type = gr.Dropdown(
                label="干扰类型",
                choices=INTERFERENCE_CHOICES,
                value="salt",
                interactive=True,
            )
            inter_key = gr.Textbox(label="密钥 key（可选）", value="")
            inter_gpu = gr.Textbox(label="GPU 编号", value="0")

        inter_button = gr.Button("运行抗干扰性评测", variant="primary")
        inter_output = gr.Textbox(label="输出日志", lines=20, show_copy_button=True)

        # 新增三件套（默认隐藏）
        inter_result_json = gr.JSON(label="评测结果（≤512KB 自动展示）", visible=False)
        inter_result_preview = gr.Code(label="结果预览（大文件截断）", language="json", visible=False)
        inter_result_file = gr.File(label="下载完整结果", visible=False)

        # -------- 联动：沿用准确性评测的 3 个事件处理器/校验器 ----------
        inter_task.change(
            _on_task_change, inputs=inter_task,
            outputs=[inter_dataset, inter_scheme, inter_infer_type, inter_eval_type, inter_allow_custom],
            queue=False
        )
        inter_dataset.change(
            _on_dataset_change, inputs=[inter_task, inter_dataset],
            outputs=[inter_scheme, inter_infer_type, inter_eval_type], queue=False
        )
        inter_scheme.change(
            _on_scheme_change, inputs=[inter_task, inter_dataset, inter_scheme],
            outputs=[inter_infer_type, inter_eval_type], queue=False
        )
        inter_allow_custom.change(
            _toggle_custom, inputs=[inter_allow_custom, inter_task, inter_dataset, inter_scheme],
            outputs=[inter_infer_type, inter_eval_type], queue=False
        )


        # -------- 运行逻辑（流式日志 + 自动抓 JSON） ----------
        def run_interference_eval(task, dataset, scheme_label, infer_type, eval_type, allow_custom,
                                  model_name, port, output_dir, interference, key, gpu):
            import time, os, json  # 确保可用
            # 1) 参数校验
            if not validate_combo(task, dataset, scheme_label, infer_type, eval_type, allow_custom):
                return (
                    "❌ 评测参数无效，请检查选择。",
                    gr.update(visible=False),  # json
                    gr.update(visible=False),  # code
                    gr.update(visible=False),  # file
                )

            try:
                start_ts = time.time()

                config = {
                    "data": [{"dataset_name": dataset, "type": None}],
                    "model": {"model_name": model_name, "port": int(port), "keys": [key] if key else []},
                    "output": {"output_dir": output_dir},
                    "evaluate": [{
                        "infer_type": infer_type,
                        "eval_type": eval_type,
                        "interference": interference,
                        "interference_kwargs": None
                    }]
                }

                with NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as temp_config:
                    yaml.dump(config, temp_config, allow_unicode=True)
                    temp_config_path = temp_config.name

                command = ()

                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True)

                output_lines = [f"启动抗干扰性评测：task={task}, dataset={dataset}, interference={interference}"]
                # ✅ 初次占位：必须返回 4 个值
                yield "\n".join(output_lines), gr.update(), gr.update(), gr.update()

                for line in iter(process.stdout.readline, ''):
                    output_lines.append(line.rstrip("\n"))
                    # ✅ 过程阶段也要返回 4 个值（其它位用占位符不变更）
                    yield "\n".join(output_lines), gr.update(), gr.update(), gr.update()

                process.wait()

                if process.returncode == 0:
                    json_path = locate_result_json(
                        "interference",
                        output_dir=output_dir,
                        model_name=model_name,
                        dataset=dataset,
                        start_ts=start_ts,
                        interference_type=interference,  # 注意：传值，不是组件
                    )
                    if json_path and os.path.exists(json_path):
                        size = os.path.getsize(json_path)
                        if size <= MAX_JSON_BYTES:
                            with open(json_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            output_lines.append(f"✅ 完成，结果文件：{json_path}（{_fmt_size(size)}）")
                            yield (
                                "\n".join(output_lines),
                                gr.update(value=data, visible=True),  # json
                                gr.update(visible=False),  # code
                                gr.update(value=json_path, visible=True),  # file
                            )
                        else:
                            preview, _ = build_json_preview_from_file(json_path)
                            output_lines.append(f"✅ 完成，结果较大（{_fmt_size(size)}），显示预览并提供下载")
                            yield (
                                "\n".join(output_lines),
                                gr.update(visible=False),  # json
                                gr.update(value=preview, visible=True),  # code
                                gr.update(value=json_path, visible=True),  # file
                            )
                    else:
                        output_lines.append("⚠️ 完成，但未找到结果 JSON（请核对 RESULT_PATTERNS 与大小写）。")
                        yield (
                            "\n".join(output_lines),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )
                else:
                    output_lines.append("❌ 评测失败，请检查日志")
                    yield (
                        "\n".join(output_lines),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                    )

            except Exception as e:
                # ✅ 异常时也必须返回 4 个值
                return (
                    f"❌ 运行失败: {str(e)}",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )


        inter_button.click(
            run_interference_eval,
            inputs=[
                inter_task, inter_dataset, inter_scheme, inter_infer_type, inter_eval_type, inter_allow_custom,
                inter_model, inter_port, inter_output_dir, interference_type, inter_key, inter_gpu
            ],
            outputs=[inter_output, inter_result_json, inter_result_preview, inter_result_file]  # ← 日志 + 三件套
        )


    # 仅本模块用到的 attack_kwargs 预设（内置，不暴露 UI）
    ATK_KW_PRESETS = {
    }


    def _auto_attack_eval_type(dataset_name: str) -> str:
        return "acc" if str(dataset_name).strip() == "RSICDObject" else "null"


    def _auto_attack_kwargs(attack: str, dataset_name: str):
        """返回 Python 对象（None 或 dict），供 yaml.dump；不返回字符串。"""
        return ATK_KW_PRESETS.get(str(attack).strip(), {}).get(str(dataset_name).strip())


    # === 抗攻击性评测 ===
    gr.Markdown("## 抗攻击性评测任务")

    with gr.Accordion("运行抗攻击性评测任务", open=False):
        # 任务 / 数据集 / 推荐搭配（与前面一致）
        with gr.Row():
            atk_task = gr.Dropdown(label="任务", choices=list(CONFIG_MATRIX.keys()), interactive=True)
            atk_dataset = gr.Dropdown(label="数据集", choices=[], interactive=True)
            atk_scheme = gr.Dropdown(label="推荐搭配", choices=[], interactive=True)

        with gr.Row():
            atk_infer_type = gr.Textbox(label="推理类型", interactive=False)
            atk_eval_type = gr.Textbox(label="评测类型", interactive=False)
        atk_allow_custom = gr.Checkbox(label="允许自定义 infer/eval（非特殊任务）", value=False)

        with gr.Row():
            atk_model = gr.Textbox(label="模型名称", value="qwen-vl")
            atk_port = gr.Textbox(label="模型端口", value="3000")

        atk_output_dir = gr.Textbox(label="输出目录", value="/path/to/output")

        # 攻击方式（用已部署的名称）
        _atk_choices_base = []
        if "ATTACK_SCRIPTS_MAP" in globals() and isinstance(ATTACK_SCRIPTS_MAP, dict):
            _atk_choices_base = list(ATTACK_SCRIPTS_MAP.keys())
        essential_attacks = ["SSA_CWA_untarget", "SSA_CWA_target"]
        ATTACK_CHOICES = sorted(set(_atk_choices_base + essential_attacks))

        with gr.Row():
            atk_attack = gr.Dropdown(label="攻击方式", choices=ATTACK_CHOICES, value="SSA_CWA_untarget",
                                     interactive=True)
            atk_attack_metric = gr.Textbox(label="攻击评测指标（自动）", value="", interactive=False,
                                           info="RSICDObject=acc，其余=null")

        # 仅 RSICDObject 需要代理 & key
        with gr.Row():
            atk_http_proxy = gr.Textbox(label="http_proxy（仅 RSICDObject）", value="", visible=False)
            atk_https_proxy = gr.Textbox(label="https_proxy（仅 RSICDObject）", value="", visible=False)
            atk_key = gr.Textbox(label="评估模型 key（仅 RSICDObject）", value="", visible=False)

        atk_gpu = gr.Textbox(label="GPU 编号", value="0")

        atk_button = gr.Button("运行抗攻击性评测", variant="primary")
        atk_output = gr.Textbox(label="评测输出", lines=20, show_copy_button=True)
        atk_result_json = gr.JSON(label="评测结果（≤2MB 自动展示）", visible=False)
        atk_result_preview = gr.Code(label="结果预览（大文件截断）", language="json", visible=False)
        atk_result_file = gr.File(label="下载完整结果", visible=False)

        # ---- 联动（与前面一致）----
        atk_task.change(
            _on_task_change, inputs=atk_task,
            outputs=[atk_dataset, atk_scheme, atk_infer_type, atk_eval_type, atk_allow_custom],
            queue=False
        )
        atk_dataset.change(
            _on_dataset_change, inputs=[atk_task, atk_dataset],
            outputs=[atk_scheme, atk_infer_type, atk_eval_type], queue=False
        )
        atk_scheme.change(
            _on_scheme_change, inputs=[atk_task, atk_dataset, atk_scheme],
            outputs=[atk_infer_type, atk_eval_type], queue=False
        )
        atk_allow_custom.change(
            _toggle_custom, inputs=[atk_allow_custom, atk_task, atk_dataset, atk_scheme],
            outputs=[atk_infer_type, atk_eval_type], queue=False
        )


        # 数据集或攻击方式变化时：自动更新“攻击评测指标”、代理/key可见性
        def _atk_update_by_ds_or_attack(dataset, attack):
            metric = _auto_attack_eval_type(dataset)
            show = (str(dataset).strip() == "RSICDObject")
            return (
                gr.update(value=metric),  # atk_attack_metric
                gr.update(visible=show,
                          value="" if not show else atk_http_proxy.value if hasattr(atk_http_proxy, "value") else ""),
                gr.update(visible=show,
                          value="" if not show else atk_https_proxy.value if hasattr(atk_https_proxy, "value") else ""),
                gr.update(visible=show, value="" if not show else atk_key.value if hasattr(atk_key, "value") else "")
            )


        atk_dataset.change(_atk_update_by_ds_or_attack, inputs=[atk_dataset, atk_attack],
                           outputs=[atk_attack_metric, atk_http_proxy, atk_https_proxy, atk_key], queue=False)
        atk_attack.change(_atk_update_by_ds_or_attack, inputs=[atk_dataset, atk_attack],
                          outputs=[atk_attack_metric, atk_http_proxy, atk_https_proxy, atk_key], queue=False)


        # ---- 运行（内置 attack_kwargs + 流式 + 结果抓取 + 大文件处理）----
        def run_attack_eval(
                task, dataset, scheme_label, infer_type, eval_type, allow_custom,
                model_name, port, output_dir,
                attack, http_proxy, https_proxy, key, gpu
        ):
            import time, os, json, yaml
            # 合法性校验（固定模式）
            if not validate_combo(task, dataset, scheme_label, infer_type, eval_type, allow_custom):
                return (
                    "❌ 评测参数无效，请检查选择。",
                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                )

            # 固定：dataset_type=None；model.keys=[]
            ds_type = None
            model_keys = []

            # 攻击评测指标 & 代理/key（仅 RSICDObject）
            attack_eval_type = _auto_attack_eval_type(dataset)
            use_proxy = (str(dataset).strip() == "RSICDObject")
            http_proxy = http_proxy if use_proxy else ""
            https_proxy = https_proxy if use_proxy else ""
            eval_keys = [key] if use_proxy and key else []

            # 内置 attack_kwargs
            atk_kwargs_obj = _auto_attack_kwargs(attack, dataset)  # None 或 dict

            try:
                start_ts = time.time()

                config = {
                    "data": [{"dataset_name": dataset, "type": ds_type}],
                    "model": {"model_name": model_name, "port": int(port), "keys": model_keys},
                    "output": {"output_dir": output_dir},
                    "evaluate": [{
                        "infer_type": infer_type,
                        "eval_type": eval_type,
                        "keys": eval_keys,  # 与 model.keys 不同
                        "proxy": {"http_proxy": http_proxy, "https_proxy": https_proxy},
                        "attack": attack,
                        "attack_kwargs": atk_kwargs_obj,  # 直接是对象，yaml.dump 会写正确结构或 null
                        "attack_eval_type": (attack_eval_type if attack_eval_type != "null" else None),
                    }]
                }

                with NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as temp_config:
                    yaml.dump(config, temp_config, allow_unicode=True, sort_keys=False)
                    temp_config_path = temp_config.name

                command = (
                )

                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True)

                # 起始提示：顺便回显我们最终使用的 attack_kwargs（只打印一行概览）
                kw_preview = "None" if atk_kwargs_obj is None else yaml.safe_dump(atk_kwargs_obj, allow_unicode=True,
                                                                                  sort_keys=False).strip().replace("\n",
                                                                                                                   " ")
                output_lines = [
                    f"启动抗攻击性评测：dataset={dataset}, attack={attack}, attack_eval_type={attack_eval_type}, kwargs={kw_preview}"]
                yield "\n".join(output_lines), gr.update(), gr.update(), gr.update()

                for line in iter(process.stdout.readline, ''):
                    output_lines.append(line.rstrip("\n"))
                    yield "\n".join(output_lines), gr.update(), gr.update(), gr.update()

                process.wait()

                if process.returncode == 0:
                    json_path = locate_result_json(
                        "attack",
                        output_dir=output_dir,
                        model_name=model_name,
                        dataset=dataset,
                        start_ts=start_ts,
                        attack=attack
                    )
                    if json_path and os.path.exists(json_path):
                        size = os.path.getsize(json_path)
                        if size <= MAX_JSON_BYTES:
                            with open(json_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            output_lines.append(f"✅ 完成，结果文件：{json_path}（{_fmt_size(size)}）")
                            yield "\n".join(output_lines), gr.update(value=data, visible=True), gr.update(
                                visible=False), gr.update(value=json_path, visible=True)
                        else:
                            preview, _ = build_json_preview_from_file(json_path)
                            output_lines.append(f"✅ 完成，结果较大（{_fmt_size(size)}），显示预览并提供下载")
                            yield "\n".join(output_lines), gr.update(visible=False), gr.update(value=preview,
                                                                                               visible=True), gr.update(
                                value=json_path, visible=True)
                    else:
                        output_lines.append(
                            "⚠️ 完成，但未找到结果 JSON（请检查 RESULT_PATTERNS['attack'] 与命名/大小写）。")
                        yield "\n".join(output_lines), gr.update(visible=False), gr.update(visible=False), gr.update(
                            visible=False)
                else:
                    output_lines.append("❌ 评测失败，请检查日志")
                    yield "\n".join(output_lines), gr.update(visible=False), gr.update(visible=False), gr.update(
                        visible=False)

            except Exception as e:
                return f"❌ 运行失败: {str(e)}", gr.update(visible=False), gr.update(visible=False), gr.update(
                    visible=False)


        atk_button.click(
            run_attack_eval,
            inputs=[
                atk_task, atk_dataset, atk_scheme, atk_infer_type, atk_eval_type, atk_allow_custom,
                atk_model, atk_port, atk_output_dir,
                atk_attack, atk_http_proxy, atk_https_proxy, atk_key, atk_gpu
            ],
            outputs=[atk_output, atk_result_json, atk_result_preview, atk_result_file]
        )



    # === 范化性评测 ===
    gr.Markdown("## 泛化性评测任务")

    with gr.Accordion("运行范化性评测", open=False):
        with gr.Row():
            gen_model = gr.Textbox(label="模型名称", value="qwen-vl")
            gen_port = gr.Textbox(label="模型端口", value="3000")
        gen_output_dir = gr.Textbox(label="输出目录",
                                    value="/path/to/output")

        with gr.Row():
            dataset1 = gr.Textbox(label="数据集 1 名称", value="RSICDObject")
            dataset1_type = gr.Textbox(label="数据集 1 类型", value="null")
        with gr.Row():
            dataset2 = gr.Textbox(label="数据集 2 名称", value="FactFailure")
            dataset2_type = gr.Textbox(label="数据集 2 类型", value="null")

        with gr.Row():
            obj_infer_type = gr.Textbox(label="数据集 1 推理类型", value="object_eval")
            obj_eval_type = gr.Textbox(label="数据集 1 评测类型", value="object_eval")
        with gr.Row():
            http_proxy = gr.Textbox(label="http_proxy", value="")
            https_proxy = gr.Textbox(label="https_proxy", value="")

        with gr.Row():
            fact_infer_type = gr.Textbox(label="数据集 2 推理类型", value="fact")
            fact_eval_type = gr.Textbox(label="数据集 2 评测类型", value="fact")

        gen_key = gr.Textbox(label="密钥 key", value="123")
        gen_gpu = gr.Textbox(label="GPU 编号", value="0")

        gen_button = gr.Button("运行范化性评测")
        gen_output = gr.Textbox(label="评测输出", lines=20)


        def run_generalization_eval(
                model_name, port, output_dir,
                dataset1, dataset1_type, dataset2, dataset2_type,
                obj_infer_type, obj_eval_type, http_proxy, https_proxy,
                fact_infer_type, fact_eval_type, key, gpu):

            config = {
                "data": [
                    {"dataset_name": dataset1, "type": None if dataset1_type == "null" else dataset1_type},
                    {"dataset_name": dataset2, "type": None if dataset2_type == "null" else dataset2_type}
                ],
                "model": {
                    "model_name": model_name,
                    "port": int(port),
                    "keys": [key] if key else []
                },
                "output": {
                    "output_dir": output_dir
                },
                "evaluate": [
                    {
                        "infer_type": obj_infer_type,
                        "eval_type": obj_eval_type,
                        "keys": [key] if key else [],
                        "proxy": {
                            "http_proxy": http_proxy,
                            "https_proxy": https_proxy
                        }
                    },
                    {
                        "infer_type": fact_infer_type,
                        "eval_type": fact_eval_type
                    }
                ]
            }

            try:
                with NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as temp_config:
                    yaml.dump(config, temp_config)
                    temp_config_path = temp_config.name

                command = ()

                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True)
                output_lines = []
                for line in iter(process.stdout.readline, ''):
                    output_lines.append(line.strip())
                    yield "\n".join(output_lines)

                process.wait()
                if process.returncode == 0:
                    yield "✅ 范化性评测完成"
                else:
                    yield "❌ 评测失败，请检查日志"

            except Exception as e:
                yield f"❌ 运行失败: {str(e)}"


        gen_button.click(
            fn=run_generalization_eval,
            inputs=[
                gen_model, gen_port, gen_output_dir,
                dataset1, dataset1_type, dataset2, dataset2_type,
                obj_infer_type, obj_eval_type, http_proxy, https_proxy,
                fact_infer_type, fact_eval_type, gen_key, gen_gpu
            ],
            outputs=gen_output
        )

    # === 运行效率评测 ===
    gr.Markdown("##  运行效率评测任务")

    with gr.Accordion("运行效率评测", open=False):
        with gr.Row():
            eff_dataset = gr.Textbox(label="数据集名称", value="RSICD")
            eff_dataset_type = gr.Textbox(label="数据集类型", value="null")
        with gr.Row():
            eff_model = gr.Textbox(label="模型名称", value="qwen-vl")
            eff_port = gr.Textbox(label="模型端口", value="3000")

        eff_output_dir = gr.Textbox(label="输出目录",
                                    value="/path/to/output")

        with gr.Row():
            eff_infer_type = gr.Textbox(label="推理类型", value="ref")
            eff_eval_type = gr.Textbox(label="评测类型", value="ref")

        eff_key = gr.Textbox(label="密钥 key", value="")
        eff_gpu = gr.Textbox(label="GPU 编号", value="0")

        eff_button = gr.Button("运行效率评测")
        eff_output = gr.Textbox(label="评测输出", lines=20)


        def run_efficiency_eval(dataset_name, dataset_type, model_name, port, output_dir,
                                infer_type, eval_type, key, gpu):

            config = {
                "data": [{
                    "dataset_name": dataset_name,
                    "type": None if dataset_type == "null" else dataset_type
                }],
                "model": {
                    "model_name": model_name,
                    "port": int(port),
                    "keys": [key] if key else []
                },
                "output": {
                    "output_dir": output_dir
                },
                "evaluate": [{
                    "infer_type": infer_type,
                    "eval_type": eval_type
                }]
            }

            try:
                with NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as temp_config:
                    yaml.dump(config, temp_config)
                    temp_config_path = temp_config.name

                command = ()

                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True)
                output_lines = []
                for line in iter(process.stdout.readline, ''):
                    output_lines.append(line.strip())
                    yield "\n".join(output_lines)

                process.wait()
                if process.returncode == 0:
                    yield "✅ 运行效率评测完成"
                else:
                    yield "❌ 评测失败，请检查日志"

            except Exception as e:
                yield f"❌ 运行失败: {str(e)}"


        eff_button.click(
            fn=run_efficiency_eval,
            inputs=[
                eff_dataset, eff_dataset_type, eff_model, eff_port,
                eff_output_dir, eff_infer_type, eff_eval_type, eff_key, eff_gpu
            ],
            outputs=eff_output
        )

    # === 防御评测 ===
    gr.Markdown("## 防御评测任务")

    with gr.Accordion("运行防御评测", open=False):
        with gr.Row():
            def_dataset = gr.Textbox(label="数据集名称", value="RSICDObject")
            def_dataset_type = gr.Textbox(label="数据集类型", value="null")
        with gr.Row():
            def_model = gr.Textbox(label="模型名称", value="qwen2-2b")
            def_port = gr.Textbox(label="模型端口", value="2000")

        def_output_dir = gr.Textbox(label="输出目录",
                                    value="/path/to/output")

        with gr.Row():
            def_infer_type = gr.Textbox(label="推理类型", value="object_eval")
            def_eval_type = gr.Textbox(label="评测类型", value="object_eval")
        with gr.Row():
            def_attack = gr.Textbox(label="攻击方式", value="APGD")
            def_attack_eval_type = gr.Textbox(label="攻击评测指标", value="acc")

        def_attack_kwargs = gr.Textbox(label="攻击参数 attack_kwargs", value="None")

        with gr.Row():
            def_http_proxy = gr.Textbox(label="http_proxy", value="")
            def_https_proxy = gr.Textbox(label="https_proxy", value="")

        def_defense = gr.Textbox(label="防御策略 defense", value="keg")

        # === 分项输入 defense_kwargs ===
        def_knowledge_base = gr.Textbox(label="知识注入模型", value="llava")
        def_dino_port = gr.Textbox(label="dino_port", value="3000")
        def_llava_port = gr.Textbox(label="llava_port", value="4000")
        def_use_label = gr.Checkbox(label="use_label", value=False)

        def_key = gr.Textbox(label="密钥 key", value="123")
        def_gpu = gr.Textbox(label="GPU 编号", value="0")

        def_button = gr.Button("运行防御评测")
        def_output = gr.Textbox(label="评测输出", lines=20)


        def run_defense_eval(
                dataset_name, dataset_type,
                model_name, port, output_dir,
                infer_type, eval_type,
                attack, attack_eval_type,
                http_proxy, https_proxy,
                attack_kwargs,
                defense,
                knowledge_base, dino_port, llava_port, use_label,
                key, gpu):

            try:
                atk_kwargs_parsed = None if attack_kwargs.strip() in ["None", "null", ""] else yaml.safe_load(
                    attack_kwargs)
            except Exception:
                atk_kwargs_parsed = None

            defense_kwargs_parsed = {
                "knowledge_base": knowledge_base,
                "dino_port": int(dino_port),
                "llava_port": int(llava_port),
                "use_label": use_label
            }

            config = {
                "data": [{
                    "dataset_name": dataset_name,
                    "type": None if dataset_type == "null" else dataset_type
                }],
                "model": {
                    "model_name": model_name,
                    "port": int(port),
                    "keys": [key] if key else []
                },
                "output": {
                    "output_dir": output_dir
                },
                "evaluate": [{
                    "infer_type": infer_type,
                    "eval_type": eval_type,
                    "keys": [key] if key else [],
                    "proxy": {
                        "http_proxy": http_proxy,
                        "https_proxy": https_proxy
                    },
                    "attack": attack,
                    "attack_kwargs": atk_kwargs_parsed,
                    "attack_eval_type": attack_eval_type,
                    "defense": defense,
                    "defense_kwargs": defense_kwargs_parsed
                }]
            }

            try:
                with NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as temp_config:
                    yaml.dump(config, temp_config)
                    temp_config_path = temp_config.name

                command = ()

                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True)
                output_lines = []
                for line in iter(process.stdout.readline, ''):
                    output_lines.append(line.strip())
                    yield "\n".join(output_lines)

                process.wait()
                if process.returncode == 0:
                    yield "✅ 防御评测完成"
                else:
                    yield "❌ 评测失败，请检查日志"

            except Exception as e:
                yield f"❌ 运行失败: {str(e)}"


        def_button.click(
            fn=run_defense_eval,
            inputs=[
                def_dataset, def_dataset_type,
                def_model, def_port,
                def_output_dir,
                def_infer_type, def_eval_type,
                def_attack, def_attack_eval_type,
                def_http_proxy, def_https_proxy,
                def_attack_kwargs,
                def_defense,
                def_knowledge_base, def_dino_port, def_llava_port, def_use_label,
                def_key, def_gpu
            ],
            outputs=def_output
        )

# 启动 WebUI
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7980)



