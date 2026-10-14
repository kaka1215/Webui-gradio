# Backend integration

The application builds the same `data`, `model`, `output`, and `evaluate` YAML structure as the original UI. Backends own model inference, dataset loading, transformations, and metric computation.

## 1. Create a local deployment file

```bash
cp configs/example.yaml configs/local.yaml
python app.py --config configs/local.yaml --check-config
python app.py --config configs/local.yaml
```

`--check-config` validates the configuration structure, not backend availability. Empty commands are allowed while setting up the interface; their Run buttons remain disabled. Restart the application after changing the file.

The local configuration is ignored by Git. Keep credentials out of shared configuration files.

## 2. Register services and tasks

```yaml
models:
  my-model:
    script: ../backend/scripts/start_model.sh
attacks:
  my-attack:
    script: ../backend/scripts/start_attack.sh

tasks:
  Object recognition:
    RSICDObject:
      - label: Object evaluation
        infer: object_eval
        eval: object_eval
        locked: false
```

These are examples of names and paths, not bundled backends. `tasks` populates the linked selectors in accuracy, interference, and adversarial evaluation. Each dataset needs at least one scheme. `locked: true` prevents custom inference/evaluation values in both the UI and server validation. The remaining three panels accept dataset and scheme names directly.

Service scripts run from their own directory and receive:

```text
Model:   bash <script> <gpu> <port>
Attack:  bash <script> <gpu>
```

Both receive `CUDA_VISIBLE_DEVICES`. Scripts must run in the foreground, ideally using `exec` to start the server. The attack panel's port field is a process-management identifier; it is not passed to the legacy attack launcher. Set the attack service's listening port in its script.

Stop controls terminate processes started by this application. They do not search for or kill unrelated processes by port. POSIX process-group cleanup is supported; Windows process-tree cleanup is not provided. Linux is the intended host for GPU service launchers; the interface and Python evaluation commands can also run on macOS.

## 3. Connect evaluation commands

Each of the six evaluation keys accepts the same structure:

```yaml
evaluations:
  accuracy:
    cwd: ../backend
    command: ['{python}', 'evaluate.py', '--config', '{config}']
    results:
      - '{output_dir}/{model_name}/{dataset}/metrics.json'
```

Available keys: `accuracy`, `interference`, `attack`, `generalization`, `efficiency`, `defense`. Configure each with the command supported by your backend. The example `evaluate.py` filename and `--config` flag are illustrative.

- `command` is an argument list, executed without a shell. Paths containing spaces stay intact. Shell operators such as `&&` and `>` are not interpreted; use a wrapper script if needed.
- `{python}` expands to the interpreter running the WebUI. Supply another interpreter path if the backend uses a different environment.
- `{config}` is required and expands to a private temporary YAML file. It is removed after execution, failure, or cancellation.
- `cwd` and service script paths resolve relative to the deployment YAML. Omitted `cwd` uses that file's directory. Output paths in the generated payload resolve according to the backend, normally relative to `cwd`.
- GPU selection is passed through `CUDA_VISIBLE_DEVICES`; `PYTHONUNBUFFERED=1` helps Python backends stream logs.

Evaluations share one queue to avoid unintentionally starting six GPU jobs at once. Model services run separately. Stop applies to a running job; a queued request has not started a subprocess yet.

## 4. YAML payload

An accuracy request produces this structure:

```yaml
data:
  - dataset_name: RSICDObject
    type: null
model:
  model_name: my-model
  port: 3000
  keys: []
output:
  output_dir: outputs/accuracy
evaluate:
  - infer_type: object_eval
    eval_type: object_eval
```

The six modes preserve their original backend fields:

| Mode | Additional payload fields |
| --- | --- |
| Accuracy | Base payload above |
| Interference | `interference`, `interference_kwargs` |
| Adversarial | `attack`, `attack_kwargs`, `attack_eval_type`, evaluation `keys` and `proxy` |
| Generalization | Two ordered `data` entries and two corresponding `evaluate` entries |
| Efficiency | Base payload with the selected backend inference/evaluation types |
| Defense | Attack fields, `defense`, and `defense_kwargs` with knowledge model, auxiliary ports, and label flag |

For adversarial evaluation, the legacy contract uses `attack_eval_type: acc` for `RSICDObject` and `null` otherwise. API keys and proxies are passed to the evaluator only for `RSICDObject`; model keys stay empty in this mode. Other modes retain their original key placement.

Optional attack presets are selected by attack name and dataset:

```yaml
attack_presets:
  my-attack:
    RSICDObject:
      steps: 10
```

Defense parameters must be a YAML/JSON mapping or `null`; malformed parameters now produce an error instead of being silently discarded. Configuration preview masks keys and proxy values. The actual temporary payload retains the submitted values needed by the backend.

## 5. Results and logs

All six panels support result preview and download. `results` accepts exact paths or glob patterns, with these placeholders:

| Placeholder | Value |
| --- | --- |
| `{output_dir}` | Submitted output directory |
| `{model_name}` | Submitted model name |
| `{dataset}` | First dataset name |
| `{dataset_lower}` / `{dataset_slug}` | Lowercase name / lowercase with spaces replaced by underscores |
| `{interference}` / `{interference_type}` | Selected interference name |
| `{attack}` | Selected attack name |

Relative result patterns resolve from the evaluation working directory. Only JSON files modified since this run started are eligible, and the newest match is selected. There is no fallback to older files and no arbitrary directory scan. For multi-dataset evaluation, configure the backend's combined summary path if you want both datasets represented in one result.

Valid JSON up to 512 KiB is rendered directly. Larger or malformed files get a bounded text preview and a full-file download. A successful command with no matching fresh result is reported explicitly. Logs show a bounded tail; retain full logs in the backend if needed. Known form credentials are masked in streamed logs, but backend-generated files are not sanitized.

## Development

```bash
python -m unittest discover -s tests -v
python app.py --config configs/example.yaml --check-config
```

Tests execute a small, temporary Python fixture to check all six YAML payloads, process output, cancellation, cleanup, error paths, and result freshness. They do not download models or run benchmarks. For formatting, use Ruff with the repository's `pyproject.toml` configuration.

The default listener is `127.0.0.1`. Use `--host 0.0.0.0` only within a trusted environment or behind authenticated access: this application starts configured local processes and is not a multi-tenant public service.
