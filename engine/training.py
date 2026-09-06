"""Functions extracted without body changes from the existing training implementation."""
from __future__ import annotations
import csv, hashlib, json, math, os, random, shutil, sys, time
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
from engine import model as base
from engine.utils import amp_optimizer_step_succeeded, classification_metrics
AMP_INITIAL_SCALE = 1024.0
AMP_GROWTH_INTERVAL = 2**31 - 1
ROOT = None
PREFLIGHT_PATH = None
def select_job_data(config, job):
    raise RuntimeError("Use the public run.py entry point to bind a prepared workspace")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def amp_dtype(config: dict):
    import torch

    mode = config["training"]["amp"]
    if mode == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("C1 v2 requires CUDA bfloat16 support")
        return torch.bfloat16
    if mode == "float16":
        return torch.float16
    raise RuntimeError(f"unsupported C1 autocast mode: {mode}")


def create_grad_scaler(config: dict):
    import torch

    enabled = config["training"]["amp"] == "float16"
    return torch.amp.GradScaler(
        "cuda",
        init_scale=AMP_INITIAL_SCALE,
        growth_interval=AMP_GROWTH_INTERVAL,
        enabled=enabled,
    )


def expected_optimizer_steps(case_count: int, epochs: int, micro: int, accumulation: int) -> int:
    return epochs * math.ceil(math.ceil(case_count / micro) / accumulation)


def set_determinism(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def train_epoch(model, optimizer, scheduler, scaler, target_loader, source_loader, config: dict, device):
    import torch

    model.train()
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    synthetic_weight = float(config["training"]["joint_training"]["synthetic_loss_weight"])
    total_microsteps = len(target_loader)
    if source_loader is not None and len(source_loader) != total_microsteps:
        raise RuntimeError("joint route source and target microstep counts differ")
    iterator = zip(target_loader, source_loader) if source_loader is not None else iter(target_loader)
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    target_count = 0
    optimizer_steps = 0
    skipped_steps = 0
    for micro_index, payload in enumerate(iterator):
        group_start = (micro_index // accumulation) * accumulation
        group_size = min(accumulation, total_microsteps - group_start)
        if source_loader is None:
            target_images, target_grades, _ = payload
            target_images = target_images.to(device, non_blocking=True)
            target_grades = target_grades.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype(config)):
                raw_loss = base.loss_for("softmax_cross_entropy", model(target_images), target_grades)
        else:
            (target_images, target_grades, _), (source_images, source_grades, _) = payload
            target_images = target_images.to(device, non_blocking=True)
            target_grades = target_grades.to(device, non_blocking=True)
            source_images = source_images.to(device, non_blocking=True)
            source_grades = source_grades.to(device, non_blocking=True)
            combined = torch.cat([target_images, source_images], dim=0)
            with torch.autocast(device_type="cuda", dtype=amp_dtype(config)):
                logits = model(combined)
                target_loss = base.loss_for("softmax_cross_entropy", logits[: len(target_grades)], target_grades)
                source_loss = base.loss_for("softmax_cross_entropy", logits[len(target_grades):], source_grades)
                raw_loss = target_loss + synthetic_weight * source_loss
        if not torch.isfinite(raw_loss):
            raise RuntimeError("non-finite C1 training loss")
        scaler.scale(raw_loss / group_size).backward()
        loss_sum += float(raw_loss.detach()) * len(target_grades)
        target_count += len(target_grades)
        if (micro_index + 1) % accumulation == 0 or micro_index + 1 == total_microsteps:
            before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            after = float(scaler.get_scale())
            optimizer.zero_grad(set_to_none=True)
            if not amp_optimizer_step_succeeded(before, after):
                skipped_steps += 1
                raise RuntimeError("AMP skipped an optimizer update; exact C1 budget violated")
            scheduler.step()
            optimizer_steps += 1
    return loss_sum / max(target_count, 1), optimizer_steps, skipped_steps, target_count


def reliability_metrics(truths: np.ndarray, probabilities: np.ndarray, bins: int) -> dict[str, float]:
    if truths.ndim != 1 or probabilities.shape != (len(truths), 3):
        raise RuntimeError("invalid reliability metric arrays")
    if not np.isfinite(probabilities).all() or np.max(np.abs(probabilities.sum(axis=1) - 1.0)) > 1e-5:
        raise RuntimeError("invalid probability simplex")
    one_hot = np.eye(3, dtype=np.float64)[truths.astype(int)]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == truths).astype(np.float64)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (confidence < upper if index < bins - 1 else confidence <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return {"ece": float(ece), "brier": brier}


def all_numeric_values_finite(value) -> bool:
    if isinstance(value, dict):
        return all(all_numeric_values_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_numeric_values_finite(item) for item in value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    return True


def evaluate(model, loader, device, ece_bins: int, config: dict):
    import torch

    model.eval()
    truths, predictions, probabilities, case_ids = [], [], [], []
    loss_sum = 0.0
    with torch.inference_mode():
        for images, grades, ids in loader:
            images = images.to(device, non_blocking=True)
            grades_device = grades.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype(config)):
                logits = model(images)
                batch_loss = base.loss_for("softmax_cross_entropy", logits, grades_device)
                batch_probabilities, batch_predictions = base.probabilities_and_predictions(
                    "softmax_cross_entropy", logits
                )
            loss_sum += float(batch_loss) * len(grades)
            truths.extend(grades.numpy().tolist())
            predictions.extend(batch_predictions.cpu().numpy().tolist())
            probabilities.extend(batch_probabilities.float().cpu().numpy().tolist())
            case_ids.extend(ids)
    truth_array = np.asarray(truths, dtype=int)
    prediction_array = np.asarray(predictions, dtype=int)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    metrics = classification_metrics(truth_array, prediction_array)
    metrics.update(reliability_metrics(truth_array, probability_array, ece_bins))
    metrics["loss"] = loss_sum / len(truths)
    rows = []
    for case_id, truth, prediction, probability in zip(case_ids, truths, predictions, probabilities):
        entropy = -sum(float(p) * math.log(max(float(p), 1e-12)) for p in probability)
        rows.append({
            "case_id": case_id,
            "truth_grade": truth,
            "predicted_grade": prediction,
            "prob_grade0": probability[0],
            "prob_grade1": probability[1],
            "prob_grade2": probability[2],
            "max_probability": max(probability),
            "predictive_entropy": entropy,
        })
    return metrics, rows


def _timed_update(config: dict, model, optimizer, scaler, target_batch, source_batch, device) -> float:
    import torch

    optimizer.zero_grad(set_to_none=True)
    target_images, target_grades, _ = target_batch
    target_images = target_images.to(device)
    target_grades = target_grades.to(device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=amp_dtype(config)):
        target_logits = model(target_images)
        loss = base.loss_for("softmax_cross_entropy", target_logits, target_grades)
        if source_batch is not None:
            source_images, source_grades, _ = source_batch
            source_images = source_images.to(device)
            source_grades = source_grades.to(device)
            source_logits = model(source_images)
            loss = loss + float(config["training"]["joint_training"]["synthetic_loss_weight"]) * base.loss_for(
                "softmax_cross_entropy", source_logits, source_grades
            )
    scaler.scale(loss).backward()
    before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    if not amp_optimizer_step_succeeded(before, float(scaler.get_scale())):
        raise RuntimeError("AMP preflight update skipped")
    torch.cuda.synchronize()
    return time.perf_counter() - start


def run_preflight(config: dict, plan: list[dict[str, str]], contract_digest: str) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("C1 preflight requires CUDA")
    representative = next(
        row for row in plan
        if row["job_type"] == "TARGET_FIT" and row["budget_percent"] == "10" and row["method_id"] == "C1_R3_NAIVE_JOINT_CE"
    )
    target, validation, source = select_job_data(config, representative)
    micro = int(config["training"]["micro_target_batch_size"])
    seed = int(representative["training_seed"])
    target_loader = base.make_loader(base.make_dataset(target, config, True, seed), micro, seed, 0, 0)
    source_loader = base.make_loader(
        base.make_dataset(source, config, True, seed), micro, seed + 17, 0, 0, replacement=True, num_samples=len(target)
    )
    target_batch = next(iter(target_loader))
    source_batch = next(iter(source_loader))
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = base.build_torch_components(config, "softmax_cross_entropy", seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = create_grad_scaler(config)
    warmup, timed = 3, 12
    target_only_times, joint_times = [], []
    for index in range(warmup + timed):
        elapsed = _timed_update(config, model, optimizer, scaler, target_batch, None, device)
        if index >= warmup:
            target_only_times.append(elapsed)
    for index in range(warmup + timed):
        elapsed = _timed_update(config, model, optimizer, scaler, target_batch, source_batch, device)
        if index >= warmup:
            joint_times.append(elapsed)
    validation_loader = base.make_eval_loader(
        base.make_dataset(validation, config, False, seed),
        int(config["training"]["evaluation_batch_size"]),
        0,
    )
    metrics, predictions = evaluate(
        model,
        validation_loader,
        device,
        int(config["evaluation"]["ece_equal_width_bins"]),
        config,
    )
    if not all_numeric_values_finite(metrics):
        raise RuntimeError("non-finite C1 preflight metric")
    peak_bytes = torch.cuda.max_memory_allocated()
    target_seconds = float(np.median(target_only_times))
    joint_seconds = float(np.median(joint_times))
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    per_job, projected_hours = {}, 0.0
    for job in plan:
        seconds = joint_seconds if job["method_id"] == "C1_R3_NAIVE_JOINT_CE" else target_seconds
        examples_per_epoch = (
            int(job["source_examples_per_epoch"])
            if job["job_type"] == "SOURCE_PRETRAIN"
            else int(job["target_examples_per_epoch"])
        )
        maximum_microsteps = math.ceil(examples_per_epoch / micro) * int(config["training"]["epochs"])
        hours = maximum_microsteps * seconds / 3600.0
        per_job[job["job_id"]] = {"maximum_microsteps_upper_bound": maximum_microsteps, "projected_gpu_hours": hours}
        projected_hours += hours
    disk = shutil.disk_usage(ROOT)
    payload = {
        "schema_version": "syn2real-ordinal-c1-resource-preflight-2",
        "decision": "PASS" if peak_bytes < 23 * 1024**3 and disk.free > 20 * 1024**3 else "FAIL",
        "contract_digest": contract_digest,
        "cuda_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "autocast_dtype": config["training"]["amp"],
        "grad_scaler_enabled": scaler.is_enabled(),
        "micro_target_batch_size": micro,
        "gradient_accumulation_steps": accumulation,
        "effective_target_batch_size": int(config["training"]["effective_target_batch_size"]),
        "median_seconds_per_target_only_microstep": target_seconds,
        "median_seconds_per_joint_microstep": joint_seconds,
        "peak_allocated_bytes": peak_bytes,
        "peak_allocated_gib": peak_bytes / 1024**3,
        "free_disk_bytes": disk.free,
        "free_disk_gib": disk.free / 1024**3,
        "maximum_projected_gpu_hours": projected_hours,
        "per_job_upper_bound": per_job,
        "preflight_metrics": metrics,
        "preflight_prediction_rows": len(predictions),
        "locked_rows_loaded": 0,
        "tpa_used": False,
        "m2_started": False,
        "training_checkpoint_written": False,
        "runner_sha256": sha256_file(Path(__file__)),
        "base_runner_sha256": sha256_file(Path(base.__file__)),
        "training_utils_sha256": sha256_file(Path(base.__file__).with_name("b0_training_utils.py")),
    }
    atomic_json(PREFLIGHT_PATH, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _write_predictions(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run_job(config: dict, job: dict[str, str], contract_digest: str) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("C1 production training requires CUDA")
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    if preflight["decision"] != "PASS" or preflight["contract_digest"] != contract_digest:
        raise RuntimeError("matching PASS C1 resource preflight is required")
    bound_hashes = {
        "runner_sha256": sha256_file(Path(__file__)),
        "base_runner_sha256": sha256_file(Path(base.__file__)),
        "training_utils_sha256": sha256_file(Path(base.__file__).with_name("b0_training_utils.py")),
    }
    for key, value in bound_hashes.items():
        if preflight[key] != value:
            raise RuntimeError(f"C1 execution code changed after preflight: {key}")
    seed = int(job["training_seed"])
    set_determinism(seed)
    target, validation, source = select_job_data(config, job)
    device = torch.device("cuda")
    model = base.build_torch_components(config, "softmax_cross_entropy", seed).to(device)
    if job["checkpoint_dependency"] == "C1_SYN_CE_PRETRAIN_V1":
        dependency = ROOT / "runs" / "c1" / "C1-SOURCE-PRETRAIN-CE" / "final.pt"
        checkpoint = torch.load(dependency, map_location="cpu", weights_only=False)
        if checkpoint["contract_digest"] != contract_digest or checkpoint["objective"] != "softmax_cross_entropy":
            raise RuntimeError("C1 source checkpoint authority mismatch")
        model.load_state_dict(checkpoint["model_state"], strict=True)
    training = config["training"]
    micro = int(training["micro_target_batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    epochs = int(training["epochs"])
    steps_per_epoch = math.ceil(math.ceil(
        (len(source) if job["job_type"] == "SOURCE_PRETRAIN" else len(target)) / micro
    ) / accumulation)
    total_steps = steps_per_epoch * epochs
    if total_steps != int(job["maximum_optimizer_steps"]):
        raise RuntimeError("computed optimizer budget differs from frozen plan")
    optimizer, scheduler = base.create_optimizer_scheduler(
        model, config, total_steps, steps_per_epoch * int(training["warmup_epochs"])
    )
    scaler = create_grad_scaler(config)
    run_dir = ROOT / "runs" / "c1" / job["job_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    completion_path = run_dir / "completion.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion["contract_digest"] != contract_digest:
            raise RuntimeError("completed C1 run has a different contract")
        print(json.dumps(completion, indent=2, sort_keys=True))
        return completion

    start_epoch = 0
    best_metrics = None
    best_epoch = None
    patience_count = 0
    optimizer_steps_done = 0
    skipped_steps_done = 0
    target_exposures_done = 0
    last_path = run_dir / "last.pt"
    if last_path.exists():
        resume = torch.load(last_path, map_location="cpu", weights_only=False)
        if resume["contract_digest"] != contract_digest or resume["job_id"] != job["job_id"]:
            raise RuntimeError("C1 resume checkpoint authority mismatch")
        model.load_state_dict(resume["model_state"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        scaler.load_state_dict(resume["scaler_state"])
        start_epoch = int(resume["epoch"]) + 1
        best_metrics = resume["best_metrics"]
        best_epoch = resume["best_epoch"]
        patience_count = int(resume["patience_count"])
        optimizer_steps_done = int(resume["optimizer_steps_done"])
        skipped_steps_done = int(resume["skipped_optimizer_steps_done"])
        target_exposures_done = int(resume["target_exposures_done"])
        base.validate_optimizer_step_authority(optimizer, optimizer_steps_done)

    if job["job_type"] == "SOURCE_PRETRAIN":
        target_dataset = base.make_dataset(source, config, True, seed)
        source_dataset = None
    else:
        target_dataset = base.make_dataset(target, config, True, seed)
        source_dataset = base.make_dataset(source, config, True, seed + 17) if source else None
    validation_loader = None
    if validation:
        validation_loader = base.make_eval_loader(
            base.make_dataset(validation, config, False, seed),
            int(training["evaluation_batch_size"]),
            int(training["num_workers"]),
        )
    epoch_log_path = run_dir / "epochs.jsonl"
    stopped_early = False
    job_start = time.time()
    last_epoch_completed = start_epoch - 1
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        target_loader = base.make_loader(
            target_dataset, micro, seed, epoch, int(training["num_workers"])
        )
        source_loader = None
        if source_dataset is not None:
            source_loader = base.make_loader(
                source_dataset,
                micro,
                seed + 17,
                epoch,
                int(training["num_workers"]),
                replacement=True,
                num_samples=len(target),
            )
        train_loss, epoch_steps, epoch_skips, epoch_target_exposures = train_epoch(
            model, optimizer, scheduler, scaler, target_loader, source_loader, config, device
        )
        optimizer_steps_done += epoch_steps
        skipped_steps_done += epoch_skips
        target_exposures_done += epoch_target_exposures
        base.validate_optimizer_step_authority(optimizer, optimizer_steps_done)
        last_epoch_completed = epoch
        validation_metrics = None
        improved = False
        if validation_loader is not None:
            validation_metrics, predictions = evaluate(
                model,
                validation_loader,
                device,
                int(config["evaluation"]["ece_equal_width_bins"]),
                config,
            )
            improved = base.better_metrics(validation_metrics, best_metrics, epoch, best_epoch)
            if improved:
                best_metrics = validation_metrics
                best_epoch = epoch
                patience_count = 0
                base.atomic_torch_save({
                    "schema_version": "syn2real-ordinal-c1-model-checkpoint-2",
                    "contract_digest": contract_digest,
                    "job_id": job["job_id"],
                    "epoch": epoch,
                    "objective": "softmax_cross_entropy",
                    "metrics": validation_metrics,
                    "model_state": model.state_dict(),
                }, run_dir / "best.pt")
                _write_predictions(run_dir / "best_validation_predictions.csv", predictions)
            else:
                patience_count += 1
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "optimizer_steps": epoch_steps,
            "optimizer_steps_done": optimizer_steps_done,
            "skipped_optimizer_steps": epoch_skips,
            "skipped_optimizer_steps_done": skipped_steps_done,
            "target_exposures": epoch_target_exposures,
            "target_exposures_done": target_exposures_done,
            "validation_metrics": validation_metrics,
            "improved": improved,
            "patience_count": patience_count,
            "elapsed_seconds": time.time() - epoch_start,
            "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        }
        with epoch_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record, sort_keys=True) + "\n")
        base.atomic_torch_save({
            "schema_version": "syn2real-ordinal-c1-resume-checkpoint-2",
            "contract_digest": contract_digest,
            "job_id": job["job_id"],
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_metrics": best_metrics,
            "best_epoch": best_epoch,
            "patience_count": patience_count,
            "optimizer_steps_done": optimizer_steps_done,
            "skipped_optimizer_steps_done": skipped_steps_done,
            "target_exposures_done": target_exposures_done,
        }, last_path)
        print(json.dumps(epoch_record, sort_keys=True), flush=True)
        if (
            validation_loader is not None
            and bool(training["early_stopping"]["enabled"])
            and patience_count >= int(training["early_stopping"]["patience"])
        ):
            stopped_early = True
            break

    if job["job_type"] == "SOURCE_PRETRAIN":
        final_payload = {
            "schema_version": "syn2real-ordinal-c1-model-checkpoint-2",
            "contract_digest": contract_digest,
            "job_id": job["job_id"],
            "epoch": last_epoch_completed,
            "objective": "softmax_cross_entropy",
            "metrics": None,
            "model_state": model.state_dict(),
        }
    else:
        best_path = run_dir / "best.pt"
        if not best_path.exists():
            raise RuntimeError("C1 target fit completed without a best Validation checkpoint")
        final_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    final_path = run_dir / "final.pt"
    base.atomic_torch_save(final_payload, final_path)
    redundant_removed = []
    for redundant_name in training["storage_policy"]["remove_after_success"]:
        redundant = run_dir / redundant_name
        if redundant.exists():
            redundant.unlink()
            redundant_removed.append(redundant_name)
    completion = {
        "schema_version": "syn2real-ordinal-c1-completion-2",
        "contract_digest": contract_digest,
        "job_id": job["job_id"],
        "job_type": job["job_type"],
        "method_id": job["method_id"],
        "budget_percent": job["budget_percent"],
        "label_budget_seed": job["label_budget_seed"],
        "epochs_completed": last_epoch_completed + 1,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "optimizer_steps_done": optimizer_steps_done,
        "maximum_optimizer_steps": int(job["maximum_optimizer_steps"]),
        "skipped_optimizer_steps_done": skipped_steps_done,
        "target_exposures_done": target_exposures_done,
        "elapsed_seconds_this_process": time.time() - job_start,
        "autocast_dtype": training["amp"],
        "grad_scaler_enabled": scaler.is_enabled(),
        "final_checkpoint_sha256": sha256_file(final_path),
        "redundant_checkpoints_removed": redundant_removed,
        "locked_rows_loaded": 0,
        "tpa_used": False,
        "m2_started": False,
        **bound_hashes,
    }
    atomic_json(completion_path, completion)
    print(json.dumps(completion, indent=2, sort_keys=True), flush=True)
    return completion
