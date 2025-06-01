import optuna
import os
from ultralytics import YOLO
from yaml_generator import generate_custom_yolov8_yaml
from search_constraints import constraints

epochs=3
imgsz=96
trials=5
nc=4
k=3

def objective(trial):
    # 1. Search space
    stages = trial.suggest_int("stages", 3, 5)
    
    init_channels = trial.suggest_categorical("init_channel", [32, 48, 64])
    scale = trial.suggest_categorical("scale", [1.5, 2.0])
    channel_sizes = [int(init_channels * (scale ** i)) for i in range(stages + 1)]
    
    c2f_repeats_choices = [
        [1, 2, 2, 2],
        [2, 4, 4, 2],
        [3, 6, 6, 3]
    ]
    c2f_repeats = trial.suggest_categorical("c2f_repeats", c2f_repeats_choices)[:stages]
    include_sppf = trial.suggest_categorical("include_sppf", [True, False])

    # 2. Generate YAML
    yaml_str = generate_custom_yolov8_yaml(nc,stages, channel_sizes, c2f_repeats, include_sppf)
    yaml_path = f"generated_yamls/model_trial_{trial.number}.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_str)

    # 3. Train
    model = YOLO(yaml_path)
    model.train(data="dataset.yaml", epochs=epochs, imgsz=imgsz, project="bo_runs", name=f"trial_{trial.number}", exist_ok=True)

    # 4. Evaluate
    metrics = model.val()
    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    score = (precision + recall + 2 * map50) / 4

    # 5. Check constraints
    info = model.info()
    model_path = f"bo_runs/trial_{trial.number}/weights/best.pt"
    size_kb = os.path.getsize(model_path) / 1000
    params_m = info[1] / 1e6
    gflops = info[3]

    if size_kb <= constraints["max_model_size_kb"] and params_m <= constraints["max_params_m"] and gflops <= constraints["max_gflops"]:
        score = score
    else:
        score = 0.0 # penalize invalid ones
        
    trial.set_user_attr("score", score)
    trial.set_user_attr("precision", precision)
    trial.set_user_attr("recall", recall)
    trial.set_user_attr("map50", map50)
    trial.set_user_attr("params_m", params_m)
    trial.set_user_attr("gflops", gflops)
    trial.set_user_attr("size_kb", size_kb)
    trial.set_user_attr("yaml_path", yaml_path)
        
    return score, size_kb, params_m

os.makedirs('generated_yamls', exist_ok=True)
# 🔁 Run the study
study = optuna.create_study(directions=["maximize", "minimize", "minimize"])
study.optimize(objective, n_trials=trials)
sorted_trials = sorted(
    [t for t in study.trials if t.values is not None],
    key=lambda t: t.values[0],  # score
    reverse=True
)
# 🏆 Show top 5 results
print(f"\n🏆 Top {k} Architectures:")

for rank, trial in enumerate(sorted_trials[:k], 1):
    print(f"\n🔹 Rank {rank}: Trial {trial.number}")
    print(f"Score:     {trial.user_attrs['score']:.4f}")
    print(f"Precision: {trial.user_attrs['precision']:.4f}")
    print(f"Recall:    {trial.user_attrs['recall']:.4f}")
    print(f"mAP50:     {trial.user_attrs['map50']:.4f}")
    print(f"Params:    {trial.user_attrs['params_m']:.2f} M")
    print(f"GFLOPs:    {trial.user_attrs['gflops']:.2f}")
    print(f"Size:      {trial.user_attrs['size_kb']:.1f} KB")
    print(f"YAML:      {trial.user_attrs['yaml_path']}")
with open("sorted_trials_user_attrs.txt", "w") as f:
    for t in sorted_trials:
        f.write(
            f"Trial#{t.number}:\n"
            f"  score     : {t.user_attrs.get('score')}\n"
            f"  precision : {t.user_attrs.get('precision')}\n"
            f"  recall    : {t.user_attrs.get('recall')}\n"
            f"  map50     : {t.user_attrs.get('map50')}\n"
            f"  params_m  : {t.user_attrs.get('params_m')}\n"
            f"  gflops    : {t.user_attrs.get('gflops')}\n"
            f"  size_kb   : {t.user_attrs.get('size_kb')}\n"
            f"  yaml_path : {t.user_attrs.get('yaml_path')}\n"
            f"{'-'*40}\n"
        )