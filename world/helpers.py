from pathlib import Path
from warnings import warn

def action_to_direction(action):
    directions = {
        0: (0, 1),   # Down
        1: (0, -1),  # Up
        2: (-1, 0),  # Left
        3: (1, 0),   # Right
    }
    return directions[action]

def save_results(file_name, world_stats, path_image, show_images):
    out_dir = Path("results/")
    if not out_dir.exists():
        warn("Evaluation output directory does not exist. Creating the "
             "directory.")
        out_dir.mkdir(parents=True, exist_ok=True)

    # Print evaluation results
    print("Evaluation complete. Results:")
    # Text file
    out_fp = out_dir / f"{file_name}.txt"
    with open(out_fp, "w") as f:
        for key, value in world_stats.items():
            f.write(f"{key}: {value}\n")
            print(f"{key}: {value}")
    
    # Image file
    out_fp = out_dir / f"{file_name}.png"
    path_image.save(out_fp)
    if show_images:
        path_image.show(f"Path Frequency")