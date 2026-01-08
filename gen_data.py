import os
import cv2
import numpy as np
import json

ROOT = "shape_step_dataset"
os.makedirs(ROOT, exist_ok=True)

num_episodes = 5
image_size = 64
num_steps = 40   # total steps per episode


def allocate_steps_for_rectangle(w, h, total_steps):
    """Distribute steps proportional to rectangle side lengths."""
    perim = 2 * (w + h)

    top = round((w / perim) * total_steps)
    right = round((h / perim) * total_steps)
    bottom = round((w / perim) * total_steps)
    left = total_steps - (top + right + bottom)  # ensure total = steps

    return top, right, bottom, left


def draw_shape(shape_name, width, height, proportional_steps=False):
    shape_dir = os.path.join(ROOT, shape_name)
    os.makedirs(shape_dir, exist_ok=True)

    for ep in range(1, num_episodes + 1):
        ep_dir = os.path.join(shape_dir, f"episode_{ep}")
        os.makedirs(ep_dir, exist_ok=True)

        canvas = np.zeros((image_size, image_size), dtype=np.uint8)
        actions = []

        # Centered placement
        x0 = image_size // 2 - width // 2
        y0 = image_size // 2 - height // 2

        right_x = x0 + width
        bottom_y = y0 + height

        cx, cy = x0, y0   # start point

        # ---- step allocation ----
        if proportional_steps:
            top_s, right_s, bottom_s, left_s = allocate_steps_for_rectangle(
                width, height, num_steps
            )
        else:
            s = num_steps // 4
            top_s = right_s = bottom_s = left_s = s

        step = 1

        # ---------- Top (move right) ----------
        for i in range(top_s):
            nx = x0 + (width * (i + 1)) // top_s
            ny = y0
            cv2.line(canvas, (cx, cy), (nx, ny), 255, 1)

            actions.append({"step": step, "line": [cx, cy, nx, ny]})
            cv2.imwrite(os.path.join(ep_dir, f"{step:03}.png"), canvas)

            cx, cy = nx, ny
            step += 1

        # ---------- Right (move down) ----------
        for i in range(right_s):
            nx = right_x
            ny = y0 + (height * (i + 1)) // right_s
            cv2.line(canvas, (cx, cy), (nx, ny), 255, 1)

            actions.append({"step": step, "line": [cx, cy, nx, ny]})
            cv2.imwrite(os.path.join(ep_dir, f"{step:03}.png"), canvas)

            cx, cy = nx, ny
            step += 1

        # ---------- Bottom (move left) ----------
        for i in range(bottom_s):
            nx = right_x - (width * (i + 1)) // bottom_s
            ny = bottom_y
            cv2.line(canvas, (cx, cy), (nx, ny), 255, 1)

            actions.append({"step": step, "line": [cx, cy, nx, ny]})
            cv2.imwrite(os.path.join(ep_dir, f"{step:03}.png"), canvas)

            cx, cy = nx, ny
            step += 1

        # ---------- Left (move up) ----------
        for i in range(left_s):
            nx = x0
            ny = bottom_y - (height * (i + 1)) // left_s
            cv2.line(canvas, (cx, cy), (nx, ny), 255, 1)

            actions.append({"step": step, "line": [cx, cy, nx, ny]})
            cv2.imwrite(os.path.join(ep_dir, f"{step:03}.png"), canvas)

            cx, cy = nx, ny
            step += 1

        # ---- Save actions JSON ----
        with open(os.path.join(ep_dir, "actions.json"), "w") as f:
            json.dump(actions, f, indent=2)


# ---------------- RUN ----------------
print("Generating Square Dataset...")
draw_shape("square", width=40, height=40, proportional_steps=False)

print("Generating Rectangle Dataset...")
draw_shape("rectangle", width=48, height=28, proportional_steps=True)

print("DONE ✔️")
