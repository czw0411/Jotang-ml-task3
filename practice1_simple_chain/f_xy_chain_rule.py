# -*- coding: utf-8 -*-
"""实践 1：简单计算链的梯度。

    f(x, y) = (x * y + x ** 2) ** 2

思路：
  1. forward()  把前向计算拆成 4 个基本步骤（u, v, s, f），并把每一步的中间结果
                存下来 —— 这就是"计算图"上的节点值；
  2. backward() 严格按照计算图的反方向（f -> s -> u/v -> x/y），每一步只乘"局部
                导数"，遇到分叉（x 同时进入 u 和 v）就把两条路径的贡献相加；
  3. 用三份互相独立的来源做对照：闭式解、中心差分、PyTorch autograd。

运行： python f_xy_chain_rule.py
产物： 控制台输出 + output.txt + computation_graph.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from graph_draw import save_figure  # noqa: E402
from text_util import pad  # noqa: E402

HERE = Path(__file__).resolve().parent
X_VAL, Y_VAL = 2.0, 3.0


def forward(x, y):
    """前向：把 f 拆成最小的步骤，每一步的结果都保留下来。

        u = x * y        (乘法节点)
        v = x * x        (平方节点，写成 x*x 是为了看清它的局部导数是 2x)
        s = u + v        (加法节点)
        f = s * s        (平方节点)
    """
    u = x * y
    v = x * x
    s = u + v
    f = s * s
    return {"u": u, "v": v, "s": s, "f": f}


def backward(x, y, values):
    """反向：从 f 出发逆着计算图走，每一步写清"局部导数 x 上游传回来的梯度"。

    ∂f/∂s = 2s        <- f = s*s
    ∂f/∂u = ∂f/∂v = ∂f/∂s * 1   <- s = u + v，加法节点的局部导数都是 1
    ∂f/∂x = ∂f/∂u * ∂u/∂x + ∂f/∂v * ∂v/∂x   <- x 有两条路径，贡献相加
    ∂f/∂y = ∂f/∂u * ∂u/∂y                   <- v 与 y 无关，只有一条路径
    """
    s = values["s"]
    d_f = 1.0                              # dL/df = 1（f 就是被求导的输出）
    d_s = d_f * (2.0 * s)                  # f = s*s    -> df/ds = 2s
    d_u = d_s * 1.0                        # s = u + v  -> ds/du = 1
    d_v = d_s * 1.0                        # s = u + v  -> ds/dv = 1
    d_x = d_u * y + d_v * (2.0 * x)        # u = x*y -> y ;  v = x*x -> 2x
    d_y = d_u * x                          # u = x*y -> x ;  v 与 y 无关
    return {"f": d_f, "s": d_s, "u": d_u, "v": d_v, "x": d_x, "y": d_y}


def analytic(x, y):
    """把 f 展开后直接写出的闭式解，用来做第一重对照。

    f = (xy + x^2)^2
    ∂f/∂x = 2(xy + x^2)(y + 2x)
    ∂f/∂y = 2x(xy + x^2)
    """
    s = x * y + x * x
    return {"f": s**2, "x": 2.0 * s * (y + 2.0 * x), "y": 2.0 * s * x}


def central_difference(x, y, h=1e-6):
    """第二重对照：不依赖任何求导公式的数值差分。"""
    d_x = (forward(x + h, y)["f"] - forward(x - h, y)["f"]) / (2.0 * h)
    d_y = (forward(x, y + h)["f"] - forward(x, y - h)["f"]) / (2.0 * h)
    return {"x": d_x, "y": d_y}


def pytorch_reference(x, y):
    """第三重对照：PyTorch autograd。"""
    import torch

    torch_x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    torch_y = torch.tensor(y, dtype=torch.float64, requires_grad=True)
    torch_f = (torch_x * torch_y + torch_x**2) ** 2
    torch_f.backward()
    return {
        "f": torch_f.item(),
        "x": torch_x.grad.item(),
        "y": torch_y.grad.item(),
    }


def build_figure(values, grads, path):
    # 节点摆成一个菱形：x/y 在左，u/v 在中，s、f 依次向右。
    # 列与列之间留出足够空隙，反向传播的标签就放在这些空隙里，不会压住方框。
    box = 2.0
    nodes_forward = [
        ("x", -3.2, 2.2, f"x = {X_VAL}", box, 0.9, "#e0f2fe"),
        ("y", -3.2, 0.2, f"y = {Y_VAL}", box, 0.9, "#e0f2fe"),
        ("u", 0.0, 0.2, f"u = x*y = {values['u']:.0f}", box, 0.9, "#eef2ff"),
        ("v", 0.0, 2.2, f"v = x*x = {values['v']:.0f}", box, 0.9, "#eef2ff"),
        ("s", 3.2, 1.2, f"s = u+v = {values['s']:.0f}", box, 0.9, "#eef2ff"),
        ("f", 6.4, 1.2, f"f = s*s = {values['f']:.0f}", box, 0.9, "#fee2e2"),
    ]
    edges_forward = [
        ("x", "u", "", "forward"),
        ("y", "u", "", "forward"),
        ("x", "v", "", "forward"),
        ("u", "s", "", "forward"),
        ("v", "s", "", "forward"),
        ("s", "f", "", "forward"),
    ]
    nodes_backward = [
        ("x", -3.2, 2.2, f"x\ndf/dx = {grads['x']}", box, 1.0, "#e0f2fe"),
        ("y", -3.2, 0.2, f"y\ndf/dy = {grads['y']}", box, 1.0, "#e0f2fe"),
        ("u", 0.0, 0.2, f"u\ndf/du = {grads['u']}", box, 1.0, "#eef2ff"),
        ("v", 0.0, 2.2, f"v\ndf/dv = {grads['v']}", box, 1.0, "#eef2ff"),
        ("s", 3.2, 1.2, f"s\ndf/ds = {grads['s']}", box, 1.0, "#eef2ff"),
        ("f", 6.4, 1.2, "f\ndf/df = 1.0", box, 1.0, "#fee2e2"),
    ]
    edges_backward = [
        # 标签都落在"列与列之间的空隙"里：横向的边把标签放在上/下方，斜边直接放中点
        ("f", "s", "df/ds = 20", "backward", (0.0, 0.75)),
        ("s", "u", "ds/du = 1", "backward", (-0.3, 0.0)),
        ("s", "v", "ds/dv = 1", "backward", (-0.3, 0.0)),
        ("u", "x", "du/dx = 3", "backward", (0.0, 0.0)),
        ("u", "y", "du/dy = 2", "backward", (0.0, 0.0)),
        ("v", "x", "dv/dx = 4", "backward", (0.0, 0.0)),
    ]
    xlim, ylim = (-4.9, 8.1), (-1.0, 3.2)
    panels = [
        dict(
            title="Forward pass: values flow right",
            nodes=nodes_forward,
            edges=edges_forward,
            xlim=xlim,
            ylim=ylim,
        ),
        dict(
            title="Backward pass: gradients flow left",
            nodes=nodes_backward,
            edges=edges_backward,
            xlim=xlim,
            ylim=ylim,
        ),
    ]
    save_figure(path, panels, ncols=1, panel_size=(6.4, 2.6))


def main():
    values = forward(X_VAL, Y_VAL)
    grads = backward(X_VAL, Y_VAL, values)
    closed = analytic(X_VAL, Y_VAL)
    numeric = central_difference(X_VAL, Y_VAL)
    torch_ref = pytorch_reference(X_VAL, Y_VAL)

    err_x = abs(grads["x"] - torch_ref["x"])
    err_y = abs(grads["y"] - torch_ref["y"])

    lines = []
    add = lines.append
    add("=" * 78)
    add("实践 1：简单计算链的梯度   f(x, y) = (x*y + x^2)^2")
    add("=" * 78)
    add(f"选取的输入： x = {X_VAL}, y = {Y_VAL}")
    add("")
    add("一、前向传播：把计算拆成 4 个基本步骤（节点值会保留下来）")
    add(f"    [1] u = x * y        = {X_VAL} * {Y_VAL} = {values['u']}")
    add(f"    [2] v = x * x        = {X_VAL} * {X_VAL} = {values['v']}")
    add(f"    [3] s = u + v        = {values['u']} + {values['v']} = {values['s']}")
    add(f"    [4] f = s * s        = {values['s']} * {values['s']} = {values['f']}")
    add("")
    add("二、反向传播：从 f 出发逆着计算图走（链式法则）")
    add(f"    [1] df/df = 1")
    add(f"    [2] df/ds = df/df * 2s            = 1 * 2*{values['s']} = {grads['s']}")
    add(f"    [3] df/du = df/ds * 1             = {grads['u']}")
    add(f"    [4] df/dv = df/ds * 1             = {grads['v']}")
    add(f"    [5] df/dx = df/du * y + df/dv * 2x")
    add(f"              = {grads['u']} * {Y_VAL} + {grads['v']} * {2 * X_VAL} = {grads['x']}")
    add(f"        df/dy = df/du * x            = {grads['u']} * {X_VAL} = {grads['y']}")
    add("        （x 同时进入 u 和 v 两条路径，所以梯度要相加；y 只有一条路径）")
    add("")
    add("三、四份结果对照")
    add("    " + pad("方法", 30) + f"{'f':>14}{'df/dx':>14}{'df/dy':>14}")
    add("    " + "-" * 70)
    rows = [
        ("NumPy 手算（按计算图反向）", values["f"], grads["x"], grads["y"]),
        ("闭式解 2s(y+2x), 2sx", closed["f"], closed["x"], closed["y"]),
        ("中心差分（h=1e-6）", None, numeric["x"], numeric["y"]),
        ("PyTorch autograd", torch_ref["f"], torch_ref["x"], torch_ref["y"]),
    ]
    for name, f_val, dx_val, dy_val in rows:
        f_col = f"{f_val:>14.6f}" if f_val is not None else f"{'--':>14}"
        add("    " + pad(name, 30) + f"{f_col}{dx_val:>14.6f}{dy_val:>14.6f}")
    add(f"    {'-' * 66}")
    add(f"    NumPy 手算 vs PyTorch 的最大绝对误差： "
        f"df/dx = {err_x:.3e}, df/dy = {err_y:.3e}")
    add(f"    中心差分 vs PyTorch 的最大绝对误差： "
        f"df/dx = {abs(numeric['x'] - torch_ref['x']):.3e}, "
        f"df/dy = {abs(numeric['y'] - torch_ref['y']):.3e}")
    verdict = "一致" if max(err_x, err_y) < 1e-9 else "不一致，需要检查"
    add(f"    结论：手算梯度与 PyTorch autograd {verdict}")
    add("")

    build_figure(values, grads, HERE / "computation_graph.png")
    add("计算图已保存到 computation_graph.png（上：前向数值；下：反向梯度）")

    text = "\n".join(lines)
    print(text)
    (HERE / "output.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover
        pass
    main()
