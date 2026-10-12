# -*- coding: utf-8 -*-
"""实践 2：两层二分类网络的梯度（NumPy 手算 vs PyTorch autograd）。

网络结构（括号里是形状，n 表示一次前向里的样本数）：

    X (n,2) -> Linear(W1 (2,H), b1 (H,)) -> Z1 (n,H)
            -> ReLU -> A1 (n,H)
            -> Linear(W2 (H,1), b2 (1,)) -> Z2 (n,1)
            -> Sigmoid -> P (n,1)
            -> BCE loss -> 标量

脚本做四件事：
  1. 从 make_moons 取样本（先 n=1，再 n=8），用固定种子初始化参数；
  2. NumPy 手写线性层 / ReLU / Sigmoid / BCE 的前向与反向（每一步都标出形状）；
  3. 用同样的输入和参数跑一遍 PyTorch，用 autograd 取梯度；
  4. 逐项比较形状、数值、最大绝对误差，并额外演示两个故意写错的版本。

运行： python nn_gradients.py
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
DTYPE = np.float64
N_HIDDEN = 4                      # 隐藏层神经元个数，刻意取小方便手算
SEED = 42
PARAM_NAMES = ("W1", "b1", "W2", "b2")


# --------------------------------------------------------------------------- #
# 数据与参数
# --------------------------------------------------------------------------- #
def make_data(n_samples=200, noise=0.10, random_state=0):
    """加载 Task 1 用过的 make_moons 数据（这里用 scikit-learn 的标准版本重新生成）。

    返回 X (n_samples, 2) 和 y (n_samples, 1)。
    标签整形成 (n, 1) 是有意的：它和 Z2 / P 的形状完全一致，做减法时不会触发广播。
    """
    from sklearn.datasets import make_moons

    features, labels = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return features.astype(DTYPE), labels.astype(DTYPE).reshape(-1, 1)


def init_params(n_features=2, n_hidden=N_HIDDEN, seed=SEED):
    """He 初始化：W ~ N(0, 2/fan_in)，b = 0。固定种子保证每次运行完全一致。"""
    rng = np.random.default_rng(seed)
    params = {
        "W1": rng.standard_normal((n_features, n_hidden)) * np.sqrt(2.0 / n_features),
        "b1": np.zeros(n_hidden),
        "W2": rng.standard_normal((n_hidden, 1)) * np.sqrt(2.0 / n_hidden),
        "b2": np.zeros(1),
    }
    return {name: value.astype(DTYPE) for name, value in params.items()}


# --------------------------------------------------------------------------- #
# NumPy 前向 / 反向
# --------------------------------------------------------------------------- #
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def forward(X, params):
    """前向传播；顺便把中间量全部返回 —— 反向传播马上要用它们。"""
    Z1 = X @ params["W1"] + params["b1"]        # (n, 2) @ (2, H) + (H,)  -> (n, H)
    A1 = np.maximum(Z1, 0.0)                    # ReLU，形状不变
    Z2 = A1 @ params["W2"] + params["b2"]       # (n, H) @ (H, 1) + (1,) -> (n, 1)
    P = sigmoid(Z2)                             # (n, 1)
    return P, {"Z1": Z1, "A1": A1, "Z2": Z2, "P": P}


def bce_loss(P, Y):
    """二元交叉熵，对 batch 取平均： -mean( y*log(p) + (1-y)*log(1-p) )。"""
    eps = 1e-12
    P_clipped = np.clip(P, eps, 1.0 - eps)
    return float(-np.mean(Y * np.log(P_clipped) + (1.0 - Y) * np.log(1.0 - P_clipped)))


def backward(X, Y, params, cache):
    """按计算图从 loss 往回推：每一步只做「上游梯度 * 局部导数」。"""
    Z1, A1, P = cache["Z1"], cache["A1"], cache["P"]
    n = X.shape[0]

    dZ2 = (P - Y) / n                    # (n, 1)  Sigmoid + BCE 合并后的结果
    dW2 = A1.T @ dZ2                     # (H, n) @ (n, 1) -> (H, 1)
    db2 = dZ2.sum(axis=0)                # (1,)    广播的逆运算 = 沿 batch 维求和
    dA1 = dZ2 @ params["W2"].T           # (n, 1) @ (1, H) -> (n, H)
    dZ1 = dA1 * (Z1 > 0.0)               # (n, H)  ReLU 的导数
    dW1 = X.T @ dZ1                      # (2, n) @ (n, H) -> (2, H)
    db1 = dZ1.sum(axis=0)                # (H,)
    dX = dZ1 @ params["W1"].T            # (n, H) @ (H, 2) -> (n, 2)（输入不更新，顺手算出）
    return {
        "W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
        "Z2": dZ2, "A1": dA1, "Z1": dZ1, "X": dX,
    }


# --------------------------------------------------------------------------- #
# PyTorch 对照
# --------------------------------------------------------------------------- #
def torch_reference(X, Y, params):
    """同样的前向，交给 PyTorch；反向完全由 autograd 完成。"""
    import torch
    import torch.nn.functional as F

    def leaf(array):
        return torch.tensor(array, dtype=torch.float64, requires_grad=True)

    X_t = torch.tensor(X, dtype=torch.float64)
    Y_t = torch.tensor(Y, dtype=torch.float64)
    W1, b1, W2, b2 = (leaf(params[name]) for name in PARAM_NAMES)

    Z1 = X_t @ W1 + b1
    A1 = torch.relu(Z1)
    Z2 = A1 @ W2 + b2
    P = torch.sigmoid(Z2)
    loss = F.binary_cross_entropy(P, Y_t, reduction="mean")

    loss.backward(retain_graph=True)          # 参数梯度走标准的 loss.backward()
    # 中间变量不是叶子节点，.grad 是 None，所以单独用 autograd.grad 取一次
    dZ1, dA1, dZ2 = torch.autograd.grad(loss, [Z1, A1, Z2])

    grads = {
        name: tensor.grad.detach().numpy().copy()
        for name, tensor in zip(PARAM_NAMES, (W1, b1, W2, b2))
    }
    grads.update({"Z1": dZ1.numpy(), "A1": dA1.numpy(), "Z2": dZ2.numpy()})
    return {"loss": loss.item(), "grads": grads}


# --------------------------------------------------------------------------- #
# 报告排版小工具
# --------------------------------------------------------------------------- #
def shaped(text, array, width=36):
    return f"{pad(text, width)} shape = {np.shape(array)}"


def fmt_array(array, precision=8, continuation=" " * 8):
    body = np.array2string(
        np.asarray(array), precision=precision, floatmode="fixed", separator=", "
    )
    return body.replace("\n", "\n" + continuation)


def fmt_numbers(array, count=None, precision=6):
    """把数组打印成固定小数位的列表；把 -0.0 和极小值统一显示成 0。"""
    values = np.asarray(array).ravel()
    if count is not None:
        values = values[:count]
    rendered = []
    for value in values:
        value = 0.0 if abs(value) < 10.0 ** (-precision) else float(value)
        rendered.append(f"{value:.{precision}f}")
    return "[" + ", ".join(rendered) + "]"


def comparison_block(name, np_grad, torch_grad, indent="    "):
    np_grad = np.asarray(np_grad)
    torch_grad = np.asarray(torch_grad)
    error = np.max(np.abs(np_grad - torch_grad))
    continuation = " " * (len(indent) + 13)   # 与等号后的数组起始列对齐
    return [
        f"{indent}dL/d{name}{pad('', 6)}形状 {np_grad.shape}",
        f"{indent}  NumPy    = {fmt_array(np_grad, continuation=continuation)}",
        f"{indent}  PyTorch  = {fmt_array(torch_grad, continuation=continuation)}",
        f"{indent}  最大绝对误差 = {error:.3e}",
    ]


def run_case(X, Y, params, title, add):
    """一个样本量下的前向 / 反向 / 对照，全部写进报告。返回最大绝对误差。"""
    n = X.shape[0]
    P, cache = forward(X, params)
    loss = bce_loss(P, Y)
    grads = backward(X, Y, params, cache)
    torch_out = torch_reference(X, Y, params)
    torch_grads = torch_out["grads"]

    add("")
    add("-" * 78)
    add(f"{title}   样本数 n = {n}")
    add("-" * 78)
    add("前向传播（形状写在右边）")
    add("    " + shaped("Z1 = X @ W1 + b1", cache["Z1"]))
    add("    " + shaped("A1 = relu(Z1)", cache["A1"]))
    add("    " + shaped("Z2 = A1 @ W2 + b2", cache["Z2"]))
    add("    " + shaped("P  = sigmoid(Z2)", cache["P"]))
    add("    " + f"{pad('loss = BCE(P, Y)', 36)} shape = ()   value = {loss:.10f}")
    add(f"    标签 Y = {[int(value) for value in Y.ravel()]}")
    add(f"    概率 P = {fmt_numbers(P, precision=6)}")
    add("")
    add("反向传播（形状写在右边，顺序就是从 loss 往回）")
    add("    " + shaped("dL/dZ2 = (P - Y) / n", grads["Z2"]))
    add("    " + shaped("dL/dW2 = A1.T @ dL/dZ2", grads["W2"]))
    add("    " + shaped("dL/db2 = dL/dZ2.sum(axis=0)", grads["b2"]))
    add("    " + shaped("dL/dA1 = dL/dZ2 @ W2.T", grads["A1"]))
    add("    " + shaped("dL/dZ1 = dL/dA1 * (Z1 > 0)", grads["Z1"]))
    add("    " + shaped("dL/dW1 = X.T @ dL/dZ1", grads["W1"]))
    add("    " + shaped("dL/db1 = dL/dZ1.sum(axis=0)", grads["b1"]))
    add("    " + shaped("dL/dX  = dL/dZ1 @ W1.T", grads["X"]))
    inactive = np.where((cache["Z1"] <= 0.0).all(axis=0))[0]
    if inactive.size:
        units = [int(index) + 1 for index in inactive]
        add(f"    提示：隐藏层第 {units} 个神经元在这批样本上全部落在 ReLU 的负半轴，"
            f"它们的 dL/dZ1、dL/db1、dL/dW1 对应位置恰好是 0。")
    dead_rows = np.where((cache["Z1"] <= 0.0).all(axis=1))[0]
    if dead_rows.size:
        samples = [int(index) for index in dead_rows]
        add(f"    另外，样本 {samples} 的四个隐藏单元全部失活（A1 整行是 0），于是 Z2 = 0、"
            f"P = 0.5（预测概率正好一半），这一行对 dL/dW1、dL/db1 的贡献同样是 0。")
    add("")
    add("NumPy 手算梯度 vs PyTorch autograd")
    add(f"    loss: NumPy = {loss:.10f}   PyTorch = {torch_out['loss']:.10f}   "
        f"绝对误差 = {abs(loss - torch_out['loss']):.3e}")
    add("")
    worst = 0.0
    for name in PARAM_NAMES:
        for line in comparison_block(name, grads[name], torch_grads[name]):
            add(line)
        add("")
        worst = max(worst, np.max(np.abs(grads[name] - torch_grads[name])))
    add("    中间量的梯度对照")
    for name in ("Z2", "A1", "Z1"):
        error = np.max(np.abs(grads[name] - torch_grads[name]))
        add(f"    dL/d{pad(name, 4)}形状 {str(grads[name].shape):<8}"
            f"NumPy 前两个元素 {fmt_numbers(grads[name], count=2, precision=8)}   "
            f"最大绝对误差 {error:.3e}")
        worst = max(worst, error)
    add("")
    verdict = "完全一致（误差 ~ 1e-16，只是浮点舍入）" if worst < 1e-12 else "存在差异，需要检查"
    add(f"    全部参数与中间量的最大绝对误差 = {worst:.3e}  ->  {verdict}")
    return worst


def build_figure(path):
    """把前向和反向的计算图画出来，节点里标注形状。"""
    # 行坐标：主链路自上而下；每一层的参数和它并排放在同一行
    row = {"X": 9.0, "Z1": 7.2, "A1": 5.4, "Z2": 3.6, "P": 1.8, "L": 0.0}
    param_x = 2.7
    boxes = {
        "X": (0.0, row["X"], "X\n(n, 2)", "#e0f2fe"),
        "Z1": (0.0, row["Z1"], "Z1 = X@W1+b1\n(n, 4)", "#eef2ff"),
        "W1": (param_x, row["Z1"], "W1\n(2, 4)", "#dcfce7"),
        "b1": (-param_x, row["Z1"], "b1\n(4,)", "#dcfce7"),
        "A1": (0.0, row["A1"], "A1 = relu(Z1)\n(n, 4)", "#eef2ff"),
        "Z2": (0.0, row["Z2"], "Z2 = A1@W2+b2\n(n, 1)", "#eef2ff"),
        "W2": (param_x, row["Z2"], "W2\n(4, 1)", "#dcfce7"),
        "b2": (-param_x, row["Z2"], "b2\n(1,)", "#dcfce7"),
        "P": (0.0, row["P"], "P = sigmoid(Z2)\n(n, 1)", "#eef2ff"),
        "L": (0.0, row["L"], "L = BCE(P, Y)\nscalar", "#fee2e2"),
        "Y": (param_x, row["L"], "Y\n(n, 1)", "#fef9c3"),
    }
    nodes_forward = [
        (key, box[0], box[1], box[2], 2.6, 1.0, box[3]) for key, box in boxes.items()
    ]
    edges_forward = [
        ("X", "Z1", "", "forward"),
        ("W1", "Z1", "", "forward"),
        ("b1", "Z1", "", "forward"),
        ("Z1", "A1", "", "forward"),
        ("A1", "Z2", "", "forward"),
        ("W2", "Z2", "", "forward"),
        ("b2", "Z2", "", "forward"),
        ("Z2", "P", "", "forward"),
        ("P", "L", "", "forward"),
        ("Y", "L", "", "forward"),
    ]
    labels_backward = {
        "X": "X\ndL/dX (n, 2)",
        "Z1": "dL/dZ1\n(n, 4)",
        "W1": "dL/dW1\nX.T @ dZ1\n(2, 4)",
        "b1": "dL/db1\nsum dZ1\n(4,)",
        "A1": "dL/dA1\n(n, 4)",
        "Z2": "dL/dZ2\n(n, 1)",
        "W2": "dL/dW2\nA1.T @ dZ2\n(4, 1)",
        "b2": "dL/db2\nsum dZ2\n(1,)",
        "P": "dL/dP\n(n, 1)",
        "L": "dL/dL = 1",
        "Y": "Y\n(n, 1)",
    }
    parameter_color = "#fee2e2"
    nodes_backward = [
        (
            key,
            box[0],
            box[1],
            labels_backward[key],
            2.6,
            1.4 if key in PARAM_NAMES else 1.0,
            parameter_color if key in PARAM_NAMES else box[3],
        )
        for key, box in boxes.items()
    ]
    edges_backward = [
        # 主链路沿竖直方向回流；局部梯度的标签放在箭头中点的行间空隙里（这一行人
        # 最稀疏，不会压到任何方框）。参数支路上不挂标签，公式直接写在参数节点内。
        ("L", "P", "1", "backward", (0.0, 0.0)),
        ("P", "Z2", "(P - Y) / n", "backward", (0.0, 0.0)),
        ("Z2", "A1", "@ W2.T", "backward", (0.0, 0.0)),
        ("A1", "Z1", "* 1[Z1 > 0]", "backward", (0.0, 0.0)),
        ("Z1", "X", "@ W1.T", "backward", (0.0, 0.0)),
        ("Z1", "W1", "", "backward"),
        ("Z1", "b1", "", "backward"),
        ("Z2", "W2", "", "backward"),
        ("Z2", "b2", "", "backward"),
    ]
    panels = [
        dict(
            title="Forward: X -> Linear -> ReLU -> Linear -> Sigmoid -> BCE",
            nodes=nodes_forward,
            edges=edges_forward,
            xlim=(-4.6, 4.6),
            ylim=(-0.9, 9.9),
        ),
        dict(
            title="Backward: chain rule, shape by shape",
            nodes=nodes_backward,
            edges=edges_backward,
            xlim=(-4.6, 4.6),
            ylim=(-0.9, 9.9),
        ),
    ]
    save_figure(path, panels, ncols=2, panel_size=(3.9, 4.8))


# --------------------------------------------------------------------------- #
# 主报告
# --------------------------------------------------------------------------- #
def main():
    features, labels = make_data()
    params = init_params()

    lines = []
    add = lines.append
    add("=" * 78)
    add("实践 2：两层二分类网络的梯度（NumPy 手算 vs PyTorch autograd）")
    add("X -> Linear(W1,b1) -> ReLU -> Linear(W2,b2) -> Sigmoid -> BCE loss")
    add("=" * 78)
    add("数据：sklearn.datasets.make_moons(n_samples=200, noise=0.10, random_state=0)")
    add("      取前 n 个样本（先 n = 1 的单样本，再 n = 8 的小 batch）")
    add(f"网络：输入 2 维，隐藏层 H = {N_HIDDEN}，输出 1 维（二分类）")
    add(f"初始化：np.random.default_rng({SEED})，W ~ N(0, 2/fan_in)（He 初始化），b = 0")
    add("精度：两边都用 float64，这样不一致只可能来自代码逻辑，而不是数值噪声")
    add("")
    add("参数一览")
    for name in PARAM_NAMES:
        add(f"    {pad(name, 4)}形状 {params[name].shape}")
        add(f"        {fmt_array(params[name], continuation=' ' * 8)}")

    worst_single = run_case(features[0:1], labels[0:1], params, "场景 A：单个样本", add)
    x_batch, y_batch = features[0:8], labels[0:8]
    worst_batch = run_case(x_batch, y_batch, params, "场景 B：小 batch（n = 8）", add)

    add("")
    add("=" * 78)
    add("单样本 (n = 1) 与 小 batch (n = 8) 的形状对比")
    add("=" * 78)
    add("    " + pad("变量", 12) + f"{'n = 1':>10}{'n = 8':>10}   说明")
    shape_rows = [
        ("X", "(1, 2)", "(8, 2)", "只有第 0 维（batch 维）变了"),
        ("W1", "(2, 4)", "(2, 4)", "权重与 batch 无关"),
        ("b1", "(4,)", "(4,)", "偏置与 batch 无关"),
        ("Z1", "(1, 4)", "(8, 4)", "逐元素运算，形状跟着 batch 走"),
        ("A1", "(1, 4)", "(8, 4)", "同 Z1"),
        ("W2", "(4, 1)", "(4, 1)", "与 batch 无关"),
        ("b2", "(1,)", "(1,)", "与 batch 无关"),
        ("Z2", "(1, 1)", "(8, 1)", "同 Z1"),
        ("P", "(1, 1)", "(8, 1)", "同 Z1"),
        ("loss", "()", "()", "对 batch 取平均，永远是标量"),
        ("dL/dW1", "(2, 4)", "(2, 4)", "沿 batch 维求和后，batch 维消失"),
        ("dL/db1", "(4,)", "(4,)", "沿 batch 维求和后，batch 维消失"),
        ("dL/dZ2", "(1, 1)", "(8, 1)", "每个样本一个误差信号 (P - Y)/n"),
    ]
    for name, one, eight, note in shape_rows:
        add("    " + pad(name, 12) + f"{one:>10}{eight:>10}   {note}")
    add("")
    add("公式层面的变化只有两处：")
    add("    1) dL/dZ2 = (P - Y) / n：n = 1 时就是 P - Y，n = 8 时每个样本只摊到 1/8；")
    add("    2) dL/db1、dL/db2 变成沿 batch 维求和 sum(axis=0)。前向里 b 被广播到了每个")
    add("       样本，反向时必须把广播出去的梯度加回来；n = 1 时求和退化成它自己，")
    add("       所以单样本的代码看不出这个动作。")
    add("    矩阵乘法的写法（X.T @ dZ1、A1.T @ dZ2）在两种 n 下完全一样，形状自动对上。")

    add("")
    add("=" * 78)
    add("故意写错的两版：练习判断误差来自哪里")
    add("=" * 78)
    probability, cache = forward(x_batch, params)
    good = backward(x_batch, y_batch, params, cache)
    torch_grads = torch_reference(x_batch, y_batch, params)["grads"]

    add("")
    add("错误 1：把 Y 写成一维 (n,)，而不是 (n, 1)")
    wrong_dz2 = probability - y_batch.ravel()
    add(f"    P 的形状 {probability.shape}，Y.ravel() 的形状 {y_batch.ravel().shape}")
    add(f"    P - Y.ravel() 的形状 = {wrong_dz2.shape}   <- 广播把 (8,1) 和 (8,) 撑成了 (8,8)")
    add(f"    dL/dW2 = A1.T @ dZ2 的形状 = {(cache['A1'].T @ wrong_dz2).shape}"
        f"   <- 正确时是 {good['W2'].shape}")
    add(f"    对角线元素 [0,0] = {wrong_dz2[0, 0]:.8f}，它只是忘了除 n 的版本（正确的 "
        f"dL/dZ2[0,0] = {good['Z2'][0, 0]:.8f}，相差 {x_batch.shape[0]} 倍）；"
        f"非对角线的 7 列全是跨样本硬凑出来的值。")
    add("    定位方式：先打印 shape。形状不对的时候，数值比较没有任何意义。")

    add("")
    add("错误 2：忘记除以 batch 大小 n")
    unscaled = cache["A1"].T @ (probability - y_batch)
    error_unscaled = np.max(np.abs(unscaled - torch_grads["W2"]))
    error_scaled = np.max(np.abs(good["W2"] - torch_grads["W2"]))
    add(f"    dL/dW2（漏掉 /n）与 PyTorch 的最大绝对误差 = {error_unscaled:.3e}")
    add(f"    dL/dW2（正确）  与 PyTorch 的最大绝对误差 = {error_scaled:.3e}")
    add(f"    漏掉 /n 的梯度恰好是正确值的 "
        f"{np.max(np.abs(unscaled)) / np.max(np.abs(good['W2'])):.1f} 倍（n = {x_batch.shape[0]}）")
    add("    指纹：形状完全正确、所有参数梯度同比例放大 —— 这正是求平均方式不一致的表现。")

    add("")
    add("错误 3：矩阵转置方向写反（例如把 dL/dW1 = X.T @ dL/dZ1 写成 dL/dZ1.T @ X）")
    add("    这种错误通常连形状都对不上；万一形状碰巧对上，就只能靠与 PyTorch 的")
    add("    逐项数值比较来定位。所以先查形状，再查缩放，最后逐元素对比。")

    add("")
    add("=" * 78)
    add("结论")
    add("=" * 78)
    add(f"    单样本  n = 1：NumPy 手算与 PyTorch 的最大绝对误差 = {worst_single:.3e}")
    add(f"    小 batch n = 8：NumPy 手算与 PyTorch 的最大绝对误差 = {worst_batch:.3e}")
    add(f"    对照：漏掉 /n 的错误版本误差量级 = {error_unscaled:.3e}，比浮点噪声大十几个数量级")
    add("    两边在 float64 下逐项一致，说明手写的链式法则、形状变换、求平均方式都正确。")

    build_figure(HERE / "computation_graph.png")
    add("")
    add("计算图已保存到 computation_graph.png（左：前向；右：反向，节点都标了形状）")

    text = "\n".join(lines)
    print(text)
    (HERE / "output.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover
        pass
    main()
