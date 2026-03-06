#!/usr/bin/env python3
"""
Advanced Calculator (safe REPL) - Upgraded

New stuff:
- Postfix factorial: 5!, (2+3)!, x!
- Comparisons + boolean logic + ternary:
    x>3, (x>0 and x<10), not (x==2)
    x>0 ? x : -x
- Stats: mean/median/stdev/var
- Number theory: isprime, factor, primes, fib
- Summation/product: summation("k^2","k",1,10), product("k","k",1,5)
- Commands: :del var, :const, :func, :state

Safety:
- AST whitelist
- No builtins
- No attributes (no obj.__class__, no module access)
"""

from __future__ import annotations

import ast
import json
import math
import random
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Tuple, Optional


# ----------------------------
# Pretty output (optional ANSI)
# ----------------------------
class Style:
    def __init__(self) -> None:
        self.enabled = True

    def c(self, s: str, code: str) -> str:
        if not self.enabled:
            return s
        return f"\033[{code}m{s}\033[0m"

    def bold(self, s: str) -> str:
        return self.c(s, "1")

    def dim(self, s: str) -> str:
        return self.c(s, "2")

    def green(self, s: str) -> str:
        return self.c(s, "32")

    def red(self, s: str) -> str:
        return self.c(s, "31")

    def cyan(self, s: str) -> str:
        return self.c(s, "36")


STYLE = Style()


def _print_err(msg: str) -> None:
    print(STYLE.red(f"Error: {msg}"))


def _print_ok(msg: str) -> None:
    print(STYLE.green(msg))


# ----------------------------
# Vector / matrix helpers
# ----------------------------
def _is_vector(x: Any) -> bool:
    return isinstance(x, (list, tuple)) and all(isinstance(v, (int, float)) for v in x)


def _is_matrix(A: Any) -> bool:
    return (
        isinstance(A, (list, tuple))
        and len(A) > 0
        and all(isinstance(r, (list, tuple)) for r in A)
        and all(len(r) == len(A[0]) for r in A)
        and all(isinstance(v, (int, float)) for r in A for v in r)
    )


def dot(a: List[float], b: List[float]) -> float:
    if not (_is_vector(a) and _is_vector(b)) or len(a) != len(b):
        raise ValueError("dot(a,b) needs two same-length vectors, like dot([1,2],[3,4])")
    return float(sum(x * y for x, y in zip(a, b)))


def cross(a: List[float], b: List[float]) -> List[float]:
    if not (_is_vector(a) and _is_vector(b)) or len(a) != 3 or len(b) != 3:
        raise ValueError("cross(a,b) needs 3D vectors, like cross([1,0,0],[0,1,0])")
    ax, ay, az = a
    bx, by, bz = b
    return [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx]


def transpose(A: List[List[float]]) -> List[List[float]]:
    if not _is_matrix(A):
        raise ValueError("transpose(A) needs a matrix like [[1,2],[3,4]]")
    return [list(row) for row in zip(*A)]


def matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    if not (_is_matrix(A) and _is_matrix(B)):
        raise ValueError("matmul(A,B) needs matrices like [[...],[...]]")
    if len(A[0]) != len(B):
        raise ValueError("matmul(A,B) dimension mismatch (cols of A must equal rows of B)")
    Bt = transpose(B)
    return [[dot(row, col) for col in Bt] for row in A]


def det(A: List[List[float]]) -> float:
    if not _is_matrix(A):
        raise ValueError("det(A) needs a square matrix like [[1,2],[3,4]]")
    n = len(A)
    if n != len(A[0]):
        raise ValueError("det(A) needs a square matrix")

    if n == 1:
        return float(A[0][0])
    if n == 2:
        return float(A[0][0] * A[1][1] - A[0][1] * A[1][0])

    total = 0.0
    for j in range(n):
        minor = [row[:j] + row[j + 1 :] for row in A[1:]]
        total += ((-1) ** j) * A[0][j] * det(minor)
    return float(total)


def nCr(n: int, r: int) -> int:
    if r < 0 or n < 0 or r > n:
        return 0
    return math.comb(n, r)


def nPr(n: int, r: int) -> int:
    if r < 0 or n < 0 or r > n:
        return 0
    return math.perm(n, r)


def gcd(a: int, b: int) -> int:
    return math.gcd(int(a), int(b))


def lcm(a: int, b: int) -> int:
    a, b = int(a), int(b)
    if a == 0 and b == 0:
        return 0
    return abs(a * b) // math.gcd(a, b)


def deg(x: float) -> float:
    return float(x) * 180.0 / math.pi


def rad(x: float) -> float:
    return float(x) * math.pi / 180.0


# ----------------------------
# Stats + number theory extras
# ----------------------------
def mean(xs: List[float]) -> float:
    if not _is_vector(xs) or len(xs) == 0:
        raise ValueError("mean([..]) needs a non-empty list of numbers")
    return float(sum(xs) / len(xs))


def median(xs: List[float]) -> float:
    if not _is_vector(xs) or len(xs) == 0:
        raise ValueError("median([..]) needs a non-empty list of numbers")
    ys = sorted(float(x) for x in xs)
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return ys[mid]
    return (ys[mid - 1] + ys[mid]) / 2.0


def var(xs: List[float], sample: bool = False) -> float:
    if not _is_vector(xs) or len(xs) == 0:
        raise ValueError("var([..]) needs a non-empty list of numbers")
    n = len(xs)
    if sample and n < 2:
        raise ValueError("sample variance needs at least 2 values")
    m = mean(xs)
    denom = (n - 1) if sample else n
    return float(sum((float(x) - m) ** 2 for x in xs) / denom)


def stdev(xs: List[float], sample: bool = False) -> float:
    return math.sqrt(var(xs, sample=sample))


def isprime(n: int) -> bool:
    n = int(n)
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def factor(n: int) -> List[int]:
    n = int(n)
    if n == 0:
        raise ValueError("factor(0) is not defined")
    if n < 0:
        n = -n
    out: List[int] = []
    while n % 2 == 0:
        out.append(2)
        n //= 2
    p = 3
    while p * p <= n:
        while n % p == 0:
            out.append(p)
            n //= p
        p += 2
    if n > 1:
        out.append(n)
    return out


def primes(n: int) -> List[int]:
    n = int(n)
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start:n + 1:step] = [False] * (((n - start) // step) + 1)
    return [i for i, ok in enumerate(sieve) if ok]


def fib(n: int) -> int:
    n = int(n)
    if n < 0:
        raise ValueError("fib(n) needs n >= 0")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def clamp(x: float, a: float, b: float) -> float:
    x = float(x); a = float(a); b = float(b)
    lo, hi = (a, b) if a <= b else (b, a)
    return max(lo, min(hi, x))


def lerp(a: float, b: float, t: float) -> float:
    return float(a) + (float(b) - float(a)) * float(t)


def sign(x: float) -> int:
    x = float(x)
    return 0 if x == 0 else (1 if x > 0 else -1)


# ----------------------------
# Safe evaluation (AST whitelist)
# ----------------------------
ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
ALLOWED_UNARYOPS = (ast.UAdd, ast.USub, ast.Not)
ALLOWED_BOOLOPS = (ast.And, ast.Or)
ALLOWED_CMPOPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)

ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Subscript,
    ast.Slice,
    ast.Load,
    ast.operator,
    ast.unaryop,
    ast.boolop,
    ast.cmpop,
)


def _bad_name(name: str) -> bool:
    return name.startswith("__") or name in {
        "eval",
        "exec",
        "open",
        "compile",
        "input",
        "globals",
        "locals",
        "vars",
        "__import__",
    }


def _validate_ast(node: ast.AST) -> None:
    for n in ast.walk(node):
        if not isinstance(n, ALLOWED_NODES):
            raise ValueError(f"Not allowed in calculator: {type(n).__name__}")

        if isinstance(n, ast.BinOp):
            if not isinstance(n.op, ALLOWED_BINOPS):
                raise ValueError("Operator not allowed")

        elif isinstance(n, ast.UnaryOp):
            if not isinstance(n.op, ALLOWED_UNARYOPS):
                raise ValueError("Unary operator not allowed")

        elif isinstance(n, ast.BoolOp):
            if not isinstance(n.op, ALLOWED_BOOLOPS):
                raise ValueError("Boolean operator not allowed")

        elif isinstance(n, ast.Compare):
            if any(not isinstance(op, ALLOWED_CMPOPS) for op in n.ops):
                raise ValueError("Comparison operator not allowed")

        elif isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name):
                raise ValueError("Only function calls like f(x) are allowed (no attributes).")
            if _bad_name(n.func.id):
                raise ValueError("Function name not allowed.")

        elif isinstance(n, ast.Name):
            if _bad_name(n.id):
                raise ValueError("Name not allowed.")


# ----------------------------
# Preprocessor:
# - ^ -> **
# - implicit multiplication: 2x, 3sin(x), (x+1)(x-1), 2(x+1), x(x+1)
# - factorial postfix: x!, (x+1)!
# - ternary: a ? b : c   -> (b if a else c)
# - skips inside quoted strings
# ----------------------------
_TOKEN_RE = re.compile(
    r"""
    (?P<WS>\s+)
  | (?P<NUMBER>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)
  | (?P<NAME>[A-Za-z_]\w*)
  | (?P<OP>\*\*|//|==|!=|<=|>=|[+\-*/%<>()\[\],!:?])
  | (?P<MISC>.)
    """,
    re.VERBOSE,
)


def _tokenize(segment: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for m in _TOKEN_RE.finditer(segment):
        kind = m.lastgroup or "MISC"
        text = m.group()
        if kind == "WS":
            continue
        if kind == "MISC":
            raise ValueError(f"Invalid character: {text!r}")
        out.append((kind, text))
    return out


def _is_left_atom(tok: Tuple[str, str]) -> bool:
    kind, text = tok
    if kind in {"NUMBER", "NAME"}:
        return True
    if kind == "OP" and text in {")", "]"}:
        return True
    return False


def _is_right_atom(tok: Tuple[str, str]) -> bool:
    kind, text = tok
    if kind in {"NUMBER", "NAME"}:
        return True
    if kind == "OP" and text in {"(", "["}:
        return True
    return False


def _callable_names(env: Dict[str, Any]) -> set[str]:
    return {k for k, v in env.items() if callable(v)}


def _insert_implicit_mul(tokens: List[Tuple[str, str]], env: Dict[str, Any]) -> List[Tuple[str, str]]:
    if not tokens:
        return []

    callables = _callable_names(env)
    out: List[Tuple[str, str]] = []

    for i in range(len(tokens) - 1):
        a = tokens[i]
        b = tokens[i + 1]
        out.append(a)

        if _is_left_atom(a) and _is_right_atom(b):
            # block function call like sin( ... ) when sin is callable
            if a[0] == "NAME" and b[0] == "OP" and b[1] == "(" and a[1] in callables:
                continue
            out.append(("OP", "*"))

    out.append(tokens[-1])
    return out


def _find_matching_open(tokens: List[Tuple[str, str]], close_i: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    for i in range(close_i, -1, -1):
        k, t = tokens[i]
        if k == "OP" and t == close_ch:
            depth += 1
        elif k == "OP" and t == open_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _apply_factorial(tokens: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    # turns atom ! into factorial(atom)
    i = 0
    out = tokens[:]
    while i < len(out):
        if out[i][0] == "OP" and out[i][1] == "!":
            if i == 0:
                raise ValueError("Nothing to factorial on the left of '!'")
            # determine atom range [j, i)
            prev = out[i - 1]
            j = i - 1

            if prev[0] == "OP" and prev[1] in {")", "]"}:
                open_ch = "(" if prev[1] == ")" else "["
                close_ch = prev[1]
                j = _find_matching_open(out, i - 1, open_ch, close_ch)
                if j == -1:
                    raise ValueError("Unmatched bracket before '!'\n")

            # wrap
            atom = out[j:i]
            repl: List[Tuple[str, str]] = [("NAME", "factorial"), ("OP", "(")] + atom + [("OP", ")")]
            out = out[:j] + repl + out[i + 1 :]
            i = j + len(repl)
            continue
        i += 1
    return out


def _pow_fix(expr: str) -> str:
    return expr.replace("^", "**")


def _ternary_fix(segment: str) -> str:
    """
    Support:  cond ? a : b
    Convert to Python: (a if cond else b)

    Notes:
    - simple transformer, handles nesting by greedy split from leftmost '?'
    - if you do crazy nesting without parentheses, it can get confusing (normal).
    """
    # quick exit
    if "?" not in segment:
        return segment

    # We'll parse with a lightweight scan that respects parentheses/brackets and strings.
    s = segment
    in_str = False
    quote = ""
    escape = False
    depth = 0

    qpos = -1
    cpos = -1

    # find first top-level '?'
    for i, ch in enumerate(s):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_str = False
            continue
        if ch in {"'", '"'}:
            in_str = True
            quote = ch
            continue
        if ch in "([": depth += 1
        elif ch in ")]": depth -= 1
        elif ch == "?" and depth == 0:
            qpos = i
            break

    if qpos == -1:
        return segment

    # find matching ':' after that, top-level
    in_str = False
    quote = ""
    escape = False
    depth = 0
    for i in range(qpos + 1, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_str = False
            continue
        if ch in {"'", '"'}:
            in_str = True
            quote = ch
            continue
        if ch in "([": depth += 1
        elif ch in ")]": depth -= 1
        elif ch == ":" and depth == 0:
            cpos = i
            break

    if cpos == -1:
        raise ValueError("Ternary needs ':' like: condition ? a : b")

    cond = s[:qpos].strip()
    a = s[qpos + 1 : cpos].strip()
    b = s[cpos + 1 :].strip()

    # recurse in case a/b have ternaries too
    cond = _ternary_fix(cond)
    a = _ternary_fix(a)
    b = _ternary_fix(b)

    return f"({a} if {cond} else {b})"


def _preprocess(expr: str, env: Dict[str, Any]) -> str:
    s = _pow_fix(expr)
    # ternary first (works better before token insertions)
    s = _ternary_fix(s)

    result: List[str] = []
    buf: List[str] = []

    in_str = False
    quote = ""
    escape = False

    def flush_buf() -> None:
        nonlocal buf
        if not buf:
            return
        seg = "".join(buf)
        tokens = _tokenize(seg)
        tokens = _insert_implicit_mul(tokens, env)
        tokens = _apply_factorial(tokens)
        result.append("".join(t[1] for t in tokens))
        buf = []

    for ch in s:
        if in_str:
            result.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_str = False
                quote = ""
            continue

        if ch in {"'", '"'}:
            flush_buf()
            in_str = True
            quote = ch
            result.append(ch)
        else:
            buf.append(ch)

    flush_buf()
    return "".join(result)


def safe_eval(expr: str, env: Dict[str, Any]) -> Any:
    expr = _preprocess(expr.strip(), env)
    node = ast.parse(expr, mode="eval")
    _validate_ast(node)
    code = compile(node, "<calc>", "eval")
    return eval(code, {"__builtins__": {}}, env)


# ----------------------------
# REPL state (global, used by diff/solve/integrate)
# ----------------------------
@dataclass
class CalcState:
    vars: Dict[str, Any] = field(default_factory=dict)
    history: List[Tuple[str, Any]] = field(default_factory=list)
    precision: int = 12
    last: Any = 0
    trig_mode: str = "rad"  # "rad" or "deg"
    snap_output: bool = True
    show_fractions: bool = False  # display-only

    def _wrap_trig(self):
        def _in(x: float) -> float:
            return rad(x) if self.trig_mode == "deg" else float(x)

        def _out(x: float) -> float:
            return deg(x) if self.trig_mode == "deg" else float(x)

        return {
            "sin": lambda x: math.sin(_in(x)),
            "cos": lambda x: math.cos(_in(x)),
            "tan": lambda x: math.tan(_in(x)),
            "asin": lambda x: _out(math.asin(float(x))),
            "acos": lambda x: _out(math.acos(float(x))),
            "atan": lambda x: _out(math.atan(float(x))),
        }

    def base_env(self) -> Dict[str, Any]:
        trig = self._wrap_trig()
        phi = (1 + 5 ** 0.5) / 2.0
        return {
            # constants
            "pi": math.pi,
            "e": math.e,
            "tau": math.tau,
            "phi": phi,

            # basic funcs
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,

            "sqrt": math.sqrt,
            "cbrt": lambda x: float(x) ** (1.0 / 3.0),
            "exp": math.exp,

            "log": math.log,
            "ln": lambda x: math.log(x),
            "log10": math.log10,

            **trig,

            "sinh": math.sinh,
            "cosh": math.cosh,
            "tanh": math.tanh,

            "floor": math.floor,
            "ceil": math.ceil,

            "factorial": math.factorial,
            "nCr": nCr,
            "nPr": nPr,
            "gcd": gcd,
            "lcm": lcm,

            "deg": deg,
            "rad": rad,

            "rand": lambda a=0.0, b=1.0: random.uniform(float(a), float(b)),
            "seed": lambda s=None: random.seed(s),

            # vectors/matrices
            "dot": dot,
            "cross": cross,
            "transpose": transpose,
            "matmul": matmul,
            "det": det,

            # stats + number theory
            "mean": mean,
            "median": median,
            "var": var,
            "stdev": stdev,

            "isprime": isprime,
            "factor": factor,
            "primes": primes,
            "fib": fib,

            "clamp": clamp,
            "lerp": lerp,
            "sign": sign,
        }

    def env_view(self) -> Dict[str, Any]:
        env = self.base_env()
        env.update(self.vars)
        env["ans"] = self.last
        return env

    def _maybe_fraction_str(self, x: float) -> Optional[str]:
        frac = Fraction(x).limit_denominator(1000)
        if abs(x - float(frac)) < 1e-10:
            return f"{frac.numerator}/{frac.denominator}"
        return None

    def _snap_float(self, x: float) -> float:
        if not self.snap_output:
            return x

        r = round(x)
        if abs(x - r) < 1e-8:
            return float(int(r))

        # keep numeric, display can show fraction if enabled
        frac = Fraction(x).limit_denominator(1000)
        if abs(x - float(frac)) < 1e-10:
            return float(frac)

        return x

    def fmt(self, value: Any) -> str:
        if isinstance(value, float):
            value = self._snap_float(value)

        if isinstance(value, int):
            return str(value)

        if isinstance(value, float):
            if self.show_fractions:
                f = self._maybe_fraction_str(value)
                if f is not None:
                    return f"{f}  (~{value:.{self.precision}g})"
            return f"{value:.{self.precision}g}"

        return str(value)

    def assign(self, name: str, value: Any) -> None:
        if not name.isidentifier() or _bad_name(name):
            raise ValueError("Invalid variable name.")
        self.vars[name] = value
        self.last = value


CURRENT_STATE = CalcState()


# ----------------------------
# Calculus / solving helpers
# ----------------------------
def _f_from_expr(expr: str, var: str, env: Dict[str, Any]):
    def f(t: float) -> float:
        local = dict(env)
        local[var] = float(t)
        return float(safe_eval(expr, local))
    return f


def diff(expr: str, var: str = "x", at: float = 0.0, h: float = 1e-6) -> float:
    env = CURRENT_STATE.env_view()
    f = _f_from_expr(expr, var, env)
    x = float(at)

    hs = [float(h), float(h) * 10.0, float(h) / 10.0]
    vals: List[float] = []
    for hh in hs:
        vals.append((-f(x + 2 * hh) + 8 * f(x + hh) - 8 * f(x - hh) + f(x - 2 * hh)) / (12 * hh))

    vals.sort()
    return float(vals[1])


def integrate(expr: str, var: str = "x", a: float = 0.0, b: float = 1.0, n: int = 2000) -> float:
    env = CURRENT_STATE.env_view()
    f = _f_from_expr(expr, var, env)
    a = float(a)
    b = float(b)
    n = int(n)

    if n <= 0:
        raise ValueError("integrate(..., n=) must be > 0")
    if n % 2 == 1:
        n += 1

    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        s += (4 if i % 2 == 1 else 2) * f(x)
    return s * h / 3.0


def _bisect(f, a: float, b: float, tol: float = 1e-12, max_iter: int = 200) -> float:
    fa = f(a)
    fb = f(b)

    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        raise ValueError("Bisection needs a bracket where f(a) and f(b) have opposite signs.")

    lo, hi = a, b
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        fm = f(mid)
        if abs(fm) <= tol or abs(hi - lo) <= tol:
            return mid
        if fa * fm < 0:
            hi = mid
            fb = fm
        else:
            lo = mid
            fa = fm
    return (lo + hi) / 2.0


def solve(
    expr: str,
    var: str = "x",
    guess_or_a: float = 0.0,
    b: Optional[float] = None,
    tol: float = 1e-10,
    max_iter: int = 80,
) -> float:
    env = CURRENT_STATE.env_view()
    f = _f_from_expr(expr, var, env)

    tol = float(tol)
    max_iter = int(max_iter)

    if b is not None:
        return _bisect(f, float(guess_or_a), float(b), tol=tol, max_iter=max_iter)

    x = float(guess_or_a)
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) <= tol:
            return x

        dfx = diff(expr, var, x)
        if abs(dfx) < 1e-14:
            x2 = x + 1e-4 if x == 0 else x * 1.0001
            f2 = f(x2)
            denom = (f2 - fx)
            if abs(denom) < 1e-14:
                break
            x = x - fx * (x2 - x) / denom
        else:
            x = x - fx / dfx

    raise ValueError("solve() did not converge. Try a different guess or use a bracket solve(expr,var,a,b).")


def summation(expr: str, var: str = "k", a: int = 1, b: int = 10) -> float:
    env = CURRENT_STATE.env_view()
    a = int(a); b = int(b)
    if a > b:
        a, b = b, a
    total = 0.0
    for k in range(a, b + 1):
        local = dict(env)
        local[var] = k
        total += float(safe_eval(expr, local))
    return float(total)


def product(expr: str, var: str = "k", a: int = 1, b: int = 10) -> float:
    env = CURRENT_STATE.env_view()
    a = int(a); b = int(b)
    if a > b:
        a, b = b, a
    total = 1.0
    for k in range(a, b + 1):
        local = dict(env)
        local[var] = k
        total *= float(safe_eval(expr, local))
    return float(total)


# Inject calculus funcs into environment (after they exist)
def _env_with_calculus(state: CalcState) -> Dict[str, Any]:
    env = state.env_view()
    env.update({
        "diff": diff,
        "integrate": integrate,
        "solve": solve,
        "summation": summation,
        "product": product,
    })
    return env


# ----------------------------
# Commands / Help
# ----------------------------
HELP_TEXT = """
Commands:
  :help                 show help
  :vars                 list variables
  :hist                 show history
  :clear                clear history
  :reset                reset variables + history
  :del NAME             delete variable (example: :del x)
  :prec N               set display precision (example :prec 12)
  :mode rad|deg         trig mode (default rad)
  :snap on|off          snap output near integers (default on)
  :frac on|off          display nice fractions when close (display-only)
  :state                show current settings
  :const                list constants
  :func                 list functions
  :save FILE.json       save session
  :load FILE.json       load session
  :color on|off         toggle ANSI colors
  :quit                 exit

Normal usage:
  - Expressions: (2+3)*4, sin(pi/2), 10^2, 5!
  - Implicit multiply: 2x, 3sin(x), (x+1)(x-1), 2(x+1)
  - Variables:
      x = 10
      y = sqrt(x) + 5
      ans is the last result

Logic / ternary:
  - Comparisons: x>3, x<=10, x==2
  - Booleans: (x>0 and x<10), not (x==2)
  - Ternary: x>0 ? x : -x

Multi-run:
  - Separate with ';'
      x=5; 2x+1; sin(30)

Solve:
  - Newton guess:
      solve("cos(x) - x", "x", 0.7)
  - Bisection bracket:
      solve("x^3 - 2x - 5", "x", 2, 3)

Sum/Product:
  - summation("k^2","k",1,10)
  - product("k","k",1,5)
""".strip()


def cmd_vars(state: CalcState) -> None:
    if not state.vars:
        print(STYLE.dim("(no variables yet)"))
        return
    for k in sorted(state.vars.keys()):
        print(f"{STYLE.cyan(k)} = {state.fmt(state.vars[k])}")


def cmd_hist(state: CalcState) -> None:
    if not state.history:
        print(STYLE.dim("(history empty)"))
        return
    start = max(1, len(state.history) - 29)
    for i, (expr, val) in enumerate(state.history[-30:], start=start):
        print(f"{STYLE.dim(str(i).rjust(3))}. {expr}  {STYLE.dim('=>')}  {state.fmt(val)}")


def cmd_save(state: CalcState, path: str) -> None:
    data = {
        "vars": state.vars,
        "history": state.history,
        "precision": state.precision,
        "last": state.last,
        "trig_mode": state.trig_mode,
        "snap_output": state.snap_output,
        "show_fractions": state.show_fractions,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    _print_ok(f"Saved to {path}")


def cmd_load(state: CalcState, path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    state.vars = data.get("vars", {})
    state.history = [tuple(x) for x in data.get("history", [])]
    state.precision = int(data.get("precision", 12))
    state.last = data.get("last", 0)
    state.trig_mode = data.get("trig_mode", "rad")
    state.snap_output = bool(data.get("snap_output", True))
    state.show_fractions = bool(data.get("show_fractions", False))
    _print_ok(f"Loaded from {path}")


def _list_consts(state: CalcState) -> None:
    env = state.base_env()
    consts = [k for k, v in env.items() if isinstance(v, (int, float))]
    for k in sorted(consts):
        print(f"{STYLE.cyan(k)} = {state.fmt(env[k])}")


def _list_funcs(state: CalcState) -> None:
    env = _env_with_calculus(state)
    funcs = [k for k, v in env.items() if callable(v)]
    for k in sorted(funcs):
        print(STYLE.cyan(k))


def _show_state(state: CalcState) -> None:
    print(f"mode={STYLE.cyan(state.trig_mode)}  prec={STYLE.cyan(str(state.precision))}  "
          f"snap={STYLE.cyan('on' if state.snap_output else 'off')}  "
          f"frac={STYLE.cyan('on' if state.show_fractions else 'off')}  "
          f"color={STYLE.cyan('on' if STYLE.enabled else 'off')}")


def handle_command(state: CalcState, line: str) -> bool:
    parts = line.strip().split()
    cmd = parts[0].lower()

    if cmd in {":q", ":quit", ":exit"}:
        return False

    if cmd == ":help":
        print(HELP_TEXT)
        return True

    if cmd == ":vars":
        cmd_vars(state)
        return True

    if cmd == ":hist":
        cmd_hist(state)
        return True

    if cmd == ":clear":
        state.history.clear()
        _print_ok("History cleared.")
        return True

    if cmd == ":reset":
        state.vars.clear()
        state.history.clear()
        state.last = 0
        _print_ok("Reset done.")
        return True

    if cmd == ":del":
        if len(parts) != 2:
            _print_err("Usage: :del NAME (example: :del x)")
            return True
        name = parts[1]
        if name in state.vars:
            del state.vars[name]
            _print_ok(f"Deleted {name}.")
        else:
            _print_err(f"{name} not found.")
        return True

    if cmd == ":prec":
        if len(parts) != 2:
            _print_err("Usage: :prec N (example: :prec 12)")
            return True
        try:
            n = int(parts[1])
        except ValueError:
            _print_err("Precision must be a number. Example: :prec 12")
            return True
        if n < 1 or n > 50:
            _print_err("Precision must be between 1 and 50.")
            return True
        state.precision = n
        _print_ok(f"Precision set to {n}.")
        return True

    if cmd == ":mode":
        if len(parts) != 2 or parts[1].lower() not in {"rad", "deg"}:
            _print_err("Usage: :mode rad | :mode deg")
            return True
        state.trig_mode = parts[1].lower()
        _print_ok(f"Trig mode set to {state.trig_mode}.")
        return True

    if cmd == ":snap":
        if len(parts) != 2 or parts[1].lower() not in {"on", "off"}:
            _print_err("Usage: :snap on | :snap off")
            return True
        state.snap_output = (parts[1].lower() == "on")
        _print_ok(f"Snap output {'on' if state.snap_output else 'off'}.")
        return True

    if cmd == ":frac":
        if len(parts) != 2 or parts[1].lower() not in {"on", "off"}:
            _print_err("Usage: :frac on | :frac off")
            return True
        state.show_fractions = (parts[1].lower() == "on")
        _print_ok(f"Fraction display {'on' if state.show_fractions else 'off'}.")
        return True

    if cmd == ":state":
        _show_state(state)
        return True

    if cmd == ":const":
        _list_consts(state)
        return True

    if cmd == ":func":
        _list_funcs(state)
        return True

    if cmd == ":save":
        if len(parts) != 2:
            _print_err("Usage: :save file.json")
            return True
        try:
            cmd_save(state, parts[1])
        except Exception as e:
            _print_err(str(e))
        return True

    if cmd == ":load":
        if len(parts) != 2:
            _print_err("Usage: :load file.json")
            return True
        try:
            cmd_load(state, parts[1])
        except Exception as e:
            _print_err(str(e))
        return True

    if cmd == ":color":
        if len(parts) != 2 or parts[1].lower() not in {"on", "off"}:
            _print_err("Usage: :color on | :color off")
            return True
        STYLE.enabled = (parts[1].lower() == "on")
        _print_ok(f"Color {'enabled' if STYLE.enabled else 'disabled'}.")
        return True

    _print_err("Unknown command. Type :help")
    return True


# ----------------------------
# Expression runner
# ----------------------------
def try_assignment(line: str) -> Optional[Tuple[str, str]]:
    s = line.strip()
    if s.lower().startswith("let "):
        s = s[4:].strip()
    if "=" in s:
        left, right = s.split("=", 1)
        name = left.strip()
        expr = right.strip()
        if name and expr:
            return name, expr
    return None


def _run_expression(state: CalcState, expr_line: str) -> None:
    line = expr_line.strip()
    if not line:
        return

    env = _env_with_calculus(state)

    assign = try_assignment(line)
    if assign:
        name, expr = assign
        val = safe_eval(expr, env)
        state.assign(name, val)
        state.history.append((line, val))
        print(f"{STYLE.cyan(name)} = {state.fmt(val)}")
        return

    val = safe_eval(line, env)
    state.last = val
    state.history.append((line, val))
    print(state.fmt(val))


# ----------------------------
# Main REPL
# ----------------------------
def main() -> int:
    state = CURRENT_STATE

    print(STYLE.bold("Advanced Calculator (Upgraded)"))
    print(STYLE.dim("Type :help for commands. Use ^ for power. 5! works. ans = last result."))
    print(STYLE.dim("Tip: separate multiple commands with ';'"))
    print()

    while True:
        try:
            raw = input(STYLE.cyan("calc> ") if STYLE.enabled else "calc> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        raw = raw.strip()
        if not raw:
            continue

        if raw.startswith(":"):
            keep = handle_command(state, raw)
            if not keep:
                return 0
            continue

        chunks = [c.strip() for c in raw.split(";") if c.strip()]
        for chunk in chunks:
            try:
                _run_expression(state, chunk)
            except Exception as e:
                _print_err(str(e))
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
