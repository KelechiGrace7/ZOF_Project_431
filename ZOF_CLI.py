
"""ZOF_CLI.py - Zero of Functions Solver (CLI)"""

import math
import sys
from typing import Callable, List, Tuple

try:
    import sympy as sp
except ImportError:
    print("Install sympy first: pip install sympy")
    sys.exit(1)


def parse_function(expr: str) -> Tuple[Callable[[float], float], Callable[[float], float]]:
    x = sp.symbols('x')
    f_sym = sp.sympify(expr)
    f = sp.lambdify(x, f_sym, modules=["math"])
    try:
        fprime_sym = sp.diff(f_sym, x)
        fprime = sp.lambdify(x, fprime_sym, modules=["math"])
    except Exception:
        def fprime(_): raise RuntimeError("Derivative unavailable")
    return f, fprime


def print_table(headers: List[str], rows: List[List]):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = ' | '.join('{:>' + str(w) + '}' for w in widths)
    print(fmt.format(*headers))
    print('-' * (sum(widths) + 3*(len(widths)-1)))
    for r in rows:
        print(fmt.format(*[f"{v:.6g}" if isinstance(v,float) else str(v) for v in r]))


# === Numerical methods ===

def bisection(f, a, b, tol, max_iter):
    rows = []
    fa, fb = f(a), f(b)
    if fa*fb>0: raise ValueError("f(a) and f(b) must have opposite signs")
    for i in range(1, max_iter+1):
        c = (a+b)/2
        fc = f(c)
        err = abs(b-a)/2
        rows.append([i, a, b, c, fc, err])
        if abs(fc)<tol or err<tol: return c, abs(fc), i, rows
        if fa*fc <0: b, fb = c, fc
        else: a, fa = c, fc
    return (a+b)/2, abs(f((a+b)/2)), max_iter, rows


def regula_falsi(f, a, b, tol, max_iter):
    rows = []
    fa, fb = f(a), f(b)
    if fa*fb>0: raise ValueError("f(a) and f(b) must have opposite signs")
    c = a
    for i in range(1,max_iter+1):
        c_prev = c
        c = (a*fb - b*fa)/(fb-fa)
        fc = f(c)
        err = abs(c-c_prev) if i>1 else abs(b-a)
        rows.append([i, a, b, c, fc, err])
        if abs(fc)<tol or err<tol: return c, abs(fc), i, rows
        if fa*fc<0: b, fb = c, fc
        else: a, fa = c, fc
    return c, abs(f(c)), max_iter, rows


def secant(f, x0, x1, tol, max_iter):
    rows=[]
    for i in range(1,max_iter+1):
        f0,f1 = f(x0), f(x1)
        if f1-f0==0: raise ZeroDivisionError("Division by zero in Secant")
        x2 = x1 - f1*(x1-x0)/(f1-f0)
        err = abs(x2-x1)
        rows.append([i,x0,x1,x2,f(x2),err])
        if abs(f(x2))<tol or err<tol: return x2, abs(f(x2)), i, rows
        x0,x1 = x1,x2
    return x1, abs(f(x1)), max_iter, rows


def newton_raphson(f,fprime,x0,tol,max_iter):
    rows=[]
    x=x0
    for i in range(1,max_iter+1):
        fx=f(x)
        fpx=fprime(x)
        if fpx==0: raise ZeroDivisionError("Zero derivative")
        x_new = x - fx/fpx
        err = abs(x_new-x)
        rows.append([i,x,fx,fpx,x_new,err])
        if abs(fx)<tol or err<tol: return x_new, abs(fx), i, rows
        x = x_new
    return x, abs(f(x)), max_iter, rows


def fixed_point(g,x0,tol,max_iter):
    rows=[]
    x=x0
    for i in range(1,max_iter+1):
        x_new = g(x)
        err = abs(x_new-x)
        rows.append([i,x,x_new,err])
        if err<tol: return x_new, abs(g(x_new)-x_new), i, rows
        x = x_new
    return x, abs(g(x)-x), max_iter, rows


def modified_secant(f,x0,delta,tol,max_iter):
    rows=[]
    x=x0
    for i in range(1,max_iter+1):
        f0 = f(x)
        denom = f(x+delta*x)-f0
        if denom==0: raise ZeroDivisionError("Division by zero in Modified Secant")
        x_new = x - delta*x*f0/denom
        err = abs(x_new-x)
        rows.append([i,x,f0,x_new,err])
        if abs(f0)<tol or err<tol: return x_new, abs(f(x_new)), i, rows
        x=x_new
    return x, abs(f(x)), max_iter, rows


# === CLI Menu ===

def cli_menu():
    print("\nZOF CLI — Zero of Functions Solver")
    expr = input("Enter function f(x): ")
    f,fprime = parse_function(expr)
    print("\nChoose method:")
    methods = {"1":"Bisection","2":"Regula Falsi","3":"Secant",
               "4":"Newton-Raphson","5":"Fixed Point","6":"Modified Secant"}
    for k,v in methods.items(): print(f"{k}. {v}")
    choice=input("Method number: ").strip()
    tol = float(input("Tolerance (e.g., 1e-6): ") or 1e-6)
    max_iter=int(input("Max iterations (e.g., 50): ") or 50)

    try:
        if choice=="1":
            a=float(input("Left endpoint a: ")); b=float(input("Right endpoint b: "))
            root,ferr,iters,rows=bisection(f,a,b,tol,max_iter)
            print_table(['iter','a','b','c','f(c)','error'],rows)
        elif choice=="2":
            a=float(input("Left endpoint a: ")); b=float(input("Right endpoint b: "))
            root,ferr,iters,rows=regula_falsi(f,a,b,tol,max_iter)
            print_table(['iter','a','b','c','f(c)','error'],rows)
        elif choice=="3":
            x0=float(input("x0: ")); x1=float(input("x1: "))
            root,ferr,iters,rows=secant(f,x0,x1,tol,max_iter)
            print_table(['iter','x0','x1','x2','f(x2)','error'],rows)
        elif choice=="4":
            x0=float(input("Initial guess x0: "))
            root,ferr,iters,rows=newton_raphson(f,fprime,x0,tol,max_iter)
            print_table(['iter','x','f(x)','f\'(x)','x_new','error'],rows)
        elif choice=="5":
            g_expr=input("Enter g(x) for Fixed Point: ")
            x_sym=sp.symbols('x'); g=sp.lambdify(x_sym,sp.sympify(g_expr),modules=["math"])
            x0=float(input("Initial guess x0: "))
            root,ferr,iters,rows=fixed_point(g,x0,tol,max_iter)
            print_table(['iter','x','g(x)','error'],rows)
        elif choice=="6":
            x0=float(input("Initial guess x0: "))
            delta=float(input("Delta (0.01 default): ") or 0.01)
            root,ferr,iters,rows=modified_secant(f,x0,delta,tol,max_iter)
            print_table(['iter','x','f(x)','x_new','error'],rows)
        else: print("Unknown choice"); return

        print(f"\nFinal estimated root: {root}")
        print(f"Final function value (abs): {ferr}")
        print(f"Iterations used: {iters}")

    except Exception as e:
        print("Error:",e)


if __name__=="__main__":
    cli_menu()
