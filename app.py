from flask import Flask, render_template, request
import sympy as sp
from ZOF_CLI import (bisection, regula_falsi, secant, newton_raphson,
                     fixed_point, modified_secant, parse_function)

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    result = None
    table = None
    if request.method=='POST':
        expr = request.form['expr']
        method = request.form['method']
        tol = float(request.form.get('tol','1e-6'))
        max_iter = int(request.form.get('max_iter','50'))
        f,fprime=parse_function(expr)
        try:
            if method=='bisection':
                a=float(request.form['a']); b=float(request.form['b'])
                root,ferr,iters,rows=bisection(f,a,b,tol,max_iter)
                headers=['iter','a','b','c','f(c)','error']
            elif method=='regula':
                a=float(request.form['a']); b=float(request.form['b'])
                root,ferr,iters,rows=regula_falsi(f,a,b,tol,max_iter)
                headers=['iter','a','b','c','f(c)','error']
            elif method=='secant':
                x0=float(request.form['x0']); x1=float(request.form['x1'])
                root,ferr,iters,rows=secant(f,x0,x1,tol,max_iter)
                headers=['iter','x0','x1','x2','f(x2)','error']
            elif method=='newton':
                x0=float(request.form['x0'])
                root,ferr,iters,rows=newton_raphson(f,fprime,x0,tol,max_iter)
                headers=['iter','x','f(x)','f\'(x)','x_new','error']
            elif method=='fixed':
                g_expr=request.form['g_expr']
                g=sp.lambdify(sp.symbols('x'), sp.sympify(g_expr), modules=["math"])
                x0=float(request.form['x0'])
                root,ferr,iters,rows=fixed_point(g,x0,tol,max_iter)
                headers=['iter','x','g(x)','error']
            elif method=='modified_secant':
                x0=float(request.form['x0'])
                delta=float(request.form.get('delta','0.01'))
                root,ferr,iters,rows=modified_secant(f,x0,delta,tol,max_iter)
                headers=['iter','x','f(x)','x_new','error']
            else: raise ValueError("Unknown method")

            result={'root':root,'ferr':ferr,'iters':iters}
            table={'headers':headers,'rows':rows}
        except Exception as e:
            result={'error':str(e)}

    return render_template('index.html', result=result, table=table)


if __name__=="__main__":
    app.run(debug=True)
