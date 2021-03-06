import random
import sys

depth, outfile = sys.argv[1], sys.argv[2]

MAX_DEPTH = int(depth)
LITERALS = [-2, -1, 0, 1, 2]
VARIABLES = ['x', 'y']
PRIM_PROCS = ['+', '-', '*', '/']

class Expression:
    def __str__(self):
        pass


class LitExpression(Expression):
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return str(self.val)


class VarExpression(Expression):
    def __init__(self, var):
        self.var = var

    def __str__(self):
        return self.var


class AppExpression(Expression):
    def __init__(self, operator, operands):
        self.operator = operator
        self.operands = operands

    def __str__(self):
        return "(%s %s)" % (str(self.operator), ' '.join(map(str, self.operands)))


class LambdaExpression(Expression):
    def __init__(self, formals, body):
        self.formals = formals
        self.body = body

    def __str__(self):
        return "(λ (%s) %s)" % (' '.join(self.formals), str(self.body))
        # return "(lambda (%s) %s)" % (' '.join(self.formals), str(self.body))


def random_valid_expression(depth=0, bindings=set()):
    if depth == 0:
        i = 5
    if depth == MAX_DEPTH:
        if len(bindings) == 0:
            i = 3
        else:
            i = random.randint(0, 4)
    elif len(bindings) == 0:
        i = random.randint(3, 10)
    else:
        i = random.randint(0, 10)

    if i < 3:
        return VarExpression(random.choice(list(bindings)))
    elif i < 5:
        return LitExpression(random.choice(LITERALS))
    else:
        if random.randint(0, 1):
            op = VarExpression(random.choice(PRIM_PROCS))
            args = [random_valid_expression(depth=depth+1), random_valid_expression(depth=depth+1)]
        else:
            formal = random.choice(VARIABLES)
            new_binds = set(bindings)
            new_binds.add(formal)
            op = LambdaExpression([formal], random_valid_expression(depth=depth+1, bindings=new_binds))
            args = [random_valid_expression(depth=depth+1, bindings=new_binds)]

        return AppExpression(op, args)


def random_expression(depth=0, bindings=set()):
    if depth == 0:
        i = 5
    if depth == MAX_DEPTH:
        if len(bindings) == 0:
            i = 3
        else:
            i = random.randint(0, 4)
    elif len(bindings) == 0:
        i = random.randint(3, 10)
    else:
        i = random.randint(0, 10)

    if i < 3:
        return VarExpression(random.choice(VARIABLES))
    elif i < 5:
        return LitExpression(random.choice(LITERALS))
    else:
        if random.randint(0, 1):
            op = VarExpression(random.choice(PRIM_PROCS))
            args = [random_expression(depth=depth+1), random_expression(depth=depth+1)]
        else:
            formal = random.choice(VARIABLES)
            new_binds = set(bindings)
            new_binds.add(formal)
            op = LambdaExpression([formal], random_expression(depth=depth+1))
            args = [random_expression(depth=depth+1)]

        return AppExpression(op, args)

f = open(outfile, 'w')
for e in LITERALS + VARIABLES:
    f.write(str(e) + "\n")

for _ in range(500000):
    e = str(random_expression())
    if len(e) > 1:
        f.write(e + "\n")
