import numpy as np
import sys

from pymoo.util.display.column import Column
from pymoo.util.display.output import Output
from pymoo.core.problem import Problem

from NSGA2_utils import evaluate

sys.path.insert(0, 'evoman')


class MyOutput(Output):

    def __init__(self):
        super().__init__()
        self.x_mean = Column("x_mean", width=15)
        self.x_max = Column('x_max', width=15)

        self.columns += [self.x_max, self.x_mean]

    def update(self, algorithm):
        super().update(algorithm)
        total_F = algorithm.pop.get('F').tolist()

        self.x_max.set("{:.8f}".format(float(min(list(map(min, total_F))))))
        self.x_mean.set("{:.8f}".format(np.mean(sum(total_F, []))))


class ProblemWrapper(Problem):

    def __init__(self, n_var, n_obj, xl, xu, env):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)

        self.ENV = env

    def _evaluate(self, designs, out, *args, **kwargs):
        """
        evluation function wrapper for inherited problem class -- mean_fit, ind_gains, & mul_gains are alternative
        evaluation metrics
        """
        res = []

        for design in designs:
            eval_result = evaluate(design, self.ENV)
            # mean_fit = eval_result[0][0]
            multiple_fit = np.array([i * -1 for i in eval_result[0][1]])
            # ind_gains = eval_result[4][0]
            # mul_gains = eval_result[4][1]
            res.append(multiple_fit)

        out['F'] = np.array(res)

