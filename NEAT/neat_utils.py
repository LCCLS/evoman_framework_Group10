import pandas as pd
import os


def average_experiment_gens(root_dir):
    """
    reads the individual dataframes from each experimetnal repetition and saves a new file which contains their average
    """

    exp1 = pd.read_csv(f"{root_dir}/EXP_1/results.txt", index_col=False)
    exp2 = pd.read_csv(f"{root_dir}/EXP_2/results.txt", index_col=False)
    exp3 = pd.read_csv(f"{root_dir}/EXP_3/results.txt", index_col=False)
    # exp4 = pd.read_csv(f"{root_dir}/EXP_4/results.txt")
    # exp5 = pd.read_csv(f"{root_dir}/EXP_5/results.txt")
    # exp6 = pd.read_csv(f"{root_dir}/EXP_6/results.txt")
    # exp7 = pd.read_csv(f"{root_dir}/EXP_7/results.txt")
    # exp8 = pd.read_csv(f"{root_dir}/EXP_8/results.txt")
    # exp9 = pd.read_csv(f"{root_dir}/EXP_9/results.txt")
    # exp10 = pd.read_csv(f"{root_dir}/EXP_10/results.txt")

    # total_exp = pd.concat([exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8, exp9, exp10]).groupby(level=0).mean()    total_exp = pd.concat([exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8, exp9, exp10]).groupby(level=0).mean()
    total_exp = pd.concat([exp1, exp2, exp3]).groupby(level=0).mean()

    if not os.path.exists(f"{root_dir}/EXP_MEAN"):
        os.makedirs(f"{root_dir}/EXP_MEAN")

    total_exp.to_csv(f"{root_dir}/EXP_MEAN/results.txt")
