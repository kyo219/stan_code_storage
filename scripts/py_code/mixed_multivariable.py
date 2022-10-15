from cmdstanpy import  CmdStanModel
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import arviz as az

salary_df = pd.read_csv('sample_data/npb_salary.csv')
salary_df.columns

### Stanファイルを読み込んでオブジェクトを生成する
stan_file_path = '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan'
model = CmdStanModel(stan_file=stan_file_path)

### MCMCを実行して事後分布をサンプリングする
iter_sampling = 10000 # サンプリングの数.デフォルト1000
iter_warmup = 2000
chains = 4 # 並列サンプリングの数．デフォルト4
class_mapping = {label:idx for idx, label in enumerate(np.unique(salary_df['チーム']))}


'''
data{
    int<lower = 0> N; //sample size
    int G; // team size (Z)
    int M; // num of X
    vector[N] y; // y
    matrix[N, M] X; //design matrix contain const
    int teamID[N]; // team id 
}
'''

X = salary_df[['打席数', '安打', '本塁打', '三振']]
X['const'] = 1

salary_df
data = {
    "N" : len(salary_df),
    'G' : len(salary_df['チーム'].unique()),
    'M' : 5,
    "y" : salary_df['打点'],
    "X" : X[['const', '打席数', '安打', '本塁打', '三振']],
    'teamID' : salary_df['チーム'].map(class_mapping) + 1
}



fit_sm = model.sample(data=data,
                      iter_sampling=iter_sampling,
                      iter_warmup = iter_warmup, 
                      chains = chains,
                      seed=1234)


la = fit_sm.stan_variables()

### 結果を解釈する
samples = az.from_cmdstanpy(posterior = fit_sm)
result = az.summary(samples)

for i in range(5):
    result[result.index == 'beta_fix['+str(i)+']']