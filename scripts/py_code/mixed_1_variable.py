from cmdstanpy import  CmdStanModel
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import arviz as az

salary_df = pd.read_csv('sample_data/npb_salary.csv')
salary_df.columns

### Stanファイルを読み込んでオブジェクトを生成する
stan_file_path = '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_1_variable.stan'
model = CmdStanModel(stan_file=stan_file_path)

### MCMCを実行して事後分布をサンプリングする
iter_sampling = 10000 # サンプリングの数.デフォルト1000
iter_warmup = 2000
chains = 4 # 並列サンプリングの数．デフォルト4
class_mapping = {label:idx for idx, label in enumerate(np.unique(salary_df['チーム']))}

data = {
    "N" : len(salary_df),
    "y" : salary_df['打点'],
    'G' : len(salary_df['チーム'].unique()),
    'x_1' : salary_df['打率'],
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
print(az.summary(samples))
