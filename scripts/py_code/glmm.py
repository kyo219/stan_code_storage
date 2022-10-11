from lib2to3.pgen2.token import SLASHEQUAL
from re import M
from cmdstanpy import  CmdStanModel
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import arviz as az

salary_df = pd.read_csv('sample_data/npb_salary.csv')
salary_df.columns

### Stanファイルを読み込んでオブジェクトを生成する
stan_file_path = '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/glmm_stan.stan'
model = CmdStanModel(stan_file=stan_file_path)

### MCMCを実行して事後分布をサンプリングする
iter_sampling = 1000 # サンプリングの数.デフォルト1000
iter_warmup = 100
chains = 4 # 並列サンプリングの数．デフォルト4


salary_df['チーム'].map({0:'Elephant',1:'Lion'})


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

plt.figure(figsize=(10,3))
bins=15
for i, k in enumerate(la.keys()):
    boxplot_list =[]
    for j in range(chains):
        x = la[k][j*iter_sampling:(j+1)*iter_sampling]
        hist, bin_edges = np.histogram(x, bins=bins) # 度数分布に変換
        #print(len(hist),len(bin_edges))
        plt.subplot(len(la.keys()), 3, 3*i+1) 
        plt.plot(bin_edges[:bins], hist, alpha=0.6, lw=0.6)
        plt.subplot(len(la.keys()), 3, 3*i+2) 
        plt.plot(x, alpha=0.5, lw=0.8)
        boxplot_list.append(x)
    plt.subplot(len(la.keys()), 3, 3*i+3) 
    plt.boxplot(boxplot_list)
plt.show()


### 結果を解釈する
samples = az.from_cmdstanpy(posterior = fit_sm)
print(az.summary(samples))