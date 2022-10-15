data{
    int<lower = 0> N; //sample size
    int G; // team size (Z)
    int M; // num of X
    vector[N] y; // y
    matrix[N, M] X; //design matrix contain const
    int teamID[N]; // team id 
}
parameters{
    vector[M] beta[G]; // fix + random effects 
    vector[M] beta_fix; // fix effects 
    real<lower=0> s_y; // sd of y
    vector<lower=0> [M] tau; // sd vector of random effects
    corr_matrix[M] omega; // corr matrix 
}
transformed parameters {
   cov_matrix[M] Tau;
   Tau = quad_form_diag(omega, tau);
}
model{
    vector[N] mu;
    for (i in 1:N){
        mu[i] = X[i] * beta[teamID[i]];
    }
    y ~ normal(mu, s_y);
    beta ~ multi_normal(beta_fix, Tau);
    beta_fix~normal(0, 100);
    s_y~cauchy(0, 5);
    tau~cauchy(0, 5);
    omega~lkj_corr(1);
}