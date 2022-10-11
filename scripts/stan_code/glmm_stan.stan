data{
    int<lower = 0> N;
    int G;
    real y[N];
    real x_1[N];
    int teamID[N];
}
parameters{
    real alpha[G];
    real alpha_fix;
    real<lower=0> tau_alpha;
    real beta[G];
    real beta_fix;
    real<lower=0> tau_beta;    
    real<lower=0> s_y;
}
model{
    s_y ~ normal(0, 100);
    for(i in 1:N){
        y[i] ~ normal(alpha[teamID[i]] + beta[teamID[i]] * x_1[i], s_y);
    }    
    alpha ~ normal(0, 100);
    beta ~ normal(beta_fix, tau_beta);
    alpha ~ normal(alpha_fix, tau_alpha);
    beta_fix  ~ normal(0, 100); 
    alpha_fix ~ normal(0, 100);
    tau_alpha ~ cauchy(0, 2);    
    tau_beta ~ cauchy(0, 2);   
}