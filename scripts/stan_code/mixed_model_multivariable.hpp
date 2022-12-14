
// Code generated by stanc v2.30.1
#include <stan/model/model_header.hpp>
namespace mixed_model_multivariable_model_namespace {

using stan::model::model_base_crtp;
using namespace stan::math;


stan::math::profile_map profiles__;
static constexpr std::array<const char*, 35> locations_array__ = 
{" (found before start of program)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 10, column 4 to column 22)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 11, column 4 to column 23)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 12, column 4 to column 22)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 13, column 4 to column 28)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 14, column 4 to column 25)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 17, column 3 to column 21)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 18, column 3 to column 36)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 21, column 11 to column 12)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 21, column 4 to column 17)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 23, column 8 to column 39)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 22, column 18 to line 24, column 5)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 22, column 4 to line 24, column 5)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 25, column 4 to column 24)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 26, column 4 to column 39)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 27, column 4 to column 28)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 28, column 4 to column 21)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 29, column 4 to column 21)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 30, column 4 to column 22)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 2, column 4 to column 21)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 3, column 4 to column 10)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 4, column 4 to column 10)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 5, column 11 to column 12)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 5, column 4 to column 16)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 6, column 11 to column 12)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 6, column 14 to column 15)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 6, column 4 to column 19)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 7, column 15 to column 16)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 7, column 4 to column 18)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 10, column 19 to column 20)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 10, column 11 to column 12)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 11, column 11 to column 12)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 13, column 21 to column 22)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 14, column 16 to column 17)",
 " (in '/Users/jmb20210028/my_dev/stan_code_storage/scripts/stan_code/mixed_model_multivariable.stan', line 17, column 14 to column 15)"};




class mixed_model_multivariable_model final : public model_base_crtp<mixed_model_multivariable_model> {

 private:
  int N;
  int G;
  int M;
  Eigen::Matrix<double, -1, 1> y_data__;
  Eigen::Matrix<double, -1, -1> X_data__;
  std::vector<int> teamID; 
  Eigen::Map<Eigen::Matrix<double, -1, 1>> y{nullptr, 0};
  Eigen::Map<Eigen::Matrix<double, -1, -1>> X{nullptr, 0, 0};
 
 public:
  ~mixed_model_multivariable_model() { }
  
  inline std::string model_name() const final { return "mixed_model_multivariable_model"; }

  inline std::vector<std::string> model_compile_info() const noexcept {
    return std::vector<std::string>{"stanc_version = stanc3 v2.30.1", "stancflags = "};
  }
  
  
  mixed_model_multivariable_model(stan::io::var_context& context__,
                                  unsigned int random_seed__ = 0,
                                  std::ostream* pstream__ = nullptr) : model_base_crtp(0) {
    int current_statement__ = 0;
    using local_scalar_t__ = double ;
    boost::ecuyer1988 base_rng__ = 
        stan::services::util::create_rng(random_seed__, 0);
    (void) base_rng__;  // suppress unused var warning
    static constexpr const char* function__ = "mixed_model_multivariable_model_namespace::mixed_model_multivariable_model";
    (void) function__;  // suppress unused var warning
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning
    try {
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      current_statement__ = 19;
      context__.validate_dims("data initialization","N","int",
           std::vector<size_t>{});
      N = std::numeric_limits<int>::min();
      
      
      current_statement__ = 19;
      N = context__.vals_i("N")[(1 - 1)];
      current_statement__ = 19;
      stan::math::check_greater_or_equal(function__, "N", N, 0);
      current_statement__ = 20;
      context__.validate_dims("data initialization","G","int",
           std::vector<size_t>{});
      G = std::numeric_limits<int>::min();
      
      
      current_statement__ = 20;
      G = context__.vals_i("G")[(1 - 1)];
      current_statement__ = 21;
      context__.validate_dims("data initialization","M","int",
           std::vector<size_t>{});
      M = std::numeric_limits<int>::min();
      
      
      current_statement__ = 21;
      M = context__.vals_i("M")[(1 - 1)];
      current_statement__ = 22;
      stan::math::validate_non_negative_index("y", "N", N);
      current_statement__ = 23;
      context__.validate_dims("data initialization","y","double",
           std::vector<size_t>{static_cast<size_t>(N)});
      y_data__ = 
        Eigen::Matrix<double, -1, 1>::Constant(N,
          std::numeric_limits<double>::quiet_NaN());
      new (&y) Eigen::Map<Eigen::Matrix<double, -1, 1>>(y_data__.data(), N);
      
      {
        std::vector<local_scalar_t__> y_flat__;
        current_statement__ = 23;
        y_flat__ = context__.vals_r("y");
        current_statement__ = 23;
        pos__ = 1;
        current_statement__ = 23;
        for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
          current_statement__ = 23;
          stan::model::assign(y, y_flat__[(pos__ - 1)],
            "assigning variable y", stan::model::index_uni(sym1__));
          current_statement__ = 23;
          pos__ = (pos__ + 1);
        }
      }
      current_statement__ = 24;
      stan::math::validate_non_negative_index("X", "N", N);
      current_statement__ = 25;
      stan::math::validate_non_negative_index("X", "M", M);
      current_statement__ = 26;
      context__.validate_dims("data initialization","X","double",
           std::vector<size_t>{static_cast<size_t>(N),
            static_cast<size_t>(M)});
      X_data__ = 
        Eigen::Matrix<double, -1, -1>::Constant(N, M,
          std::numeric_limits<double>::quiet_NaN());
      new (&X) Eigen::Map<Eigen::Matrix<double, -1, -1>>(X_data__.data(), N, M);
        
      
      {
        std::vector<local_scalar_t__> X_flat__;
        current_statement__ = 26;
        X_flat__ = context__.vals_r("X");
        current_statement__ = 26;
        pos__ = 1;
        current_statement__ = 26;
        for (int sym1__ = 1; sym1__ <= M; ++sym1__) {
          current_statement__ = 26;
          for (int sym2__ = 1; sym2__ <= N; ++sym2__) {
            current_statement__ = 26;
            stan::model::assign(X, X_flat__[(pos__ - 1)],
              "assigning variable X", stan::model::index_uni(sym2__),
                                        stan::model::index_uni(sym1__));
            current_statement__ = 26;
            pos__ = (pos__ + 1);
          }
        }
      }
      current_statement__ = 27;
      stan::math::validate_non_negative_index("teamID", "N", N);
      current_statement__ = 28;
      context__.validate_dims("data initialization","teamID","int",
           std::vector<size_t>{static_cast<size_t>(N)});
      teamID = std::vector<int>(N, std::numeric_limits<int>::min());
      
      
      current_statement__ = 28;
      teamID = context__.vals_i("teamID");
      current_statement__ = 29;
      stan::math::validate_non_negative_index("beta", "G", G);
      current_statement__ = 30;
      stan::math::validate_non_negative_index("beta", "M", M);
      current_statement__ = 31;
      stan::math::validate_non_negative_index("beta_fix", "M", M);
      current_statement__ = 32;
      stan::math::validate_non_negative_index("tau", "M", M);
      current_statement__ = 33;
      stan::math::validate_non_negative_index("omega", "M", M);
      current_statement__ = 33;
      stan::math::validate_non_negative_index("omega", "M", M);
      current_statement__ = 34;
      stan::math::validate_non_negative_index("Tau", "M", M);
      current_statement__ = 34;
      stan::math::validate_non_negative_index("Tau", "M", M);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    num_params_r__ = (G * M) + M + 1 + M + ((M * (M - 1)) / 2);
    
  }
  
  template <bool propto__, bool jacobian__ , typename VecR, typename VecI, 
  stan::require_vector_like_t<VecR>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr> 
  inline stan::scalar_type_t<VecR> log_prob_impl(VecR& params_r__,
                                                 VecI& params_i__,
                                                 std::ostream* pstream__ = nullptr) const {
    using T__ = stan::scalar_type_t<VecR>;
    using local_scalar_t__ = T__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning
    static constexpr const char* function__ = "mixed_model_multivariable_model_namespace::log_prob";
    (void) function__;  // suppress unused var warning
    
    try {
      std::vector<Eigen::Matrix<local_scalar_t__, -1, 1>> beta =
         std::vector<Eigen::Matrix<local_scalar_t__, -1, 1>>(G, 
           Eigen::Matrix<local_scalar_t__, -1, 1>::Constant(M, DUMMY_VAR__));
      current_statement__ = 1;
      beta = in__.template read<
               std::vector<Eigen::Matrix<local_scalar_t__, -1, 1>>>(G, M);
      Eigen::Matrix<local_scalar_t__, -1, 1> beta_fix =
         Eigen::Matrix<local_scalar_t__, -1, 1>::Constant(M, DUMMY_VAR__);
      current_statement__ = 2;
      beta_fix = in__.template read<Eigen::Matrix<local_scalar_t__, -1, 1>>(
                   M);
      local_scalar_t__ s_y = DUMMY_VAR__;
      current_statement__ = 3;
      s_y = in__.template read_constrain_lb<local_scalar_t__, jacobian__>(0,
              lp__);
      Eigen::Matrix<local_scalar_t__, -1, 1> tau =
         Eigen::Matrix<local_scalar_t__, -1, 1>::Constant(M, DUMMY_VAR__);
      current_statement__ = 4;
      tau = in__.template read_constrain_lb<
              Eigen::Matrix<local_scalar_t__, -1, 1>, jacobian__>(0, lp__, M);
      Eigen::Matrix<local_scalar_t__, -1, -1> omega =
         Eigen::Matrix<local_scalar_t__, -1, -1>::Constant(M, M, DUMMY_VAR__);
      current_statement__ = 5;
      omega = in__.template read_constrain_corr_matrix<
                Eigen::Matrix<local_scalar_t__, -1, -1>, jacobian__>(lp__, M);
      Eigen::Matrix<local_scalar_t__, -1, -1> Tau =
         Eigen::Matrix<local_scalar_t__, -1, -1>::Constant(M, M, DUMMY_VAR__);
      current_statement__ = 7;
      stan::model::assign(Tau, stan::math::quad_form_diag(omega, tau),
        "assigning variable Tau");
      current_statement__ = 6;
      stan::math::check_cov_matrix(function__, "Tau", Tau);
      {
        current_statement__ = 8;
        stan::math::validate_non_negative_index("mu", "N", N);
        Eigen::Matrix<local_scalar_t__, -1, 1> mu =
           Eigen::Matrix<local_scalar_t__, -1, 1>::Constant(N, DUMMY_VAR__);
        current_statement__ = 12;
        for (int i = 1; i <= N; ++i) {
          current_statement__ = 10;
          stan::model::assign(mu,
            stan::math::multiply(
              stan::model::rvalue(X, "X", stan::model::index_uni(i)),
              stan::model::rvalue(beta, "beta",
                stan::model::index_uni(stan::model::rvalue(teamID, "teamID",
                                         stan::model::index_uni(i))))),
            "assigning variable mu", stan::model::index_uni(i));
        }
        current_statement__ = 13;
        lp_accum__.add(stan::math::normal_lpdf<propto__>(y, mu, s_y));
        current_statement__ = 14;
        lp_accum__.add(
          stan::math::multi_normal_lpdf<propto__>(beta, beta_fix, Tau));
        current_statement__ = 15;
        lp_accum__.add(stan::math::normal_lpdf<propto__>(beta_fix, 0, 100));
        current_statement__ = 16;
        lp_accum__.add(stan::math::cauchy_lpdf<propto__>(s_y, 0, 5));
        current_statement__ = 17;
        lp_accum__.add(stan::math::cauchy_lpdf<propto__>(tau, 0, 5));
        current_statement__ = 18;
        lp_accum__.add(stan::math::lkj_corr_lpdf<propto__>(omega, 1));
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
    } // log_prob_impl() 
    
  template <typename RNG, typename VecR, typename VecI, typename VecVar, 
  stan::require_vector_like_vt<std::is_floating_point, VecR>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr, 
  stan::require_vector_vt<std::is_floating_point, VecVar>* = nullptr> 
  inline void write_array_impl(RNG& base_rng__, VecR& params_r__,
                               VecI& params_i__, VecVar& vars__,
                               const bool emit_transformed_parameters__ = true,
                               const bool emit_generated_quantities__ = true,
                               std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    static constexpr bool propto__ = true;
    (void) propto__;
    double lp__ = 0.0;
    (void) lp__;  // dummy to suppress unused var warning
    int current_statement__ = 0; 
    stan::math::accumulator<double> lp_accum__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    constexpr bool jacobian__ = false;
    (void) DUMMY_VAR__;  // suppress unused var warning
    static constexpr const char* function__ = "mixed_model_multivariable_model_namespace::write_array";
    (void) function__;  // suppress unused var warning
    
    try {
      std::vector<Eigen::Matrix<double, -1, 1>> beta =
         std::vector<Eigen::Matrix<double, -1, 1>>(G, 
           Eigen::Matrix<double, -1, 1>::Constant(M,
             std::numeric_limits<double>::quiet_NaN()));
      current_statement__ = 1;
      beta = in__.template read<
               std::vector<Eigen::Matrix<local_scalar_t__, -1, 1>>>(G, M);
      Eigen::Matrix<double, -1, 1> beta_fix =
         Eigen::Matrix<double, -1, 1>::Constant(M,
           std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 2;
      beta_fix = in__.template read<Eigen::Matrix<local_scalar_t__, -1, 1>>(
                   M);
      double s_y = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 3;
      s_y = in__.template read_constrain_lb<local_scalar_t__, jacobian__>(0,
              lp__);
      Eigen::Matrix<double, -1, 1> tau =
         Eigen::Matrix<double, -1, 1>::Constant(M,
           std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 4;
      tau = in__.template read_constrain_lb<
              Eigen::Matrix<local_scalar_t__, -1, 1>, jacobian__>(0, lp__, M);
      Eigen::Matrix<double, -1, -1> omega =
         Eigen::Matrix<double, -1, -1>::Constant(M, M,
           std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 5;
      omega = in__.template read_constrain_corr_matrix<
                Eigen::Matrix<local_scalar_t__, -1, -1>, jacobian__>(lp__, M);
      Eigen::Matrix<double, -1, -1> Tau =
         Eigen::Matrix<double, -1, -1>::Constant(M, M,
           std::numeric_limits<double>::quiet_NaN());
      for (int sym1__ = 1; sym1__ <= M; ++sym1__) {
        for (int sym2__ = 1; sym2__ <= G; ++sym2__) {
          out__.write(beta[(sym2__ - 1)][(sym1__ - 1)]);
        }
      }
      out__.write(beta_fix);
      out__.write(s_y);
      out__.write(tau);
      out__.write(omega);
      if (stan::math::logical_negation((stan::math::primitive_value(
            emit_transformed_parameters__) || stan::math::primitive_value(
            emit_generated_quantities__)))) {
        return ;
      } 
      current_statement__ = 7;
      stan::model::assign(Tau, stan::math::quad_form_diag(omega, tau),
        "assigning variable Tau");
      current_statement__ = 6;
      stan::math::check_cov_matrix(function__, "Tau", Tau);
      if (emit_transformed_parameters__) {
        out__.write(Tau);
      } 
      if (stan::math::logical_negation(emit_generated_quantities__)) {
        return ;
      } 
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    } // write_array_impl() 
    
  template <typename VecVar, typename VecI, 
  stan::require_vector_t<VecVar>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr> 
  inline void transform_inits_impl(VecVar& params_r__, VecI& params_i__,
                                   VecVar& vars__,
                                   std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    
    try {
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      std::vector<Eigen::Matrix<local_scalar_t__, -1, 1>> beta =
         std::vector<Eigen::Matrix<local_scalar_t__, -1, 1>>(G, 
           Eigen::Matrix<local_scalar_t__, -1, 1>::Constant(M, DUMMY_VAR__));
      for (int sym1__ = 1; sym1__ <= M; ++sym1__) {
        for (int sym2__ = 1; sym2__ <= G; ++sym2__) {
          stan::model::assign(beta, in__.read<local_scalar_t__>(),
            "assigning variable beta", stan::model::index_uni(sym2__),
                                         stan::model::index_uni(sym1__));
        }
      }
      out__.write(beta);
      Eigen::Matrix<local_scalar_t__, -1, 1> beta_fix =
         Eigen::Matrix<local_scalar_t__, -1, 1>::Constant(M, DUMMY_VAR__);
      for (int sym1__ = 1; sym1__ <= M; ++sym1__) {
        stan::model::assign(beta_fix, in__.read<local_scalar_t__>(),
          "assigning variable beta_fix", stan::model::index_uni(sym1__));
      }
      out__.write(beta_fix);
      local_scalar_t__ s_y = DUMMY_VAR__;
      s_y = in__.read<local_scalar_t__>();
      out__.write_free_lb(0, s_y);
      Eigen::Matrix<local_scalar_t__, -1, 1> tau =
         Eigen::Matrix<local_scalar_t__, -1, 1>::Constant(M, DUMMY_VAR__);
      for (int sym1__ = 1; sym1__ <= M; ++sym1__) {
        stan::model::assign(tau, in__.read<local_scalar_t__>(),
          "assigning variable tau", stan::model::index_uni(sym1__));
      }
      out__.write_free_lb(0, tau);
      Eigen::Matrix<local_scalar_t__, -1, -1> omega =
         Eigen::Matrix<local_scalar_t__, -1, -1>::Constant(M, M, DUMMY_VAR__);
      for (int sym1__ = 1; sym1__ <= M; ++sym1__) {
        for (int sym2__ = 1; sym2__ <= M; ++sym2__) {
          stan::model::assign(omega, in__.read<local_scalar_t__>(),
            "assigning variable omega", stan::model::index_uni(sym2__),
                                          stan::model::index_uni(sym1__));
        }
      }
      out__.write_free_corr_matrix(omega);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    } // transform_inits_impl() 
    
  inline void get_param_names(std::vector<std::string>& names__) const {
    
    names__ = std::vector<std::string>{"beta", "beta_fix", "s_y", "tau",
      "omega", "Tau"};
    
    } // get_param_names() 
    
  inline void get_dims(std::vector<std::vector<size_t>>& dimss__) const {
    
    dimss__ = std::vector<std::vector<size_t>>{std::vector<size_t>{
                                                                   static_cast<size_t>(G)
                                                                   ,
                                                                   static_cast<size_t>(M)
                                                                   },
      std::vector<size_t>{static_cast<size_t>(M)}, std::vector<size_t>{
      }, std::vector<size_t>{static_cast<size_t>(M)},
      std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(M)},
      std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(M)}};
    
    } // get_dims() 
    
  inline void constrained_param_names(
                                      std::vector<std::string>& param_names__,
                                      bool emit_transformed_parameters__ = true,
                                      bool emit_generated_quantities__ = true) const
    final {
    
    for (int sym1__ = 1; sym1__ <= M; ++sym1__) {
      {
        for (int sym2__ = 1; sym2__ <= G; ++sym2__) {
          {
            param_names__.emplace_back(std::string() + "beta" + '.' + std::to_string(sym2__) + '.' + std::to_string(sym1__));
          } 
        }
      } 
    }
    for (int sym1__ = 1; sym1__ <= M; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "beta_fix" + '.' + std::to_string(sym1__));
      } 
    }
    param_names__.emplace_back(std::string() + "s_y");
    for (int sym1__ = 1; sym1__ <= M; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "tau" + '.' + std::to_string(sym1__));
      } 
    }
    for (int sym1__ = 1; sym1__ <= M; ++sym1__) {
      {
        for (int sym2__ = 1; sym2__ <= M; ++sym2__) {
          {
            param_names__.emplace_back(std::string() + "omega" + '.' + std::to_string(sym2__) + '.' + std::to_string(sym1__));
          } 
        }
      } 
    }
    if (emit_transformed_parameters__) {
      for (int sym1__ = 1; sym1__ <= M; ++sym1__) {
        {
          for (int sym2__ = 1; sym2__ <= M; ++sym2__) {
            {
              param_names__.emplace_back(std::string() + "Tau" + '.' + std::to_string(sym2__) + '.' + std::to_string(sym1__));
            } 
          }
        } 
      }
    }
    
    if (emit_generated_quantities__) {
      
    }
    
    } // constrained_param_names() 
    
  inline void unconstrained_param_names(
                                        std::vector<std::string>& param_names__,
                                        bool emit_transformed_parameters__ = true,
                                        bool emit_generated_quantities__ = true) const
    final {
    
    for (int sym1__ = 1; sym1__ <= M; ++sym1__) {
      {
        for (int sym2__ = 1; sym2__ <= G; ++sym2__) {
          {
            param_names__.emplace_back(std::string() + "beta" + '.' + std::to_string(sym2__) + '.' + std::to_string(sym1__));
          } 
        }
      } 
    }
    for (int sym1__ = 1; sym1__ <= M; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "beta_fix" + '.' + std::to_string(sym1__));
      } 
    }
    param_names__.emplace_back(std::string() + "s_y");
    for (int sym1__ = 1; sym1__ <= M; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "tau" + '.' + std::to_string(sym1__));
      } 
    }
    for (int sym1__ = 1; sym1__ <= ((M * (M - 1)) / 2); ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "omega" + '.' + std::to_string(sym1__));
      } 
    }
    if (emit_transformed_parameters__) {
      for (int sym1__ = 1; sym1__ <= (M + ((M * (M - 1)) / 2)); ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "Tau" + '.' + std::to_string(sym1__));
        } 
      }
    }
    
    if (emit_generated_quantities__) {
      
    }
    
    } // unconstrained_param_names() 
    
  inline std::string get_constrained_sizedtypes() const {
    
    return std::string("[{\"name\":\"beta\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(G) + ",\"element_type\":{\"name\":\"vector\",\"length\":" + std::to_string(M) + "}},\"block\":\"parameters\"},{\"name\":\"beta_fix\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(M) + "},\"block\":\"parameters\"},{\"name\":\"s_y\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"tau\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(M) + "},\"block\":\"parameters\"},{\"name\":\"omega\",\"type\":{\"name\":\"matrix\",\"rows\":" + std::to_string(M) + ",\"cols\":" + std::to_string(M) + "},\"block\":\"parameters\"},{\"name\":\"Tau\",\"type\":{\"name\":\"matrix\",\"rows\":" + std::to_string(M) + ",\"cols\":" + std::to_string(M) + "},\"block\":\"transformed_parameters\"}]");
    
    } // get_constrained_sizedtypes() 
    
  inline std::string get_unconstrained_sizedtypes() const {
    
    return std::string("[{\"name\":\"beta\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(G) + ",\"element_type\":{\"name\":\"vector\",\"length\":" + std::to_string(M) + "}},\"block\":\"parameters\"},{\"name\":\"beta_fix\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(M) + "},\"block\":\"parameters\"},{\"name\":\"s_y\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"tau\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(M) + "},\"block\":\"parameters\"},{\"name\":\"omega\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(((M * (M - 1)) / 2)) + "},\"block\":\"parameters\"},{\"name\":\"Tau\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string((M + ((M * (M - 1)) / 2))) + "},\"block\":\"transformed_parameters\"}]");
    
    } // get_unconstrained_sizedtypes() 
    
  
    // Begin method overload boilerplate
    template <typename RNG>
    inline void write_array(RNG& base_rng,
                            Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
                            Eigen::Matrix<double,Eigen::Dynamic,1>& vars,
                            const bool emit_transformed_parameters = true,
                            const bool emit_generated_quantities = true,
                            std::ostream* pstream = nullptr) const {
      const size_t num_params__ = 
  (((((G * M) + M) + 1) + M) + (M * M));
      const size_t num_transformed = emit_transformed_parameters * 
  (M * M);
      const size_t num_gen_quantities = emit_generated_quantities * 0;
      const size_t num_to_write = num_params__ + num_transformed +
        num_gen_quantities;
      std::vector<int> params_i;
      vars = Eigen::Matrix<double, Eigen::Dynamic, 1>::Constant(num_to_write,
        std::numeric_limits<double>::quiet_NaN());
      write_array_impl(base_rng, params_r, params_i, vars,
        emit_transformed_parameters, emit_generated_quantities, pstream);
    }

    template <typename RNG>
    inline void write_array(RNG& base_rng, std::vector<double>& params_r,
                            std::vector<int>& params_i,
                            std::vector<double>& vars,
                            bool emit_transformed_parameters = true,
                            bool emit_generated_quantities = true,
                            std::ostream* pstream = nullptr) const {
      const size_t num_params__ = 
  (((((G * M) + M) + 1) + M) + (M * M));
      const size_t num_transformed = emit_transformed_parameters * 
  (M * M);
      const size_t num_gen_quantities = emit_generated_quantities * 0;
      const size_t num_to_write = num_params__ + num_transformed +
        num_gen_quantities;
      vars = std::vector<double>(num_to_write,
        std::numeric_limits<double>::quiet_NaN());
      write_array_impl(base_rng, params_r, params_i, vars,
        emit_transformed_parameters, emit_generated_quantities, pstream);
    }

    template <bool propto__, bool jacobian__, typename T_>
    inline T_ log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r,
                       std::ostream* pstream = nullptr) const {
      Eigen::Matrix<int, -1, 1> params_i;
      return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
    }

    template <bool propto__, bool jacobian__, typename T__>
    inline T__ log_prob(std::vector<T__>& params_r,
                        std::vector<int>& params_i,
                        std::ostream* pstream = nullptr) const {
      return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
    }


    inline void transform_inits(const stan::io::var_context& context,
                         Eigen::Matrix<double, Eigen::Dynamic, 1>& params_r,
                         std::ostream* pstream = nullptr) const final {
      std::vector<double> params_r_vec(params_r.size());
      std::vector<int> params_i;
      transform_inits(context, params_i, params_r_vec, pstream);
      params_r = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,1>>(
        params_r_vec.data(), params_r_vec.size());
    }

  inline void transform_inits(const stan::io::var_context& context,
                              std::vector<int>& params_i,
                              std::vector<double>& vars,
                              std::ostream* pstream__ = nullptr) const {
     constexpr std::array<const char*, 5> names__{"beta", "beta_fix", "s_y",
      "tau", "omega"};
      const std::array<Eigen::Index, 5> constrain_param_sizes__{(G * M), 
       M, 1, M, (M * M)};
      const auto num_constrained_params__ = std::accumulate(
        constrain_param_sizes__.begin(), constrain_param_sizes__.end(), 0);
    
     std::vector<double> params_r_flat__(num_constrained_params__);
     Eigen::Index size_iter__ = 0;
     Eigen::Index flat_iter__ = 0;
     for (auto&& param_name__ : names__) {
       const auto param_vec__ = context.vals_r(param_name__);
       for (Eigen::Index i = 0; i < constrain_param_sizes__[size_iter__]; ++i) {
         params_r_flat__[flat_iter__] = param_vec__[i];
         ++flat_iter__;
       }
       ++size_iter__;
     }
     vars.resize(num_params_r__);
     transform_inits_impl(params_r_flat__, params_i, vars, pstream__);
    } // transform_inits() 
    
};
}

using stan_model = mixed_model_multivariable_model_namespace::mixed_model_multivariable_model;

#ifndef USING_R

// Boilerplate
stan::model::model_base& new_model(
        stan::io::var_context& data_context,
        unsigned int seed,
        std::ostream* msg_stream) {
  stan_model* m = new stan_model(data_context, seed, msg_stream);
  return *m;
}

stan::math::profile_map& get_stan_profile_data() {
  return mixed_model_multivariable_model_namespace::profiles__;
}

#endif


