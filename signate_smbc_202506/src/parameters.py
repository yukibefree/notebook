
class TrainingParameters:
    def __init__(self, category) -> None:
        self.category = category

        self.base_param = {
            'boosting_type': 'gbdt',
            'force_col_wise': 'true',
            'learning_rate': 0.01,
            'min_child_samples': 100,
            'n_estimators': 1000,
            'max_bin': 9999,
            'n_jobs': -1,
            'objective': 'poisson',
            'random_state': 1234,
            'lambda_l1': 1,
            'lambda_l2': 2,
            'feature_fraction': 0.2,
            'metric': 'rmse',
            'feval': None
        }
        self.param_rmse1 = {
            'boosting_type': 'gbdt',
            'force_col_wise': 'true',
            'learning_rate': 0.01,
            'min_child_samples': 100,
            'n_estimators': 1000,
            'max_bin': 9999,
            'n_jobs': -1,
            'objective': 'poisson',
            'random_state': 1234,
            'lambda_l1': 1,
            'lambda_l2': 2,
            'feature_fraction': 0.2,
            'metric': 'rmse',
            'feval': None,
        }
        self.param_mae1 = {
            'boosting_type': 'gbdt',
            'force_col_wise': 'true',
            'learning_rate': 0.01,
            'min_child_samples': 100,
            'n_estimators': 1000,
            'max_bin': 9999,
            'n_jobs': -1,
            'objective': 'poisson',
            'random_state': 1234,
            'lambda_l1': 1,
            'lambda_l2': 2,
            'feature_fraction': 0.2,
            'metric': 'mae',
            'feval': None
        }
        # ----- 以下未設定 -----
        self.param_2 = {
            'boosting_type': 'gbdt',
            'force_col_wise': 'true',
            'learning_rate': 0.01,
            'min_child_samples': 100,
            'n_estimators': 1000,
            'max_bin': 9999,
            'n_jobs': -1,
            'objective': 'poisson',
            'random_state': 1234,
            'lambda_l1': 1,
            'lambda_l2': 3,
            'feature_fraction': 0.2,
            'metric': None,
            'feval': None
        }
        self.param_3 = {
            'boosting_type': 'gbdt',
            'force_col_wise': 'true',
            'learning_rate': 0.01,
            'min_child_samples': 100,
            'n_estimators': 750,
            'max_bin': 9999,
            'n_jobs': -1,
            'objective': 'poisson',
            'random_state': 1234,
            'lambda_l1': 1,
            'lambda_l2': 2,
            'feature_fraction': 0.2,
            'metric': None,
            'feval': None
        }
        self.param_4 = {
            'boosting_type': 'gbdt',
            'force_col_wise': 'true',
            'learning_rate': 0.01,
            'min_child_samples': 100,
            'n_estimators': 1000,
            'max_bin': 9999,
            'n_jobs': -1,
            'objective': 'poisson',
            'random_state': 1234,
            'lambda_l1': 1,
            'lambda_l2': 2,
            'feature_fraction': 0.2,
            'metric': None,
            'feval': None
        }
        self.param_5 = {
            'boosting_type': 'gbdt',
            'force_col_wise': 'true',
            'learning_rate': 0.01,
            'min_child_samples': 100,
            'n_estimators': 1000,
            'max_bin': 9999,
            'n_jobs': -1,
            'objective': 'poisson',
            'random_state': 1234,
            'lambda_l1': 1,
            'lambda_l2': 1,
            'feature_fraction': 0.2,
            'metric': None,
            'feval': None
        }
        self.param_6 = {
            'boosting_type': 'gbdt',
            'force_col_wise': 'true',
            'learning_rate': 0.01,
            'min_child_samples': 200,
            'n_estimators': 1000,
            'max_bin': 9999,
            'n_jobs': -1,
            'objective': 'poisson',
            'random_state': 1234,
            'lambda_l1': 1,
            'lambda_l2': 1,
            'feature_fraction': 0.2,
            'metric': None,
            'feval': None,
            'force_col_wise': True
        }
        self.param_7 = {
            'boosting_type': 'gbdt',
            'force_col_wise': 'true',
            'learning_rate': 0.01,
            'min_child_samples': 100,
            'n_estimators': 750,
            'max_bin': 9999,
            'n_jobs': -1,
            'objective': 'rmse',
            'random_state': 1234,
            'lambda_l1': 1,
            'lambda_l2': 3,
            'feature_fraction': 0.2,
            'metric': None,
            'feval': None
        }
        self.param_8 = {
            'boosting_type': 'gbdt',
            'force_col_wise': 'true',
            'learning_rate': 0.01,
            'min_child_samples': 100,
            'n_estimators': 1000,
            'max_bin': 9999,
            'n_jobs': -1,
            'objective': 'poisson',
            'random_state': 1234,
            'lambda_l1': 1,
            'lambda_l2': 3,
            'feature_fraction': 0.2,
            'metric': 'rmse',
            'feval': None
        }
        self.param_10 = {
            'boosting_type': 'gbdt',
            'force_col_wise': 'true',
            'learning_rate': 0.01,
            'min_child_samples': 100,
            'n_estimators': 1000,
            'max_bin': 9999,
            'n_jobs': -1,
            'objective': 'poisson',
            'random_state': 1234,
            'lambda_l1': 1,
            'lambda_l2': 1,
            'feature_fraction': 0.2,
            'metric': None,
            'feval': None
        }
        self.param_44 = {
            'boosting_type': 'gbdt',
            'force_col_wise': 'true',
            'learning_rate': 0.01,
            'min_child_samples': 50,
            'n_estimators': 750,
            'max_bin': 9999,
            'n_jobs': -1,
            'objective': 'poisson',
            'random_state': 1234,
            'lambda_l1': 1,
            'lambda_l2': 3,
            'feature_fraction': 0.2,
            'metric': None,
            'feval': None
        }
        self.param_99 = {
            'boosting_type': 'gbdt',
            'force_col_wise': 'true',
            'learning_rate': 0.01,
            'min_child_samples': 50,
            'n_estimators': 500,
            'n_jobs': -1,
            'objective': 'tweedie',
            'random_state': 1234,
            'lambda_l1': 1,
            'lambda_l2': 3,
            'feature_fraction': 0.2,
            'metric': 'rmse',
            'feval': None
        }
        self.param_quantile = {
            'boosting_type': 'gbdt',
            'force_col_wise': 'true',
            'learning_rate': 0.01,
            'min_child_samples': 100,
            'n_estimators': 500,
            'max_bin': 9999,
            'n_jobs': -1,
            'objective': 'quantile',
            'alpha': 0.68,
            'random_state': 1234,
            'lambda_l1': 1,
            'lambda_l2': 1,
            'feature_fraction': 0.2,
            'feval': None
        }

    def get_params(self):
        if hasattr(self, f'param_{self.category}'):
            return getattr(self, f'param_{self.category}')
        else:
            return self.base_param