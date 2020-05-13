import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def get_vif(exogs, data):

    vif_dict, tolerance_dict = {}, {}

    for exog in exogs:
        not_exog = [i for i in exogs if i != exog]
        formula = f"{exog} ~ {' + '.join(not_exog)}"
        #print(formula)
        # extract r-squared from the fit
        r_squared = smf.ols(formula, data=data).fit().rsquared

        # calculate VIF
        vif = 1 / (1 - r_squared)
        vif_dict[exog] = vif

        # calculate tolerance
        tolerance = 1 - r_squared
        tolerance_dict[exog] = tolerance

        # return VIF DataFrame
    df_vif = pd.DataFrame({'VIF': vif_dict, 'Tolerance': tolerance_dict})

    return df_vif

#estimate linear regression
class linear_regression():

    def __init__(self, training, dep_var):

        self.training = training
        self.variable = list(training)

        self.dep_var = dep_var

    def regression(self):

        # perform unbiased estimation
        self.variable.remove(self.dep_var)

        # df_vif = get_vif(self.variable, self.training)
        #
        # while np.max(df_vif["VIF"]) >= 10 or len(np.isinf(df_vif["VIF"])[np.isinf(df_vif["VIF"]) == True])>0:
        #     for var, vf in zip(df_vif.index.values, df_vif["VIF"]):
        #         if vf==np.max(df_vif["VIF"]):
        #             self.variable.remove(var)
        #
        #     df_vif = get_vif(self.variable, self.training)


        linear_model = sm.OLS(self.training[self.dep_var], self.training[self.variable])
        final_model = linear_model.fit()


        # while len(np.isnan(final_model.pvalues)[np.isnan(final_model.pvalues) == True]) > 0:
        #
        #     for i in final_model.pvalues.index.get_values():
        #         if np.isnan(final_model.pvalues[i]):
        #
        #             self.variable.remove(i)
        #
        #             trainX = self.trainingX[self.variable]
        #             linear_model = sm.OLS(self.trainingY, trainX)
        #             final_model = linear_model.fit()
        #
        #
        #     if np.max(final_model.pvalues) > 0.1:
        #         rem_var = final_model.pvalues[final_model.pvalues ==
        #                                       np.max(final_model.pvalues)].index.get_values()
        #
        #         self.variable.remove(rem_var)
        #
        #         trainX = self.trainingX[self.variable]
        #         linear_model = sm.OLS(self.trainingY, trainX)
        #         final_model = linear_model.fit()


        return final_model, self.variable

