import pandas as pd

from data import get_data

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer


ORDINAL_MAPPINGS = {
    'BsmtCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4},
    'BsmtQual': {'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
    'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
    'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'BsmtFinType2': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'Electrical': {'Mix': 0, 'FuseP': 1, 'FuseF': 2, 'FuseA': 3, 'SBrkr': 4},
    'ExterQual': {'Fa': 0, 'TA': 1, 'Gd': 2, 'Ex': 3},
    'ExterCond': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
    'Fence': {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4},
    'FireplaceQu': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'Functional': {'Sev': 0, 'Maj2': 1, 'Maj1': 2, 'Mod': 3, 'Min2': 4, 'Min1': 5, 'Typ': 6},
    'GarageCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
    'GarageQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'HeatingQC': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
    'KitchenQual': {'Fa': 0, 'TA': 1, 'Gd': 2, 'Ex': 3},
    'LandContour': {'Low': 0, 'Bnk': 1, 'HLS': 2, 'Lvl': 3},
    'LandSlope': {'Sev': 0, 'Mod': 1, 'Gtl': 2},
    'LotShape': {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3},
    'PavedDrive': {'N': 0, 'P': 1, 'Y': 2},
    'PoolQC': {'None': 0, 'Fa': 1, 'Gd': 2, 'Ex': 3},
    'Utilities': {'NoSeWa': 0, 'AllPub': 1},
}


def ordinally_encode(data):
    global ORDINAL_MAPPINGS

    data.replace(ORDINAL_MAPPINGS, inplace=True)

    return data

def build_pipeline():
    global ORDINAL_MAPPINGS

    #------- VARIABLES -------#

    # List of columns where nan has meaning
    nan_features = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 
        'GarageCond', 'GarageFinish', 'GarageQual', 'BsmtFinType2', 'BsmtExposure', 
        'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'MasVnrType'
    ]


    #------- PIPELINE -------#

    ordinal_encoder = Pipeline([
        ('ordinal_encoder', FunctionTransformer(ordinally_encode))
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown = 'ignore', sparse = False, drop = 'if_binary'))
    ])

    # ColumnTransformer to apply preprocessing pipelines
    preprocessor = ColumnTransformer([
        ('num_imputer', SimpleImputer(), ['LotFrontage', 'GarageYrBlt', 'MasVnrArea']), # REMINDER: numerical columns only had missing values in these columns
        ('num_scaler', RobustScaler(), make_column_selector(dtype_include = ['int', 'float'])),

        ('nan_imputer', SimpleImputer(strategy = 'constant', fill_value = 'None'), nan_features), # This takes care of categoricals except Electrical
        ('ord', ordinal_encoder, list(ORDINAL_MAPPINGS.keys())),
        ('cat', categorical_pipeline, make_column_selector(dtype_include = ['object'])) # This will apply the imputer to Electrical as it's the only one left with a NaN
    ], remainder = 'drop')

    final_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

    return final_pipe

def fit_pipeline():
    data = get_data()

    #------- VARIABLES -------#

    X = data.drop(columns = ['SalePrice'])
    y = data.SalePrice

    pipeline = build_pipeline()

    return pipeline.fit(X, y)

if __name__ == '__main__':
    fitted_pipeline = fit_pipeline()
    breakpoint()
