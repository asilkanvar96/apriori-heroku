import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import OneHotEncoder
pd.set_option('display.max_columns', None)
pd.options.display.max_colwidth = 200

path = r"C:\Users\User\PycharmProjects\ARL_Apriori_OCI_Test\dataset\Dataset_sample-3_fs.csv"

def df_preprocessing(df_path):
    data = pd.read_csv(df_path)

    data.fillna(0, inplace=True)

    bins = [0, 0.1, 0.3, 1]
    labels =["On Target","Within Upper Limit","Above Upper Limit"]
    data['Surface_Crack'] = pd.cut(data['WIP Quality Result (Average) (Item: G15-Pinion Gear Operation: 20 (TST) Quality Element: Surface Crack (mm))'], bins,labels=labels)
    len(data.Surface_Crack.unique())
    data.Surface_Crack.nunique()

    data = data.drop(["WIP Quality Result (Average) (Item: G15-Pinion Gear Operation: 20 (TST) Quality Element: Surface Crack (mm))",
                     "WIP Quality Result (Average) (Item: G15-Pinion Gear Operation: 20 (TST) Quality Element: Tooth Depth (mm))",
                     "WIP Quality Result (Average) (Item: G15-Pinion Gear Operation: 20 (TST) Quality Element: Gear Hardness)",
                     "Work Order First Pass Yield (Work Order Yield)",
                     "Operation First Pass Yield (Operation: 20 (TST) Department: Testing)",
                     ], axis = 1)

    columns = [0, 70, 71, 72, 73, 44, 74]
    data_fs = data.iloc[:, columns]
    data_fs.set_index("Work Order", inplace=True)

    data_fs["Internal_Temperature"] = pd.qcut(data_fs[
                                                  "Average (Operation: 10 (3DP) Department: Machining Equipment: 3D-PRINT Parameter: OW_Internal Temperature Time segment: Full)"],
                                              3)
    data_fs["Vibration"] = pd.qcut(data_fs[
                                       "Maximum (Operation: 10 (3DP) Department: Machining Equipment: 3D-PRINT Parameter: OW_Vibration Time segment: Full)"],
                                   3)
    data_fs["Tensile_Strength"] = pd.qcut(data_fs[
                                              "Component Quality Result (Weighted Average) (Item: 3DP Main Mtl Quality Element: Comp Tensile Strength)"],
                                          3)
    data_fs["AISI_Grade"] = pd.qcut(
        data_fs["Component Quality Result (Weighted Average) (Item: 3DP Main Mtl Quality Element: AISI Grade)"], 3,
        duplicates="drop")

    data_fs.drop([
                     "Average (Operation: 10 (3DP) Department: Machining Equipment: 3D-PRINT Parameter: OW_Internal Temperature Time segment: Full)",
                     "Maximum (Operation: 10 (3DP) Department: Machining Equipment: 3D-PRINT Parameter: OW_Vibration Time segment: Full)",
                     "Component Quality Result (Weighted Average) (Item: 3DP Main Mtl Quality Element: Comp Tensile Strength)",
                     "Component Quality Result (Weighted Average) (Item: 3DP Main Mtl Quality Element: AISI Grade)"
                     ], axis=1, inplace=True)
    return data_fs

data_fs = df_preprocessing(path)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

#binary_cols = [col for col in data.columns if data[col].dtype not in [int, float] and data[col].nunique() == 2]

#for col in binary_cols:
#	data = label_encoder(data, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

cols = ["Surface_Crack", "Internal_Temperature", "Vibration", "Tensile_Strength", "AISI_Grade",
        "Supplier of Lot (Operation: 10 (3DP) Department: Machining Component: 3DP Support Mtl)"]
final_df = one_hot_encoder(data_fs, cols)


def create_rules(dataframe):

    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

rules_apriori = create_rules(final_df)

insights = []
insights_df = pd.DataFrame(columns=["Rules", "Condition", "Support", "Factor_Influence"])

def insights_filter(rules_df,dataframe, condition = "Surface_Crack_On Target", support = 0.05, confidence = 0.5, lift = 2):
    sorted_rules = rules_df[(rules_df["support"] > support) & (rules_df["confidence"] > confidence) & (rules_df["lift"] > lift)]. \
        sort_values("confidence", ascending=False)

    for i, rules in enumerate(sorted_rules["consequents"]):
        for j in list(rules):
            if j == condition:
                insights.append(list(sorted_rules.iloc[i]["antecedents"]))
                dataframe = dataframe.append({"Rules": list(sorted_rules.iloc[i]["antecedents"]),
                                              "Condition": j,
                                              "Support": sorted_rules.iloc[i]["support"],
                                              "Factor_Influence": sorted_rules.iloc[i]["confidence"]},
                                             ignore_index = True)
    return dataframe
