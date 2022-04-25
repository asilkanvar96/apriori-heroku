from fastapi import FastAPI
from pydantic import BaseModel
from apriori_flow import *
from mlxtend.frequent_patterns import apriori, association_rules
import uvicorn
import json


class Data(BaseModel):
    Rules: str
    Surface_Crack: str
    Support: float
    Factor_Influence: float

app = FastAPI()

@app.get("/insights")
def insights(data: str):
    return{"data": "test"}

@app.post("/apriori")
def apriori( data:Data,a: str, limit, published: bool):
    path = r"C:\Users\User\PycharmProjects\ARL_Apriori_OCI_Test\dataset\Dataset_sample-3_fs.csv"

    data_fs = df_preprocessing(path)
    final_df = one_hot_encoder(data_fs, cols)
    rules_apriori = create_rules(final_df)
    insights_df = insights_filter(rules_apriori, pd.DataFrame(columns=["Rules", "Surface_Crack", "Support", "Factor_Influence"]),
                    a)
    data = insights_df.to_dict()
    rules = data["Rules"]
    surface_crack = data["Surface_Crack"]
    support = data["Support"]
    factor_influence = data["Factor_Influence"]
    if published:
        return{"insights": data}
    else:
        return {"insights_2": data}

data_fs = df_preprocessing(path)
final_df = one_hot_encoder(data_fs, cols)
rules_apriori = create_rules(final_df)
@app.get("/ARL")
def apriori( a: str, limit, published: bool):
    path = r"C:\Users\User\PycharmProjects\ARL_Apriori_OCI_Test\dataset\Dataset_sample-3_fs.csv"

    insights_df = insights_filter(rules_apriori, pd.DataFrame(columns=["Rules", "Condition", "Support", "Factor_Influence"]),
                    a)
    list_dict = []

    for index, row in list(insights_df.iterrows()):
        list_dict.append(dict(row))
    if published:
        return{"insights": list_dict}
    else:
        return {"insights_2": list_dict}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn fast_api:app --reload