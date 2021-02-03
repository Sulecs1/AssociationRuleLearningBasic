############################################################
# Customized Product Recommendations Development Project   #
############################################################
#<<<Şule AKÇAY>>>

# Amacımız retail_II veri setine birliktelik analizi uygulamak.

# 1. Veri Ön İşleme
#   1. Eksik değer, aykırı değer vs (rfm'deki klasik işler)
#   2. Invoice product (basket product) matrisini oluşturmak
# 2. Birliktelik Kurallarnın Çıkarılması

import datetime as dt
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_columns', None)

#veri okuma işlemi
data = pd.read_excel(r"C:\Users\Suleakcay\PycharmProjects\pythonProject6\datasets\online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = data.copy()
df.info()


########################
#Data Preprocessing
########################


from helpers.helpers import check_df
check_df(df) #verinin detayını aldık
from helpers.helpers import crm_data_prep
#eksik değerleri uçurma,düzeltme işlemi ve hsaplana yaptık

df = crm_data_prep(df)
check_df(df) #veri temizleme işlemini gerçekleştirdik

df_fr = df[df['Country'] == "Germany"]
check_df(df_fr) #Sadece Germany için bilgi işlemlerini gerçekleştridm
#shape (541910,8)
#NA Description -> 1454
#Customer ID ->135080

#invoiceları tekilleştirdik Quantity lere göre sum larını aldım(herbir faturada ne kadar ürün olduğu
#burada her ürün tekilleşti faturalar için bir şey diyemeyeceğiz
df_fr.groupby(['Invoice', 'StockCode']).agg({"Quantity": "sum"}).head(200)
#unstack() ->Dizinleri eşitleme işlemi yapar
df_fr.groupby(['Invoice', 'StockCode']).agg({"Quantity": "sum"}).unstack().iloc[0:20, 0:20]
#Doğrulama işlemi gerçekleştirildi bu bize Quantityi gösterir
df[(df["StockCode"] == 16016) & (df["Invoice"] == 536983)]
df[(df["StockCode"] == 16235) & (df["Invoice"] == 538174)]
df[(df["StockCode"] == 17003) & (df["Invoice"] == 537894)]


#fiilna() -> boşluklara sıfır koyduk
df_fr.groupby(['Invoice', 'StockCode']).\
    agg({"Quantity": "sum"}).\
    unstack().fillna(0).iloc[0:5, 0:5] #nan değerlerine 0 koyduk yukarıdaki çıktıyı elde etmek için!

#Yukarıdaki quantity bir değer olarak değilde 1  ya da 0 olarak almak istediğimiz için aşağıdaki işemi yaptık
#df_fr.groupby(['Invoice', 'StockCode']).\
#   agg({"Quantity": "sum"}).\
#    unstack().fillna(0).\
#   applymap(lambda x: 1 if x > 0 else 0).iloc[0:20, 0:20]
#applymap() -> Tüm elemanlara çalışmasını istersek
#apply() ->satır ve sutun için çalışır :)

def create_invoice_product_df(dataframe):  #matrisi dataframe olarak aldık
    #StockCode
    return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
        applymap(lambda x: 1 if x > 0 else 0)



germany_inv_pro_df = create_invoice_product_df(df_fr)
#!!!! 1  ve 0 lar bir sepette ne kadar ürün olup olmadığını gösteriyor
germany_inv_pro_df.head(220)


############################################
# Birliktelik Kurallarının Çıkarılması
############################################
#apriori fonskiynu bize itemlerin frekanslarını verecek #supportları hesapladık
frequent_itemsets = apriori(germany_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False)

#supportlara göre, min_thresholda göre kuralları çıkardık
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()
rules.sort_values("lift", ascending=False).head()


############################################
# Çalışmanın Fonksiyonlaştırılması
############################################


import pandas as pd
pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules
from helpers.helpers import crm_data_prep, create_invoice_product_df  #istenilen fonksiyonlar

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

df = crm_data_prep(df)

def create_rules(dataframe, country=False, head=5):
    if country:
        dataframe = dataframe[dataframe['Country'] == country]
        dataframe = create_invoice_product_df(dataframe)
        frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01) #birliktelik kurallarını çıkar
        print(rules.sort_values("lift", ascending=False).head(head))
    else:
        dataframe = create_invoice_product_df(dataframe)
        frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
        print(rules.sort_values("lift", ascending=False).head(head))

    return rules


rules = create_rules(df)