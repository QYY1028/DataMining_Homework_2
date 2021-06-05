import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

def perform_rule_calculation(transact_items_matrix, rule_type="fpgrowth", min_support=0.05):
    """
    desc: this function performs the association rule calculation 
    @params:
        - transact_items_matrix: the transaction X Items matrix
        - rule_type: 
                    - apriori or Growth algorithms (default="fpgrowth")
                    
        - min_support: minimum support threshold value (default = 0.001)
        
    @returns:
        - the matrix containing 3 columns:
            - support: support values for each combination of items
            - itemsets: the combination of items
            - number_of_items: the number of items in each combination of items
            
        - the excution time for the corresponding algorithm
        
    """
    start_time = 0
    total_execution = 0
    
    if(not rule_type=="fpgrowth"):
        start_time = time.time()
        rule_items = apriori(transact_items_matrix, 
                       min_support=min_support, 
                       use_colnames=True)
        total_execution = time.time() - start_time
        print("Computed Apriori!")
        
    else:
        start_time = time.time()
        rule_items = fpgrowth(transact_items_matrix, 
                       min_support=min_support, 
                       use_colnames=True).sort_values(by='support', ascending=False)
        total_execution = time.time() - start_time
        print("Computed Fp Growth!")
    
    rule_items['number_of_items'] = rule_items['itemsets'].apply(lambda x: len(x))
    
    return rule_items, total_execution

def compute_association_rule(rule_matrix, metric="lift", min_thresh=0.9):
    """
    @desc: Compute the final association rule
    @params:
        - rule_matrix: the corresponding algorithms matrix
        - metric: the metric to be used (default is lift)
        - min_thresh: the minimum threshold (default is 1)
        
    @returns:
        - rules: all the information for each transaction satisfying the given metric & threshold
    """
    rules = association_rules(rule_matrix, 
                              metric=metric, 
                              min_threshold=min_thresh)
    
    return rules

def plot_metrics_relationship(rule_matrix, col1, col2):
    """
    desc: shows the relationship between the two input columns 
    @params:
        - rule_matrix: the matrix containing the result of a rule (apriori or Fp Growth)
        - col1: first column
        - col2: second column
    """
    fit = np.polyfit(rule_matrix[col1], rule_matrix[col2], 1)
    fit_funt = np.poly1d(fit)
    plt.plot(rule_matrix[col1], rule_matrix[col2], 'yo', rule_matrix[col1], 
    fit_funt(rule_matrix[col1]))
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('{} vs {}'.format(col1, col2))



data_name = r'D:\研一下学期\数据挖掘\互评作业1\Wine reviews\winemag-data_first150k.csv'
file = pd.read_csv(data_name)
print(file.columns)
file_cp = file

print(file.columns)  
# Index(['Unnamed: 0', 'country', 'description', 'designation', 'points',
       # 'price', 'province', 'region_1', 'region_2', 'variety', 'winery'],
      # dtype='object')
      
file_cp = file.loc[file['country']=='US']

# file_cp.loc[(file_cp['country']!='France') & (file_cp['country']!='US'),'country']='other country'
# file_cp.loc[(file_cp['province']!='California') & (file_cp['province']!='Washington'),'province']='other province'

file_cp.loc[(file_cp['variety']!='Pinot Noir') & (file_cp['variety']!='Chardonnay')& (file_cp['variety']!='Cabernet Sauvignon')& (file_cp['variety']!='Red Blend')& (file_cp['variety']!='Bordeaux-style Red Blend'),'variety']='other variety'

file_cp.loc[file_cp['price']<=16,'price']=10
file_cp.loc[(file_cp['price']>16) & (file_cp['price']<=24.0),'price']=20
file_cp.loc[(file_cp['price']>24) & (file_cp['price']<=40.0),'price']=30
file_cp.loc[(file_cp['price']>30) & (file_cp['price']<=2301.0),'price']=100

file_cp.loc[(file_cp['price'].astype(str)=='10.0'),'price']='low price'
file_cp.loc[(file_cp['price'].astype(str)=='20.0'),'price']='medium low price'
file_cp.loc[(file_cp['price'].astype(str)=='30.0'),'price']='medium high price'
file_cp.loc[(file_cp['price'].astype(str)=='100.0'),'price']='high price'

file_cp.loc[:,'price']

file_cp.loc[:,'price']
dataset = []
att_list = ['points', 'price', 'province','variety']
value_dict={}

for idx,att in enumerate(att_list):
    for i,point in enumerate(file_cp.loc[:,att].astype(str)):
        
        if idx==0:
            dataset.append([point])
        else:
            dataset[i].append(point)
print(dataset[:10])
trans_encoder = TransactionEncoder() # Instanciate the encoder
trans_encoder_matrix = trans_encoder.fit(dataset).transform(dataset)
trans_encoder_matrix = pd.DataFrame(trans_encoder_matrix, columns=trans_encoder.columns_)

trans_encoder_matrix.head()

#频繁模式
fpgrowth_matrix, fp_growth_exec_time = perform_rule_calculation(trans_encoder_matrix)
fpgrowth_matrix.head()
fpgrowth_matrix.tail()

#关联规则
fp_growth_rule_lift = compute_association_rule(fpgrowth_matrix)
fp_growth_rule_lift.sort_values(by='confidence', ascending=False, inplace=True)  
fp_growth_rule_lift.head()

#可视化
plot_metrics_relationship(fp_growth_rule_lift, col1='lift', col2='confidence')

fp_growth_rule = compute_association_rule(fpgrowth_matrix, metric="confidence", min_thresh=0.2)
fp_growth_rule.head()
