import pandas as pd 

def build_set():
    sheetNames = ['HC_test_(rfel)', 'PwD_test_(rfel)', 'HC_train_(rfel)', 'PwD_train_(rfel)']
    master_df_list = []
    for sheetName in sheetNames:
        print(f'Processing sheet: {sheetName}')
        workingDF = pd.read_excel('SupplementaryTableS1.xlsx', sheet_name=sheetName, header=2)
        workingDF = workingDF.iloc[:, 1:]
        depression = 1 if 'PwD' in sheetName else 0
        workingDF.insert(0, 'PwD', depression)
        master_df_list.append(workingDF)
    master_df = pd.concat(master_df_list, ignore_index=True)
    master_df.fillna(0, inplace=True)
    master_df.to_csv('master_set.csv', index=False)

def build_study1_set():
    columns_to_drop = ['Samples', 'Median HC', 'Median PwD', 'Wilcoxon', 'FDR']
    df = pd.read_excel('SupplementaryTableS6.xlsx', sheet_name='Supplementary Table S6b', header=1)
    filtered_df = df.drop(columns_to_drop, axis=1)
    transposed_df = filtered_df.transpose()
    transposed_df.reset_index(inplace=True)
    transposed_df.iloc[0, 0] = 'PwD'
    transposed_df.iloc[1:, 0] = transposed_df.iloc[1:, 0].apply(lambda x: 0 if 'HC' in str(x) else 1)
    transposed_df.to_csv('study1_master.csv',index=False, header=False)

def build_presplit_sets():
    setDict = {'train':[], 'test':[]}
    for setSplit in setDict.keys():
        for sheetName in [f'HC_{setSplit}_(rfel)', f'PwD_{setSplit}_(rfel)']:
            df = pd.read_excel('SupplementaryTableS1.xlsx', sheet_name=sheetName, header=2)
            df = df.iloc[:,1:]
            depression = 1 if 'PwD' in sheetName else 0
            df.insert(0, 'PwD', depression)
            setDict[setSplit].append(df)
        setDict[setSplit] = pd.concat(setDict[setSplit], ignore_index=True)
        setDict[setSplit].fillna(0, inplace=True)
        setDict[setSplit].to_csv(f'{setSplit}_master.csv', index=False)

def build_presplit_from_study1():
    columns_to_drop = ['Samples', 'Median HC', 'Median PwD', 'Wilcoxon', 'FDR']
    df = pd.read_excel('SupplementaryTableS6.xlsx', sheet_name='Supplementary Table S6b', header=1)
    filtered_df = df.drop(columns_to_drop, axis=1)
    transposed_df = filtered_df.transpose()
    transposed_df.reset_index(inplace=True)
    setDict = {'train':[], 'test':[]}
    id_dict = {'test':{'HC':(31,40), 'PwD':(28,37)},'train':{'HC':(1,30), 'PwD':(2,27)}}
    for setSplit in setDict.keys():
        for class_attribute, rangeTuple in id_dict[setSplit].items():
            for index, row in transposed_df.iterrows():
                full_ID = row['index'].split('_')
                if len(full_ID)<2:
                    if class_attribute=='HC': #hack to get header
                        setDict[setSplit].append(row.to_dict())
                    continue
                current_class = full_ID[0]
                set_ID = int(full_ID[1])
                if current_class!=class_attribute:
                    continue
                start, end = rangeTuple[0], rangeTuple[1]+1
                if set_ID in range(start, end):
                    row['index']= 0 if class_attribute=='HC' else 1
                    setDict[setSplit].append(row.to_dict())
        set_df = pd.DataFrame(setDict[setSplit])
        set_df.fillna(0,inplace=True)
        set_df.iloc[0, 0] = 'PwD'
        set_df.to_csv(f'study1_{setSplit}.csv', index=False, header=False)  

def filter_sets_by_feature_importance():
    df = pd.read_csv('feature_importances.csv',header=None)
    feature_list = df[0][1:251].to_list() # save only the 250 highest-ranked features by importance
    feature_list.append('PwD')
    for setSplit in ['train', 'test']:
        set_df = pd.read_csv(f'study1_{setSplit}.csv')
        for column in set_df.columns:
            if column not in feature_list:
                set_df = set_df.drop(column, axis=1)
        set_df.to_csv(f'250F_study1_{setSplit}.csv', index=False, header=True)
    
def filter_set_by_feature_importance():
    df = pd.read_csv('feature_importances.csv',header=None)
    feature_list = df[0][1:251].to_list() # save only the 250 highest-ranked features by importance
    feature_list.append('PwD')
    set_df = pd.read_csv(f'study1_master.csv')
    for column in set_df.columns:
        if column not in feature_list:
            set_df = set_df.drop(column, axis=1)
    set_df.to_csv(f'250F_study1_master.csv', index=False, header=True)

def build_study2_sets():
    setDict = {'train':[], 'test':[]}
    for setSplit in setDict.keys():
        for sheetName in [f'HC_{setSplit}_(rfel)', f'PwD_{setSplit}_(rfel)']:
            df = pd.read_excel('SupplementaryTableS1.xlsx', sheet_name=sheetName, header=2)
            df = df.iloc[:,1:]
            depression = 1 if 'PwD' in sheetName else 0
            df.insert(0, 'PwD', depression)
            setDict[setSplit].append(df)
        setDict[setSplit] = pd.concat(setDict[setSplit], ignore_index=True, join='inner')
        setDict[setSplit].fillna(0, inplace=True)
        setDict[setSplit].to_csv(f'study2_{setSplit}.csv', index=False)

def build_study2_master_set():
    sheetNames = ['HC_test_(rfel)', 'PwD_test_(rfel)', 'HC_train_(rfel)', 'PwD_train_(rfel)']
    master_df_list = []
    for sheetName in sheetNames:
        print(f'Processing sheet: {sheetName}')
        workingDF = pd.read_excel('SupplementaryTableS1.xlsx', sheet_name=sheetName, header=2)
        workingDF = workingDF.iloc[:, 1:]
        depression = 1 if 'PwD' in sheetName else 0
        workingDF.insert(0, 'PwD', depression)
        master_df_list.append(workingDF)
    master_df = pd.concat(master_df_list, ignore_index=True, join='inner')
    master_df.fillna(0, inplace=True)
    master_df.to_csv('study2_master.csv', index=False)

def build_yolo_master_set():
    sheetNames = ['HC_real_train_(yolo)', 'PwD_real_train_(yolo)', 'HC_syn_train_(yolo)', 'PwD_syn_train_(yolo)']
    master_df_list = []
    for sheetName in sheetNames:
        print(f'Processing sheet: {sheetName}')
        workingDF = pd.read_excel('SupplementaryTableS1.xlsx', sheet_name=sheetName, header=2)
        workingDF = workingDF.iloc[:, 1:]
        depression = 1 if 'PwD' in sheetName else 0
        workingDF.insert(0, 'PwD', depression)
        master_df_list.append(workingDF)
    master_df = pd.concat(master_df_list, ignore_index=True, join='inner')
    master_df.fillna(0, inplace=True)
    master_df.to_csv('study2_yolo_train_master.csv', index=False)

    sheetNames = ['HC_test_(yolo)', 'PwD_test_(yolo)']
    master_df_list = []
    for sheetName in sheetNames:
        print(f'Processing sheet: {sheetName}')
        workingDF = pd.read_excel('SupplementaryTableS1.xlsx', sheet_name=sheetName, header=2)
        workingDF = workingDF.iloc[:, 1:]
        depression = 1 if 'PwD' in sheetName else 0
        workingDF.insert(0, 'PwD', depression)
        master_df_list.append(workingDF)
    master_df = pd.concat(master_df_list, ignore_index=True, join='inner')
    master_df.fillna(0, inplace=True)
    master_df.to_csv('study2_yolo_test_master.csv', index=False)