import pandas as pd 
import os

class Processor:
    def __init__(self):
        self.path='../datasets/'
        self.classes=['HC','PwD']
        self.sheetDict={'study1':{'Supplementary Table S6a':None, 
                                  'Supplementary Table S6b':None}, 
                        'study2_classic':{'HC_test_(rfel)':None, 
                                        'PwD_test_(rfel)':None, 
                                        'HC_train_(rfel)':None, 
                                        'PwD_train_(rfel)':None},
                        'study2_yolo':{'HC_real_train_(yolo)':None, 
                                        'PwD_real_train_(yolo)':None, 
                                        'HC_syn_train_(yolo)':None, 
                                        'PwD_syn_train_(yolo)':None,
                                        'HC_test_(yolo)':None, 
                                        'PwD_test_(yolo)':None}}
        self.study2_split_IDs={'train':[],'test':[]}

    def processSet(self, study, split, useSynthetic=False, useSpecies=False):
        print('Preprocessing datasets\n====================')
        #verify split IDs have been stored, collect them if not
        if split and len(self.study2_split_IDs['train'])<1:
            self.load_classic_splits()
        if study==1:
            self.process_study1_set(split, useSpecies)
        else: #study 2
            self.process_study2_set(split, useSynthetic)


    def process_study1_set(self, split, useSpecies):
        study1DropColumns=['Samples', 'Median HC', 'Median PwD', 'Wilcoxon', 'FDR']
        #get study 1 sheets if we don't have them already
        if self.sheetDict['study1']['Supplementary Table S6a']==None:
            self.loadSheets('study1')
        workingDF=self.sheetDict['study1'][f'Supplementary Table S6{'b' if useSpecies else 'a'}']
        workingDF=workingDF.drop(study1DropColumns,axis=1)
        workingDF=workingDF.transpose()
        workingDF.reset_index(inplace=True)
        if split: 

            train=[]
            test=[]
            # iterate over rows & find train/test IDs
            for index, row in workingDF.iterrows():
                full_ID = row['index']
                if '_' not in full_ID:
                    train.append(row.to_dict())
                    test.append(row.to_dict())
                    continue
                
        else:
            workingDF.iloc[0,0]='PwD'
            workingDF.iloc[1:,0]=workingDF.iloc[1:,0].apply(lambda x: 0 if 'HC' in str(x) else 1)
            workingDF.to_csv(self.path+'processed/study1_full.csv',index=False, header=False)

    def process_study2_set(self,split,useSynthetic):
        if not useSynthetic:
            if split: #get the IDs for each split if requested
                pass
            else: #fullset
                pass
        else: #yoloSets
            if split: #get the IDs for each split if requested
                pass
            else: # fullset
                pass

    def load_classic_splits(self):
        if not self.sheetDict['study2_classic']['HC_test_(rfel)']: #load the study2 classic sheets if not already
            self.loadSheets('study2_classic')
        for split in self.study2_split_IDs.keys():
            for classType in self.classes:
                df = self.sheetDict['study2_classic'][f'{classType}_{split}_(rfel)']
                IDs = df.iloc[:, 0].to_list()
                self.study2_split_IDs[split].append(IDs)

    def loadSheets(self, sheetBlock):
        print(f'Loading sheet block: {sheetBlock}')
        filename, header ='SupplementaryTableS6.xlsx', 1
        if '1' not in sheetBlock: # apply study 2 csv settings
            filename, header = 'SupplementaryTableS1.xlsx', 2
        sheetNames=self.sheetDict[sheetBlock].keys()
        for sheet in sheetNames:
            df = pd.read_excel(self.path+filename,sheet_name=sheet, header=header)
            self.sheetDict[sheetBlock][sheet]=df
        print('Sheets loaded.')

    def protoProcessor():
        sheetNames = []
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

if __name__ == "__main__":
    #assumes execution occurs from /scripts/
    processor = Processor()
    #Check which set user wants to generate
    processor.processSet(1,False,False)