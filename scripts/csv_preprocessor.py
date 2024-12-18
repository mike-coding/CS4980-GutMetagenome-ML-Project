import pandas as pd 

class Processor:
    def __init__(self):
        self.path='../datasets/'
        self.csvPathDict={1:'SupplementaryTableS6.xlsx', 2:'SupplementaryTableS1.xlsx'}
        self.study1DropColumns=['Samples', 'Median HC', 'Median PwD', 'Wilcoxon', 'FDR']
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

    def processSet(self, study,split,useSynthetic):
        print('Preprocessing datasets\n====================')
        if study==1:
            if split:
                if len(self.study2_split_IDs['train'])<1:
                    self.load_classic_splits()
                pass
            else:
                pass

        else: #study2
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
        # default to study 1 csv settings
        filename=self.csvPathDict[1]
        header=1
        if '1' not in sheetBlock: # apply study 2 csv settings
            filename=self.csvPathDict[2]
            header=2
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
    processor = Processor()
    #Check which set user wants to generate
    processor.processSets()
    pass