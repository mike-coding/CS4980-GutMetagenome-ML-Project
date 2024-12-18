import pandas as pd 
import os

class Processor:
    def __init__(self):
        self.path='../datasets/'
        self.classes=['HC','PwD']
        self.sheetDict={'study1':{'loaded':False,
                                  'Supplementary Table S6a':None, 
                                  'Supplementary Table S6b':None}, 
                        'study2_classic':{'loaded':False,
                                        'HC_test_(rfel)':None, 
                                        'PwD_test_(rfel)':None, 
                                        'HC_train_(rfel)':None, 
                                        'PwD_train_(rfel)':None},
                        'study2_yolo':{'loaded':False,
                                        'HC_real_train_(yolo)':None, 
                                        'PwD_real_train_(yolo)':None, 
                                        'HC_syn_train_(yolo)':None, 
                                        'PwD_syn_train_(yolo)':None,
                                        'HC_test_(yolo)':None, 
                                        'PwD_test_(yolo)':None}}
        self.study2_split_IDs={'loaded':False,'train':[],'test':[]}

    def processSet(self, **kwargs):
        """
        kwargs:
            - study (int): data source- study 1 or 2
            - split (bool): Whether we mirror the study 2 train/test split
            - useSynthetic (bool, optional): study 2-specific. Yolo data flag. Default False.
            - useSpecies (bool, optional): study 1-specific. genus/species flag. Default False.
        """
        study=kwargs.get('study', 2)
        print('Preprocessing datasets\n====================')
        #verify split IDs have been stored, collect them if not
        if study==1:
            self.load_sheets('study1')
            self.process_study1_set(**kwargs)
        else: #study 2
            self.process_study2_set(**kwargs)

    def process_study1_set(self, **kwargs):
        #unpack kwargs
        split= kwargs.get('split',False)
        useSpecies=kwargs.get('useSpecies',True)
        #get genus/species-level sheet
        working_df=self.sheetDict['study1'][f'Supplementary Table S6{"b" if useSpecies else "a"}']
        working_df=working_df.drop(['Samples', 'Median HC', 'Median PwD', 'Wilcoxon', 'FDR'],axis=1)
        working_df=working_df.transpose()
        working_df.reset_index(inplace=True)
        if not split: 
            working_df.iloc[0,0]='PwD'
            working_df.iloc[1:,0]=working_df.iloc[1:,0].apply(lambda x: 0 if 'HC' in str(x) else 1)
            working_df.to_csv(self.path+'processed/study1_full.csv',index=False, header=False)
        else:
            frames=self.perform_split_processing(working_df)
            for _set, df in frames.items():
                df.to_csv(self.path+f'/processed/study1_{_set}_{"species" if useSpecies else "genus"}.csv', index=False, header=False)

    def perform_split_processing(self, working_df):
        self.load_classic_splits()
        sets = {'test':[], 'train':[]}
        # iterate over rows & find train/test IDs
        for row in working_df.iterrows():
            full_ID = row['index']
            if '_' not in full_ID: #handle header row
                print(full_ID)
                row['index']='PwD'
                sets['train'].append(row.to_dict())
                sets['test'].append(row.to_dict())
                print(f'featurespace len: {len(row.to_dict())}')
                continue
            if full_ID in self.study2_split_IDs['train']:
                row['index']= 0 if 'HC' in full_ID else 1
                sets['train'].append(row.to_dict())
            elif full_ID in self.study2_split_IDs['test']:
                row['index']= 0 if 'HC' in full_ID else 1
                sets['test'].append(row.to_dict())
            else:
                self.prompt_user_for_error('study_1_split_row_no_class')
        frames={}
        for _set in sets.keys():
            df = pd.DataFrame(sets[_set])
            df.fillna(0,inplace=True)
            df.iloc[0,0]='PwD'
            frames[_set]=(df)
        return frames
             
    def process_study2_set(self, **kwargs):
        useSynthetic = kwargs.get('useSynthetic',False)
        split=kwargs.get('split',False)
        if not useSynthetic:
            self.load_sheets('study2_classic')
            if split: #get the IDs for each split if requested
                self.load_classic_splits()
                df_train = pd.concat()
            else: #fullset
                pass
        else: #yoloSets
            self.load_sheets('study2_yolo')
            if split: #get the IDs for each split if requested
                pass
            else: # fullset
                pass

    def load_classic_splits(self):
        if not self.study2_split_IDs['loaded']:
            self.load_sheets('study2_classic') 
            for split in self.make_keys_list(self.study2_split_IDs):
                for classType in self.classes:
                    df = self.sheetDict['study2_classic'][f'{classType}_{split}_(rfel)']
                    IDs = df.iloc[:, 0].to_list()
                    [self.study2_split_IDs[split].append(x) for x in IDs]
                    #self.study2_split_IDs[split].append(IDs)
            self.study2_split_IDs['loaded']=True

    def load_sheets(self, sheetBlock):
        if not self.sheetDict[sheetBlock]['loaded']:
            print(f'Loading sheet block: {sheetBlock}')
            filename, header ='SupplementaryTableS6.xlsx', 1
            if '1' not in sheetBlock: # apply study 2 csv settings
                filename, header = 'SupplementaryTableS1.xlsx', 2
            sheetNames=self.make_keys_list(self.sheetDict[sheetBlock])
            for sheet in sheetNames:
                df = pd.read_excel(self.path+filename,sheet_name=sheet, header=header)
                self.sheetDict[sheetBlock][sheet]=df
            self.sheetDict[sheetBlock]['loaded']=True
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

    def prompt_user_for_error(self, error):
        print(f'PREPROCESSOR ERROR OCCURRED: {error}')
        userChoice = input('Continue anyway? Y/N:\n')
        if userChoice.strip().upper() in ['N', 'NO']:
            exit()

    def make_keys_list(self, dict):
        key_list = list(dict.keys())
        key_list.remove('loaded')
        return key_list


if __name__ == "__main__":
    #assumes execution occurs from /scripts/
    processor = Processor()
    #Check which set user wants to generate
    processor.processSet(study=1,split=True)