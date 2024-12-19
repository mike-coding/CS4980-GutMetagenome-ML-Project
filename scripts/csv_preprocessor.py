import pandas as pd 
import os
import xml.etree.ElementTree as et

class Processor:
    def __init__(self):
        self.path = self.get_wd_path()
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
        self.demographicData= pd.DataFrame()

    def get_wd_path(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        datasets_abs_path = os.path.join(project_root, 'datasets')
        return datasets_abs_path + os.sep

    def build_all_sets(self):
        print(f'Building all datasets. This may take a moment...')
        #study1
        processor.process_set(study=1)
        processor.process_set(study=1,split=True)
        processor.process_set(study=1, useSpecies=False)
        processor.process_set(study=1,split=True,useSpecies=False)
        processor.process_set(study=1,addDemoData=True)
        processor.process_set(study=1,split=True,addDemoData=True)
        processor.process_set(study=1, useSpecies=False,addDemoData=True)
        processor.process_set(study=1,split=True,useSpecies=False,addDemoData=True)

        #study2
        processor.process_set(study=2)
        processor.process_set(study=2,split=True)
        processor.process_set(study=2, split=True, useSynthetic=True)
        processor.process_set(study=2,addDemoData=True)
        processor.process_set(study=2,split=True,addDemoData=True)
        print('Datasets built.')

    def process_set(self, **kwargs):
        """
        kwargs:
            - study (int): data source- study 1 or 2
            - split (bool, optional): Whether we mirror the study 2 train/test split. Default False
            - useSynthetic (bool, optional): study 2-specific. Yolo data flag. Default False.
            - useSpecies (bool, optional): study 1-specific. genus/species flag. Default True.
            - addDemoData (bool, optional): include bioproject age & sex per subject. Default False
        """
        study=kwargs.get('study', 2)
        print(f'\nPreprocessing dataset for study {study}\n====================================')
        self.report_kwargs(**kwargs)
        if kwargs.get('addDemoData',False):
            self.build_demographic_data()
        if study==1:
            self.process_study1_set(**kwargs)
        else: #study 2
            self.process_study2_set(**kwargs)

    def process_study1_set(self, **kwargs):
        split= kwargs.get('split',False)
        useSpecies=kwargs.get('useSpecies',True)
        addDemoData=kwargs.get('addDemoData',False)

        self.load_sheets('study1')
        working_df=self.sheetDict['study1'][f'Supplementary Table S6{"b" if useSpecies else "a"}']
        working_df=working_df.drop(['Samples', 'Median HC', 'Median PwD', 'Wilcoxon', 'FDR'],axis=1)
        working_df=working_df.transpose()
        working_df.reset_index(inplace=True)
        if addDemoData:
                working_df = self.join_demo_data(working_df)
        if not split:
            working_df.iloc[0,0]='PwD'
            working_df.iloc[1:,0]=working_df.iloc[1:,0].apply(lambda x: 0 if 'HC' in str(x) else 1)
            self.write_csv(working_df,'full', False, **kwargs)
        else:
            frames=self.split_frame(working_df)
            for _set, df in frames.items():
                self.write_csv(df,_set,False,**kwargs)

    def split_frame(self, working_df):
        self.load_classic_splits()
        sets = {'test':[], 'train':[]}
        # find train/test IDs per row
        for index, row in working_df.iterrows():
            if len(sets['test'])<1 or len(sets['train'])<1: #handle header row
                sets['train'].append(row.to_dict())
                sets['test'].append(row.to_dict())
                continue
            try:
                full_ID = row['index']
            except:
                print("[ERROR]: Couldn't get full_ID from row['index'].....")
                print(row)
                quit()
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
        addDemoData=kwargs.get('addDemoData',False)
        sheet_block = 'study2_classic' if not useSynthetic else 'study2_yolo'
        self.load_sheets(sheet_block)
        if split:
            for _set in ['train', 'test']:
                df= self.concat_frames(sheet_block, _set, addDemoData)
                self.write_csv(df, _set, **kwargs)
        else: # fullset
            df = self.concat_frames(sheet_block,"_",addDemoData)
            self.write_csv(df,'full',**kwargs)

    def write_csv(self, df, _set, useHeaders=True, **kwargs): 
        '''
        build dataset name from parameters
        create subdirectories where necessary
        write dataFrame to csv in respective directory
        '''
        study_number = kwargs.get('study',2)
        csv_name='study'
        csv_name+=str(study_number)
        csv_path = os.path.join(self.path, 'processed', f"{csv_name}_sets")
        if study_number==2:
            csv_name+= '_yolo' if kwargs.get('useSynthetic',False) else '_classic'
        if not kwargs.get('useSpecies',True):
            csv_name+='_genus'
        csv_name = f"{csv_name}_{_set}"
        if kwargs.get('addDemoData',False):
            csv_name+='_demo'
            csv_path = os.path.join(csv_path, 'demographic_features')
        os.makedirs(csv_path, exist_ok=True)
        full_csv_string= os.path.join(csv_path, csv_name+'.csv')
        df.to_csv(full_csv_string,index=False,header=useHeaders)

    def report_kwargs(self, **kwargs):
        for k, v in kwargs.items():
            print(f'{k}: {v}')

    def concat_frames(self, sheet_block, _set, addDemoData):
        frames= [self.sheetDict[sheet_block][sheet] for sheet in self.sheetDict[sheet_block].keys() if _set in sheet]
        df = pd.concat(frames,ignore_index=True,join='inner')
        df.fillna(0, inplace=True)
        if addDemoData:
            df = self.join_demo_data(df)
        df.rename(columns={df.columns[0]:'PwD'}, inplace=True)
        df.iloc[0:,0]=df.iloc[0:,0].apply(lambda x: 0 if 'HC' in str(x) else 1)
        return df

    def join_demo_data(self, df):
        right_target=df.columns[0]
        df = self.demographicData.merge(df, right_on=right_target, left_on='ID', how='right')
        df=df.drop(df.columns[3],axis=1)
        if right_target=='index': #apply fixes for frames not using header (study 1)
            for pair in [(0,'PwD'),(1,'sex'),(2,'age')]:
                df.iloc[0,pair[0]] = pair[1]
            df.rename(columns={'ID': 'index'}, inplace=True)
        return df

    def load_classic_splits(self):
        if not self.study2_split_IDs['loaded']:
            self.load_sheets('study2_classic') 
            for split in self.make_keys_list(self.study2_split_IDs):
                for classType in self.classes:
                    df = self.sheetDict['study2_classic'][f'{classType}_{split}_(rfel)']
                    IDs = df.iloc[:, 0].to_list()
                    [self.study2_split_IDs[split].append(x) for x in IDs]
            self.study2_split_IDs['loaded']=True

    def load_sheets(self, sheetBlock):
        if not self.sheetDict[sheetBlock]['loaded']:
            print(f'Loading sheet block: {sheetBlock}')
            filename, header ='SupplementaryTableS6.xlsx', 1
            if '1' not in sheetBlock: # apply study 2 csv settings
                filename, header = 'SupplementaryTableS1.xlsx', 2
            sheetNames=self.make_keys_list(self.sheetDict[sheetBlock])
            for sheet in sheetNames:
                df = pd.read_excel(os.path.join(self.path, filename),sheet_name=sheet, header=header)
                self.sheetDict[sheetBlock][sheet]=df
            self.sheetDict[sheetBlock]['loaded']=True
            print('Sheets loaded.')

    def prompt_user_for_error(self, error):
        print(f'PREPROCESSOR ERROR OCCURRED: {error}')
        userChoice = input('Continue anyway? Y/N:\n')
        if userChoice.strip().upper() in ['N', 'NO']:
            exit()

    def make_keys_list(self, dict):
        key_list = list(dict.keys())
        key_list.remove('loaded')
        return key_list

    def build_demographic_data(self):
        demo_path = os.path.join(self.path, 'processed', 'demographic_data.csv')
        if self.demographicData.empty:
            if os.path.exists(demo_path):
                print('Loading demographic data.')
                self.demographicData = pd.read_csv(demo_path)
            else:
                print('Building demographic data from BioProject XML...')
                xml_path = os.path.join(self.path, 'BioProject_PRJNA762199_summary.xml')
                tree = et.parse(xml_path)
                root = tree.getroot()
                subject_dict={}
                for biosample in root.findall('BioSample'):
                    description = biosample.find('Description')
                    subject_ID = description.find('Title').text.replace('PD','PwD')
                    text = description.find('Comment').find('Paragraph').text
                    text = text.split('(')[1]
                    text = text.split(')')[0]
                    texts = text.split(',')
                    sex,age = texts[0],texts[1]
                    sex='F' if 'f' in sex.lower() else 'M'
                    age = "".join([x for x in age if x.isdigit()])
                    subject_dict[subject_ID]={'sex':sex,'age':age}
                self.demographicData=subject_dict
                print('Demographic data built.')
                df = pd.DataFrame(subject_dict)
                df = df.transpose()
                df.index.name = 'ID'
                df = df.reset_index()
                df.to_csv(demo_path,index=False)
                self.demographicData=df
        
if __name__ == "__main__":
    processor = Processor()
    processor.build_all_sets()