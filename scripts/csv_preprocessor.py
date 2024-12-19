import pandas as pd 
import os
import json
import xml.etree.ElementTree as et

class DataPreprocessor:
    def __init__(self):
        self.path = self.get_data_path()
        self.classes=['HC','PwD']
        self.valid_experiments={1:['classic', 'classic_demo','genus','genus_demo'],2:['classic', 'demo', 'yolo']}
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
    # ============================
    # Prequesite Loading / Building
    # ============================
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
                    sex=0 if 'f' in sex.lower() else 1
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

    # ============================
    # Processing
    # ============================  
    def process_set(self, **kwargs):
        """
        kwargs:
            - study (int): data source- study 1 or 2
            - split (bool, optional): Whether we mirror the study 2 train/test split. Default False
            - useSynthetic (bool, optional): study 2-specific. Yolo data flag. Default False.
            - useSpecies (bool, optional): study 1-specific. genus/species flag. Default True.
            - addDemoData (bool, optional): include bioproject age & sex per subject. Default False
        """
        if self.already_built(**kwargs):
            return
        study=kwargs.get('study', 2)
        print(f'\nPreprocessing dataset\n====================================')
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

    def make_keys_list(self, dict):
        key_list = list(dict.keys())
        key_list.remove('loaded')
        return key_list

    # ============================
    # Export / Interact
    # ============================    
    def write_csv(self, df, _set, useHeaders=True, **kwargs):
        """
        Write DataFrame to CSV in the directory structure:
        datasets/processed/study{study_number}/{experiment_type}/{_set}.csv

        Parameters:
        - df (pd.DataFrame): The DataFrame to save.
        - _set (str): One of 'full', 'test', or 'train'.
        - useHeaders (bool): Whether to include header in CSV.
        - **kwargs: Additional parameters to determine study number and experiment type.
        """
        study_number = kwargs.get('study', 2)
        experiment_type = self.assign_experiment_type(**kwargs)

        # Construct the directory path
        csv_path = os.path.join(self.path, 'processed', f"study{study_number}", experiment_type)
        os.makedirs(csv_path, exist_ok=True)

        # The CSV filename is now simply the set name (full.csv, test.csv, or train.csv)
        csv_filename = f"{_set}.csv"
        full_csv_path = os.path.join(csv_path, csv_filename)

        df.to_csv(full_csv_path, index=False, header=useHeaders)
        print(f"Saved: {full_csv_path}")
        self.update_build_receipt(study_number, experiment_type, _set)

    def assign_experiment_type(self, **kwargs):
        """
        Parameters:
            - study (int): 1 or 2 (defaults to 2)
            - useSpecies (bool): For study 1; genus/species flag. Default True means species-level => 'classic', False => 'genus'.
            - addDemoData (bool): If True, adds '_demo' suffix to experiment type.
            - useSynthetic (bool): For study 2; if True => 'yolo', else 'classic' or 'demo'.

        Returns:
            str: experiment_type (e.g., 'classic', 'genus_demo', 'yolo', 'demo', etc.)
        """
        study_number = kwargs.get('study', 2)
        useSpecies = kwargs.get('useSpecies', True)
        addDemoData = kwargs.get('addDemoData', False)
        useSynthetic = kwargs.get('useSynthetic', False)

        if study_number == 1:
            experiment_type = 'classic'
            if not useSpecies:
                experiment_type = 'genus'
            if addDemoData:
                experiment_type +="_demo"
        else:
            if useSynthetic:
                experiment_type = 'yolo'
            else:
                experiment_type = 'classic'
                if addDemoData:
                    experiment_type = 'demo'
                    

        return experiment_type

    def experiment_type_to_kwargs(self, study: int, experiment_type: str):
        """
        Study 1:
          - classic => useSpecies=True, addDemoData=False
          - genus => useSpecies=False, addDemoData=False
          - classic_demo => useSpecies=True, addDemoData=True
          - genus_demo => useSpecies=False, addDemoData=True

        Study 2:
          - classic => useSynthetic=False, addDemoData=False
          - demo => useSynthetic=False, addDemoData=True
          - yolo => useSynthetic=True
        """
        addDemoData = 'demo' in experiment_type
        if study==1:
            useSpecies=True
            if 'genus' in experiment_type:
                useSpecies = False
            return {
                'study': study,
                'useSpecies': useSpecies,
                'addDemoData': addDemoData,
                }
        else:
            useSynthetic=False
            if experiment_type == 'yolo':
                useSynthetic = True
                addDemoData = False
            return {
                'study': study,
                'useSynthetic': useSynthetic,
                'addDemoData': addDemoData,
                }

    def get_experiment(self, experiment_type: str, study: int = 2):
        """
        Check if the experiment sets exist, if not build them.
        Then load them into DataFrames and return.

        Returns:
            dict: { 'full': df_full, 'test': df_test, 'train': df_train } 
                  depending on availability.
        """
        
        if type(study)!=int:
            print(f'[ERROR] Attempted to load experiment with invalid study: {study}')
            quit()
        elif experiment_type not in self.valid_experiments[study]:
            print(f'[ERROR] Attempted to load experiment with invalid experiment: {experiment_type}')
            quit()
        kwargs = self.experiment_type_to_kwargs(study, experiment_type)
        kwargs['experiment_type'] = experiment_type

        if not self.already_built(**kwargs):
            print(f"Experiment sets not built for study{study}/{experiment_type}. Building now...")
            self.process_set(**kwargs)
            kwargs['split']=True
            self.process_set(**kwargs)
        base_path = os.path.join(self.path, 'processed', f"study{study}", experiment_type)
        dataframes = {}
        candidates = ['full', 'test', 'train']
        for c in candidates:
            csv_file = os.path.join(base_path, f"{c}.csv")
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                dataframes[c] = df
        return dataframes

    # ============================
    # Debug , Reporting , Safeguards
    # ============================
    def prompt_user_for_error(self, error):
        print(f'PREPROCESSOR ERROR OCCURRED: {error}')
        userChoice = input('Continue anyway? Y/N:\n')
        if userChoice.strip().upper() in ['N', 'NO']:
            exit()

    def report_kwargs(self, **kwargs):
        for k, v in kwargs.items():
            print(f'{k}: {v}')
    
    def update_build_receipt(self, study_number, experiment_type, _set):
        """
        Update the build receipt (build_receipt.json) to record that a particular
        study/experiment_type/set combination has been processed.

        Parameters:
        - study_number (int): The study number (e.g. 1 or 2).
        - experiment_type (str): The experiment type (e.g. 'classic', 'genus_demo', 'yolo', etc.).
        - _set (str): One of 'full', 'test', or 'train'.
        """
        receipt_path = os.path.join(self.path, 'processed', 'build_receipt.json')
        receipt_data = {}

        # Load existing receipt if it exists
        if os.path.exists(receipt_path):
            with open(receipt_path, 'r') as f:
                try:
                    receipt_data = json.load(f)
                except json.JSONDecodeError:
                    # If file is corrupt or empty, start fresh
                    receipt_data = {}

        study_key = f"study{study_number}"
        if study_key not in receipt_data:
            receipt_data[study_key] = {}
        if experiment_type not in receipt_data[study_key]:
            receipt_data[study_key][experiment_type] = []

        # Add this set to the receipt if not already recorded
        if _set not in receipt_data[study_key][experiment_type]:
            receipt_data[study_key][experiment_type].append(_set)

        # Write back to the receipt
        with open(receipt_path, 'w') as f:
            json.dump(receipt_data, f, indent=4)
        print(f"Updated build receipt at: {receipt_path}")

    def already_built(self, **kwargs):
        study_number = kwargs.get('study', 2)
        experiment_type = self.assign_experiment_type(**kwargs)
        receipt_path = os.path.join(self.path, 'processed', 'build_receipt.json')
        required_sets = ['full', 'test', 'train']
        if experiment_type == 'yolo':
            required_sets = ['test', 'train'] 
        if not os.path.exists(receipt_path):
            return False
        try:
            with open(receipt_path, 'r') as f:
                receipt_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return False
        study_key = f"study{study_number}"
        if study_key not in receipt_data:
            return False
        if experiment_type not in receipt_data[study_key]:
            return False
        built_sets = receipt_data[study_key][experiment_type]
        return all(s in built_sets for s in required_sets)

    def get_data_path(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        data_abs_path = os.path.join(project_root, 'data')
        return data_abs_path + os.sep


if __name__ == "__main__":
    processor = DataPreprocessor()
    processor.build_all_sets()