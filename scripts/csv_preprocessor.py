import pandas as pd 

class Processor:
    def __init__(self):
        self.path='../datasets/'
        self.csvPathDict={1:'SupplementaryTableS6.xlsx', 2:'SupplementaryTableS1.xlsx'}
        self.study1DropColumns=['Samples', 'Median HC', 'Median PwD', 'Wilcoxon', 'FDR']

    def processSet(study,split,useSynthetic):
        print('Preprocessing datasets\n====================')
        if split: #get the IDs for each split if requested

        pass

    def protoProcessor():
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



if __name__ == "__main__":
    processor = Processor()
    #Check which set user wants to generate
    processor.processSets()
    pass