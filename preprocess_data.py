# import library
import pandas as pd
from sklearn.model_selection import train_test_split

# load data (MIMIC-IV ED)
edstay = pd.read_csv('./data/edstays.csv.gz', compression = 'gzip')
diagnosis = pd.read_csv('./data/diagnosis.csv.gz', compression = 'gzip')
med = pd.read_csv('./data/medrecon.csv.gz', compression = 'gzip')
pyxis = pd.read_csv('./data/pyxis.csv.gz', compression = 'gzip')
triage = pd.read_csv('./data/triage.csv.gz', compression = 'gzip')
vital = pd.read_csv('./data/vitalsign.csv.gz', compression = 'gzip')

## process edstay df
# drop useless columns
edstay_1 = edstay.loc[:, ['subject_id', 'stay_id', 'disposition', 'intime', 'outtime', 'gender', 'race']]
# keep home & admitted only
edstay_2 = edstay_1[(edstay_1['disposition'] == 'HOME')| (edstay_1['disposition'] == 'ADMITTED')]
# calculate ed stay time
edstay_2.intime = pd.to_datetime(edstay_2.intime, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
edstay_2.outtime = pd.to_datetime(edstay_2.outtime, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
edstay_2['staytime'] = round((edstay_2['outtime'] - edstay_2['intime']).dt.total_seconds() / (24 * 60), 2)
edstay_2['label'] = edstay_2['disposition'] == 'ADMITTED'
edstay_2['label'] = edstay_2['label'].astype(int)
# create df for joining
edstay_df = edstay_2.drop(columns = ['intime', 'outtime', 'disposition'])
edstay_df = edstay_df.rename(columns = {'label' : 'disposition'})
edstay_df = edstay_df.fillna('')
edstay_df.head()

## process diagnosis
# drop useless columns
diagnosis_1 = diagnosis.loc[:, ['stay_id', 'icd_title']]
# put all icd title into 1 string for each id
diagnosis_1.icd_title = diagnosis_1.icd_title.str.lower()
diagnosis_1 = diagnosis_1.fillna('')
diagnosis_df = diagnosis_1.groupby('stay_id')['icd_title'].apply(lambda x: list(set(x))).reset_index()
diagnosis_df['icd_title'] = diagnosis_df['icd_title'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

## process med
med_1 = med.loc[:, ['stay_id', 'name']]
med_1 = med_1.rename(columns = {'name' : 'current_med'})
# put all current medication into 1 string for each id
med_1.current_med = med_1.current_med.str.lower()
med_1 = med_1.fillna('')
med_df = med_1.groupby('stay_id')['current_med'].apply(lambda x: list(set(x))).reset_index()
med_df['current_med'] = med_df['current_med'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

## process pyxis
pyxis_1 = pyxis.loc[:, ['stay_id', 'name']]
pyxis_1 = pyxis_1.rename(columns = {'name' : 'prescription'})
# put all prescription into 1 string for each id
pyxis_1.prescription = pyxis_1.prescription.str.lower()
pyxis_1 = pyxis_1.fillna('')
pyxis_df = pyxis_1.groupby('stay_id')['prescription'].apply(lambda x: list(set(x))).reset_index()
pyxis_df['prescription'] = pyxis_df['prescription'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

## process vital
# find most extreme value for each vital per individual
vital_1 = vital.drop(columns = ['charttime', 'rhythm', 'pain'])
# maximum vitals
max = vital_1.groupby(by = 'stay_id').max()
max = max.drop(columns = 'subject_id')
max = max.add_prefix('max_')
# minimum vitals
min = vital_1.groupby(by = 'stay_id').min()
min = min.drop(columns = 'subject_id')
min = min.add_prefix('min_')
# merge max and min vitals
vital_2 = pd.merge(max, min, on = ['stay_id'], how = 'outer')
vital_2.shape
# get all rhythm for each id & merge
rhythm = vital.loc[:, ['stay_id', 'rhythm']].dropna(subset = ['rhythm'])
rhythm_1 = rhythm.groupby(by = 'stay_id')['rhythm'].apply(lambda x: list(set(x.str.lower()))).reset_index()
rhythm_1['rhythm'] = rhythm_1['rhythm'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
vital_3 = pd.merge(vital_2, rhythm_1, on = ['stay_id'], how = 'left')
# get all pain for each id & merge
pain = vital.loc[:, ['stay_id', 'pain']].dropna(subset = ['pain'])
pain_1 = pain.groupby(by = 'stay_id')['pain'].apply(lambda x: list(set(x.str.lower()))).reset_index()
pain_1['pain'] = pain_1['pain'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
vital_df = pd.merge(vital_3, pain_1, on = ['stay_id'], how = 'left')
vital_df = vital_df.fillna('')

## process triage
# change triage column names
triage_df = triage.add_prefix('triage_')
triage_df = triage_df.rename(columns = {'triage_subject_id' : 'subject_id',
                                        'triage_stay_id' : 'stay_id',
                                        'triage_acuity' : 'acuity',
                                        'triage_chiefcomplaint' : 'chiefcomplaint'})
triage_df = triage_df.drop(columns = 'subject_id')
triage_df = triage_df.fillna('')

# join all tables together
df_1 = pd.merge(edstay_df, diagnosis_df, on = 'stay_id', how = 'left')
df_2 = pd.merge(df_1, med_df, on = 'stay_id', how = 'left')
df_3 = pd.merge(df_2, pyxis_df, on = 'stay_id', how = 'left')
df_4 = pd.merge(df_3, triage_df, on = 'stay_id', how = 'left')
all_df = pd.merge(df_4, vital_df, on = 'stay_id', how = 'left')

# create column names
column_name = {'staytime' : 'emergency department stay time in hours',
               'icd_title' : 'diagnosis icd title',
               'current_med' : 'current medication',
               'triage_temperature' : 'temperature at triage',
               'triage_heartrate' : 'heartrate at triage',
               'triage_resprate' : 'respiratory rate at triage',
               'triage_o2sat' : 'oxygen saturation at triage',
               'triage_sbp' : 'sbp at triage',
               'triage_dbp' : 'dbp at triage', 
               'triage_pain' : 'pain level at triage',
               'max_temperature' : 'maximum temperature',
               'max_heartrate' : 'maximum heartrate',
               'max_resprate' : 'maximum respiratory rate', 
               'max_o2sat' : 'maximum oxygen saturation', 
               'max_sbp' : 'maximum sbp',
               'max_dbp' : 'maximum dbp',
               'min_temperature' : 'minimum temperature',
               'min_heartrate' : 'minimum heartrate',
               'min_resprate' : 'minimum respiratory rate', 
               'min_o2sat' : 'minimum oxygen saturation', 
               'min_sbp' : 'minimum sbp',
               'min_dbp' : 'minimum dbp',
               'rhythm' : 'heart rhythm'}
all_df = all_df.rename(columns = column_name)

# change acuity level to text
acuity_dict = {1.0 : 'esi 1, triage process stops, patient taken directly to a room and imeediate physician intervention requrested',
               2.0 : 'esi 2, triage nurse notifies resource nurse and appropriate placement tbd',
               3.0 : 'esi 3, patient requires two or more resources',
               4.0 : 'esi 4, patient requires one resource',
               5.0 : 'esi 5, patient not requires any resource'}
all_df_1 = all_df.replace({'acuity' : acuity_dict})

# define function to generate text chunck using cleaned df
def generate_text_df(df):
    col_name = list(df.columns)
    out_df = pd.DataFrame()
    
    for i in range(len(col_name)):
        if col_name[i] == 'stay_id' or col_name[i] == 'disposition':
            out_df[col_name[i]] = df[col_name[i]]
        else:
            # concat column name and value
            out_df[col_name[i]] = col_name[i] + ' : ' + df[col_name[i]].astype(str)
    
    return out_df

# generate a df with text chunck
text_df = generate_text_df(all_df_1)
text_df = text_df.fillna('')
clinical_text = text_df.drop(columns = ['stay_id', 'subject_id', 'disposition']).apply(lambda row: '. '.join([str(x) for x in row]), axis = 1)
clinical_text_df = text_df.loc[:, ['stay_id']]
clinical_text_df['TEXT'] = clinical_text.str.lower()
clinical_text_df['Label'] = text_df['disposition']
clinical_text_df = clinical_text_df.rename(columns = {'stay_id' : 'ID'})

'''clinical_text_df is used to train the baseline model (clinicalBERT)'''
clinical_text_df.head()


### below are processing steps to add numerical values from vital for clinicalEDBERT training
# create a list of extreme vitals
max = max.fillna(0)
min = min.fillna(0)
vital_max = max.apply(lambda row: row.tolist(), axis = 1)
vital_min = min.apply(lambda row: row.tolist(), axis = 1)

# delete vital information from text chunck
drop_vital = [col for col in all_df.columns if 'maximum' in col or 'minimum' in col]
all_but_vital = all_df.drop(columns = drop_vital)
text_no_vital_df = generate_text_df(all_but_vital)
text_no_vital_df = text_no_vital_df.fillna('')
clinical_no_vital = text_no_vital_df.drop(columns = ['stay_id', 'subject_id', 'disposition']).apply(lambda row: '. '.join([str(x) for x in row]), axis = 1)
clinical_no_vital_df = text_no_vital_df.loc[:, ['stay_id']]
clinical_no_vital_df['TEXT'] = clinical_no_vital.str.lower()
clinical_no_vital_df['Label'] = text_no_vital_df['disposition']
clinical_no_vital_df = clinical_no_vital_df.rename(columns = {'stay_id' : 'ID'})

# create vital columns and merge with the main df with text
vitals = vital_max + vital_min
vitals = vitals.to_frame().reset_index()
vitals.columns = ['ID', 'vitals']
clinical_no_vital_df = pd.merge(clinical_no_vital_df, vitals, on = 'ID', how = 'left')


## train test split function
## model requires train, test, and validation dataset
## splited using ratio 8 : 1 : 1
def split_train_test_val(df, size):
    df_split = df.sample(n = size)

    y = df_split.loc[:, 'Label']
    X = df_split.drop(columns = 'Label')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
    X_test, X_eval, y_test, y_eval = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

    X_train['Label'] = y_train
    X_test['Label'] = y_test
    X_eval['Label'] = y_eval

    X_train.to_csv('train_data.csv', index = False)
    X_test.to_csv('test_data.csv', index = False)
    X_eval.to_csv('val_data.csv', index = False)