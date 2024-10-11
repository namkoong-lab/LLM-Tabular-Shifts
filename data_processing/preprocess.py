import pandas as pd
import numpy as np
import functools
from sklearn.model_selection import train_test_split
from whyshift.folktables import adult_filter
import pickle

def flatten_list(list_of_lists):
        return functools.reduce(lambda x,y:x+y, list_of_lists)

rs = 0
s = 'AL,AK,AZ,AR,CA,CO,CT,DE,FL,GA,HI,ID,IL,IN,IA,KS,KY,LA,ME,MD,MA,MI,MN,MS,MO,MT,NE,NV,NH,NJ,NM,NY,NC,ND,OH,OK,OR,PA,RI,SC,SD,TN,TX,UT,VT,VA,WA,WV,WI,WY'
all_states = s.split(',')

sex_dict = {1: 'male',
            2: 'female'}

mar_dict = {1: 'married',
            2: 'widowed',
            3: 'divorced',
            4: 'separated',
            5: 'never married'}

dis_dict = {1: 'has a disability',
            2: 'does not have a disability'}

esp_dict = {1: "is living with two parents: both parents in labor force",
            2: "is living with two parents and only Father is in labor force",
            3: "is living with two parents and only mother is in labor force",
            4: "is living with two parents and neither parent in labor force",
            5: "is living with father and father in the labor force",
            6: "is living with father and father not in labor force",
            7: "is living with mother and mother is in the labor force",
            8: "is living with mother and mother is not in labor force"}

cit_dict = {1: 'is born in the U.S.',
            2: "is born in Puerto Rico, Guam, the U.S. Virgin Islands, or the Northern Marianas",
            3: "is born abroad of American parent(s)",
            4: "is a U.S. citizen by naturalization",
            5: "is not a citizen of the U.S."}


mig_dict = {1: 'lives in the same house as they did one year ago',
            2: 'does not live in the same house as they did one year ago; they were living outside the United States and Puerto Rico',
            3: 'does not live in the same house as they did one year ago; they were living within the United States or Puerto Rico'}

mil_dict = {1: 'is now on active duty',
            2: "is on active duty in the past, but not now",
            3: 'is only on active duty for training in Reserves/National Guard',
            4: "is never served in the military"}
anc_dict = {1: "single",
            2: "multiple",
            3: "unclassified",
            4: "not reported",
            8: "suppressed for data year 2018 for select PUMAs"}

nat_dict = {1: 'native',
            2: 'foreign born'}

dear_dict = {1: 'has hearing difficulty',
             2: 'does not have hearing difficulty'}

deye_dict = {1: 'has vision difficulty',
             2: 'does not have vision difficulty'}

drem_dict = {1: 'has cognitive difficulty',
             2: 'does not have cognitive difficulty'}

esr_dict = {1: "is civilian employed, at work",
            2: "is civilian employed, with a job but not at work",
            3: "is unemployed",
            4: "is in the armed forces, at work",
            5: "is in the armed forces, with a job but not at work",
            6: "is not in labor force"}

fer_dict = {1: "gave birth to child within the past 12 months",
            2: "did not give birth to child within the past 12 months"}

race_dict = {1: 'white',
             2: 'black or African American',
             3: 'American Indian',
             4: 'Alaska native',
             5: 'American Indian and Alaska native',
             6: 'Asian',
             7: 'native Hawaiian and other Pacific islander',
             8: 'other race',
             9: 'two or more races'}

relp_dict = {0: 'reference person', 
            1: 'husband or wife',
            2: 'biological son or daughter',
            3: 'adopted son or daughter',
            4: 'stepson or stepdaughter',
            5: 'brother or sister',
            6: 'father or mother',
            7: 'grandchild',
            8: 'parent-in-law',
            9: 'son-in-law or daughter-in-law',
            10: 'other relative',
            11: 'roomer or boarder',
            12: 'housemate or roommate',
            13: 'unmarried partner',
            14: 'foster child',
            15: 'other nonrelative',
            16: 'institutionalized group quarters population',
            17: 'noninstitutionalized group quarters population'}

gcl_dict = {1: "is a grandparent living with grandchildren", 
            2: "is not a grandparent living with grandchildren"}

cow_dict = {1: 'is an employee of a private for-profit company or business, or of an individual, for wages, salary, or commissions',
            2: 'is an employee of a private not-for-profit, tax-exempt, or charitable organization',
            3: 'is a local government employee (city, county, etc.)',
            4: 'is a state government employee',
            5: 'is a federal government employee',
            6: 'is self-employed in own not incorporated business, professional practice, or farm',
            7: 'is self-employed in own incorporated business, professional practice or farm',
            8: 'is working without pay in family business or farm',
            9: 'is unemployed and last worked 5 years ago or earlier or never worked'}

schl_dict = {1: "no schooling completed",
                2: "nursery school or preschool",
                3: "Kindergarten",
                4: "Grade 1",
                5: "Grade 2",
                6: "Grade 3",
                7: "Grade 4",
                8: "Grade 5",
                9: "Grade 6",
                10: "Grade 7",
                11: "Grade 8",
                12: "Grade 9",
                13: "Grade 10",
                14: "Grade 11",
                15: "12th Grade - no diploma",
                16: "regular high school diploma",
                17: "GED or alternative credential",
                18: "some college but less than 1 year",
                19: "1 or more years of college credit but no degree",
                20: "associate's degree",
                21: "Bachelor's degree",
                22: "Master's degree",
                23: "professional degree beyond a Bachelor's degree",
                24: "Doctorate degree"}

with open('./occp_dict.pkl', 'rb') as f:
    occp_dict = pickle.load(f)


feature_categories = ['AGEP','RACE','SCHL','MAR','MIL','CIT','MIG','DIS','SEX','NATIVITY']
jwtr_vals = ['car', 'bus', 'streetcar', 'subway', 'railroad', 'ferryboat', 'taxicab', 'motocycle', 'bicycle', 'walk', 'home', 'other']

OCCP_vals = ['MGR1', 'MGR2', 'MGR3', 'MGR4', 'MGR5', 'BUS1', 'BUS2', 'BUS3', 'FIN1', 'FIN2', 
             'CMM1', 'CMM2', 'CMM3', 'ENG1', 'ENG2', 'ENG3', 'SCI1', 'SCI2', 'SCI3', 'SCI4', 
             'CMS1', 'LGL1', 'EDU1', 'EDU2', 'EDU3', 'EDU4', 'ENT1', 'ENT2', 'ENT3', 'ENT4', 
             'MED1', 'MED2', 'MED3', 'MED4', 'MED5', 'MED6', 'HLS1', 'PRT1', 'PRT2', 'PRT3',  
             'EAT1', 'EAT2', 'CLN1', 'PRS1', 'PRS2', 'PRS3', 'PRS4', 'SAL1', 'SAL2', 'SAL3', 
             'OFF1', 'OFF2', 'OFF3', 'OFF4', 'OFF5', 'OFF6', 'OFF7', 'OFF8', 'OFF9', 'OFF10',
             'FFF1', 'FFF2', 'CON1', 'CON2', 'CON3', 'CON4', 'CON5', 'CON6', 'EXT1', 'EXT2', 
             'RPR1', 'RPR2', 'RPR3', 'RPR4', 'RPR5', 'RPR6', 'RPR7', 'PRD1', 'PRD2', 'PRD3', 
             'PRD4', 'PRD5', 'PRD6', 'PRD7', 'PRD8', 'PRD9', 'PRD10', 'PRD11', 'PRD12', 'PRD13', 
             'TRN1', 'TRN2', 'TRN3', 'TRN4', 'TRN5', 'TRN6', 'TRN7', 'TRN8', 'MIL1', "no1"]

Big_OCCP_vals = ['MGR', 'BUS', 'FIN', 'CMM', 'ENG', 'SCI', 'CMS', 'LGL', 'EDU', 'ENT', 'MED', 'HLS', 'PRT', 'EAT', 'CLN', 'PRS',
                'SAL', 'OFF', 'FFF', 'CON', 'EXT', 'RPR', 'PRD', 'TRN', 'MIL', 'no']
Big_OCCP_threshold = [[0,4], [5,7], [8,9], [10,12], [13,15], [16,19], [20, 20], [21, 21], [22, 25],\
                    [26, 29], [30,35], [36,36], [37,39], [40,41], [42, 42], [43,46], [47,49], [50,59],\
                    [60,61], [62,67], [68,69], [70,76], [77,89], [90,97], [98,98],[99,99]]
Large_OCCP_vals = ['Lg', 'Semi_Lg', 'Non_Lg']
CIT_vals = ['us', 'pr', 'abroad', 'citizen', 'not']
ESR_vals = ['employed', 'partial_employed', 'unemployed', 'armed', 'partial_armed', 'no']

def add_esr_indicators(t):
    for idx, esr in enumerate(ESR_vals):
        t['ESR_%s'%esr] = t.ESR == (idx+1)

def add_cit_indicators(t):
    for idx, cit in enumerate(CIT_vals):
        t['CIT_%s'%cit] = t.CIT == (idx+1)

def add_occp_indicators(t):
    for idx, occp in enumerate(OCCP_vals):
        t['occp_%s'%occp] = t.OCCP == idx
    return t

def add_big_OCCP_indicators(t):
    for idx, big_occp in enumerate(Big_OCCP_vals):
        t['big_occp_%s'%big_occp] = ((t.OCCP >= Big_OCCP_threshold[idx][0]) & (t.OCCP <= Big_OCCP_threshold[idx][1]))
    return t 

def add_large_occp_indicator(t):
    t['large_occp_Lg'] = ((t.big_occp_MGR == 1)|(t.big_occp_BUS == 1)|(t.big_occp_FIN == 1)|\
                          (t.big_occp_LGL == 1)|(t.big_occp_EDU == 1)|(t.big_occp_ENT == 1))

    t['large_occp_Semi_Lg'] = ((t.big_occp_CMM == 1)|(t.big_occp_ENG == 1)|(t.big_occp_SCI == 1)|\
                          (t.big_occp_CMS == 1)|(t.big_occp_MED == 1)|(t.big_occp_HLS == 1)|(t.big_occp_PRS == 1)|\
                          (t.big_occp_SAL == 1)|(t.big_occp_OFF == 1)|(t.big_occp_RPR == 1)|(t.big_occp_PRD == 1))

    t['large_occp_Non_Lg'] = ((t.big_occp_PRT == 1)|(t.big_occp_EAT == 1)|(t.big_occp_CLN == 1)|\
                          (t.big_occp_FFF == 1)|(t.big_occp_CON == 1)|(t.big_occp_EXT == 1)|(t.big_occp_TRN == 1)\
                          |(t.big_occp_MIL == 1)|(t.big_occp_no == 1))


def age_map(num):
    int_num = int(num)  # Convert float to int
    if int_num == 0:
        return "less than 1"
    else:
        return int_num

def preprocess_age(t):
    t["age"] = t['AGEP'].map(age_map)

def preprocess_sex(t):
    t['sex'] = t['SEX'].map(lambda x: sex_dict.get(x, None))

def WKHP_map(num):
    try:
        int_num = int(num)  # Convert float to int
        if int_num >= 99:
            return "greater than or equal to 99"
        else:
            return int_num
    except:
        return 

def preprocess_WKHP(t):
    t['usual hours worked per week last year'] = t['WKHP'].map(WKHP_map)

def preprocess_race(t):
    t['race'] = t['RAC1P'].map(lambda x: race_dict.get(x, None))

def preprocess_relp(t):
    t['relationship to householder'] = t['RELP'].map(lambda x: relp_dict.get(x, None))

def preprocess_mar(t):
    t['marital status'] = t['MAR'].map(lambda x: mar_dict.get(x, None))

def preprocess_school(t):
    t['educational attainment'] = t['SCHL'].map(lambda x: schl_dict.get(x, None))

def preprocess_cow(t):
    t['class of worker'] = t['COW'].map(lambda x: cow_dict.get(x, None))

def preprocess_occp(t):
    t['occupation'] = t['OCCP'].map(lambda x: occp_dict.get(int(x), None))

def preprocess_dis(t):
    t['disability'] = t['DIS'].map(lambda x: dis_dict.get(x, None))

def preporcess_esp(t):
    t['employment status of parents'] = t['ESP'].map(lambda x: esp_dict.get(x, None))

def preprocess_cit(t):
    t['citizenship'] = t['CIT'].map(lambda x: cit_dict.get(x, None))

def preprocess_mig(t):
    t['mobility'] = t['MIG'].map(lambda x: mig_dict.get(x, None))

def preprocess_mil(t):
    t['military service'] = t['MIL'].map(lambda x: mil_dict.get(x, None))

def preprocess_anc(t):
    t['ancestry'] = t['ANC'].map(lambda x: anc_dict.get(x, None))

def preprocess_nat(t):
    t['nativity'] = t['NATIVITY'].map(lambda x: nat_dict.get(x, None))

def preprocess_dear(t):
    t['hearing difficulty'] = t['DEAR'].map(lambda x: dear_dict.get(x, None))

def preprocess_deye(t):
    t['vision difficulty'] = t['DEYE'].map(lambda x: deye_dict.get(x, None))

def preprocess_drem(t):
    t['cognitive difficulty'] = t['DREM'].map(lambda x: drem_dict.get(x, None))

def preprocess_gcl(t):
    t['grandparent living with grandchildren'] = t['GCL'].map(lambda x: gcl_dict.get(x, None))

def PINCP_map(num):
    int_num = int(num)  # Convert float to int
    if int_num <= -19998:
        return "less than or equal to -19998"
    elif int_num >= 4209995:
        return "greater than or equal to 4209995"
    else:
        return int_num

def preprocess_pincp(t):
    t["person's total annual income"] = t['PINCP'].map(PINCP_map)

def preprocess_esr(t):
    t['employment status'] = t['ESR'].map(lambda x: esr_dict.get(x, None))

def preprocess_fer(t):
    t['gave birth to child within the past 12 months'] = t['FER'].map(lambda x: fer_dict.get(x, None))

def JWMNP_map(num):
    try:
        int_num = int(num)  # Convert float to int
        if int_num >= 200:
            return "greater than or equal to 200 minutes"
        else:
            return f"{int_num} minutes"
    except:
        return 

def preprocess_jwmnp(t):
    t["travel time to work"] = t['JWMNP'].map(JWMNP_map)


def add_citizenship_indicators(t):
    t['cit_born_us']=t.CIT==1
    t['cit_born_territory']=t.CIT==2
    t['cit_am_parents']=t.CIT==3
    t['cit_naturalized']=t.CIT==4
    t['cit_not_citizen']=t.CIT==5
    return t

def add_mobility_indicators(t):
    t['mig_moved']=t.MIG!=1
    return t
        
def add_jwtr_indicators(t):
    for idx, relp in enumerate(jwtr_vals):
        t['jwtr_%s'%relp] = t.JWTR == (idx+1)
    return t

def preprocess_ACSIncome(t):
    preprocess_age(t)
    preprocess_cow(t)
    preprocess_school(t)
    preprocess_mar(t)
    preprocess_occp(t)
    preprocess_relp(t)
    preprocess_WKHP(t)
    preprocess_sex(t)
    preprocess_race(t)
    return t

def preprocess_ACSPubCov(t):
    preprocess_age(t)
    preprocess_school(t)
    preprocess_mar(t)
    preprocess_sex(t)
    preprocess_dis(t)
    preporcess_esp(t)
    preprocess_cit(t)
    preprocess_mig(t)
    preprocess_mil(t)
    preprocess_anc(t)
    preprocess_nat(t)
    preprocess_dear(t)
    preprocess_deye(t)
    preprocess_drem(t) 
    preprocess_esr(t) 
    preprocess_pincp(t)
    preprocess_fer(t)
    preprocess_race(t)
    return t

def preprocess_ACSMobility(t):
    preprocess_age(t)
    preprocess_school(t)
    preprocess_mar(t)
    preprocess_sex(t)
    preprocess_dis(t)
    preporcess_esp(t)
    preprocess_cit(t)
    preprocess_mil(t)
    preprocess_anc(t)
    preprocess_nat(t)
    preprocess_relp(t)
    preprocess_dear(t)
    preprocess_deye(t)
    preprocess_drem(t) 
    preprocess_race(t)
    preprocess_gcl(t)
    preprocess_cow(t)
    preprocess_esr(t) 
    preprocess_WKHP(t)
    preprocess_jwmnp(t)
    preprocess_pincp(t)
    return t

'''
def add_indicators_year(t):
    adult_filter(t)
    add_race_indicators(t)
    # add_relp_indicators(t)
    add_school_indicators(t)
    add_married_indicator(t)
    
    # add_military_indicators(t)
    # add_citizenship_indicators(t)
    # add_mobility_indicators(t)
    return t

def add_indicators_pubcov(t):
    add_race_indicators(t)
    add_school_indicators(t)
    add_married_indicator(t)
    add_cit_indicators(t)
    add_esr_indicators(t)
    return t

def add_indicators_traveltime(t):
    add_race_indicators(t)
    add_school_indicators(t)
    add_married_indicator(t)
    add_jwtr_indicators(t)
    return t

binarized_features = ['race_'+x for x in rac1p_vals] + ['married',
                                                    'schl_at_least_bachelor',
                                                    'schl_at_least_high_school_or_ged',
                                                    'schl_postgrad',
                                                    'active_military','vet',
                                                    'cit_born_us',
                                                    'cit_born_territory',
                                                    'cit_am_parents',
                                                    'cit_naturalized',
                                                    'cit_not_citizen','mig_moved']
'''