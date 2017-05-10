import pandas as pd
import numpy as np
import math
import csv
import sys
from scipy import stats
from collections import defaultdict

def transform(filename,drop = False):
    """ preprocess the training data"""
    """ your code here """
    index = list(range(2,109))
    df_pre = pd.read_csv(filename)#keep_default_na=False)
    if (drop == True):
        df_pre = df_pre.dropna()
    df_transed = df_pre
    data = df_pre[index[0:1]]
    i = 0
    gend_column = df_pre[index[0:1]]
    column = df_pre[index[i:i+1]]

    def Gender_map(v):   #Female-0 Male-1
        if v == 'Female':
            return int(0)
        if v == 'Male':
            return int(1)

    df_transed['Gender'] = df_pre['Gender'].map(Gender_map)
    #while i != 107:
    def Income_map(v):
        if v == 'under $25,000':
            return int(1)
        elif v == '$25,001 - $50,000':
            return int(2)
        elif v == '$50,000 - $74,999':
            return int(3)
        elif v == '$75,000 - $100,000':
            return int(4)
        elif v == '$100,001 - $150,000':
            return int(5)
        elif v == 'over $150,000':
            return int(6)
    df_transed['Income'] = df_pre['Income'].map(Income_map)
    def HouseholdS_map(v):
        if v == 'Single (no kids)':
            return int(0)
        elif v == 'Single (w/kids)':
            return int(1)
        elif v == 'Married (no kids)':
            return int(2)
        elif v == 'Married (w/kids)':
            return int(3)
        elif v == 'Domestic Partners (no kids)':
            return int(4)
        elif v == 'Domestic Partners (w/kids)':
            return int(5)
    df_transed['HouseholdStatus'] = df_pre['HouseholdStatus'].map(HouseholdS_map)
    def EduLv_map(v):
        if v == 'Current K-12':
            return int(1)
        elif v == 'High School Diploma':
            return int(2)
        elif v == "Current Undergraduate":
            return int(3)
        elif v == "Associate's Degree":
            return int(4)
        elif v == "Bachelor's Degree":
            return int(5)
        elif v == "Master's Degree":
            return int(6)
        elif v == 'Doctoral Degree':
            return int(7)
    df_transed['EducationLevel'] = df_pre['EducationLevel'].map(EduLv_map)
    def Party_map(v):
        if v == 'Independent':
            return int(0)
        elif v == 'Democrat':
            return int(1)
        elif v == 'Republican':
            return int(2)
        elif v == 'Libertarian':
            return int(3)
        elif v == 'Other':
            return int(4)
    df_transed['Party'] = df_pre['Party'].map(Party_map)

    df_transed=df_pre.replace(to_replace = ['Yes','No'],value = [1,0])
    df_pre = df_pre.replace(to_replace = ['Yes','No'],value = [1,0])
    #df_transed=df_pre.replace(to_replace = 'No',value = 0)

    df_transed=df_pre.replace(to_replace = ['Public','Art','Try first','Giving','Idealist','Hot headed','Standard hours','Happy',
                                            'A.M.','Start','Me','TMI','Talk','Technology','Supportive','PC',
                                            'Cautious','Yes!','Space','In-person','Yay people!','Own','Optimist'],value = 1)
    df_pre=df_pre.replace(to_replace = ['Public','Art','Try first','Giving','Idealist','Hot headed','Standard hours','Happy',
                                            'A.M.','Start','Me','TMI','Talk','Technology','Supportive','PC',
                                            'Cautious','Yes!','Space','In-person','Yay people!','Own','Optimist'],value = 1)
    df_transed=df_pre.replace(to_replace = ['Private','Science','Study first','Receiving','Pragmatist','Cool headed','Odd hours','Right',
                                            'P.M.','End','Circumstances','Mysterious','Tunes','People','Demanding','Mac',
                                            'Risk-friendly','Umm...','Socialize','Online','Grrr people','Rent','Pessimist'],value = 0)
    df_pre=df_pre.replace(to_replace = ['Private','Science','Study first','Receiving','Pragmatist','Cool headed','Odd hours','Right',
                                            'P.M.','End','Circumstances','Mysterious','Tunes','People','Demanding','Mac',
                                            'Risk-friendly','Umm...','Socialize','Online','Grrr people','Rent','Pessimist'],value = 0)
    df_transed=df_pre.replace(to_replace = ['Mom','Check!'],value = 1)
    df_pre=df_pre.replace(to_replace = ['Mom','Check!'],value = 1)
    df_transed=df_pre.replace(to_replace = ['Dad','Nope','Only-child'],value = 0)
    df_pre=df_pre.replace(to_replace = ['Dad','Nope','Only-child'],value = 0)
    #trans dataframe to dict
    #data = df_transed.to_dict(orient = 'dict')
    #df_transed.pop('Happy')
    data = df_transed.as_matrix()
    data = data[:,1:]
    #data = np.delete(data,6,1)
    test_id = df_pre['UserID'].as_matrix()
    if 'Happy' in df_pre:
        target = df_pre['Happy'].as_matrix()

        transformed_data = dict()
        transformed_data['data'] = data
        transformed_data['target'] = target
        return {'data':data,'target':target}
    else:
        transformed_data = dict()
        transformed_data['data'] = data
        return {'data':data,'id':test_id}
    #print(type(data[92,0]))

def fill_missing(X, strategy, isClassified):
    """
     @X: input matrix with missing data filled by nan
     @strategy: string, 'median', 'mean', 'most_frequent'
     @isclassfied: boolean value, if isclassfied == true, then you need build a
     decision tree to classify users into different classes and use the
     median/mean/mode values of different classes to fill in the missing data;
     otherwise, just take the median/mean/most_frequent values of input data to
     fill in the missing data
    """

    feature_row = np.shape(X)[0]
    sample_col = np.shape(X)[1]

    def get_mean(X,col):  #if there is a nan ，jump over it
        sum_of_valid = 0
        mean = 0
        for i in range(np.shape(X)[0]):
            if math.isnan(float(X[i,col])) :
                pass
            else:
                sum_of_valid += 1
                mean += X[i,col]
                #print(mean,sum_of_valid)

        #print("_mean_:",mean/sum_of_valid)
        return mean/sum_of_valid

    def get_median(X,col):
        X_ = np.reshape(X[:,col],(np.shape(X)[0]))
        X_ = [x for x in X_ if str(x) != 'nan'] #delete the NAN
        return np.median(X_)

    def get_mode(X,col):
        X_ = np.reshape(X[:,col],(np.shape(X)[0]))
        X_ = [x for x in X_ if str(x) != 'nan'] #delete the NAN
        Mode = stats.mode(X_)
        return Mode.mode[0]
    #basic, without classfication
    '''
    basic way ,without classfication
    '''
    def replace_directly(X,strategy):
        if strategy == 'mean':
            for i in range(0,np.shape(X)[1]):
                temp = get_mean(X,i)
                for k in range(0,np.shape(X)[0]):
                    if str(X[k,i]) == 'nan':
                        X[k,i] = temp

        elif strategy == 'median':
            for i in range(0,np.shape(X)[1]):
                temp = get_median(X,i)
                for k in range(0,np.shape(X)[0]):
                    if str(X[k,i]) == 'nan':
                        X[k,i] = temp
        elif strategy == 'mode':
            for i in range(0,np.shape(X)[1]):
                temp = get_mode(X,i)
                for k in range(0,np.shape(X)[0]):
                    if str(X[k,i]) == 'nan':
                        X[k,i] = temp
        return X

    #to classfi the group accrouding to feature

    #feature: 1. Gender(2) 2. Income(6) 3. Household(6) 4. Education(7) 5. Party(5)
    ''' not used briefly
    def classify_method(X,feature):
        X_modified = defaultdict(list)

        if feature == 'Gender':#col = 1
            for k in range (0,feature_row):
                if X[k,1] == 1:
                    X_modified['Male'].append(X[k,:])
                elif X[k,1] == 0:
                    X_modified['Female'].append(X[k,:])

        elif feature == 'Income':#col = 2
            for k in range (0,feature_row):
                if X[k,2] == 1:
                    X_modified['under $25,000'].append(X[k,:])
                elif X[k,2] == 2:
                    X_modified['$25,001 - $50,000'].append(X[k,:])
                elif X[k,2] == 3:
                    X_modified['$50,000 - $74,999'].append(X[k,:])
                elif X[k,2] == 4:
                    X_modified['$75,000 - $100,000'].append(X[k,:])
                elif X[k,2] == 5:
                    X_modified['$100,001 - $150,000'].append(X[k,:])
                elif X[k,2] == 6:
                    X_modified['over $150,000'].append(X[k,:])
        elif feature == 'Household':
            for k in range (0,feature_row):
                if X[k,3] == 0:
                    X_modified['Single (no kids)'].append(X[k,:])
                elif X[k,3] == 1:
                    X_modified['Single (w/kids)'].append(X[k,:])
                elif X[k,3] == 2:
                    X_modified['Married (no kids)'].append(X[k,:])
                elif X[k,3] == 3:
                    X_modified['Married (w/kids)'].append(X[k,:])
                elif X[k,3] == 4:
                    X_modified['Domestic Partners (no kids)'].append(X[k,:])
                elif X[k,3] == 5:
                    X_modified['Domestic Partners (w/kids)'].append(X[k,:])
        elif feature == 'Education':
            for k in range (0,feature_row):
                if X[k,4] == 1:
                    X_modified['Current K-12'].append(X[k,:])
                elif X[k,4] == 2:
                    X_modified['High School Diploma'].append(X[k,:])
                elif X[k,4] == 3:
                    X_modified['Current Undergraduate'].append(X[k,:])
                elif X[k,4] == 4:
                    X_modified["Associate's Degree"].append(X[k,:])
                elif X[k,4] == 5:
                    X_modified["Bachelor's Degree"].append(X[k,:])
                elif X[k,4] == 6:
                    X_modified["Master's Degree"].append(X[k,:])
                elif X[k,4] == 7:
                    X_modified['Doctoral Degree'].append(X[k,:])
        elif feature == 'Party':
            for k in range (0,feature_row):
                if X[k,5] == 0:
                    X_modified['Independent'].append(X[k,:])
                elif X[k,5] == 1:
                    X_modified['Democrat'].append(X[k,:])
                elif X[k,5] == 2:
                    X_modified['Republican'].append(X[k,:])
                elif X[k,5] == 3:
                    X_modified['Libertarian'].append(X[k,:])
                elif X[k,5] == 4:
                    X_modified['Other'].append(X[k,:])

        return X_modified
        '''
#classfication tree
#1-sex 60% is man and 40% is woman(not NaN)
#2-Income use mean value to predict NaN
#3-Party use random to predict NaN

    def Predict_Gender(X):
       feature_row,sample_col = np.shape(X)
       np.random.seed(seed = 1)
       rand_value =np.random.random(size = feature_row)
       #print("random value",rand_value)
       for i in range(0,feature_row):
           if str(X[i,1]) == 'nan':
               if rand_value[i] > 0.6:
                   X[i,1] = 1
               else:
                   X[i,1] = 0
       return X

    def Predict_Income(X):
        feature_row,sample_col = np.shape(X)
        mean_income = math.floor(get_mean(X,2))
        for i in range(0,feature_row):
            if str(X[i,2]) == 'nan':
                X[i,2] = mean_income
        return X

    def Predict_Party(X):
        feature_row,sample_col = np.shape(X)
        rand_value =[np.random.randint(0,4) for x in range(feature_row)]
        #print(rand_value)
        for i in range(0,feature_row):
            if str(X[i,5]) == 'nan':
                X[i,5] = rand_value[i]
        return X

    def Predict_Age(X):
        feature_row,sample_col = np.shape(X)
        rand_value =[np.random.randint(1,4) for x in range(feature_row)]
        #print(rand_value)
        for i in range(0,feature_row):
            if str(X[i,0]) == 'nan':
                X[i,0] = rand_value[i]
        return X

    def classfication_age_first(X):

        count_Age_lv1 = 0
        count_Age_lv2 = 0
        count_Age_lv3 = 0
        count_Age_lv4 = 0

        Age_ = X[:,0]

        temp1 = 0
        temp2 = 0
        temp3 = 0
        temp4 = 0

        for i in range(0,np.shape(X)[0]):
            if Age_[i] == 1:
                count_Age_lv1 += 1
            elif Age_[i] == 2:
                count_Age_lv2 += 1
            elif Age_[i] == 3:
                count_Age_lv3 += 1
            elif Age_[i] == 4:
                count_Age_lv4 += 1
        X_Age_lv1 = np.empty([count_Age_lv1,sample_col])
        X_Age_lv2 = np.empty([count_Age_lv2,sample_col])
        X_Age_lv3 = np.empty([count_Age_lv3,sample_col])
        X_Age_lv4 = np.empty([count_Age_lv4,sample_col])
        #print(count_income_lv1,count_income_lv2,count_income_lv3,count_income_lv4,count_income_lv5,count_income_lv6)
        for k in range(0,np.shape(X)[0]):
            if Age_[k] == 1:
                X_Age_lv1[temp1,:] = X[k,:]
                temp1 = temp1 + 1
            elif Age_[k] == 2:
                X_Age_lv2[temp2,:] = X[k,:]
                temp2 = temp2 + 1
            elif Age_[k] == 3:
                X_Age_lv3[temp3,:] = X[k,:]
                temp3 = temp3 + 1
            elif Age_[k] == 4:
                X_Age_lv4[temp4,:] = X[k,:]
                temp4 = temp4 + 1


        return X_Age_lv1,X_Age_lv2,X_Age_lv3,X_Age_lv4

    def classfication_income_second(X):#input X_male or X_female
        count_income_lv1 = 0
        count_income_lv2 = 0
        count_income_lv3 = 0
        count_income_lv4 = 0
        count_income_lv5 = 0
        count_income_lv6 = 0

        Income_ = X[:,2]

        temp1 = 0
        temp2 = 0
        temp3 = 0
        temp4 = 0
        temp5 = 0
        temp6 = 0

        for i in range(0,np.shape(X)[0]):
            if Income_[i] == 1:
                count_income_lv1 += 1
            elif Income_[i] == 2:
                count_income_lv2 += 1
            elif Income_[i] == 3:
                count_income_lv3 += 1
            elif Income_[i] == 4:
                count_income_lv4 += 1
            elif Income_[i] == 5:
                count_income_lv5 += 1
            elif Income_[i] == 6:
                count_income_lv6 += 1
        X_Income_lv1 = np.empty([count_income_lv1,sample_col])
        X_Income_lv2 = np.empty([count_income_lv2,sample_col])
        X_Income_lv3 = np.empty([count_income_lv3,sample_col])
        X_Income_lv4 = np.empty([count_income_lv4,sample_col])
        X_Income_lv5 = np.empty([count_income_lv5,sample_col])
        X_Income_lv6 = np.empty([count_income_lv6,sample_col])
        #print(count_income_lv1,count_income_lv2,count_income_lv3,count_income_lv4,count_income_lv5,count_income_lv6)
        for k in range(0,np.shape(X)[0]):
            if Income_[k] == 1:
                X_Income_lv1[temp1,:] = X[k,:]
                temp1 = temp1 + 1
            elif Income_[k] == 2:
                X_Income_lv2[temp2,:] = X[k,:]
                temp2 = temp2 + 1
            elif Income_[k] == 3:
                X_Income_lv3[temp3,:] = X[k,:]
                temp3 = temp3 + 1
            elif Income_[k] == 4:
                X_Income_lv4[temp4,:] = X[k,:]
                temp4 = temp4 + 1
            elif Income_[k] == 5:
                X_Income_lv5[temp5,:] = X[k,:]
                temp5 = temp5 + 1
            elif Income_[k] == 6:
                X_Income_lv6[temp6,:] = X[k,:]
                temp6 = temp6 + 1
        return X_Income_lv1,X_Income_lv2,X_Income_lv3,X_Income_lv4,X_Income_lv5,X_Income_lv6


    '''
    def classfication_tree_first(X):
        temp1 = 0
        temp2 = 0
        Gender_ = X[:,1]
        X_temp = X
        count_male = 0
        feature_row,sample_col = np.shape(X)

        for i in range(0,feature_row):
            if Gender_[i] == 1:
                count_male += 1

        count_female = feature_row - count_male


        X_male = np.empty([count_male,sample_col])
        X_female = np.empty([count_female,sample_col]) #first tree

        for k in range(0,feature_row):#0-3694 train data
            if Gender_[k] == 1:
                X_male[temp1,:] = X[k,:]
                temp1 = temp1 + 1
            elif Gender_[k] == 0:
                X_female[temp2,:] = X[k,:]
                temp2 = temp2 + 1

        return X_male,X_female
    '''
    '''
    def classfication_tree_second(X):
        count_Party_lv1 = 0
        count_Party_lv2 = 0
        count_Party_lv3 = 0
        count_Party_lv4 = 0
        count_Party_lv5 = 0


        Party_ = X[:,5]
        temp1 = 0
        temp2 = 0
        temp3 = 0
        temp4 = 0
        temp5 = 0

        for i in range(0,np.shape(X)[0]):
            if Party_[i] == 0:
                count_Party_lv1 += 1
            elif Party_[i] == 1:
                count_Party_lv2 += 1
            elif Party_[i] == 2:
                count_Party_lv3 += 1
            elif Party_[i] == 3:
                count_Party_lv4 += 1
            elif Party_[i] == 4:
                count_Party_lv5 += 1

        X_Party_lv1 = np.empty([count_Party_lv1,sample_col])
        X_Party_lv2 = np.empty([count_Party_lv2,sample_col])
        X_Party_lv3 = np.empty([count_Party_lv3,sample_col])
        X_Party_lv4 = np.empty([count_Party_lv4,sample_col])
        X_Party_lv5 = np.empty([count_Party_lv5,sample_col])
        #print(count_Party_lv1,count_Party_lv2,count_Party_lv3,count_Party_lv4,count_Party_lv5,count_Party_lv6)
        for k in range(0,np.shape(X)[0]):
            if Party_[k] == 0:
                X_Party_lv1[temp1,:] = X[k,:]
                temp1 = temp1 + 1
            elif Party_[k] == 1:
                X_Party_lv2[temp2,:] = X[k,:]
                temp2 = temp2 + 1
            elif Party_[k] == 2:
                X_Party_lv3[temp3,:] = X[k,:]
                temp3 = temp3 + 1
            elif Party_[k] == 3:
                X_Party_lv4[temp4,:] = X[k,:]
                temp4 = temp4 + 1
            elif Party_[k] == 4:
                X_Party_lv5[temp5,:] = X[k,:]
                temp5 = temp5 + 1

            #print("ss",temp1,temp2,temp3,temp4,temp5)
        return X_Party_lv1,X_Party_lv2,X_Party_lv3,X_Party_lv4,X_Party_lv5
    '''
    '''
    def classfication_tree_third(X):#input X_male or X_female
        count_income_lv1 = 0
        count_income_lv2 = 0
        count_income_lv3 = 0
        count_income_lv4 = 0
        count_income_lv5 = 0
        count_income_lv6 = 0

        Income_ = X[:,2]

        temp1 = 0
        temp2 = 0
        temp3 = 0
        temp4 = 0
        temp5 = 0
        temp6 = 0

        for i in range(0,np.shape(X)[0]):
            if Income_[i] == 1:
                count_income_lv1 += 1
            elif Income_[i] == 2:
                count_income_lv2 += 1
            elif Income_[i] == 3:
                count_income_lv3 += 1
            elif Income_[i] == 4:
                count_income_lv4 += 1
            elif Income_[i] == 5:
                count_income_lv5 += 1
            elif Income_[i] == 6:
                count_income_lv6 += 1
        X_Income_lv1 = np.empty([count_income_lv1,sample_col])
        X_Income_lv2 = np.empty([count_income_lv2,sample_col])
        X_Income_lv3 = np.empty([count_income_lv3,sample_col])
        X_Income_lv4 = np.empty([count_income_lv4,sample_col])
        X_Income_lv5 = np.empty([count_income_lv5,sample_col])
        X_Income_lv6 = np.empty([count_income_lv6,sample_col])
        #print(count_income_lv1,count_income_lv2,count_income_lv3,count_income_lv4,count_income_lv5,count_income_lv6)
        for k in range(0,np.shape(X)[0]):
            if Income_[k] == 1:
                X_Income_lv1[temp1,:] = X[k,:]
                temp1 = temp1 + 1
            elif Income_[k] == 2:
                X_Income_lv2[temp2,:] = X[k,:]
                temp2 = temp2 + 1
            elif Income_[k] == 3:
                X_Income_lv3[temp3,:] = X[k,:]
                temp3 = temp3 + 1
            elif Income_[k] == 4:
                X_Income_lv4[temp4,:] = X[k,:]
                temp4 = temp4 + 1
            elif Income_[k] == 5:
                X_Income_lv5[temp5,:] = X[k,:]
                temp5 = temp5 + 1
            elif Income_[k] == 6:
                X_Income_lv6[temp6,:] = X[k,:]
                temp6 = temp6 + 1
        return X_Income_lv1,X_Income_lv2,X_Income_lv3,X_Income_lv4,X_Income_lv5,X_Income_lv6
    '''
    '''optional delete data with unknown gender,sample = 3262'''
    def delete_missing_gender(X):
        X_gender = X[:,1]
        m,n = np.shape(X)
        M = 0

        for x in range(m):
            if str(X[x,1]) != 'nan':
                M += 1

        X_renewed = np.zeros((M,n))
        count = 0
        for i in range(m):

            if str(X[i,1]) != 'nan':
                #print('sdsd',X[i,:])
                X_renewed[count] = np.r_[X[i,:]]
                count += 1
        return X_renewed
    '''optional:group by age '''
    def group_by_age(X):
        m,n = np.shape(X)
        age_col = X[:,0]

        for  i in range(m):
            if float(age_col[i]) < 1965.0:
                X[i,0] = 1.0
            elif float(age_col[i]) >= 1965.0 and float(age_col[i]) < 1980.0:
                X[i,0] = 2.0
            elif float(age_col[i]) >= 1980.0 and float(age_col[i]) < 1995.0:
                X[i,0] = 3.0
            elif float(age_col[i]) >= 1995.0:
                X[i,0] = 4.0
        return X
    #print("renewed",delete_missing_gender(X))
    #X_full = Predict_Gender(X)
    #X_full = Predict_Income(X)
    #X_full = Predict_Party(X)
    X = Predict_Age(X)
    X = group_by_age(X)
    X_Predicted_Gender = Predict_Gender(X)
    #'''delete missing gender data'''
    #X_Predicted_Gender = Predict_Gender(X)#'''no deleting, predict with mean'''
    #X_Predicted_Gender = delete_missing_gender(X)
    #print("test group by age",group_by_age(X))
    #X_Predicted_G_Income = Predict_Income(X_Predicted_Gender)
    #X_Predicted_G_I_Party = Predict_Party(X_Predicted_G_Income)
    #X_Predicted_G_Party = Predict_Party(X_Predicted_Gender)
    '''
    X_classfied_Gender_Male,X_classfied_Gender_Female is the first classfication result
    '''
    #X_classfied_Gender_Male,X_classfied_Gender_Female = classfication_tree_first(X_Predicted_G_Income)
    #X_classfied_Gender_Male,X_classfied_Gender_Female = classfication_tree_first(X_Predicted_G_Party)
    '''4/30 new method, age first,income second'''
    X_classfied_age_1,X_classfied_age_2,X_classfied_age_3,X_classfied_age_4 = classfication_age_first(X_Predicted_Gender)
    X_classfied_age_1 = Predict_Income(X_classfied_age_1)
    X_classfied_age_2 = Predict_Income(X_classfied_age_2)
    X_classfied_age_3 = Predict_Income(X_classfied_age_3)
    X_classfied_age_4 = Predict_Income(X_classfied_age_4)
    X_classfied_A_1_Income_1,X_classfied_A_1_Income_2,X_classfied_A_1_Income_3,X_classfied_A_1_Income_4,X_classfied_A_1_Income_5,X_classfied_A_1_Income_6 = classfication_income_second(X_classfied_age_1)
    X_classfied_A_2_Income_1,X_classfied_A_2_Income_2,X_classfied_A_2_Income_3,X_classfied_A_2_Income_4,X_classfied_A_2_Income_5,X_classfied_A_2_Income_6 = classfication_income_second(X_classfied_age_2)
    X_classfied_A_3_Income_1,X_classfied_A_3_Income_2,X_classfied_A_3_Income_3,X_classfied_A_3_Income_4,X_classfied_A_3_Income_5,X_classfied_A_3_Income_6 = classfication_income_second(X_classfied_age_3)
    X_classfied_A_4_Income_1,X_classfied_A_4_Income_2,X_classfied_A_4_Income_3,X_classfied_A_4_Income_4,X_classfied_A_4_Income_5,X_classfied_A_4_Income_6 = classfication_income_second(X_classfied_age_4)

    X_classfied_A_1_Income_1 = replace_directly(X_classfied_A_1_Income_1,strategy)
    X_classfied_A_1_Income_2 = replace_directly(X_classfied_A_1_Income_2,strategy)
    X_classfied_A_1_Income_3 = replace_directly(X_classfied_A_1_Income_3,strategy)
    X_classfied_A_1_Income_4 = replace_directly(X_classfied_A_1_Income_4,strategy)
    X_classfied_A_1_Income_5 = replace_directly(X_classfied_A_1_Income_5,strategy)
    X_classfied_A_1_Income_6 = replace_directly(X_classfied_A_1_Income_6,strategy)

    X_classfied_A_2_Income_1 = replace_directly(X_classfied_A_2_Income_1,strategy)
    X_classfied_A_2_Income_2 = replace_directly(X_classfied_A_2_Income_2,strategy)
    X_classfied_A_2_Income_3 = replace_directly(X_classfied_A_2_Income_3,strategy)
    X_classfied_A_2_Income_4 = replace_directly(X_classfied_A_2_Income_4,strategy)
    X_classfied_A_2_Income_5 = replace_directly(X_classfied_A_2_Income_5,strategy)
    X_classfied_A_2_Income_6 = replace_directly(X_classfied_A_2_Income_6,strategy)

    X_classfied_A_3_Income_1 = replace_directly(X_classfied_A_3_Income_1,strategy)
    X_classfied_A_3_Income_2 = replace_directly(X_classfied_A_3_Income_2,strategy)
    X_classfied_A_3_Income_3 = replace_directly(X_classfied_A_3_Income_3,strategy)
    X_classfied_A_3_Income_4 = replace_directly(X_classfied_A_3_Income_4,strategy)
    X_classfied_A_3_Income_5 = replace_directly(X_classfied_A_3_Income_5,strategy)
    X_classfied_A_3_Income_6 = replace_directly(X_classfied_A_3_Income_6,strategy)

    X_classfied_A_4_Income_1 = replace_directly(X_classfied_A_4_Income_1,strategy)
    X_classfied_A_4_Income_2 = replace_directly(X_classfied_A_4_Income_2,strategy)
    X_classfied_A_4_Income_3 = replace_directly(X_classfied_A_4_Income_3,strategy)
    X_classfied_A_4_Income_4 = replace_directly(X_classfied_A_4_Income_4,strategy)
    X_classfied_A_4_Income_5 = replace_directly(X_classfied_A_4_Income_5,strategy)
    X_classfied_A_4_Income_6 = replace_directly(X_classfied_A_4_Income_6,strategy)

    X_full_A_1 = np.vstack((X_classfied_A_1_Income_1,X_classfied_A_1_Income_2,X_classfied_A_1_Income_3,X_classfied_A_1_Income_4,X_classfied_A_1_Income_5,X_classfied_A_1_Income_6))
    X_full_A_2 = np.vstack((X_classfied_A_2_Income_1,X_classfied_A_2_Income_2,X_classfied_A_2_Income_3,X_classfied_A_2_Income_4,X_classfied_A_2_Income_5,X_classfied_A_2_Income_6))
    X_full_A_3 = np.vstack((X_classfied_A_3_Income_1,X_classfied_A_3_Income_2,X_classfied_A_3_Income_3,X_classfied_A_3_Income_4,X_classfied_A_3_Income_5,X_classfied_A_3_Income_6))
    X_full_A_4 = np.vstack((X_classfied_A_4_Income_1,X_classfied_A_4_Income_2,X_classfied_A_4_Income_3,X_classfied_A_4_Income_4,X_classfied_A_4_Income_5,X_classfied_A_4_Income_6))
    if isClassified == True:
        X_full = np.vstack((X_full_A_1,X_full_A_2,X_full_A_3,X_full_A_4))
        np.random.shuffle(X_full)
    '''
    is second classfied tree for male
    '''
    #X_classfied_M_Income_lv1,X_classfied_M_Income_lv2,X_classfied_M_Income_lv3,X_classfied_M_Income_lv4,X_classfied_M_Income_lv5,X_classfied_M_Income_lv6 = classfication_tree_second(X_classfied_Gender_Male)
    #X_classfied_M_Party_lv1,X_classfied_M_Party_lv2,X_classfied_M_Party_lv3,X_classfied_M_Party_lv4,X_classfied_M_Party_lv5 = classfication_tree_second(X_classfied_Gender_Male)
    '''
    is second classfied tree for female
    '''
    #X_classfied_F_Income_lv1,X_classfied_F_Income_lv2,X_classfied_F_Income_lv3,X_classfied_F_Income_lv4,X_classfied_F_Income_lv5,X_classfied_F_Income_lv6 = classfication_tree_second(X_classfied_Gender_Female)
    #X_classfied_F_Party_lv1,X_classfied_F_Party_lv2,X_classfied_F_Party_lv3,X_classfied_F_Party_lv4,X_classfied_F_Party_lv5 = classfication_tree_second(X_classfied_Gender_Female)
    '''
    Then we can use  replace_directly(X,strategy) function below
    X is the array,strategy = 'mean','median','mode'
    '''
    '''
    Third-level tree - Income
    '''
    '''
    X_classfied_F_P_lv1_Income_lv1, X_classfied_F_P_lv1_Income_lv2,X_classfied_F_P_lv1_Income_lv3,X_classfied_F_P_lv1_Income_lv4,X_classfied_F_P_lv1_Income_lv5,X_classfied_F_P_lv1_Income_lv6 = classfication_tree_third(X_classfied_F_Party_lv1)
    X_classfied_F_P_lv2_Income_lv1, X_classfied_F_P_lv2_Income_lv2,X_classfied_F_P_lv2_Income_lv3,X_classfied_F_P_lv2_Income_lv4,X_classfied_F_P_lv2_Income_lv5,X_classfied_F_P_lv2_Income_lv6 = classfication_tree_third(X_classfied_F_Party_lv2)
    X_classfied_F_P_lv3_Income_lv1, X_classfied_F_P_lv3_Income_lv2,X_classfied_F_P_lv3_Income_lv3,X_classfied_F_P_lv3_Income_lv4,X_classfied_F_P_lv3_Income_lv5,X_classfied_F_P_lv3_Income_lv6 = classfication_tree_third(X_classfied_F_Party_lv3)
    X_classfied_F_P_lv4_Income_lv1, X_classfied_F_P_lv4_Income_lv2,X_classfied_F_P_lv4_Income_lv3,X_classfied_F_P_lv4_Income_lv4,X_classfied_F_P_lv4_Income_lv5,X_classfied_F_P_lv4_Income_lv6 = classfication_tree_third(X_classfied_F_Party_lv4)
    X_classfied_F_P_lv5_Income_lv1, X_classfied_F_P_lv5_Income_lv2,X_classfied_F_P_lv5_Income_lv3,X_classfied_F_P_lv5_Income_lv4,X_classfied_F_P_lv5_Income_lv5,X_classfied_F_P_lv5_Income_lv6 = classfication_tree_third(X_classfied_F_Party_lv5)

    X_classfied_M_P_lv1_Income_lv1, X_classfied_M_P_lv1_Income_lv2,X_classfied_M_P_lv1_Income_lv3,X_classfied_M_P_lv1_Income_lv4,X_classfied_M_P_lv1_Income_lv5,X_classfied_M_P_lv1_Income_lv6 = classfication_tree_third(X_classfied_M_Party_lv1)
    X_classfied_M_P_lv2_Income_lv1, X_classfied_M_P_lv2_Income_lv2,X_classfied_M_P_lv2_Income_lv3,X_classfied_M_P_lv2_Income_lv4,X_classfied_M_P_lv2_Income_lv5,X_classfied_M_P_lv2_Income_lv6 = classfication_tree_third(X_classfied_M_Party_lv2)
    X_classfied_M_P_lv3_Income_lv1, X_classfied_M_P_lv3_Income_lv2,X_classfied_M_P_lv3_Income_lv3,X_classfied_M_P_lv3_Income_lv4,X_classfied_M_P_lv3_Income_lv5,X_classfied_M_P_lv3_Income_lv6 = classfication_tree_third(X_classfied_M_Party_lv3)
    X_classfied_M_P_lv4_Income_lv1, X_classfied_M_P_lv4_Income_lv2,X_classfied_M_P_lv4_Income_lv3,X_classfied_M_P_lv4_Income_lv4,X_classfied_M_P_lv4_Income_lv5,X_classfied_M_P_lv4_Income_lv6 = classfication_tree_third(X_classfied_M_Party_lv4)
    X_classfied_M_P_lv5_Income_lv1, X_classfied_M_P_lv5_Income_lv2,X_classfied_M_P_lv5_Income_lv3,X_classfied_M_P_lv5_Income_lv4,X_classfied_M_P_lv5_Income_lv5,X_classfied_M_P_lv5_Income_lv6 = classfication_tree_third(X_classfied_M_Party_lv5)
    '''
    '''
    X_classfied_M_Income_lv1 = replace_directly(X_classfied_M_Income_lv1,'median')
    X_classfied_M_Income_lv2 = replace_directly(X_classfied_M_Income_lv2,'median')
    X_classfied_M_Income_lv3 = replace_directly(X_classfied_M_Income_lv3,'median')
    X_classfied_M_Income_lv4 = replace_directly(X_classfied_M_Income_lv4,'median')
    X_classfied_M_Income_lv5 = replace_directly(X_classfied_M_Income_lv5,'median')
    X_classfied_M_Income_lv6 = replace_directly(X_classfied_M_Income_lv6,'median')

    X_classfied_F_Income_lv1 = replace_directly(X_classfied_F_Income_lv1,'median')
    X_classfied_F_Income_lv2 = replace_directly(X_classfied_F_Income_lv2,'median')
    X_classfied_F_Income_lv3 = replace_directly(X_classfied_F_Income_lv3,'median')
    X_classfied_F_Income_lv4 = replace_directly(X_classfied_F_Income_lv4,'median')
    X_classfied_F_Income_lv5 = replace_directly(X_classfied_F_Income_lv5,'median')
    X_classfied_F_Income_lv6 = replace_directly(X_classfied_F_Income_lv6,'median')
    '''
    '''
    X_classfied_M_Party_lv1 = replace_directly(X_classfied_M_Party_lv1,strategy)
    X_classfied_M_Party_lv2 = replace_directly(X_classfied_M_Party_lv2,strategy)
    X_classfied_M_Party_lv3 = replace_directly(X_classfied_M_Party_lv3,strategy)
    X_classfied_M_Party_lv4 = replace_directly(X_classfied_M_Party_lv4,strategy)
    X_classfied_M_Party_lv5 = replace_directly(X_classfied_M_Party_lv5,strategy)

    X_classfied_F_Party_lv1 = replace_directly(X_classfied_F_Party_lv1,strategy)
    X_classfied_F_Party_lv2 = replace_directly(X_classfied_F_Party_lv2,strategy)
    X_classfied_F_Party_lv3 = replace_directly(X_classfied_F_Party_lv3,strategy)
    X_classfied_F_Party_lv4 = replace_directly(X_classfied_F_Party_lv4,strategy)
    X_classfied_F_Party_lv5 = replace_directly(X_classfied_F_Party_lv5,strategy)
    '''
    #Female-third
    '''
    X_classfied_F_P_lv1_Income_lv1 = replace_directly(X_classfied_F_P_lv1_Income_lv1,strategy)
    X_classfied_F_P_lv1_Income_lv2 = replace_directly(X_classfied_F_P_lv1_Income_lv2,strategy)
    X_classfied_F_P_lv1_Income_lv3 = replace_directly(X_classfied_F_P_lv1_Income_lv3,strategy)
    X_classfied_F_P_lv1_Income_lv4 = replace_directly(X_classfied_F_P_lv1_Income_lv4,strategy)
    X_classfied_F_P_lv1_Income_lv5 = replace_directly(X_classfied_F_P_lv1_Income_lv5,strategy)
    X_classfied_F_P_lv1_Income_lv6 = replace_directly(X_classfied_F_P_lv1_Income_lv6,strategy)

    X_classfied_F_P_lv2_Income_lv1 = replace_directly(X_classfied_F_P_lv2_Income_lv1,strategy)
    X_classfied_F_P_lv2_Income_lv2 = replace_directly(X_classfied_F_P_lv2_Income_lv2,strategy)
    X_classfied_F_P_lv2_Income_lv3 = replace_directly(X_classfied_F_P_lv2_Income_lv3,strategy)
    X_classfied_F_P_lv2_Income_lv4 = replace_directly(X_classfied_F_P_lv2_Income_lv4,strategy)
    X_classfied_F_P_lv2_Income_lv5 = replace_directly(X_classfied_F_P_lv2_Income_lv5,strategy)
    X_classfied_F_P_lv2_Income_lv6 = replace_directly(X_classfied_F_P_lv2_Income_lv6,strategy)

    X_classfied_F_P_lv3_Income_lv1 = replace_directly(X_classfied_F_P_lv3_Income_lv1,strategy)
    X_classfied_F_P_lv3_Income_lv2 = replace_directly(X_classfied_F_P_lv3_Income_lv2,strategy)
    X_classfied_F_P_lv3_Income_lv3 = replace_directly(X_classfied_F_P_lv3_Income_lv3,strategy)
    X_classfied_F_P_lv3_Income_lv4 = replace_directly(X_classfied_F_P_lv3_Income_lv4,strategy)
    X_classfied_F_P_lv3_Income_lv5 = replace_directly(X_classfied_F_P_lv3_Income_lv5,strategy)
    X_classfied_F_P_lv3_Income_lv6 = replace_directly(X_classfied_F_P_lv3_Income_lv6,strategy)

    X_classfied_F_P_lv4_Income_lv1 = replace_directly(X_classfied_F_P_lv4_Income_lv1,strategy)
    X_classfied_F_P_lv4_Income_lv2 = replace_directly(X_classfied_F_P_lv4_Income_lv2,strategy)
    X_classfied_F_P_lv4_Income_lv3 = replace_directly(X_classfied_F_P_lv4_Income_lv3,strategy)
    X_classfied_F_P_lv4_Income_lv4 = replace_directly(X_classfied_F_P_lv4_Income_lv4,strategy)
    X_classfied_F_P_lv4_Income_lv5 = replace_directly(X_classfied_F_P_lv4_Income_lv5,strategy)
    X_classfied_F_P_lv4_Income_lv6 = replace_directly(X_classfied_F_P_lv4_Income_lv6,strategy)

    X_classfied_F_P_lv5_Income_lv1 = replace_directly(X_classfied_F_P_lv5_Income_lv1,strategy)
    X_classfied_F_P_lv5_Income_lv2 = replace_directly(X_classfied_F_P_lv5_Income_lv2,strategy)
    X_classfied_F_P_lv5_Income_lv3 = replace_directly(X_classfied_F_P_lv5_Income_lv3,strategy)
    X_classfied_F_P_lv5_Income_lv4 = replace_directly(X_classfied_F_P_lv5_Income_lv4,strategy)
    X_classfied_F_P_lv5_Income_lv5 = replace_directly(X_classfied_F_P_lv5_Income_lv5,strategy)
    X_classfied_F_P_lv5_Income_lv6 = replace_directly(X_classfied_F_P_lv5_Income_lv6,strategy)
    #Male-third
    X_classfied_M_P_lv1_Income_lv1 = replace_directly(X_classfied_M_P_lv1_Income_lv1,strategy)
    X_classfied_M_P_lv1_Income_lv2 = replace_directly(X_classfied_M_P_lv1_Income_lv2,strategy)
    X_classfied_M_P_lv1_Income_lv3 = replace_directly(X_classfied_M_P_lv1_Income_lv3,strategy)
    X_classfied_M_P_lv1_Income_lv4 = replace_directly(X_classfied_M_P_lv1_Income_lv4,strategy)
    X_classfied_M_P_lv1_Income_lv5 = replace_directly(X_classfied_M_P_lv1_Income_lv5,strategy)
    X_classfied_M_P_lv1_Income_lv6 = replace_directly(X_classfied_M_P_lv1_Income_lv6,strategy)

    X_classfied_M_P_lv2_Income_lv1 = replace_directly(X_classfied_M_P_lv2_Income_lv1,strategy)
    X_classfied_M_P_lv2_Income_lv2 = replace_directly(X_classfied_M_P_lv2_Income_lv2,strategy)
    X_classfied_M_P_lv2_Income_lv3 = replace_directly(X_classfied_M_P_lv2_Income_lv3,strategy)
    X_classfied_M_P_lv2_Income_lv4 = replace_directly(X_classfied_M_P_lv2_Income_lv4,strategy)
    X_classfied_M_P_lv2_Income_lv5 = replace_directly(X_classfied_M_P_lv2_Income_lv5,strategy)
    X_classfied_M_P_lv2_Income_lv6 = replace_directly(X_classfied_M_P_lv2_Income_lv6,strategy)

    X_classfied_M_P_lv3_Income_lv1 = replace_directly(X_classfied_M_P_lv3_Income_lv1,strategy)
    X_classfied_M_P_lv3_Income_lv2 = replace_directly(X_classfied_M_P_lv3_Income_lv2,strategy)
    X_classfied_M_P_lv3_Income_lv3 = replace_directly(X_classfied_M_P_lv3_Income_lv3,strategy)
    X_classfied_M_P_lv3_Income_lv4 = replace_directly(X_classfied_M_P_lv3_Income_lv4,strategy)
    X_classfied_M_P_lv3_Income_lv5 = replace_directly(X_classfied_M_P_lv3_Income_lv5,strategy)
    X_classfied_M_P_lv3_Income_lv6 = replace_directly(X_classfied_M_P_lv3_Income_lv6,strategy)

    X_classfied_M_P_lv4_Income_lv1 = replace_directly(X_classfied_M_P_lv4_Income_lv1,strategy)
    X_classfied_M_P_lv4_Income_lv2 = replace_directly(X_classfied_M_P_lv4_Income_lv2,strategy)
    X_classfied_M_P_lv4_Income_lv3 = replace_directly(X_classfied_M_P_lv4_Income_lv3,strategy)
    X_classfied_M_P_lv4_Income_lv4 = replace_directly(X_classfied_M_P_lv4_Income_lv4,strategy)
    X_classfied_M_P_lv4_Income_lv5 = replace_directly(X_classfied_M_P_lv4_Income_lv5,strategy)
    X_classfied_M_P_lv4_Income_lv6 = replace_directly(X_classfied_M_P_lv4_Income_lv6,strategy)

    X_classfied_M_P_lv5_Income_lv1 = replace_directly(X_classfied_M_P_lv5_Income_lv1,strategy)
    X_classfied_M_P_lv5_Income_lv2 = replace_directly(X_classfied_M_P_lv5_Income_lv2,strategy)
    X_classfied_M_P_lv5_Income_lv3 = replace_directly(X_classfied_M_P_lv5_Income_lv3,strategy)
    X_classfied_M_P_lv5_Income_lv4 = replace_directly(X_classfied_M_P_lv5_Income_lv4,strategy)
    X_classfied_M_P_lv5_Income_lv5 = replace_directly(X_classfied_M_P_lv5_Income_lv5,strategy)
    X_classfied_M_P_lv5_Income_lv6 = replace_directly(X_classfied_M_P_lv5_Income_lv6,strategy)
    '''
    '''
    X_full_M = np.vstack((X_classfied_M_Party_lv1,X_classfied_M_Party_lv2,X_classfied_M_Party_lv3,X_classfied_M_Party_lv4,X_classfied_M_Party_lv5))
    X_full_F = np.vstack((X_classfied_F_Party_lv1,X_classfied_F_Party_lv2,X_classfied_F_Party_lv3,X_classfied_F_Party_lv4,X_classfied_F_Party_lv5))
    X_full = np.vstack((X_full_F,X_full_M))#返回的是分类-mean 的结果
    #X_full = replace_directly(X,'mean')#直接返回的是无分类-mean 的结果
    '''
    '''
    X_full_F_1 = np.vstack((X_classfied_F_P_lv1_Income_lv1,X_classfied_F_P_lv1_Income_lv2,X_classfied_F_P_lv1_Income_lv3,X_classfied_F_P_lv1_Income_lv4,X_classfied_F_P_lv1_Income_lv5,X_classfied_F_P_lv1_Income_lv6))
    X_full_F_2 = np.vstack((X_classfied_F_P_lv2_Income_lv1,X_classfied_F_P_lv2_Income_lv2,X_classfied_F_P_lv2_Income_lv3,X_classfied_F_P_lv2_Income_lv4,X_classfied_F_P_lv2_Income_lv5,X_classfied_F_P_lv2_Income_lv6))
    X_full_F_3 = np.vstack((X_classfied_F_P_lv3_Income_lv1,X_classfied_F_P_lv3_Income_lv2,X_classfied_F_P_lv3_Income_lv3,X_classfied_F_P_lv3_Income_lv4,X_classfied_F_P_lv3_Income_lv5,X_classfied_F_P_lv3_Income_lv6))
    X_full_F_4 = np.vstack((X_classfied_F_P_lv4_Income_lv1,X_classfied_F_P_lv4_Income_lv2,X_classfied_F_P_lv4_Income_lv3,X_classfied_F_P_lv4_Income_lv4,X_classfied_F_P_lv4_Income_lv5,X_classfied_F_P_lv4_Income_lv6))
    X_full_F_5 = np.vstack((X_classfied_F_P_lv5_Income_lv1,X_classfied_F_P_lv5_Income_lv2,X_classfied_F_P_lv5_Income_lv3,X_classfied_F_P_lv5_Income_lv4,X_classfied_F_P_lv5_Income_lv5,X_classfied_F_P_lv5_Income_lv6))

    X_full_M_1 = np.vstack((X_classfied_M_P_lv1_Income_lv1,X_classfied_M_P_lv1_Income_lv2,X_classfied_M_P_lv1_Income_lv3,X_classfied_M_P_lv1_Income_lv4,X_classfied_M_P_lv1_Income_lv5,X_classfied_M_P_lv1_Income_lv6))
    X_full_M_2 = np.vstack((X_classfied_M_P_lv2_Income_lv1,X_classfied_M_P_lv2_Income_lv2,X_classfied_M_P_lv2_Income_lv3,X_classfied_M_P_lv2_Income_lv4,X_classfied_M_P_lv2_Income_lv5,X_classfied_M_P_lv2_Income_lv6))
    X_full_M_3 = np.vstack((X_classfied_M_P_lv3_Income_lv1,X_classfied_M_P_lv3_Income_lv2,X_classfied_M_P_lv3_Income_lv3,X_classfied_M_P_lv3_Income_lv4,X_classfied_M_P_lv3_Income_lv5,X_classfied_M_P_lv3_Income_lv6))
    X_full_M_4 = np.vstack((X_classfied_M_P_lv4_Income_lv1,X_classfied_M_P_lv4_Income_lv2,X_classfied_M_P_lv4_Income_lv3,X_classfied_M_P_lv4_Income_lv4,X_classfied_M_P_lv4_Income_lv5,X_classfied_M_P_lv4_Income_lv6))
    X_full_M_5 = np.vstack((X_classfied_M_P_lv5_Income_lv1,X_classfied_M_P_lv5_Income_lv2,X_classfied_M_P_lv5_Income_lv3,X_classfied_M_P_lv5_Income_lv4,X_classfied_M_P_lv5_Income_lv5,X_classfied_M_P_lv5_Income_lv6))

    X_full_F = np.vstack((X_full_F_1,X_full_F_2,X_full_F_3,X_full_F_4,X_full_F_5))
    X_full_M = np.vstack((X_full_M_1,X_full_M_2,X_full_M_3,X_full_M_4,X_full_M_5))
    #X_full = np.vstack((X_full_F,X_full_M))
    '''
    if isClassified == False:
        #X = delete_missing_gender(X)
        X_full = replace_directly(X,strategy)

    return X_full



    #X_full = replace_directly(X,'mode')
    #X_full = get_mean(X,1)
    #if isClassified == False:

    #elif isClassified == True:

    # I have complete the missing data filled by nan in transform



'''
code belowing is just for test
'''
'''
def main():
    filename_train = './data/train.csv'
    train_dataset = transform(filename_train)
    X = train_dataset['data']#includes ‘Happy’
    y = train_dataset['target']
    data = fill_missing(X, 'mean', False)

    m,n = np.shape(data)
    #print(m,n)
    np.savetxt("foo.csv",data,delimiter = ",")

if __name__ == '__main__':
    main()
'''
