# функция, возвращающая таблицу с процентом пропусков
def miss_values_table(df, numeric_cols_0):
    import numpy as np
    import pandas as pd
    means = np.zeros(df.shape[1]) 
    for j in range(df.shape[1]):
        if df.keys()[j] in numeric_cols_0:
            means[j] = df[df.columns[j]].mean()
            means[j] = round(means[j], 3)
        else:
            means[j] = -1
    means_data = pd.Series(means, df.columns)
       
        # Total missing values
    mis_val = df.isnull().sum()
        
        # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent, means_data], axis=1)
        
        # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values', 2 : 'Means'})
        
        # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns.sort_values(
    '% of Total Values', ascending=False)
        
    # Return the dataframe with missing information
    return mis_val_table_ren_columns
# проверяет, является ли переменная вещественной
def isfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
# проверяет, является ли переменная int
def isint(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
# проверяет, может ли быть переменная int
def can_be_int(s):
    try:
        s.astype(int)
        return True
    except ValueError:
        return False

#функция, счтиающая средние (можно сделать просто методом .mean())
def calculate_means(numeric_data):
    import numpy as np
    means = np.zeros(numeric_data.shape[1])
    for j in range(numeric_data.shape[1]):
        to_sum = numeric_data.iloc[:,j]
        indices = np.nonzero(~numeric_data.iloc[:,j].isnull())[0]
        correction = np.amax(to_sum[indices])
        to_sum /= correction
        for i in indices:
            means[j] += to_sum[i]
        means[j] /= indices.size
        means[j] *= correction
        means[j] = round(means[j], 2)
    return pd.Series(means, numeric_data.columns)

#функция, автоматически определяющая тип переменной, параметр 'n_uniq_to_be_cat' - необходимое число уникальных значений переменной,чтобы считать ее категориальной 

def split_to_diff_types(X_train, n_uniq_to_be_cat):
    real_features = []
    discrete_features = []
    cat_features = []
    X_train = X_train.dropna(how='all', axis=1)
    X_train = X_train.loc[:, (X_train != 0.0).any(axis=0)]
    for name in X_train.columns:
        if X_train[name].nunique() <= n_uniq_to_be_cat:
            cat_features.append(name)   #можно отнести к категориальным переменными числовые,имеющие мало уникальных значений
        else:
            if X_train[name].dtype == 'float64' and can_be_int(X_train[name]) == False:
                real_features.append(name)
            if X_train[name].dtype == 'float64' and can_be_int(X_train[name]) == True:
                discrete_features.append(name)
            if X_train[name].dtype == 'int64':
                discrete_features.append(name)
            if X_train[name].dtype == 'object':
                if isfloat(list(X_train[name].dropna())[0]) == False:
                    cat_features.append(name)
                else :
                    if isint(list(X_train[name].dropna())[0]) == True:
                        discrete_features.append(name)
                    else:
                        real_features.append(name)
    return real_features, discrete_features, cat_features

# функция, отбирающая только те переменные, в которых доля нулей менее заданного параметра k
def sec_filtr(first_filtr_vars, X_train, k):
    second_filtr = []
    for x in first_filtr_vars:
        amount=0
        for y in X_train[x]:
            if y == 0.0:
                amount+=1
        if amount/len(X_train[x]) <= k:
            second_filtr.append(x)
    return second_filtr

# функция, сшивающая два интервала, если конец левого правее начала правого 
def merge_ranges(meetings):
    sorted_meetings = sorted(meetings)
    merged_meetings = []
    previous_meeting_start, previous_meeting_end = sorted_meetings[0]
    for current_meeting_start, current_meeting_end in sorted_meetings[1:]:
        if current_meeting_start <= previous_meeting_end:
            previous_meeting_end = max(current_meeting_end, previous_meeting_end)       
        else:
            merged_meetings.append([previous_meeting_start, previous_meeting_end])
            previous_meeting_start, previous_meeting_end = \
                current_meeting_start, current_meeting_end
    merged_meetings.append([previous_meeting_start, previous_meeting_end])
    return merged_meetings

# функция, возвращающая пару индексов тех групп, которые лучшего всего 'сшить' на данном шаге (для вещественных) с точки зрения  статистики  хи-квадрат
def sew_inds(table, chi_crit):
    import scipy
    indexes_for_sewing = []
    if table != []:
        for i in range(len(table)-1):
            elem1 = table[i][:-1]
            elem2 = table[i+1][:-1]
            pair = [elem1, elem2]
            chi2_, prob, df, expected = scipy.stats.chi2_contingency(pair)
            if chi2_ < chi_crit:
                indexes_for_sewing.append([chi2_, [table[i][-1],table[i+1][-1]]])
        zz = sorted(indexes_for_sewing)
        if zz != []:
            return zz[0][1]
        else:
            return []
         
# функция, возвращающая пару индексов тех групп, которые лучшего всего 'сшить' на данном шаге (для категориальных) с точки зрения  статистики  хи-квадрат
def sew_inds_cat(table, chi_crit):
    import scipy
    indexes_for_sewing = []
    if table != []:
        meanings = []
        [meanings.append(x[:-1]) for x in table]
        print(meanings)
        bins = []
        [bins.append(x[2]) for x in table]
        print(bins)
        for i in range(len(meanings)):
            for j in range (i+1, len (meanings)):
                pair = [meanings[i],meanings[j]]
                chi2_, prob, df, expected = scipy.stats.chi2_contingency(pair)
                if chi2_ < chi_crit:
                    indexes_for_sewing.append([chi2_, [bins[i],bins[j]]])
        
        zz = sorted(indexes_for_sewing)
        if zz != []:
            return zz[0][1]
        else:
            return []
        
#получение списка вещественных и категориальных переменных после применения фильтров к исходным признакам
def return_new_cats_after_filtrs(second_filtr, numeric_cols_0):
    numeric_cols = []
    cat_features_new = []
    for x in second_filtr:
        if x in numeric_cols_0:
            numeric_cols.append(x)
        else:
            cat_features_new.append(x)
    return numeric_cols, cat_features_new
        
# нормально работающий optimal binning, нужно добавить ограничение на максимальное количество бакетов как в SAS (отличается от биннинга выше работой с пропусками - если их менее 5%, то они заменяются на среднее) 
def chi_sqare_binning(X_train, y_train, numeric_cols, chi_crit):
    import numpy as np
    import pandas as pd
    import scipy
    from drr import sew_inds
    num_bad = len(y_train[y_train == 1]) # общее количество бэдов
    num_good = len(y_train[y_train == 0]) # общее количество гудов
    d = pd.DataFrame({},index=[]) # создание пустого датасета(позднее он будет заполнен WOE, IV)
    for z in numeric_cols:
        max_bin = 20
        n = max_bin
        X = X_train[z]
        #разделяем датасет на пропуски и не пропуски
        df1 = pd.DataFrame({"X": X, "Y": y_train})
        justmiss = df1[['X','Y']][df1.X.isnull()]
        notmiss = df1[['X','Y']][df1.X.notnull()]
        #если доля пропусков меньше 5%, то заменяем их на среднее значение(в исходном датасете в том числе)
        if len(justmiss) > 0 and len(justmiss)/len(X_train) < 0.05:
            mean_val = X.mean()
            X = X.fillna(mean_val)
            X_train[z] = X
            df1 = pd.DataFrame({"X": X, "Y": y_train})
            #в этом случае непропущенные значения - это весь столбец, пропущенные - пусты
            notmiss = df1[['X','Y']][df1.X.notnull()]
            justmiss = []

        while n >= 1:
            try:    
                #пытаемся разделить на максимальное число квантилей
                cutting = pd.qcut(notmiss.X, n, retbins = True, duplicates = 'drop', labels = False)  
                print(len(cutting[1]))                                                          
                d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": cutting[0]})                                                   
                d2 = d1.groupby('Bucket', as_index=True)
                needed_inds = [2,3]               
            
                while needed_inds != [] :  
                
                    # таблица для подсчета хи квадрат
                    gg = pd.DataFrame({},index=[])
                    gg["EVENT"] = d2.sum().Y
                    gg["NONEVENT"] = d2.count().Y - d2.sum().Y
                    gg['buck'] = gg.T.iloc[0].keys()
                    #print(gg.T.iloc[0].keys())
                    gg=gg.reset_index(drop=True)
                    #print(gg)
                    table = []
                    for io, ou, ia in zip(list(gg["EVENT"]), list(gg["NONEVENT"]), list(gg['buck'])):
                        table.append([io, ou, ia])                      
                    # поиск индексов для 'сшития' на данном шаге
                
                    needed_inds = sew_inds(table, chi_crit)
                    print(needed_inds, z)
                   
                    if needed_inds != []:
                        jo = needed_inds[0]
                        ji = needed_inds[1]
                        d1['Bucket'][(d1['Bucket']<= ji) & (d1['Bucket']>= jo)] = jo
                        d2 = d1.groupby('Bucket', as_index=True)
            
                    else:                     
                        break
            
            
                # создание таблицы с гудами и бэдами для каждого бина
                d3 = pd.DataFrame({},index=[])
                d3["COUNT"] = d2.count().Y
                d3["EVENT"] = d2.sum().Y
                d3["NONEVENT"] = d2.count().Y - d2.sum().Y
                d3['Var'] = z
                d3["MIN_VALUE"] = d2.min().X
                d3["MAX_VALUE"] = d2.max().X
                d3['MAX_VALUE'][d3['MAX_VALUE'] == max(d3['MAX_VALUE'])] = np.inf
                d3['MIN_VALUE'][d3['MIN_VALUE'] == min(d3['MIN_VALUE'])] = -np.inf
                d3=d3.reset_index(drop=True)
                # если доля пропусков переменной меньше 0.05, то мы эти пропуски отбрасываем, а не выделяем в отдельны бин
                if len(justmiss) > 0 and len(justmiss)/len(X_train) >= 0.05:
                    d4 = pd.DataFrame({'COUNT': justmiss.count().Y},index=[0])
                    d4["EVENT"] = justmiss.sum().Y
                    d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
                    d4['Var'] = z
                    d4["MIN_VALUE"] = np.nan
                    d4["MAX_VALUE"] = np.nan
                    #print(d4)
                    d3 = d3.append(d4,ignore_index=True)  
                # если доля пропусков больше 0.05, то добавляем их как отдельный бин
                d3['Part_good'] = d3.NONEVENT/num_good
                d3['Part_bad'] = d3.EVENT/num_bad
                if len(d3["COUNT"][d3["COUNT"]/len(X_train) >= 0.05]) == len(d3["COUNT"]) and \
                                                        len(d3['Part_bad'][d3['Part_bad'] > 0]) == len(d3['Part_bad']):
                    d3["WOE"] = np.log(d3.Part_good/d3.Part_bad)
                    d3["Log_for_every_bin"] = (d3.Part_good-d3.Part_bad)*np.log(d3.Part_good/d3.Part_bad)
                    d3["IV"] = d3.sum().Log_for_every_bin 
                    #print(d3)
                    n = 0
                # если доля бэдов меньше, то делим дальше 
                else:
                    n = n-1
            except Exception as e:
                n = n-1
        d = d.append(d3,ignore_index=True)
    return d

# нормально работающий optimal binning с ограничением на максимальное количество бакетов как в SAS 
def chi_sqare_binning_limit(X_train, y_train, numeric_cols, chi_crit, max_bins):
    import numpy as np
    import pandas as pd
    import scipy
    from drr import sew_inds
    num_bad = len(y_train[y_train == 1]) # общее количество бэдов
    num_good = len(y_train[y_train == 0]) # общее количество гудов
    d = pd.DataFrame({},index=[]) # создание пустого датасета(позднее он будет заполнен WOE, IV)
    for z in numeric_cols:
        max_bin = 20
        n = max_bin
        X = X_train[z]
        #разделяем датасет на пропуски и не пропуски
        df1 = pd.DataFrame({"X": X, "Y": y_train})
        justmiss = df1[['X','Y']][df1.X.isnull()]
        notmiss = df1[['X','Y']][df1.X.notnull()]
        #если доля пропусков меньше 5%, то заменяем их на среднее значение(в исходном датасете в том числе)
        flag = True
        if len(justmiss) > 0 and len(justmiss)/len(X_train) < 0.05:
            flag = False
            mean_val = X.mean()
            X = X.fillna(mean_val)
            X_train[z] = X
            df1 = pd.DataFrame({"X": X, "Y": y_train})
            #в этом случае непропущенные значения - это весь столбец, пропущенные - пусты
            notmiss = df1[['X','Y']][df1.X.notnull()]
            justmiss = []
        
        while n >= 1:
            
            try:    
                #пытаемся разделить на максимальное число квантилей
                cutting = pd.qcut(notmiss.X, n, retbins = True, duplicates = 'drop', labels = False)  
                print(len(cutting[1]))                                                          
                d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": cutting[0]})                                                   
                d2 = d1.groupby('Bucket', as_index=True)
                needed_inds = [2,3]               
                ll = len(cutting[1])
                while needed_inds != []:  
                    ll = ll-1
                    # таблица для подсчета хи квадрат
                    gg = pd.DataFrame({},index=[])
                    gg["EVENT"] = d2.sum().Y
                    gg["NONEVENT"] = d2.count().Y - d2.sum().Y
                    gg['buck'] = gg.T.iloc[0].keys()
                    #print(gg.T.iloc[0].keys())
                    gg=gg.reset_index(drop=True)
                    #print(gg)
                    table = []
                    for io, ou, ia in zip(list(gg["EVENT"]), list(gg["NONEVENT"]), list(gg['buck'])):
                        table.append([io, ou, ia])                      
                    # поиск индексов для 'сшития' на данном шаге
                
                    needed_inds = sew_inds(table, chi_crit)
                    print(needed_inds, z)
                   
                    if needed_inds != []:
                        jo = needed_inds[0]
                        ji = needed_inds[1]
                        d1['Bucket'][(d1['Bucket']<= ji) & (d1['Bucket']>= jo)] = jo
                        d2 = d1.groupby('Bucket', as_index=True)
            
                    else:                     
                        break
                # если есть отдельный бакет с пропусками, то мы его учитываем здесь, чтобы не было 6 бакетов вместо 5
                if flag:
                    ll = ll+1
                        
                while ll > max_bins:
                    ll=ll-1
                    print(ll)
                    # таблица для подсчета хи квадрат
                    gg = pd.DataFrame({},index=[])
                    gg["EVENT"] = d2.sum().Y
                    gg["NONEVENT"] = d2.count().Y - d2.sum().Y
                    gg['buck'] = gg.T.iloc[0].keys()
                    #print(gg.T.iloc[0].keys())
                    gg=gg.reset_index(drop=True)
                    #print(gg)
                    table = []
                    for io, ou, ia in zip(list(gg["EVENT"]), list(gg["NONEVENT"]), list(gg['buck'])):
                        table.append([io, ou, ia])                      
                    # поиск индексов для 'сшития' на данном шаге
                
                    needed_inds = sew_inds_limit(table)
                    print(needed_inds, z)
                   
                       
                    jo = needed_inds[0]
                    ji = needed_inds[1]
                    d1['Bucket'][(d1['Bucket']<= ji) & (d1['Bucket']>= jo)] = jo
                    d2 = d1.groupby('Bucket', as_index=True)
            
                    
                # создание таблицы с гудами и бэдами для каждого бина
                d3 = pd.DataFrame({},index=[])
                d3["COUNT"] = d2.count().Y
                d3["EVENT"] = d2.sum().Y
                d3["NONEVENT"] = d2.count().Y - d2.sum().Y
                d3['Var'] = z
                d3["MIN_VALUE"] = d2.min().X
                d3["MAX_VALUE"] = d2.max().X
                d3['MAX_VALUE'][d3['MAX_VALUE'] == max(d3['MAX_VALUE'])] = np.inf
                d3['MIN_VALUE'][d3['MIN_VALUE'] == min(d3['MIN_VALUE'])] = -np.inf
                d3=d3.reset_index(drop=True)
                # если доля пропусков переменной меньше 0.05, то мы эти пропуски отбрасываем, а не выделяем в отдельны бин
                if len(justmiss) > 0 and len(justmiss)/len(X_train) >= 0.05:
                    d4 = pd.DataFrame({'COUNT': justmiss.count().Y},index=[0])
                    d4["EVENT"] = justmiss.sum().Y
                    d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
                    d4['Var'] = z
                    d4["MIN_VALUE"] = np.nan
                    d4["MAX_VALUE"] = np.nan
                    #print(d4)
                    d3 = d3.append(d4,ignore_index=True)  
                # если доля пропусков больше 0.05, то добавляем их как отдельный бин
                d3['Part_good'] = d3.NONEVENT/num_good
                d3['Part_bad'] = d3.EVENT/num_bad
                if len(d3["COUNT"][d3["COUNT"]/len(X_train) >= 0.05]) == len(d3["COUNT"]) and \
                                                        len(d3['Part_bad'][d3['Part_bad'] > 0]) == len(d3['Part_bad']):
                    d3["WOE"] = np.log(d3.Part_good/d3.Part_bad)
                    d3["Log_for_every_bin"] = (d3.Part_good-d3.Part_bad)*np.log(d3.Part_good/d3.Part_bad)
                    d3["IV"] = d3.sum().Log_for_every_bin 
                    #print(d3)
                    n = 0
                # если доля бэдов меньше, то делим дальше 
                else:
                    n = n-1
            except Exception as e:
                n = n-1
        d = d.append(d3,ignore_index=True)
    return d


# биннинг для категориальных переменных, требует доработок
def cat_bin(X_train, y_train, cat_features, chi_crit):
    import numpy as np
    import pandas as pd
    import scipy
    from drr import sew_inds_cat
    num_bad = len(y_train[y_train == 1])
    num_good = len(y_train[y_train == 0])
    d0 = pd.DataFrame({},index=[])
    cat_feat_norm = [] # список для будующих "хороших" переменных
    for z in cat_features:       #cat_features
        X = X_train[z]
        df1 = pd.DataFrame({"X": X, "Y": y_train})
        justmiss = df1[['X','Y']][df1.X.isnull()]
        notmiss = df1[['X','Y']][df1.X.notnull()]       
        d1 = notmiss
        norm_feat =[]
        list_of_cats = pd.unique(d1["X"].values.ravel('K'))
        for x in list_of_cats:
            if len(d1['X'][d1['X'] == x])/len(X_train) >= 0.05:
                norm_feat.append(x)
        # должен создасться список "хороших" категорий переменных
            
            
            
        if len(norm_feat) != len(list_of_cats):
            continue 
        # только если все категории переменной - хорошие, мы продолжаем 
    
        cat_feat_norm.append(z)
        #print(norm_feat)
        d1 = d1.loc[d1['X'].isin(norm_feat)]
        #print(d1)       
    
        # каждому значению категории присваиваем уникальный номер
        fg = sorted(norm_feat)  
        gf = range(len(fg))
    
        kyes_for_cat = []
        for m,n in zip(fg, gf):
            kyes_for_cat.append([m,n])
        
        #создаем столбец бакет 
        d1['Bucket'] = d1['X']
        for i in range(len(kyes_for_cat)):
            d1['Bucket'][d1['Bucket'] == kyes_for_cat[i][0]] = kyes_for_cat[i][1]
        
        
        #df2 = d1.groupby('X',as_index=True)      
        #группируем по бакетам 
        d2 = d1.groupby('Bucket',as_index=True)  
        needed_inds = [2,3]
                #print(needed_inds)
                # на каждом сшаге сшиваем те бины, которые можно сшить по критерию хи квадрат 
        while needed_inds != []:  
                
                # таблица для подсчета хи квадрат
                    gg = pd.DataFrame({},index=[])
                    gg["EVENT"] = d2.sum().Y
                    gg["NONEVENT"] = d2.count().Y - d2.sum().Y
                    gg['buck'] = gg.T.iloc[0].keys()
                    #print(gg.T.iloc[0].keys())
                    gg=gg.reset_index(drop=True)
                    #print(gg)
                    table = []
                    for io, ou, ia in zip(list(gg["EVENT"]), list(gg["NONEVENT"]), list(gg['buck'])):
                        table.append([io, ou, ia])  
                    #print(table)
                    # поиск индексов для 'сшития' на данном шаге
                
                    needed_inds = sew_inds_cat(table, chi_crit)
                    print(needed_inds, z)
               
                    if needed_inds != []:
                        jo = needed_inds[0]
                        ji = needed_inds[1]          
                        d1['X'][d1['Bucket'].isin(needed_inds)] = str(str(d1['X'][d1['Bucket'] == jo].iloc[0]) +', '\
                                                                      + str(d1['X'][d1['Bucket'] == ji].iloc[0]))
                        d1['Bucket'][d1['Bucket'].isin(needed_inds)] = jo
                        d2 = d1.groupby('Bucket', as_index=True)
            
                    else:
                        #print(d2)
                        break
            
        d3 = pd.DataFrame({},index=[])
        inds_surv_cats0 = list(d2.sum().Y.index)
        #dfg = np.array(list(d2['X']))
        #print(dfg)
        loi0 = d2['X'].unique()
        loi1 = []
        [loi1.append(*lopy) for lopy in loi0]
        loi =[]
        for elk in loi1:
            if isfloat(elk) == True:
                loi.append(float(elk))
            else:
                try :
                    kle = list(map(float, elk.split(', ')))
                    #print(kle)
                    loi.append(kle)
                except:
                    loi.append(elk)       
        
    #surv_cats = []
    #inds_surv_cats = []
    #[inds_surv_cats.append(int(ol)) for ol in inds_surv_cats0]
    #print(inds_surv_cats)
    #for ele in inds_surv_cats:
        #surv_cats.append(fg[ele])
    #print(surv_cats)
        d3["COUNT"] = d2.count().Y
        d3["EVENT"] = d2.sum().Y
        d3["NONEVENT"] = d2.count().Y - d2.sum().Y
        d3['Var'] = z
        d3["MIN_VALUE"] = loi
        d3["MAX_VALUE"] = d3["MIN_VALUE"]
        #d3=d3.reset_index(drop=True)
        #print(d3)
        if len(justmiss) > 0 and len(justmiss)/len(X_train) >= 0.05:
            d4 = pd.DataFrame({'COUNT': justmiss.count().Y},index=[0])
            d4["EVENT"] = justmiss.sum().Y
            d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
            d4['Var'] = z
            d4["MIN_VALUE"] = np.nan
            d4["MAX_VALUE"] = np.nan
            d3 = d3.append(d4,ignore_index=True)
        d3['Part_good'] = d3.NONEVENT/num_good
        d3['Part_bad'] = d3.EVENT/num_bad
        #print(d3)
        #if len(d3['Part_bad'][d3['Part_bad'] >= 0.05]) == len(d3['Part_bad']):
        d3["WOE"] = np.log(d3.Part_good/d3.Part_bad)
        d3["Log_for_every_bin"] = (d3.Part_good-d3.Part_bad)*np.log(d3.Part_good/d3.Part_bad)
        d3["IV"] = d3.sum().Log_for_every_bin 
        #print(d3)
        d0 = d0.append(d3,ignore_index=True)
    return d0

# функция для анализа мультиколлинеарности
def calculate_vif(X_):
    import pandas as pd
    vif = pd.DataFrame()
    to_vif = []
    vif["Features"] = X_.columns
    for i in range(X_.shape[1]):
        try:
            to_vif.append(variance_inflation_factor(X_.values, i))
        except:
            to_vif.append(0)
    vif["VIF"] = to_vif
    return(vif)


# основываясь на полученном бинниге, данная функция возвращает датасет, состоящий из WOE
def transform_data_to_woe1(IV_final, X_train, good_r, good_cat):
    import numpy as np
    import pandas as pd
    X_to_fact = pd.DataFrame({},index=[])
    X_for_model_fact = pd.DataFrame({},index=[])
    for x in X_train.columns:
        if x in good_r:
            lit_data0 = IV_final[IV_final['Var'] == x]
            lit_data = lit_data0.sort_values(['MIN_VALUE'])
            X_to_fact[x] = X_train[x]
            X_for_model_fact[x] = X_to_fact[x]
            if len(lit_data) > 2:
                for i in range(len(lit_data)):
                    if lit_data.iloc[-1].isnull().values.any() == False:
                        if i < len(lit_data)-1:
                            im = lit_data.iloc[i]
                            mok = lit_data.iloc[i+1]
                            if im.isnull().values.any() == False:
                                X_for_model_fact[x][(X_to_fact[x] >= im['MIN_VALUE']) & (X_to_fact[x] < mok['MIN_VALUE'])] = float(im['WOE'])
                            else:
                                X_for_model_fact[x][X_to_fact[x].isnull()] = float(im['WOE'])
                        else:
                            im = lit_data.iloc[i]
                        
                            if im.isnull().values.any() == False:
                                X_for_model_fact[x][(X_to_fact[x] >= im['MIN_VALUE']) & (X_to_fact[x] < im['MAX_VALUE'])] = float(im['WOE'])
                            else:
                                X_for_model_fact[x][X_to_fact[x].isnull()] = float(im['WOE'])
                    else:
                        if i < len(lit_data)-2:
                            im = lit_data.iloc[i]
                            mok = lit_data.iloc[i+1]
                        
                            X_for_model_fact[x][(X_to_fact[x] >= im['MIN_VALUE']) & (X_to_fact[x] < mok['MIN_VALUE'])] = float(im['WOE'])
                        if i == len(lit_data)-2:
                            im = lit_data.iloc[i]
                        
                            X_for_model_fact[x][(X_to_fact[x] >= im['MIN_VALUE']) & (X_to_fact[x] < im['MAX_VALUE'])] = float(im['WOE'])
                        else:
                            im = lit_data.iloc[i]
                            X_for_model_fact[x][X_to_fact[x].isnull()] = float(im['WOE'])
            else:
                for i in range(len(lit_data)):
                    im = lit_data.iloc[i]
                    #print(im)
                    #print(im['WOE'])
                    if im.isnull().values.any() == False:
                        X_for_model_fact[x][(X_to_fact[x] >= im['MIN_VALUE']) & (X_to_fact[x] <= im['MAX_VALUE'])] = float(im['WOE'])
                    else:
                        X_for_model_fact[x][X_to_fact[x].isnull()] = float(im['WOE'])                        
        if x in good_cat:
            lit_data = IV_final[IV_final['Var'] == x]
            X_to_fact[x] = X_train[x]
            X_for_model_fact[x] = X_to_fact[x]
            #keu = X_to[x].keys()
            for i in range(len(lit_data)):
                im = lit_data.iloc[i]
                #print(im)
                #print(im['WOE'])
                if im.isnull().values.any() == False:
                    try:
                         X_for_model_fact[x][(X_to_fact[x] == im['MIN_VALUE'])] = float(im['WOE'])
                    except:
                        X_for_model_fact[x][(X_to_fact[x].isin(im['MIN_VALUE']))] = float(im['WOE'])
                else:
                    X_for_model_fact[x][X_to_fact[x].isnull()] = float(im['WOE'])
    return X_for_model_fact

# поиск оптимального cutoff для заданного таргет и полученных предсказаний
def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]

    return roc_t['threshold']
# сшивает бакеты с наименьшим хи-квадрат
def sew_inds_limit(table):
    import scipy
    indexes_for_sewing = []
    for i in range(len(table)-1):
        elem1 = table[i][:-1]
        elem2 = table[i+1][:-1]
        pair = [elem1, elem2]
        chi2_, prob, df, expected = scipy.stats.chi2_contingency(pair)
        indexes_for_sewing.append([chi2_, [table[i][-1],table[i+1][-1]]])
    zz = sorted(indexes_for_sewing)   
    if zz != []:
        return zz[0][1]
    else:
        return []

# сшивает бакеты для монотонного биннинга
def sew_inds_entropy(table, tg, max_bins):
    import math as mt
    import scipy
    indexes_for_sewing = []
    indexes_for_sewing0 = []
    for i in range(len(table)-1):
        elem1 = table[i][:-3]
        elem2 = table[i+1][:-3]
        a1 = elem1[0] #event
        b1 = elem1[1] #count
        a2 = elem2[0] #event
        b2 = elem2[1] #count
        a0 = a1 + a2 
        b0 = b1 + b2
        dr0 = a0/b0
        H0 = -dr0*np.log(dr0)-(1-dr0)*np.log(1-dr0)
        dr1 = a1/b1
        dr2 = a2/b2
        H1 = (-dr1*np.log(dr1)-(1-dr1)*np.log(1-dr1))*b1/(b1+b2)
        
        
        #WOE1 = np.log((1-dr1)/dr1)
        #WOE2 = np.log((1-dr2)/dr2)
        #PP1 = -dr1*mt.log2(dr1)
        #G1 = 2*dr1*(1-dr1)*b1/(b1+b2)
        
        H2 = (-dr2*np.log(dr2)-(1-dr2)*np.log(1-dr2))*b2/(b1+b2)
        dd = H0 - H1 - H2
        #dd = abs(H1 - H2)
        dist = np.abs(dr2 - dr1)
        
        #PP1 = -dr2*mt.log2(dr2)
        #G2 = 2*dr2*(1-dr2)*b2/(b1+b2)
        delta = np.abs(dd)
        #print(H1,H2, delta)
        indexes_for_sewing0.append([delta, [table[i][-3],table[i+1][-3]], [table[i+1][-2], table[i+1][-1]]])
        if tg < 0 :
            print(1)
            if dr2 > dr1:
                indexes_for_sewing.append([dist, [table[i][-3],table[i+1][-3]], [table[i+1][-2], table[i+1][-1]]])
        if tg >= 0 :
            if dr2 < dr1:
                print(2)
                indexes_for_sewing.append([dist, [table[i][-3],table[i+1][-3]], [table[i+1][-2], table[i+1][-1]]])
    print(indexes_for_sewing)
    bound = []
    [bound.append([x[0], x[-1]]) for x in indexes_for_sewing]
    print(bound)
    zz = sorted(indexes_for_sewing) 
    print(zz)
    if zz != []:
        print('сшиваю по монотонности')
        return zz[-1][1]
       
    else:
        if len(indexes_for_sewing0) >= max_bins:
            print('kkk')
            bound = []
            [bound.append([x[0], x[-1]]) for x in indexes_for_sewing0]
            #print(bound)
            zz = sorted(indexes_for_sewing0) 
            if zz != []:
                print('сшиваю по энтропии')
                return zz[0][1]
                
            else:
                return []
        else:
            return []  
        
        
# реализация биннинга с монотонным event rate
def monotonig_event_rate_binning(X_train, y_train, numeric_cols, chi_crit, max_bins):
    import numpy as np
    import pandas as pd
    import math
    import scipy
    from drr import sew_inds
    from sklearn.linear_model import LinearRegression
    num_bad = len(y_train[y_train == 1]) # общее количество бэдов
    num_good = len(y_train[y_train == 0]) # общее количество гудов
    d = pd.DataFrame({},index=[]) # создание пустого датасета(позднее он будет заполнен WOE, IV)
    #flag = True
    for z in numeric_cols[35:36]:
        max_bin = 20
        X = X_train[z]
        #разделяем датасет на пропуски и не пропуски
        df1 = pd.DataFrame({"X": X, "Y": y_train})
        justmiss = df1[['X','Y']][df1.X.isnull()]
        notmiss = df1[['X','Y']][df1.X.notnull()]
        n = math.floor(max_bin*(1-len(justmiss)/(len(notmiss)+len(justmiss))))
        max_bins0 = max_bins - 1
        #если доля пропусков меньше 5%, то заменяем их на среднее значение(в исходном датасете в том числе)
        if len(justmiss) >= 0 and len(justmiss)/len(X_train) < 0.05:
            #flag = False
            max_bins0 = max_bins 
            mean_val = X.mean()
            X = X.fillna(mean_val)
            X_train[z] = X
            df1 = pd.DataFrame({"X": X, "Y": y_train})
            #в этом случае непропущенные значения - это весь столбец, пропущенные - пусты
            notmiss = df1[['X','Y']][df1.X.notnull()]
            justmiss = []
        
        while n >= 1:
            
            try:    
                #пытаемся разделить на максимальное число квантилей
                cutting = pd.qcut(notmiss.X, n, retbins = True, duplicates = 'drop', labels = False)  
                print(len(cutting[1])) 
                print(max_bins0)
                #print(cutting[1])
                d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": cutting[0]})     
                d2 = d1.groupby('Bucket', as_index=True)
                ggg = pd.DataFrame({},index=[])
                ggg["dr"] = d2.sum().Y/d2.count().Y
                ggg['buck'] = ggg.T.iloc[0].keys()
                regr = LinearRegression()
                fitt_regr = regr.fit(np.array(ggg['buck']).reshape(-1, 1), ggg["dr"])
                tg = fitt_regr.coef_
                needed_inds = [2,3]               
                ll = len(cutting[1])
#                 if flag:
#                     max_bins0 = max_bins - 1 
#                     flag = False
                while needed_inds != [] : #and ll > max_bins + 1:  
                    ll = ll-1
                    #print(ll)
                    # таблица для подсчета хи квадрат
                    gg = pd.DataFrame({},index=[])
                    gg["EVENT"] = d2.sum().Y
                    gg["COUNT"] = d2.count().Y
                    gg['buck'] = gg.T.iloc[0].keys()
                    gg["MIN_VALUE"] = d2.min().X
                    gg["MAX_VALUE"] = d2.max().X
                    #print(len(gg.T.iloc[0].keys()))
                    gg=gg.reset_index(drop=True)
                    #print(gg)
                    table = []
                    for io, ou, ia, mn, mx in zip(list(gg["EVENT"]), list(gg["COUNT"]), list(gg['buck']),
                                          list(gg["MIN_VALUE"]),list(gg["MAX_VALUE"])):
                        table.append([io, ou, ia, mn, mx])                      
                    # поиск индексов для 'сшития' на данном шаге
                
                    needed_inds = sew_inds_entropy(table, tg, max_bins0)
                    print(needed_inds, z)
                   
                    if needed_inds != []:
                        jo = needed_inds[0]
                        ji = needed_inds[1]
                        #d1['Bucket'][(d1['Bucket']<= ji) & (d1['Bucket']>= jo)] = jo
                        d1['Bucket'][(d1['Bucket']== ji)] = jo
                        d2 = d1.groupby('Bucket', as_index=True)
            
                    else:                     
                        break
                print(d2)
                # если есть отдельный бакет с пропусками, то мы его учитываем здесь, чтобы не было 6 бакетов вместо 5
#                 if flag:
#                     ll = ll+1
                        
#                 while ll > max_bins:
#                     ll=ll-1
#                     print(ll)
#                     # таблица для подсчета хи квадрат
#                     gg = pd.DataFrame({},index=[])
#                     gg["EVENT"] = d2.sum().Y
#                     gg["NONEVENT"] = d2.count().Y - d2.sum().Y
#                     gg['buck'] = gg.T.iloc[0].keys()
#                     #print(gg.T.iloc[0].keys())
#                     gg=gg.reset_index(drop=True)
#                     #print(gg)
#                     table = []
#                     for io, ou, ia in zip(list(gg["EVENT"]), list(gg["NONEVENT"]), list(gg['buck'])):
#                         table.append([io, ou, ia])                      
#                     # поиск индексов для 'сшития' на данном шаге
                
#                     needed_inds = sew_inds_limit(table)
#                     print(needed_inds, z)
                   
                       
#                     jo = needed_inds[0]
#                     ji = needed_inds[1]
#                     d1['Bucket'][(d1['Bucket']<= ji) & (d1['Bucket']>= jo)] = jo
#                     d2 = d1.groupby('Bucket', as_index=True)
            
                    
                # создание таблицы с гудами и бэдами для каждого бина
                d3 = pd.DataFrame({},index=[])
                d3["COUNT"] = d2.count().Y
                d3["EVENT"] = d2.sum().Y
                d3["NONEVENT"] = d2.count().Y - d2.sum().Y
                d3['Var'] = z
                d3["MIN_VALUE"] = d2.min().X
                d3["MAX_VALUE"] = d2.max().X
                d3['MAX_VALUE'][d3['MAX_VALUE'] == max(d3['MAX_VALUE'])] = np.inf
                d3['MIN_VALUE'][d3['MIN_VALUE'] == min(d3['MIN_VALUE'])] = -np.inf
                d3=d3.reset_index(drop=True)
                # если доля пропусков переменной меньше 0.05, то мы эти пропуски отбрасываем, а не выделяем в отдельны бин
                if len(justmiss) > 0 and len(justmiss)/len(X_train) >= 0.05:
                    d4 = pd.DataFrame({'COUNT': justmiss.count().Y},index=[0])
                    d4["EVENT"] = justmiss.sum().Y
                    d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
                    d4['Var'] = z
                    d4["MIN_VALUE"] = np.nan
                    d4["MAX_VALUE"] = np.nan
                    #print(d4)
                    d3 = d3.append(d4,ignore_index=True)  
                # если доля пропусков больше 0.05, то добавляем их как отдельный бин
                d3['Part_good'] = d3.NONEVENT/num_good
                d3['Part_bad'] = d3.EVENT/num_bad
                d3['dr'] = d3["EVENT"]/d3["COUNT"]
#                 if len(d3["COUNT"][d3["COUNT"]/len(X_train) >= 0.05]) == len(d3["COUNT"]) and \
#                                                         len(d3['Part_bad'][d3['Part_bad'] > 0]) == len(d3['Part_bad']):
                d3["WOE"] = np.log(d3.Part_good/d3.Part_bad)
                d3["Log_for_every_bin"] = (d3.Part_good-d3.Part_bad)*np.log(d3.Part_good/d3.Part_bad)
                d3["IV"] = d3.sum().Log_for_every_bin 
                    #print(d3)
                n = 0
                # если доля бэдов меньше, то делим дальше 
#                 else:
#                     n = n-1
#                     print("что-то с условиями")
            except Exception as e:
                n = n-1
        d = d.append(d3,ignore_index=True)
    return d