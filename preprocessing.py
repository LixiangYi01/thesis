import csv
import os.path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight

# Table to transform key names in datasets to common format
def init_keynames():
    keys={}
    # keep the same
    for c in 'abcdefghijklmnopqrstuvwxyz0123456789':
        keys[c]=c
    for s in ['enter','space','backspace','windows','command','end','home','insert','menu','unknown','alt','f12','ctrl']:
        keys[s]=s
    for s in ['numpad_'+d for d in list('0123456789')+['decimal']]:
        keys[s]=s
    for s in ['page_'+s for s in ['down','up']]:
        keys[s]=s
    # CMU-specific
    keys['zero'],keys['one']='0','1'
    keys['two'],keys['three']='2','3'
    keys['four'],keys['five']='4','5'
    keys['six'],keys['seven']='6','7'
    keys['eight'],keys['nine']='8','9'
    keys['Return']='enter'
    keys['Delete']='backspace'    
    keys['Control']='ctrl'
    for s in ['Tab','Select']:
        keys[s]=s.lower()
    # Villani-specific
    keys['back_slash']='\\'
    keys['num_lock']='numlock'
    keys['dead_acute']='acute'
    keys['F12']='f12'
    keys['F13']='f13'
    # Comon
    keys['comma']=','
    keys['period']='.'
    keys['semicolon']=';'
    keys['slash']='/'
    keys['backslash']='\\'
    for s in ['apostrophe','quote']:
        keys[s]='\''
    for s in ['minus','dash']:
        keys[s]='-'
    for s in ['bracketleft','open_bracket']:
        keys[s]='('
    for s in ['bracketright','close_bracket']:
        keys[s]=')'
    for s in ['caps_lock','Caps_Lock']:
        keys[s]='capslock'
    for s in ['equal','equals']:
        keys[s]='='
    for s in ['grave','back_quote']:
        keys[s]='`'
    for s in ['delete','Remove']:
        keys[s]='del'
    for s in ['down','Down','up','Up','left','Left','right','Right','shift','Shift']:
        keys[s]=s.lower()
    return keys

# Transform CMU dataset to common format
def to_csv_cmu():
    # keyname table
    keynames=init_keynames()
    # File names
    p_dd=os.path.join('..','data','cmu_free_vs_transcribed','TimingFeatures-DD.txt')
    p_hold=os.path.join('..','data','cmu_free_vs_transcribed','TimingFeatures-Hold.txt')
    p_map=os.path.join('..','data','cmu_free_vs_transcribed','SessionMap.txt')
    p_out=os.path.join('..','data','00_cmu.csv')
    # Open input and output files
    with open(p_dd) as f_dd,open(p_hold) as f_hold,open(p_map) as f_map,open(p_out,'w') as f_out:
        writer=csv.writer(f_out)
        # Write header to output file
        writer.writerow(['user','session','screen','inputtype','keyname','duration','transition'])
        # Skip headers in input files
        f_dd.readline(),f_hold.readline(),f_map.readline()
        sessionIndex,screenIndex='',''
        for line in f_hold:
            row=[]
            l_hold=line.split()
            # Set transition and inputtype
            if l_hold[1]!=sessionIndex:
                transition='0'
                inputType=f_map.readline()[-7:-2]
                # Trim leading space from 'Free'
                if inputType[0]==' ':
                    inputType=inputType[1:]
                # Rename value according to other dataset
                if inputType=='Trans':
                    inputType='fixed'
                elif inputType=='Free':
                    inputType='free'
            elif l_hold[2]!=screenIndex:
                # Ignore transitions between screens
                f_dd.readline()
                transition='0'
            else:
                transition=f_dd.readline().split()[-1]
            # Set key (only remember second key if a combination)
            key=keynames[l_hold[4].split('.')[-1]]
            row+=l_hold[0:3]     # Write subject,sessionIndex and screenIndex
            row+=[inputType,key] # Write inputtype and key
            row+=[l_hold[5]]     # Write duration
            row+=[transition]    # Write transition
            # Set current session and screen
            sessionIndex,screenIndex=row[1],row[2]
            # Write line to output CSV
            writer.writerow(row)

# Transform Villani dataset to common format
def to_csv_villani():
    def filter_users(df,n=36):
        sessions_per_user=df.groupby('user')['session'].nunique()
        users=sessions_per_user.sort_values(ascending=False)[:n].index.tolist()
        return df[df['user'].isin(users)]
    def transition(group):
        return group - group.shift(1)
    df=pd.read_csv('../data/villani/keystroke.csv')
    # Drop unneeded features
    df.drop(['task','repetition','gender','agegroup','handedness','awareness'],axis=1,inplace=True)
    # Dropping useless sessions (just same character repeated)
    df=df[df['session']!=308]
    df=df[df['session']!=892]
    df=df[df['session']!=963]
    # Keep sessions for 36 users with most sessions
    df=filter_users(df)
    # Calculate duration
    df=filter_users(df)
    # Calculate transitions within every session
    by_session=df.groupby('session')['timepress']
    df['transition']=by_session.apply(lambda group: transition(group))
    # First row in every session has no transition (NaN)
    df['transition'].fillna(0,inplace=True)
    # Drop old timing features
    df.drop(['timepress','timerelease'],axis=1,inplace=True)
    # Translate keynames
    keynames=init_keynames()
    df['keyname']=df['keyname'].apply(lambda x:keynames[x])
    # Save CSV
    df.to_csv('../data/00_villani.csv',index=False)

def init_categories():
    cons='bcdfghjklmnpqrstvwxyz'
    vowels='aeiou'
    letters=vowels+cons
    cons_cons=[c+d for c in cons for d in cons]
    vowel_vowel=[c+d for c in vowels for d in vowels]
    vowel_cons=[c+d for c in vowels for d in cons]
    cons_vowel=[c+d for c in cons for d in vowels]
    letter_space=[c+'space' for c in letters]
    space_letter=['space'+c for c in letters]
    categories={}
    # durations
    for c in 'eaoiu':
        categories[c]=[c,'vowels','all-letters']
    for c in 'tnsrh':
        categories[c]=[c,'freq-cons','all-letters']
    for c in 'ldcpf':
        categories[c]=[c,'next-freq-cons','all-letters']
    for c in 'mwybg':
        categories[c]=[c,'least-freq-cons','all-letters']
    for c in 'jkqvxz':
        categories[c]=['other','least-freq-cons','all-letters']
    for c in ['vowels','freq-cons','next-freq-cons','least-freq-cons']:
        categories[c]=[c,'all-letters']
    categories['other']=['other','least-freq-cons','all-letters']
    categories['space']=['space']
    # transitions
    for s in cons_cons:
        if s in ['th','st','nd']:
            categories[s]=[s,'cons-cons','letter-letter']
        else:
            categories[s]=['cons-cons','letter-letter']
    for s in vowel_vowel:
        if s=='ea':
            categories[s]=[s,'vowel-vowel','letter-letter']
        else:
            categories[s]=['vowel-vowel','letter-letter']
    for s in vowel_cons:
        if s in ['an','in','er','es','on','at','en']:
            categories[s]=[s,'vowel-cons','letter-letter']
        else:
            categories[s]=['vowel-cons','letter-letter']
    for s in cons_vowel:
        if s in ['he','re','ti']:
            categories[s]=[s,'cons-vowel','letter-letter']
        else:
            categories[s]=['cons-vowel','letter-letter']
    for s in ['cons-cons','vowel-vowel','vowel-cons','cons-vowel']:
        categories[s]=[s,'letter-letter']
    for s in letter_space:
        categories[s]=['letter-space']
    for s in space_letter:
        categories[s]=['space-letter']
    return categories

def init_features():
    names=list(init_categories().values())
    names=[f for sublist in names for f in sublist]
    features={}
    for c in set(names):
        features[c]=[]
    return features

def feature_names():
    features=[]
    features+=[['mean_'+f,'std_'+f] for f in list(init_features().keys())]
    features=[f for sublist in features for f in sublist]
    return features

def read_sample_transition(categories,features,line,key2):
    # read next key
    key1=key2
    key2=line[4]
    # read duration
    duration_categories=categories.get(key2)
    if duration_categories!=None:
        duration=float(line[-2])
        for c in duration_categories:
            features[c]+=[duration]
    # read transition
    transition_categories=categories.get(key1+key2)
    transition=float(line[-1])
    if transition!=0 and transition_categories!=None:
        for c in transition_categories:
            features[c]+=[transition]
    return key2

def process_features(feature_categories,features):
    names=feature_names()
    row=[]
    for i in range(0,len(names),2):
        name=names[i].split('_')[1]
        # find measurements,and use fallback if absent
        a=features[name]
        while len(a)==0:
            name=feature_categories[name][1]
            a=features[name]
        a=np.array(a)
        # remove outliers (more than 1.5 IQR removed from Q1 or Q3)
        q1=np.percentile(a,25)
        q3=np.percentile(a,75)
        iqr=q3 - q1
        a=a[q1 - 1.5*iqr <= a]
        a=a[a <= q3+1.5*iqr]
        # calculate feature value and store in row
        row+=[np.mean(a),np.std(a)]
    return row

def to_csv_baseline(p_in,p_out,headers_prefix,read_sample):
    # Init dictionaries
    categories=init_categories()
    # Open input and output files
    with open(p_in) as f_in,open(p_out,'w') as f_out:
        # Read input file into memory
        lines=f_in.readlines()
        # reverse so we can use pop and append for fast removal and insertion at end of list
        lines.reverse()
        # Skip header in input file
        lines.pop()
        # Create CSV writer
        writer=csv.writer(f_out)
        # Write header
        writer.writerow(headers_prefix+feature_names())
        # Write rows
        while True:
            row=read_sample(categories,lines)
            if row==[]:
                break
            writer.writerow(row) 

def read_sample_cmu(feature_categories,lines):
    # init feature map
    features=init_features()
    # determine if finished or not
    if lines==[]:
        return []
    # read subject,sessionIndex,screenIndex and inputType
    line=lines.pop().split(',')
    user,session,screen,inputtype=line[0],line[1],line[2],line[3]
    # collect durations and transitions
    key2=line[4]
    while line[2]==screen:
        key2=read_sample_transition(feature_categories,features,line,key2)
        # read next line if there is one
        if lines==[]:
            break
        line=lines.pop().split(',')
    # Push last line back
    if lines!=[]:
        lines.append(','.join(line))
    row=[user,session,screen,inputtype]
    row+=process_features(feature_categories,features)
    return row

def to_csv_baseline_cmu():
    # File names
    p_in=os.path.join('..','data','00_cmu.csv')
    p_out=os.path.join('..','data','01_cmu.csv')
    headers_prefix=['user','session','screen','inputtype']
    to_csv_baseline(p_in,p_out,headers_prefix,read_sample_cmu)

def read_sample_villani(feature_categories,lines):
    # init feature map
    features=init_features()
    # determine if finished or not
    if lines==[]:
        return []
    # read subject,sessionIndex,screenIndex and inputType
    line=lines.pop().split(',')
    user,session,inputtype,platform=line[0],line[1],line[2],line[3]
    # collect durations and transitions
    key2=line[4]
    while line[1]==session:
        key2=read_sample_transition(feature_categories,features,line,key2)
        # read next line if there is one
        if lines==[]:
            break
        line=lines.pop().split(',')
    # Push last line back of hold file
    if lines!=[]:
        lines.append(','.join(line))
    row=[user,session,inputtype,platform]
    row+=process_features(feature_categories,features)
    return row

def to_csv_baseline_villani():
    # File names
    p_in=os.path.join('..','data','00_villani.csv')
    p_out=os.path.join('..','data','01_villani.csv')
    headers_prefix=['user','session','inputtype','platform']
    to_csv_baseline(p_in,p_out,headers_prefix,read_sample_villani)

def init_categories():
    categories={}
    for c in 'eaoiutnsrh':
        categories[c]=c
    for c in 'ldcpfmwybg':
        categories[c]='mid_freq_cons'
    for c in 'jkqvxz':
        categories[c]='least_freq_cons'
    for s in ['space','shift','backspace']:
        categories[s]=s
    for s in ['.',',','`',';']:
        categories[s]='punctuation'
    for c in '0123456789':
        categories[c]='numbers'
    for s in ['numpad_'+d for d in list('0123456789')+['decimal']]+['numlock']:        
        categories[s]='non_letters_other'
    for s in ['=','(',')','acute','alt','\\','\'','command','ctrl','del','down','end','f12','f13',
             'home','insert','left','menu','-','page_down','page_up','right','/','unknown',
             'up','windows','enter','capslock','select','tab']:
        categories[s]='non_letters_other'
    return categories

def feature_engineering_local(f_in,f_out):
    df=pd.read_csv('../data/'+f_in)
    # remove location column from Tappert dataset
    if 'location' in df.columns:
        df.drop('location',axis=1,inplace=True)
    # replace keynames
    categories=init_categories()
    df['keyname']=df['keyname'].apply(lambda x:categories[x])
    # one-hot encode keynames
    df=one_hot_encode(df,'keyname',prefix='key')
    # save result
    df.to_csv('../data/'+f_out,index=False)

def save_data(X_train,y_train,X_test,y_test,name):
    # Write to files
    file=os.path.join('..','data',name)
    np.savez_compressed(file=file,X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)
    # Return shapes
    return X_train.shape,y_train.shape,X_test.shape,y_test.shape

def prepare_mixed(f_in,f_out,to_tensors,test_size=0.2,random_state=999):
    df=pd.read_csv('../data/'+f_in)
    X,y=to_tensors(df)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=random_state,stratify=y)    
    return save_data(X_train,y_train,X_test,y_test,name=f_out)

def prepare_ideal(f_in,f_out,to_tensors,inputtype,platform=None,test_size=0.2,random_state=999):
    df=pd.read_csv('../data/'+f_in)
    if platform==None:
        X,y=to_tensors(df[df['inputtype']==inputtype])
        f_out+='_'+inputtype
    else:
        X,y=to_tensors(df[(df['inputtype']==inputtype) & (df['platform']==platform)])
        f_out+='_'+inputtype+'_'+platform
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=random_state,stratify=y)
    return save_data(X_train,y_train,X_test,y_test,name=f_out)

def prepare_nonideal(f_in,f_out,to_tensors,inputtype,platform=None):
    df=pd.read_csv('../data/'+f_in)
    if platform==None:
        X_train,y_train=to_tensors(df[df['inputtype']==inputtype])
        X_test,y_test=to_tensors(df[df['inputtype']!=inputtype])
        f_out+='_'+inputtype
    else:
        X_train,y_train=to_tensors(df[(df['inputtype']==inputtype) & (df['platform']==platform)])
        X_test,y_test=to_tensors(df[(df['inputtype']!=inputtype) & (df['platform']!=platform)])
        f_out+='_'+inputtype+'_'+platform
    return save_data(X_train,y_train,X_test,y_test,name=f_out)

def to_tensors_baseline(df):
    # Replace user id's with integers
    df=df.copy()
    df['user']=pd.factorize(df['user'])[0]
    # Tensor dimensions
    n_samples=len(df)
    n_features=len(df.columns) - 4
    # Tensors for samples and labels
    X=np.empty((n_samples,n_features))
    y=np.empty((n_samples,),dtype='int')
    # Write rows
    for i in range(n_samples):
        X[i,:]=df.iloc[i,4:].as_matrix()
        y[i]=df.iloc[i,0]
    return X,y

def to_tensors_rnn(df,sample_cols):
    # Replace user id's with integers
    df=df.copy()
    df['user']=pd.factorize(df['user'])[0]
    # Tensor dimensions
    samples=df.groupby(sample_cols)
    n_samples=len(samples)
    n_timesteps=samples[sample_cols[-1]].count().max()
    n_features=len(df.columns) - 4
    # Tensors for samples and labels
    X=np.zeros((n_samples,n_timesteps,n_features)) # Zeros for padding timesteps
    y=np.empty((n_samples,),dtype='int')
    i=0
    for _,group in samples:
        steps=group.shape[0]
        X[i,:steps,:]=group.iloc[:,4:].as_matrix()
        y[i]=group['user'].iloc[0]        
        i+=1
    return X,y

def main():
    # Feature Engineering (1): Create single format for both datasets.
    to_csv_cmu()
    to_csv_villani()
    # Feature Engineering (2): Baseline model
    to_csv_baseline_cmu()
    to_csv_baseline_villani()
    # Feature Engineering (3): CRNN model
    feature_engineering_local('00_cmu.csv','02_cmu.csv')
    feature_engineering_local('00_villani.csv','02_villani.csv')
    # Convert baseline models to numpy arrays
    prepare_mixed('01_cmu.csv','baseline_cmu_mixed',to_tensors_baseline)
    prepare_mixed('01_villani.csv','baseline_villani_mixed',to_tensors_baseline)
    prepare_ideal('01_cmu.csv','baseline_cmu_ideal',to_tensors_baseline,'free')
    prepare_ideal('01_villani.csv','baseline_villani_ideal',to_tensors_baseline,'free','laptop')
    prepare_nonideal('01_cmu.csv','baseline_cmu_nonideal',to_tensors_baseline,'fixed')
    prepare_nonideal('01_villani.csv','baseline_villani_nonideal',to_tensors_baseline,'free','laptop')
    # Convert baseline models to numpy arrays
    to_tensors_rnn_cmu=lambda x:to_tensors_rnn(x,sample_cols=['user','session','screen'])
    to_tensors_rnn_villani=lambda x:to_tensors_rnn(x,sample_cols=['user','session'])
    prepare_mixed('02_cmu.csv','rnn_cmu_mixed',to_tensors_rnn_cmu)
    prepare_mixed('02_villani.csv','rnn_villani_mixed',to_tensors_rnn_villani)
    prepare_ideal('02_cmu.csv','rnn_cmu_ideal',to_tensors_rnn_cmu,'free')
    prepare_ideal('02_villani.csv','rnn_villani_ideal',to_tensors_rnn_villani,'free','laptop')
    prepare_nonideal('02_cmu.csv','rnn_cmu_nonideal',to_tensors_rnn_cmu,'fixed')
    prepare_nonideal('02_villani.csv','rnn_villani_nonideal',to_tensors_rnn_villani,'free','laptop')

if __name__=='__main__':
    main()
