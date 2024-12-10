import pandas as pd
rt=input('enter root path')

### utrnet
utr_root='utrset/UTRSet-Real'
df1 = pd.read_csv(rt+utr_root+'/train/gt.txt', delimiter='.jpg', header=None)
df1[0] = utr_root+'/train/'+df1[0] + ".jpg"
#
df2 = pd.read_csv(rt+utr_root+'/test/gt.txt', delimiter='.jpg', header=None)
df2[0] = utr_root+'/test/'+df2[0] + ".jpg"
df2
#
#### nust
nus_root='nust-uhwr-dataset/DataSet/UHWR/UHWR/'
df3 = pd.read_csv(rt+nus_root+'/train.txt', delimiter='.jpg', header=None)
df3[0] = nus_root.replace('Dataset/','')+df3[0] + ".jpg"
#
df4 = pd.read_csv(rt+nus_root+'/test.txt', delimiter='.jpg', header=None)
df4[0] = nus_root.replace('Dataset/','')+df4[0] + ".jpg"
#
df5 = pd.read_csv(rt+nus_root+'/val.txt', delimiter='.jpg', header=None)
df5[0] = nus_root.replace('Dataset/','')+df5[0] + ".jpg"
#
#### upti
upt_root='upti-ocr/UPTI'
df6 = pd.read_csv(rt+upt_root+'/data/gt.txt', delimiter='.png', header=None)
df6[0] = upt_root+'/data/'+df6[0] + ".png"
df6
#
df7 = pd.read_csv(rt+upt_root+'/data_valid/gt.txt', delimiter='.png', header=None)
df7[0] = upt_root+'/data_valid/'+df7[0] + ".png"
df7
#

##### ghalib
glb_root='urdu-poetry-dataset/Urdu_poetry_ocr_dataset/'
df8=pd.read_csv(rt+glb_root+'annotations.csv',header=None)
df8=df8[df8[0].str.contains('Ghalib', na=False)]
df8.reset_index(drop=True, inplace=True)
df8[0] = glb_root+df8[0]
df8
#
### iqbal
iqb_root='iqbal-ocr/FYP_OCR_2024'
df9=pd.read_csv(rt+iqb_root+'/iqbal_gt_labels_v1.csv', delimiter=',',header=None)
df9.dropna(inplace=True)
df9.reset_index(drop=True, inplace=True)
df9[0] = df9[0].str.replace('/content/drive/MyDrive/FYP', iqb_root)
df9
#

#### unhd
und_root='chkhunhd/'
df10=pd.read_csv(rt+und_root+'data1.csv')
df10.columns = [1, 0]
df10[0]=df10[0].str.replace(rt, '')
df10
###
#### salook
slk_root='salook-sulemani/salook_sulemani/'
df11 = pd.read_csv(rt+slk_root+'gt_labels.txt', delimiter='.jpg', header=None)
df11[0] = slk_root+'rois/'+df11[0]+ '.jpg'
df11
###
### concatenate
df = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11], ignore_index=True)
df.columns = ['file_name', 'text']
#df

fcheck=[os.path.exists(rt+x) for x in df['file_name']]
df['check']=fcheck
df=df[df['check']==True]
df.reset_index(drop=True, inplace=True)
df = df.drop('check', axis=1)
df
