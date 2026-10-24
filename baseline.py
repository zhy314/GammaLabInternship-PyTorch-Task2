"""Python 标准库 kNN 教学基线；只读公开训练数据。运行 python baseline.py。"""
import csv,math,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
with open(DATA/'train.csv',encoding='utf-8-sig') as f:train=list(csv.DictReader(f))
with open(DATA/'test.csv',encoding='utf-8-sig') as f:test=list(csv.DictReader(f))
FEATURES=['f1', 'f2', 'f3', 'f4']
REGRESSION=True
# 只用训练集估计缺失值填补与标准化参数。
def value(row,k):
 try:return float(row[k])
 except (ValueError,TypeError):return None
means={k:sum(value(r,k) for r in train if value(r,k) is not None)/max(1,sum(value(r,k) is not None for r in train)) for k in FEATURES}
scales={k:max(1e-8,math.sqrt(sum(((value(r,k) if value(r,k) is not None else means[k])-means[k])**2 for r in train)/len(train))) for k in FEATURES}
def vec(row):return [((value(row,k) if value(row,k) is not None else means[k])-means[k])/scales[k] for k in FEATURES]
x=[vec(r) for r in train]
# 自己划分训练/验证集调参；不要根据榜单反复猜测隐藏标签。
with open(ROOT/'submission.csv','w',newline='',encoding='utf-8') as f:
 writer=csv.writer(f);writer.writerow(['id','prediction'])
 for row in test:
  v=vec(row);near=sorted(range(len(train)),key=lambda i:sum((a-b)**2 for a,b in zip(x[i],v)))[:5]
  ys=[train[i]['target'] for i in near]
  pred=sum(map(float,ys))/len(ys) if REGRESSION else collections.Counter(ys).most_common(1)[0][0]
  writer.writerow([row['id'],pred])
print('已生成 submission.csv；可上传平台。')
