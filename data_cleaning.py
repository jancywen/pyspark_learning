# -*- coding: utf-8 -*-

"""
    数据清洗
"""


from pyspark.sql import SparkSession
import pyspark.sql.functions as fn

import pyspark.sql.types as typ
import matplotlib.pyplot as plt


spark = SparkSession.builder.getOrCreate()


'''
df = spark.createDataFrame([(1, 144.5, 5.9, 33, 'M'),
                            (2, 167.2, 5.4, 45, 'M'),
                            (3, 124.1, 5.2, 23, 'F'),
                            (4, 144.5, 5.9, 33, 'M'),
                            (5, 133.2, 5.7, 54, 'F'),
                            (3, 124.1, 5.2, 23, 'F'),
                            (5, 129.2, 5.3, 42, 'M')
                            ], ['id', 'weight', 'height', 'age', 'gender'])
# 去重
print('Count of ids: {}'.format(df.count()))
print('Count of distinct rows: {}'.format(df.distinct().count()))

df.distinct().show()

df = df.drop_duplicates()
df.show()

# 去除除 id 以外的相同行
print('Count of ids: {}'.format(df.count()))
print('Count of distinct ids: {}'.format(
    df.select([c for c in df.columns if c != 'id']).distinct().count()
))

df = df.drop_duplicates(subset=[
    c for c in df.columns if c != 'id'
])

df.show()


# 查看id的总数和id唯一数


df.agg(
    fn.count('id').alias('count'),
    fn.countDistinct('id').alias('distinct')
).show()

# 添加新id
df.withColumn('new_id', fn.monotonically_increasing_id()).show()

'''

'''
# 处理缺失值

df_miss = spark.createDataFrame([
    (1, 144.5, 5.9, 33, 'M', 10000),
    (2, 167.2, 5.4, 45, 'M', None),
    (3, None, 5.2, None, None, None),
    (4, 144.5, 5.9, 33, 'M', None),
    (5, 133.2, 5.7, 54, 'F', None),
    (6, 124.1, 5.2, None, 'F', None),
    (7, 129.2, 5.3, 42, 'M', 7600)
], ['id', 'weight', 'height', 'age', 'gender', 'income'])

# 每行缺失情况
statistics = df_miss.rdd.map(
    lambda row: (row['id'], sum([c == None for c in row]))
).collect()

print(statistics)

df_miss.where('id == 3').show()

# 每列缺失百分比
df_miss.agg(*[
    (1 - (fn.count(c)/fn.count('*'))).alias(c + '_missing') for c in df_miss.columns
]).show()


# 去除 'income' 特征
df_miss_no_income = df_miss.select([
    c for c in df_miss.columns if c != 'income'
])

# 去除缺失多的行
df_miss_no_income.dropna(thresh=3).show()


# 填充缺失的值
# 先创建一个带值的字典,然后把它传递给 .fillna(...)

means = df_miss_no_income.agg(
    *[fn.mean(c).alias(c) for c in df_miss_no_income.columns if c != 'gender']
).toPandas().to_dict('records')[0]
means['gender'] = 'missing'
df_miss_no_income.fillna(means).show()

'''

'''
# 离群值
df_outliers = spark.createDataFrame([(1, 144.5, 5.9, 33),
                            (2, 167.2, 5.4, 45),
                            (3, 342.1, 5.2, 99),
                            (4, 144.5, 5.9, 33),
                            (5, 133.2, 5.7, 54),
                            (6, 124.1, 5.2, 23),
                            (7, 129.2, 5.3, 42)
                            ], ['id', 'weight', 'height', 'age'])


# 计算特征截断点

cols = ['weight', 'height', 'age']
bounds = {}

for col in cols:
    quantiles = df_outliers.approxQuantile(
        col, [0.25, 0.75], 0.05
    )

    IQR = quantiles[1] - quantiles[0]

    bounds[col] = [
        quantiles[0] - 1.5 * IQR,
        quantiles[1] + 1.5 * IQR
    ]

print(bounds)

# 标记离散值
outliers = df_outliers.select(*['id'] + [
    (
        (df_outliers[c] < bounds[c][0]) |
        (df_outliers[c] > bounds[c][1])
    ).alias(c + '_o') for c in cols if c != 'id'
])

outliers.show()

# 打印离散值
df_outliers = df_outliers.join(outliers, on='id')
df_outliers.show()
df_outliers.filter('weight_o').select('id', 'weight').show()
df_outliers.filter('age_o').select('id', 'age').show()
'''


# 可视化数据

fraud = spark.sparkContext.textFile('./resources/ccFraud.csv')



header = fraud.first()

print(header)

fraud = fraud\
    .filter(lambda row: row != header)\
    .map(lambda row: [int(elem) for elem in row.split(',')])


fields = [
    *[
        typ.StructField(h[1:-1], typ.IntegerType(), True) for h in header.split(',')
    ]
]

schema = typ.StructType(fields)
fraud_df = spark.createDataFrame(fraud, schema)


fraud_df.printSchema()
fraud_df.show(5)
print(fraud_df.count())

fraud_df.groupBy('gender').count().show()

# 查看数值特征  .describe(...)
fraud_df.describe().show()

# 列表聚合函数


# 相关性 .corr(...)
# 构建相关矩阵
numerical = ['balance', 'numTrans', 'numIntlTrans']
n_numerical = len(numerical)
corr = []
for i in range(0, n_numerical):
    temp = [None] *1
    for j in range(i, n_numerical):
        temp.append(fraud_df.corr(numerical[i], numerical[j]))
    corr.append(temp)


plt.style.use('ggplot')

hists = fraud_df.select('balance').rdd.flatMap(
    lambda row: row
).histogram(20)

data = {
    'bins': hists[0][:-1],
    'freq': hists[1]
}
plt.bar(data['bins'], data['freq'], width=2000)
plt.title("Histogram of \'balance\'")
plt.show()
spark.stop()
