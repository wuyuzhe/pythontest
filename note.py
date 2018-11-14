#######################################################	数据处理
check_random_state
turn seed into a np.random.RandomState instance
用法：
random_state = check_random_state(42)
index = random_state.randint(low=0, high=stats_c_[key], size=num_samples)

np.vstack  拼接数据增加行数np.vstack((a,b)) 将b数据追加到a的下面

train_test_split 将数据矩阵分成训练组和验证组

####################################################### 模型选择和训练
GradientBoostingRegressor  梯度渐进回归
GradientBoostingClassifier  梯度提升决策树
MLPClassifier   多层感知器分类器

clf.fit(x_train, y_train)
pre = clf.predict(x_test)
#平均绝对值误差
mse = mean_absolute_error(y_test, pre)

predict 返回一个大小为n的一维数组，一维数组中的第i个值为模型预测第i个预测样本的标签
predict_proba返回的是一个n行k列的数组，第i行第j列上的数值是模型预测第i个预测样本的标签为j的概率。此时每一行的和应该等于

####################################################### 模型保存
joblib  
#保存模型
joblib.dump(clf, 'hexin' + str(k) + '_data/clf_r_4.pkl')
#加载模型
clf = joblib.load("train_model.pkl")
clf.predict
######################################################## 误差计算
#均方差
mean_squared_error
#可释方差得分
explained_variance_score
#中值绝对误差
median_absolute_error