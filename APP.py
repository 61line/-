import datetime
import json
import os
import random
from flask import Flask, send_file, jsonify, request
import joblib
import numpy as np
import pandas
import warnings

warnings.filterwarnings('ignore')

df = pandas.read_csv('./Stroke_data.csv')


# 因为用的是vue框架是{{}}语法, 与flask的jinja2模板语法冲突所以不能使用render渲染方法返回数据
def sendTemp(name: str):
    return send_file(os.path.join(os.getcwd(), 'templates', name))


# 创建一个APP
app = Flask(__name__, static_url_path='/static')


# home页面
@app.route('/')
def loginHtml():
    return sendTemp('login.html')


@app.route('/home')
def homeHtml():
    return sendTemp('home.html')


@app.route('/user/login', methods=['POST'])
def userLogin():
    data = request.get_json()
    print(data)
    if data['username'] == 'admin' and data['password'] == 'admin':
        return jsonify(code=200)
    else:
        return jsonify(code=500, msg='请检查用户名密码')


@app.route('/user/page')
def userPage():
    pageSize = int(request.args.get('pageSize', 10))
    pageNum = int(request.args.get('pageNum', 1))
    dd = df.iloc[(pageNum - 1) * pageSize: pageNum * pageSize]
    dd.fillna("缺失值", inplace=True)
    return jsonify(dict(
        data=dd.to_dict('records'),
        count=int(str(df['id'].count()))
    ))


@app.route('/model')
def model():
    age = float(request.args.get('age'))
    heart_disease = int(request.args.get('heart_disease'))
    avg_glucose_level = float(request.args.get('avg_glucose_level'))
    hypertension = int(request.args.get('hypertension'))
    ever_married = int(request.args.get('ever_married'))
    modelA = joblib.load('随机森林--test.pickle')
    result = modelA.predict(
        np.array([age, heart_disease, avg_glucose_level, hypertension, ever_married]).reshape(1, -1))[0]

    if result >= 0.5:
        output = f"{int(result * 100)}%的概率会患有脑卒中,建议医嘱预防"
    else:
        output = "未患有"

    return jsonify(code=200, data=output)


if __name__ == '__main__':
    app.run(debug=True)
