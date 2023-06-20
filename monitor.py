import streamlit as st
import pandas as pd
import altair as alt
import json
from flask import Flask
import psutil
import requests
from sklearn.tree import DecisionTreeClassifier
import random
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import schedule
import subprocess

import threading


import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

import numpy as np
import plotly.graph_objs as go

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

num_lines = 5
container_name = "ali-tlg"
keyword = "16434"
webhook_url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=3ad387e5-f071-4603-982f-a1c3bf11a483'
begin_schedule = True

# 共享变量锁
send_notification_to_wechat = 0

def loop_verify(clf):
    print("********************************开始预警*******************************************")
    resources = calculate_resources()

    predict_data = pd.DataFrame({
        'CPU': [resources[0]],
        'Memory': [resources[1]],
        'Disk': [resources[2]],
        'Status': [resources[3]]
    })

    # 使用已训练的决策树模型进行预测
    predictions = clf.predict(predict_data)

    # 添加预测结果
    predict_data['Result'] = predictions
    print(predict_data)

    global send_notification_to_wechat
    print("^^^", send_notification_to_wechat)

    for i in predict_data['Result']:
        if i == 1:
            # TODO: 使用缓存技术，将发送的消息存储到mysql中，每30秒发送一次，如果一样，则掠过。
            print('报警-发送警告信息')


            # send_alert(resources)
        else:
            print('不报警-等待下轮预测')

    print("********************************结束预警*******************************************")
    time.sleep(1)


def calculate_resources():
    resource = []

    # CPU
    cpu_usage = psutil.cpu_percent()
    print("CPU使用率:", cpu_usage)

    # Memory
    memory = psutil.virtual_memory()
    print("已使用内存:", int(memory.used / memory.total * 100))
    memory_usage = int(memory.used / memory.total * 100)

    # Disk
    disk = psutil.disk_usage('/')
    print("已使用磁盘空间:", int(disk.used / disk.total * 100))
    disk_usage = int(disk.used / disk.total * 100)

    # Status
    command = f"docker logs -n {num_lines} {container_name}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output, _ = process.communicate()

    # 将输出按行拆分
    logs = output.decode().split('\n')

    # 过滤日志并打印匹配的行
    status = 0
    for log in logs:
        if keyword in log:
            status = 1

    resource.append(cpu_usage)
    resource.append(memory_usage)
    resource.append(disk_usage)
    resource.append(status)

    return resource

def send_alert(resources):
    msg = f'CPU {resources[0]}; Memory: {resources[1]}; Disk: {resources[2]}; Status: {resources[3]}'

    payload = json.dumps({
        'msgtype':'text',
        'text': {
            "content": msg,
            "mentioned_list": ["@all"],
        }
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", webhook_url, headers=headers, data=payload)
    if response.status_code == 200:
        print('消息发送成功')
    else:
        print('消息发送失败')

# 返回模型
def build_model():
    data = []

    for i in range(50):
        item = []
        # utc_time = datetime.datetime.utcfromtimestamp(time.time())
        # item.append(utc_time.strftime('%Y%m%d%H%M%S'))
        item.append(random.randint(0, 100))
        item.append(random.randint(0, 100))
        item.append(random.randint(0, 100))
        item.append(random.randint(0, 1))
        data.append(item)

    df = pd.DataFrame(data, columns=['CPU', 'Memory', 'Disk', 'Status'])
    df['Result'] = df.apply(lambda row: 1 if row['CPU'] > 80 or row['Memory'] > 80 or row['Status'] == 1 else 0, axis=1)

    print(df)
    # 将特征列作为X
    X = df[['CPU', 'Memory', 'Disk', 'Status']]

    # 将目标列作为Y
    Y = df['Result']

    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    # 创建决策树模型
    clf = DecisionTreeClassifier()

    # 训练模型
    clf.fit(X_train, y_train)
    return clf


def schedule_loop():
    clf = build_model()
    while True:
        loop_verify(clf)
        time.sleep(5)

def init_program():
    # 创建定时任务的线程
    schedule_thread = threading.Thread(target=lambda: schedule_loop())
    schedule_thread.start()


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H3("智能AI服务预警系统", className="display-4"),
        html.Hr(),
        html.P(
            "智能AI服务预警系统主要是为我们现有的客户组件提供分析预警服务，并在不改变现有组件功能基础上，该服务预警系统能够独立提供日志的收集，分析以及预测等功能，并根据纳入的算法模型，向用户提供智能化的信息预警服务。", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Page 1", href="/page-1", active="exact"),
                dbc.NavLink("Page 2", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        # return html.P("This is the content of the home page!")
        return dbc.Container(
            [
                dcc.Store(id="store"),
                html.H1("Dynamically rendered tab content"),
                html.Hr(),
                dbc.Button(
                    "Regenerate graphs",
                    color="primary",
                    id="button",
                    className="mb-3",
                ),
                dbc.Tabs(
                    [
                        dbc.Tab(label="Scatter", tab_id="scatter"),
                        dbc.Tab(label="Histograms", tab_id="histogram"),
                    ],
                    id="tabs",
                    active_tab="scatter",
                ),
                html.Div(id="tab-content", className="p-4"),
            ]
        )
    elif pathname == "/page-1":
        return html.P("This is the content of page 1. Yay!")
    elif pathname == "/page-2":
        return html.P("Oh cool, this is page 2!")
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )

@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"), Input("store", "data")],
)
def render_tab_content(active_tab, data):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if active_tab and data is not None:
        if active_tab == "scatter":
            return dcc.Graph(figure=data["scatter"])
        elif active_tab == "histogram":
            return dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=data["hist_1"]), width=6),
                    dbc.Col(dcc.Graph(figure=data["hist_2"]), width=6),
                ]
            )
    return "No tab selected"

@app.callback(Output("store", "data"), [Input("button", "n_clicks")])
def generate_graphs(n):
    """
    This callback generates three simple graphs from random data.
    """
    if not n:
        # generate empty graphs when app loads
        return {k: go.Figure(data=[]) for k in ["scatter", "hist_1", "hist_2"]}

    # simulate expensive graph generation process
    time.sleep(2)

    # generate 100 multivariate normal samples
    data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)

    scatter = go.Figure(
        data=[go.Scatter(x=data[:, 0], y=data[:, 1], mode="markers")]
    )
    hist_1 = go.Figure(data=[go.Histogram(x=data[:, 0])])
    hist_2 = go.Figure(data=[go.Histogram(x=data[:, 1])])

    # save figures in a dictionary for sending to the dcc.Store
    return {"scatter": scatter, "hist_1": hist_1, "hist_2": hist_2}


if __name__ == '__main__':
    # 初始化定时任务
    # init_program()
    app.run_server()






