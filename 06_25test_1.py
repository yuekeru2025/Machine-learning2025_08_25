# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 11:36:34 2025

@author: yueke
"""



import os
from urllib.request import urlretrieve
import seaborn as sns
from seaborn.utils import get_dataset_names  # 添加这行导入

def load_dataset(name, cache=True, data_home=None, **kws):
    path = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/{}.csv"
    full_path = path.format(name)

    if cache:
        cache_path = os.path.join(get_data_home(data_home),
                    os.path.basename(full_path))
        if not os.path.exists(cache_path):
            if name not in get_dataset_names():  # 现在这个函数已定义
                raise ValueError(f"'{name}' is not one of the example datasets.")
            urlretrieve(full_path, cache_path)
        full_path = cache_path
    
    # 返回数据集（实际使用时可能需要用pandas读取）
    return full_path

def get_data_home(data_home=None):
    if data_home is None:
        data_home = os.environ.get('SEABORN_DATA',
                     os.path.join('~', 'seaborn-data'))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home

# 使用示例
dataset = sns.load_dataset(name="iris", cache=True, data_home="./seaborn-data")
print(dataset.head())


# 自己打代码
#显示代码
import streamlit as st
import seaborn as sns
import plotly.express as px
st.title("Welcome to the word of :red[streamlit] :sunglasses:")
st.header("Pandas DataFrame")
st.markdown("Load :blue[Iris Data Set]")
df = sns.load_dataset('iris')
st.dataframe(df)
st.header("Visualize Using Heatmap")
fig = px.imshow(df.iloc[:,:-1])
st.plotly_chart(fig)


#可视化
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
x1 = np.linspace(-3,3,301)
x2 = np.linspace(-3,3,301)
xx1,xx2 = np.meshgrid(x1,x2)#二维网格坐标
ff = xx1 * np.exp(-xx1 ** 2 -xx2 **2)#二元函数的曲面数据
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111,projection = '3d')
ax.plot_wireframe(xx1,xx2,ff,rstride = 10,cstride = 10)
st.write(fig)

fig = go.Figure(data = [go.Surface(z = ff, x = xx1, y = xx2, colorscale = 'RdYlBu')])
st.write(fig)


#输入工具
import streamlit as st
st.button('Click me')
st.checkbox('Ture or False')
st.radio('Choose one:',['Option_1','Option_2','Option_3'])
st.multiselect('Choose many:',["A","B","C"])
st.slider("select one:",0.0, 10.0, 5.0, 0.01)
st.text_input("Enter your name")
st.text_area("Enter your message")
st.number_input("Select a number",50,100,75)
st.date_input("Select a date")
st.time_input("Select a time")
st.file_uploader("Upload a file")
st.color_picker("pick a color")
st.selectbox("Select one",['Option_1','Option_2','Option_3'])
st.select_slider("Select a value",[1,2,3,4,5])


#侧边栏
import streamlit as st
import numpy as np
from sympy import symbols,lambdify
import matplotlib.pyplot as plt
with st.sidebar:
    st.header("Choose coefficients")
    st.latex("f(x) = ax^2 + bx + c")
    a = st.slider("a",-5.00, 5.00, 1.00, 0.01)
    b = st.slider("b",-5.00, 5.00, -2.00, 0.01)
    c = st.slider("c",-5.00, 5.00, -3.00, 0.01)
    
#抛物线
x = symbols('x')
f_x = a*x**2 + b*x + c
x_array = np.linspace(-5,5,301)

# 主页面
st.title("Qudratic function")
st.latex(r'f(x)=')
st.latex(f_x)
f_x_fcn = lambdify(x,f_x)
y = f_x_fcn(x_array)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111)
ax.set_aspect('equal',adjustable = 'box')
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.grid(True)
ax.plot(x_array,y)
st.write(fig)

#多列布局
import streamlit as st
col1,col2 = st.columns(2)
with col1:
    st.header("col1")
    st.subheader("this is the col1")
with col2:
    st.header("col2")
    st.subheader("this is the col2")
    

#多选项卡
tabA,tabB = st.tabs(["Tab_A","Tab_B"])
tabA.header("Tab_A")
tabB.header("Tab_B")
tabA.write("this is the tabA")


#可展开区域
import streamlit as st
import seaborn as sns
import plotly.express as px
df = sns.load_dataset('iris')
with st.expander("Open and View Iris Set"):
    df = sns.load_dataset('iris')
    st.write(df)

with st.expander("Open and View Heatmap"):
    fig = px.imshow(df.iloc[:,:-1])
    st.plotly_chart(fig,key = 'unique_heatmap2')


























































