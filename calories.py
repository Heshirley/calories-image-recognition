import streamlit as st
import os
from fastai.vision.all import *
import pathlib
import sys


# 根据不同的操作系统设置正确的pathlib.Path
if sys.platform == "win32":
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
else:
    temp = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath

# 获取当前文件所在的文件夹路径
path = os.path.dirname(os.path.abspath(__file__))

# 添加CSS样式
st.markdown(
    """
    <style>
        .img-classifier {
            text-align: center;
            background-color: #E0FFFF; /* 海洋蓝色 */
            border: 2px solid #40E0D0;
            border-radius: 10px;
            padding: 20px;
            margin: 20px auto;
        }
        .img-classifier h1 {
            color: #40E0D0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 开始定义带有class的div
st.markdown(
    '<div class="img-classifier">',
    unsafe_allow_html=True
)

# 在div内部添加所有内容
st.markdown("<h1>图像分类应用</h1>", unsafe_allow_html=True)
st.write("上传一张图片，应用将预测对应的标签。")

# 创建一个下拉菜单供用户选择模型
model_selection = st.selectbox('请选择要使用的模型', ('model1-resnet', 'model2-mobilenet_v3_large'))
# 根据用户的选择加载模型
if model_selection == 'model1-resnet':
    model_path = os.path.join(path, "model1.pkl")
elif model_selection == 'model2-mobilenet_v3_large':
    model_path = os.path.join(path, "model2.pkl")

# 加载模型
learn_inf = load_learner(model_path)

# 恢复pathlib.Path的原始值
if sys.platform == "win32":
    pathlib.PosixPath = temp
else:
    pathlib.WindowsPath = temp

# 允许用户上传图片
uploaded_file = st.file_uploader("选择一张照片...", type=["jpg", "jpeg", "png"])

# 如果用户已上传图片
if uploaded_file is not None:
    image = PILImage.create(uploaded_file)
    st.image(image, caption="上传的图片",use_column_width=True)
    
    # 假设learn_inf.predict()是你的预测函数
    pred, pred_idx, probs = learn_inf.predict(image)
    st.markdown(f"<h3>预测结果: {pred}; 概率: {probs[pred_idx]:.04f}</h3>", unsafe_allow_html=True)

# 结束div标签
st.markdown('</div>', unsafe_allow_html=True)


import streamlit as st
from litellm import completion

# Streamlit 应用程序界面
st.markdown(
    """
    <style>
        .translation-tool {
            background-color: #F0FFF0;
            border: 2px solid #008000;
            border-radius: 10px;
            padding: 20px;
            margin: 20px auto;
        }
        .translation-tool h1 {
            color: #008000;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="translation-tool">',
    unsafe_allow_html=True
)
st.markdown("<h1>菜品名中英文翻译器</h1>", unsafe_allow_html=True)

# 用户输入中文文本
input_text = st.text_input('请输入菜品', '')

# 当用户输入文本后, 进行翻译并显示结果
if st.button('点击翻译'):
    if input_text:
        # 使用 LiteLLM调用 Deepseek模型进行翻译
        response = completion(
            model='deepseek/deepseek-chat',
            messages=[
                {
                    "content": "你是一个优秀的翻译官, 请根据用户输入，判断所输入的语言，\
                    如果输入为中文，请依据英文读者的阅读习惯, 把内容原原本本翻译成对应的英文。\
                    如果输入为英文，请依据中文读者的阅读习惯，把内容原原本本翻译成对应的中文",
                    "role": "system"
                },
                {
                    "content": input_text,
                    "role": "user"
                }
            ]
        )
        translated_text = response['choices'][0]['message']['content']
        st.write('翻译结果:', translated_text)
    else:
        st.write('请输入正确文本后再进行翻译')
st.markdown('</div>', unsafe_allow_html=True)

import streamlit as st
from joblib import load

# 加载模型
model = load('random_forest_model.joblib')

# 加载TF-IDF矢量化器
tfidf_vectorizer = load('tfidf_vectorizer.joblib')

# 设置页面标题
st.markdown(
    """
    <style>
        .calorie-classifier {
            background-color: #FFFACD;
            border: 2px solid #DAA520;
            border-radius: 10px;
            padding: 20px;
            margin: 20px auto;
            text-align: right;
        }
        .calorie-classifier h1 {
            color: #DAA520;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="calorie-classifier">',
    unsafe_allow_html=True
)
st.markdown("<h1>食物卡路里水平分类器</h1>", unsafe_allow_html=True)

# 用户输入字段
food_name = st.text_input('请输入食物名称')

# 预测按钮
if st.button('预测'):
    # 使用TF-IDF向量化输入的文本
    input_vector = tfidf_vectorizer.transform([food_name])
    
    # 使用模型进行预测
    prediction = model.predict(input_vector)[0]
    
    # 输出预测结果
    st.write(f'预测的卡路里水平: {prediction}')
    # 描述卡路里等级的含义
    st.write("### 卡路里等级说明")
    st.write("我们采用以下标准对食物的卡路里含量进行分级：")
    st.markdown("""
    - **等级0 - 低卡路里**: 卡路里含量小于100。
    - **等级1 - 中卡路里**: 卡路里含量介于100到300之间。
    - **等级2 - 高卡路里**: 卡路里含量大于300。
    """)
st.markdown('</div>', unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import random

# 加载数据
low_cal_foods = pd.read_excel('Low_Calorie_Foods.xlsx')
medium_cal_foods = pd.read_excel('Medium_Calorie_Foods.xlsx')
high_cal_foods = pd.read_excel('High_Calorie_Foods.xlsx')

# 设置页面标题
st.markdown(
    """
    <style>
        .food-recommender {
            background-color: #191970;
            color: white;
            border: 2px solid #ADD8E6;
            border-radius: 10px;
            padding: 20px;
            margin: 20px auto;
            text-align: center;
        }
        .food-recommender h1 {
            color: #ADD8E6;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="food-recommender">',
    unsafe_allow_html=True
)
st.markdown("<h1>食物卡路里推荐器</h1>", unsafe_allow_html=True)

# 定义函数用于获取随机食物
def get_random_foods(food_df, num_foods):
    return food_df.sample(n=num_foods)

import streamlit as st

# ...

# 创建按钮和选择器
category = st.selectbox('选择食物类别', ['低卡路里', '中卡路里', '高卡路里'])

# 根据选择的食物类别创建滑块，并存储在session_state中
if 'num_foods' not in st.session_state:
    st.session_state.num_foods = 1  # 默认值

if category == '低卡路里':
    st.session_state.num_foods = st.slider('选择推荐的食物数量', 
                                           min_value=1, 
                                           max_value=len(low_cal_foods), 
                                           value=st.session_state.num_foods)
elif category == '中卡路里':
    st.session_state.num_foods = st.slider('选择推荐的食物数量', 
                                           min_value=1, 
                                           max_value=len(medium_cal_foods), 
                                           value=st.session_state.num_foods)
else:
    st.session_state.num_foods = st.slider('选择推荐的食物数量', 
                                           min_value=1, 
                                           max_value=len(high_cal_foods), 
                                           value=st.session_state.num_foods)

# ...

if st.button('获取推荐'):
    if category == '低卡路里':
        foods = get_random_foods(low_cal_foods, st.session_state.num_foods)
    elif category == '中卡路里':
        foods = get_random_foods(medium_cal_foods, st.session_state.num_foods)
    else:
        foods = get_random_foods(high_cal_foods, st.session_state.num_foods)
        
    st.table(foods[['Food', 'Calories']])
st.markdown('</div>', unsafe_allow_html=True)
