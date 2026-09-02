import os

import streamlit as st
import pandas as pd
import joblib

# 输入所有建模参数，顺序必须与 logistic_regression.pkl 的 feature_names_in_ 一致
vars = ["address", "grip_max", "IADL", "chronic_lung_diseases",
        "Sleep_time", "Pain", "Hope", "Hospital", "Retire", "UA"]

# 二分类变量的下拉选项
yes_no = {"No": 0, "Yes": 1}

# 每个变量对应的侧边栏控件配置
inputs = {
    "address": {"kind": "select", "label": "Address", "options": {"Urban": 1, "Rural": 2}},
    "grip_max": {"kind": "number", "label": "Grip max(kg)", "min": 0.0, "max": 100.0, "value": 26.0, "step": 0.1},
    "IADL": {"kind": "select", "label": "IADL", "options": yes_no},
    "chronic_lung_diseases": {"kind": "select", "label": "Chronic lung diseases", "options": yes_no},
    "Sleep_time": {"kind": "number", "label": "Sleep time(h)", "min": 0.0, "max": 24.0, "value": 6.0, "step": 0.5},
    "Pain": {"kind": "select", "label": "Pain", "options": yes_no},
    "Hope": {"kind": "select", "label": "Hope", "options": yes_no},
    "Hospital": {"kind": "select", "label": "Hospital visit in the past month", "options": yes_no},
    "Retire": {"kind": "select", "label": "Retire", "options": yes_no},
    "UA": {"kind": "number", "label": "UA(mg/dL)", "min": 0.0, "max": 20.0, "value": 4.5, "step": 0.1},
}

# 初始化 session_state 中的 data
# 创建一个空的DataFrame来存储预测数据
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame(columns=vars + ['Probability(%)', 'Label'])

# 在主页面上显示数据
st.header('Depression risk of adult patients with arthritis or rheumatism based on LR')

# 创建两列布局
left_column, col1, col2, col3, right_column = st.columns(5)

# 在左侧列中添加其他内容
left_column.write("")

# 在右侧列中显示图像
dirs = os.getcwd()

# 在右侧列中显示图像
right_column.image('./hospital.png', caption='', width=100)

# 创建一个侧边栏
st.sidebar.header('Input parameters')

# 按配置依次生成输入控件
values = {}
for v in vars:
    cfg = inputs[v]
    if cfg["kind"] == "select":
        choice = st.sidebar.selectbox(cfg["label"], list(cfg["options"].keys()))
        values[v] = cfg["options"][choice]
    else:
        values[v] = st.sidebar.number_input(cfg["label"], min_value=cfg["min"],
                                            max_value=cfg["max"], value=cfg["value"],
                                            step=cfg["step"])

# Unpickle classifier
mm = joblib.load('./logistic_regression.pkl')

# If button is pressed
if st.sidebar.button("Submit"):
    # Store inputs into dataframe
    X = pd.DataFrame([[values[v] for v in vars]], columns=vars)

    # Get prediction
    result_prob_pos = round(float(mm.predict_proba(X)[0][1]) * 100, 2)  # 预测概率

    # Output prediction
    st.text(f"The probability of LR is: {str(result_prob_pos)}%")

    # 创建一个新的DataFrame来存储用户输入的数据
    new_data = pd.DataFrame([[values[v] for v in vars] + [result_prob_pos, None]],
                            columns=st.session_state['data'].columns)

    # 将预测结果添加到新数据中
    st.session_state['data'] = pd.concat([st.session_state['data'], new_data], ignore_index=True)

# 上传文件按钮
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    # 读取 Excel 文件
    df = pd.read_excel(uploaded_file)

    # 列名映射字典,左为Excel字段，右为模型参数名
    column_mapping = {v: v for v in vars}

    # 假设 'Label' 列在 Excel 文件中存在并且不参与计算
    label_column = 'label'  # 这是 Excel 文件中未参与计算的列名

    # 进行列名映射
    df = df.rename(columns=column_mapping)

    # 检查是否所有必需的列都存在
    missing_cols = [col for col in vars if col not in df.columns]

    if missing_cols:
        st.error(f"Missing columns in the uploaded file: {', '.join(missing_cols)}")
    else:
        # 提取建模所需的列并进行预测
        X = df[vars]
        result_prob = (mm.predict_proba(X)[:, 1] * 100).round(2)

        # 获取标签列的值
        label = df[label_column] if label_column in df.columns else None

        # 将结果添加到 session_state 的 data 中
        new_data = X.copy()
        new_data['Probability(%)'] = result_prob
        new_data['Label'] = label if label is not None else None
        st.session_state['data'] = pd.concat([st.session_state['data'], new_data], ignore_index=True)

# 显示更新后的 data
st.write(st.session_state['data'])

# Footer

st.write(
    "<p style='font-size: 12px;'>Disclaimer: This mini app is designed to provide general information and is not a substitute for professional medical advice or diagnosis. Always consult with a qualified healthcare professional if you have any concerns about your health.</p>",
    unsafe_allow_html=True)
st.markdown('<div style="font-size: 12px; text-align: right;">Powered by MyLab+ i-Research Consulting Team</div>',
            unsafe_allow_html=True )
